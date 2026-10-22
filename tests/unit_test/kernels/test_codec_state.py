# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange
from sglang_omni.platforms import current_platform

REQUIRES_CUDA = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the exchange kernels run on CUDA"
)


def history_exchange_reference(
    padded: torch.Tensor,
    history_pool: torch.Tensor,
    slot_index: torch.Tensor,
    new_samples: int,
    is_cold: bool,
) -> None:
    history_samples = int(history_pool.shape[1])
    if is_cold:
        padded[:, :history_samples].zero_()
    else:
        padded[:, :history_samples].copy_(history_pool.index_select(0, slot_index))
    padded[:, history_samples + new_samples :].zero_()
    history_pool.index_copy_(
        0, slot_index, padded[:, new_samples : new_samples + history_samples]
    )


def overlap_exchange_reference(
    output: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    emitted_samples: int,
    is_cold: bool,
) -> None:
    overlap_samples = int(overlap_pool.shape[1])
    if is_cold:
        pass
    else:
        output[:, :overlap_samples] += overlap_pool.index_select(0, slot_index)
    overlap_pool.index_copy_(
        0, slot_index, output[:, emitted_samples : emitted_samples + overlap_samples]
    )


@pytest.mark.accelerator
@REQUIRES_CUDA
@pytest.mark.parametrize("history_samples", [2, 6, 54])
@pytest.mark.parametrize("chunks", [(10, 10, 7), (1, 2, 1, 3, 40), (32, 1, 32)])
def test_history_exchange_builds_the_causal_window_of_the_whole_stream(
    history_samples: int, chunks: tuple[int, ...]
) -> None:
    torch.manual_seed(history_samples + len(chunks))
    slots, channels = 5, 3
    slot_index = torch.tensor([3, 1], device="cuda")
    history_pool = torch.randn(slots, history_samples, channels, device="cuda")
    untouched = history_pool[[0, 2, 4]].clone()
    stream = torch.randn(2, sum(chunks), channels, device="cuda")
    whole = F.pad(stream, (0, 0, history_samples, 0))
    start = 0
    for index, new_samples in enumerate(chunks):
        padded = torch.full(
            (2, history_samples + new_samples + 2, channels), 7.0, device="cuda"
        )
        padded[:, history_samples : history_samples + new_samples] = stream[
            :, start : start + new_samples
        ]

        history_exchange(
            padded,
            history_pool,
            slot_index,
            new_samples=new_samples,
            is_cold=index == 0,
        )

        torch.testing.assert_close(
            padded[:, : history_samples + new_samples],
            whole[:, start : start + history_samples + new_samples],
            rtol=0,
            atol=0,
        )
        assert not padded[:, history_samples + new_samples :].any()
        start += new_samples
    torch.testing.assert_close(
        history_pool[slot_index], whole[:, -history_samples:], rtol=0, atol=0
    )
    torch.testing.assert_close(history_pool[[0, 2, 4]], untouched, rtol=0, atol=0)


@pytest.mark.accelerator
@REQUIRES_CUDA
@pytest.mark.parametrize("stride", [3, 5, 8])
@pytest.mark.parametrize("chunks", [(4, 4, 2), (1, 1, 1, 5)])
def test_overlap_exchange_streams_the_whole_transposed_conv(
    monkeypatch: pytest.MonkeyPatch, stride: int, chunks: tuple[int, ...]
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(stride + len(chunks))
    weight = torch.randn(4, 3, 2 * stride, device="cuda")
    inputs = torch.randn(2, 4, sum(chunks), device="cuda")
    whole = F.conv_transpose1d(inputs, weight, stride=stride)
    overlap_pool = torch.randn(4, stride, 3, device="cuda")
    slot_index = torch.tensor([2, 0], device="cuda")
    emitted = []
    start = 0
    for index, frames in enumerate(chunks):
        output = F.conv_transpose1d(
            inputs[..., start : start + frames], weight, stride=stride
        ).transpose(1, 2)

        overlap_exchange(
            output,
            overlap_pool,
            slot_index,
            emitted_samples=frames * stride,
            is_cold=index == 0,
        )

        emitted.append(output[:, : frames * stride])
        start += frames
    torch.testing.assert_close(
        torch.cat(emitted, dim=1).transpose(1, 2), whole[..., : start * stride]
    )


@pytest.mark.accelerator
@REQUIRES_CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_channels_first", [False, True])
@pytest.mark.parametrize("is_cold", [False, True])
@pytest.mark.parametrize(
    "history_samples,new_samples,tail_samples",
    [
        (2, 1, 0),
        (6, 2, 0),
        (6, 4, 0),
        (6, 40, 2),
        (18, 320, 0),
        (54, 32, 2),
        (54, 320, 8),
    ],
)
def test_history_exchange_kernel_equals_the_reference(
    dtype: torch.dtype,
    is_channels_first: bool,
    is_cold: bool,
    history_samples: int,
    new_samples: int,
    tail_samples: int,
) -> None:
    torch.manual_seed(history_samples * new_samples)
    channels = 200
    slot_index = torch.tensor([4, 1, 6], device="cuda")
    history_pool = torch.randn(8, history_samples, channels, device="cuda").to(dtype)
    samples = history_samples + new_samples + tail_samples
    if is_channels_first:
        padded = (
            torch.randn(3, channels, samples, device="cuda").to(dtype).transpose(1, 2)
        )
    else:
        padded = torch.randn(3, samples, channels, device="cuda").to(dtype)
    expected_padded, expected_pool = padded.clone(), history_pool.clone()

    history_exchange_reference(
        expected_padded, expected_pool, slot_index, new_samples, is_cold
    )
    history_exchange(padded, history_pool, slot_index, new_samples, is_cold)

    assert torch.equal(padded, expected_padded)
    assert torch.equal(history_pool, expected_pool)


@pytest.mark.accelerator
@REQUIRES_CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_channels_first", [False, True])
@pytest.mark.parametrize("is_cold", [False, True])
@pytest.mark.parametrize(
    "overlap_samples,emitted_samples", [(3, 3), (4, 640), (5, 160), (8, 32), (8, 320)]
)
def test_overlap_exchange_kernel_equals_the_reference(
    dtype: torch.dtype,
    is_channels_first: bool,
    is_cold: bool,
    overlap_samples: int,
    emitted_samples: int,
) -> None:
    torch.manual_seed(overlap_samples * emitted_samples)
    channels = 200
    slot_index = torch.tensor([4, 1, 6], device="cuda")
    overlap_pool = torch.randn(8, overlap_samples, channels, device="cuda").to(dtype)
    samples = emitted_samples + overlap_samples
    if is_channels_first:
        output = (
            torch.randn(3, channels, samples, device="cuda").to(dtype).transpose(1, 2)
        )
    else:
        output = torch.randn(3, samples, channels, device="cuda").to(dtype)
    expected_output, expected_pool = output.clone(), overlap_pool.clone()

    overlap_exchange_reference(
        expected_output, expected_pool, slot_index, emitted_samples, is_cold
    )
    overlap_exchange(output, overlap_pool, slot_index, emitted_samples, is_cold)

    assert torch.equal(output, expected_output)
    assert torch.equal(overlap_pool, expected_pool)
