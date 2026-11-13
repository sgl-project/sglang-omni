# SPDX-License-Identifier: Apache-2.0
"""The fused codebook heads draw what the codebook sampler draws from the same
logits, and their logits are the head projection's up to bf16 rounding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.fishaudio_drama3.audio_decoder import CODEBOOK_TOP_K
from sglang_omni.models.fishaudio_drama3.head_sampling import (
    HeadSampling,
    head_has_forward_hooks,
    head_sampling_unsupported_reason,
)
from sglang_omni.models.fishaudio_drama3.sampling import MINIMUM_TEMPERATURE
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_CODEBOOK_HEAD_STREAM,
    CodebookSampler,
    sample_codebook_head,
)
from sglang_omni.models.fishaudio_drama3.tower_tiles import MAX_FUSED_ROWS
from sglang_omni.platforms import current_platform
from tests.unit_test.fishaudio_drama3.test_tower_kernels import (  # noqa: F401
    published_server_args,
    require_validated_gpu,
)

CODEBOOK_SIZE = 4096
DIM = 2048
accelerator = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="the fused codebook heads run on CUDA only",
)


def test_hook_detection_keeps_the_eager_path_when_torch_changes() -> None:
    assert head_has_forward_hooks(SimpleNamespace()) is True
    assert head_has_forward_hooks(SimpleNamespace(_forward_hooks={})) is False
    assert head_has_forward_hooks(SimpleNamespace(_forward_hooks={"probe": 1})) is True


def test_hook_detection_sees_a_real_forward_hook() -> None:
    head = nn.Linear(2, 2)
    assert head_has_forward_hooks(head) is False
    handle = head.register_forward_hook(lambda module, inputs, output: output)
    assert head_has_forward_hooks(head) is True
    handle.remove()
    assert head_has_forward_hooks(head) is False


def random_head(device: torch.device, seed: int) -> nn.Linear:
    head = nn.Linear(
        DIM, CODEBOOK_SIZE, bias=False, device=device, dtype=torch.bfloat16
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    head.weight.data.copy_(
        0.05 * torch.randn(head.weight.shape, device=device, generator=generator)
    )
    return head


def head_sampling(device: torch.device) -> HeadSampling:
    return HeadSampling.allocate(
        codebook_size=CODEBOOK_SIZE,
        dim=DIM,
        top_k_limit=CODEBOOK_TOP_K,
        minimum_temperature=MINIMUM_TEMPERATURE,
        device=device,
    )


def sampling_parameters(
    row_count: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = 0.3 + torch.rand(row_count, device=device, generator=generator)
    top_k = torch.randint(-1, 80, (row_count,), device=device, generator=generator)
    top_p = 0.3 + 0.7 * torch.rand(row_count, device=device, generator=generator)
    return temperature, top_k, top_p


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", [1, 5, MAX_FUSED_ROWS])
def test_fused_heads_draw_what_the_codebook_sampler_draws_from_their_logits(
    row_count: int,
) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    fused = head_sampling(device)
    head = random_head(device, seed=row_count)
    sampler = CodebookSampler(
        step=torch.zeros((), dtype=torch.int64, device=device), seed=4242
    )
    generator = torch.Generator(device=device).manual_seed(100 + row_count)
    frames = torch.full((row_count, 11), -1, dtype=torch.int64, device=device)
    with torch.no_grad():
        for step in range(40):
            sampler.step.fill_(step)
            hidden = (1 + step % 4) * torch.randn(
                (row_count, DIM), device=device, generator=generator
            ).bfloat16()
            temperature, top_k, top_p = sampling_parameters(
                row_count, generator, device
            )
            stream_id = FIRST_CODEBOOK_HEAD_STREAM + step % 9
            out = frames[:, 1 + step % 9]
            logits = fused.sample(
                head=head,
                hidden_states=hidden,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sampler=sampler,
                stream_id=stream_id,
                out=out,
            ).clone()
            expected = sample_codebook_head(
                logits,
                temperature,
                top_k,
                top_p,
                sampler,
                stream_id,
                MINIMUM_TEMPERATURE,
                top_k_limit=CODEBOOK_TOP_K,
            )
            assert torch.equal(out, expected), step
            plain = head(hidden).float()
            # One bf16 rounding of each logit, plus fp32 reordering on the row's scale.
            tolerance = (
                plain.abs() * 2.0**-7 + plain.abs().amax(dim=1, keepdim=True) * 2.0**-10
            )
            assert torch.all((logits.float() - plain).abs() <= tolerance), step
            assert not torch.any(fused.counters), step


@accelerator
@pytest.mark.accelerator
def test_a_captured_fused_head_replays_the_eager_codes() -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    fused = head_sampling(device)
    head = random_head(device, seed=3)
    row_count = 6
    sampler = CodebookSampler(
        step=torch.zeros((), dtype=torch.int64, device=device), seed=77
    )
    generator = torch.Generator(device=device).manual_seed(5)
    hidden = torch.randn(
        (row_count, DIM), device=device, generator=generator
    ).bfloat16()
    temperature, top_k, top_p = sampling_parameters(row_count, generator, device)
    out = torch.zeros(row_count, dtype=torch.int64, device=device)

    def draw() -> None:
        fused.sample(
            head=head,
            hidden_states=hidden,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sampler=sampler,
            stream_id=FIRST_CODEBOOK_HEAD_STREAM,
            out=out,
        )

    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            draw()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            draw()
        for step in range(5):
            sampler.step.fill_(step)
            hidden.copy_(
                torch.randn((row_count, DIM), device=device, generator=generator)
            )
            graph.replay()
            torch.cuda.synchronize()
            replayed = out.clone()
            draw()
            assert torch.equal(replayed, out), step


@pytest.mark.usefixtures("published_server_args")
def test_the_resolver_needs_a_cap_within_the_selected_candidates() -> None:
    if not torch.cuda.is_available() or not current_platform.is_cuda():
        pytest.skip("needs a CUDA device")
    else:
        pass
    require_validated_gpu()
    device = torch.device("cuda")
    head = random_head(device, seed=0)
    assert (
        head_sampling_unsupported_reason(
            heads=[head], top_k_limit=CODEBOOK_TOP_K, device=device
        )
        is None
    )
    assert "top-k cap" in str(
        head_sampling_unsupported_reason(heads=[head], top_k_limit=64, device=device)
    )
    biased = nn.Linear(DIM, CODEBOOK_SIZE, device=device, dtype=torch.bfloat16)
    assert "bias-free" in str(
        head_sampling_unsupported_reason(
            heads=[biased], top_k_limit=CODEBOOK_TOP_K, device=device
        )
    )
    small = nn.Linear(DIM, 64, bias=False, device=device, dtype=torch.bfloat16)
    assert "does not fit" in str(
        head_sampling_unsupported_reason(
            heads=[small], top_k_limit=CODEBOOK_TOP_K, device=device
        )
    )
