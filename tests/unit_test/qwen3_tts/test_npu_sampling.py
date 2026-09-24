# SPDX-License-Identifier: Apache-2.0
"""Coverage for the Qwen3-TTS NPU seeded sampling path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sglang_omni.platforms import current_platform

_MODULE_PATH = (
    Path(__file__).parents[3] / "sglang_omni/models/qwen3_tts/sampling_kernels.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen3_tts_sampling_kernels", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
sampling_kernels = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sampling_kernels)


_UINT32_MASK = 0xFFFFFFFF


def _npu_available() -> bool:
    return current_platform.is_npu()


def _rotl32(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & _UINT32_MASK


def _mix32(hash_value: int, key: int) -> int:
    key = (key * 0xCC9E2D51) & _UINT32_MASK
    key = _rotl32(key, 15)
    key = (key * 0x1B873593) & _UINT32_MASK
    hash_value ^= key
    hash_value = _rotl32(hash_value, 13)
    return (hash_value * 5 + 0xE6546B64) & _UINT32_MASK


def _fmix32(hash_value: int) -> int:
    hash_value ^= hash_value >> 16
    hash_value = (hash_value * 0x85EBCA6B) & _UINT32_MASK
    hash_value ^= hash_value >> 13
    hash_value = (hash_value * 0xC2B2AE35) & _UINT32_MASK
    return (hash_value ^ (hash_value >> 16)) & _UINT32_MASK


def _reference_hash(seed: int, position: int, column: int) -> int:
    seed &= 0xFFFFFFFFFFFFFFFF
    hash_value = 0
    hash_value = _mix32(hash_value, seed & _UINT32_MASK)
    hash_value = _mix32(hash_value, (seed >> 32) & _UINT32_MASK)
    hash_value = _mix32(hash_value, position & _UINT32_MASK)
    hash_value = _mix32(hash_value, column & _UINT32_MASK)
    return _fmix32(hash_value ^ 16)


def test_murmur_hash32_pytorch_matches_scalar_reference() -> None:
    seeds = torch.tensor([0, 17, -1], dtype=torch.int64)
    positions = torch.tensor([1_707_985_137, 3, 9], dtype=torch.int64)

    actual = sampling_kernels.murmur_hash32_pytorch(seeds, positions, 4)
    expected = torch.tensor(
        [
            [_reference_hash(seed, position, column) for column in range(4)]
            for seed, position in zip(seeds.tolist(), positions.tolist())
        ],
        dtype=torch.int64,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_float32_seeded_sampling_is_repeatable() -> None:
    logprobs = torch.log_softmax(
        torch.tensor([[1.0, 0.5, -0.5], [-1.0, 2.0, 0.0]], dtype=torch.float32),
        dim=-1,
    )
    seeds = torch.tensor([11, 22], dtype=torch.int64)
    positions = torch.tensor([4, 8], dtype=torch.int64)

    first = sampling_kernels.seeded_gumbel_argmax_float32(logprobs, seeds, positions)
    second = sampling_kernels.seeded_gumbel_argmax_float32(logprobs, seeds, positions)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.dtype == torch.long
    assert first.shape == (2,)


def test_seeded_sampling_gumbel_math_stays_float32(monkeypatch) -> None:
    observed_dtypes = []
    original_log = torch.log

    def record_log_dtype(value):
        observed_dtypes.append(value.dtype)
        return original_log(value)

    monkeypatch.setattr(torch, "log", record_log_dtype)
    sampling_kernels.seeded_gumbel_argmax_float32(
        torch.tensor([[0.0, -1.0]], dtype=torch.float32),
        torch.tensor([5], dtype=torch.int64),
        torch.tensor([7], dtype=torch.int64),
    )

    assert observed_dtypes == [torch.float32, torch.float32]


def test_float32_seeded_sampling_caps_maximum_hash_uniform() -> None:
    seeds = torch.tensor([0], dtype=torch.int64)
    positions = torch.tensor([1_707_985_137], dtype=torch.int64)
    hashes = sampling_kernels.murmur_hash32_pytorch(seeds, positions, 2)
    assert hashes[0, 0].item() == _UINT32_MASK

    sampled = sampling_kernels.seeded_gumbel_argmax_float32(
        torch.tensor([[-100.0, 0.0]], dtype=torch.float32), seeds, positions
    )

    assert sampled.item() == 1


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
def test_npu_murmur_hash_and_float32_gumbel_execute_on_device() -> None:
    device = torch.device("npu:0")
    seeds = torch.tensor([0, 17, -1], device=device, dtype=torch.int64)
    positions = torch.tensor([1_707_985_137, 3, 9], device=device, dtype=torch.int64)

    hashes = sampling_kernels.murmur_hash32_pytorch(seeds, positions, 4)
    sampled = sampling_kernels.seeded_gumbel_argmax_float32(
        torch.tensor(
            [[0.0, -1.0, -2.0, -3.0]] * 3,
            device=device,
            dtype=torch.float32,
        ),
        seeds,
        positions,
    )
    torch.npu.synchronize(device)

    expected_hashes = torch.tensor(
        [
            [_reference_hash(seed, position, column) for column in range(4)]
            for seed, position in zip(seeds.cpu().tolist(), positions.cpu().tolist())
        ],
        dtype=torch.int64,
    )
    torch.testing.assert_close(hashes.cpu(), expected_hashes, rtol=0, atol=0)
    assert sampled.device.type == "npu"
    assert sampled.shape == (3,)


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "position,logprobs,expected",
    [(1_707_985_137, [-100.0, 0.0], [9]), (1_625_054_877, [100.0, 0.0], [7])],
)
def test_sorted_seeded_sampler_executes_float32_path_on_npu(
    position: int, logprobs: list[float], expected: list[int]
) -> None:
    device = torch.device("npu:0")
    sampled = sampling_kernels.sample_from_sorted_logprobs_with_seed_small_k(
        torch.tensor([logprobs], device=device, dtype=torch.float32),
        torch.tensor([[7, 9]], device=device, dtype=torch.long),
        torch.tensor([0], device=device, dtype=torch.int64),
        torch.tensor([position], device=device, dtype=torch.int64),
    )
    torch.npu.synchronize(device)

    assert sampled is not None
    assert sampled.device.type == "npu"
    assert sampled.cpu().tolist() == expected


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("num_cols", [1, 50, 257, 2048])
def test_npu_fused_gumbel_matches_reference(num_cols: int) -> None:
    generator = torch.Generator().manual_seed(1234)
    storage = torch.randn(256, num_cols * 2, generator=generator)
    logprobs = storage[:, ::2]
    logprobs[0].fill_(-float("inf"))
    logprobs[1].fill_(float("inf"))
    logprobs[2, -1] = float("nan")
    seeds = torch.tensor([0, -1, 2**40 + 123, -(2**63)] * 64)
    positions = torch.arange(256, dtype=torch.int64) + 1_707_985_137
    expected = sampling_kernels.seeded_gumbel_argmax_float32(logprobs, seeds, positions)
    actual = sampling_kernels.seeded_gumbel_argmax_float32(
        storage.npu()[:, ::2], seeds.npu(), positions.npu()
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
def test_npu_fused_gumbel_graph_replay_uses_updated_inputs() -> None:
    generator = torch.Generator().manual_seed(5678)
    logprobs_cpu = torch.randn(16, 50, generator=generator)
    seeds_cpu = torch.tensor([0, -1, 2**40 + 123, -(2**63)] * 4)
    positions_cpu = torch.arange(16, dtype=torch.int64)
    logprobs = logprobs_cpu.npu()
    seeds, positions = seeds_cpu.npu(), positions_cpu.npu()
    for _ in range(2):
        sampling_kernels.seeded_gumbel_argmax_float32(logprobs, seeds, positions)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = sampling_kernels.seeded_gumbel_argmax_float32(
            logprobs, seeds, positions
        )
    for _ in range(32):
        logprobs_cpu = torch.randn(16, 50, generator=generator)
        seeds_cpu = seeds_cpu.roll(1)
        positions_cpu += 1
        logprobs.copy_(logprobs_cpu)
        seeds.copy_(seeds_cpu)
        positions.copy_(positions_cpu)
        graph.replay()
        expected = sampling_kernels.seeded_gumbel_argmax_float32(
            logprobs_cpu, seeds_cpu, positions_cpu
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("width", [1, 8, 50, 128, 1024])
def test_npu_top_k_fusion_matches_probability_reference(width: int) -> None:
    from sglang_omni.models.qwen3_tts.npu_sampling import sample_top_k_npu

    generator = torch.Generator().manual_seed(1234)
    scores = torch.randn(256, width, generator=generator).sort(descending=True)[0]
    scores[0].fill_(-float("inf"))
    scores[1].fill_(0)
    scores[2, 0] = float("nan")
    scores[3] *= 100
    indices = torch.arange(width).flip(0).expand(256, -1).contiguous()
    top_ks = torch.randint(1, width + 1, (256,), generator=generator)
    seeds = torch.tensor([0, -1, 2**40 + 123, -(2**63)] * 64)
    positions = torch.arange(256) + 1_707_985_137
    masked = scores.masked_fill(
        torch.arange(width)[None] >= top_ks[:, None], -float("inf")
    )
    probabilities = masked.softmax(dim=-1)
    logprobs = torch.where(probabilities > 0, probabilities.log(), -float("inf"))
    ranks = sampling_kernels.seeded_gumbel_argmax_float32(logprobs, seeds, positions)
    expected = indices.gather(1, ranks[:, None]).flatten()
    actual = sample_top_k_npu(
        scores.npu(), indices.npu(), top_ks.npu(), seeds.npu(), positions.npu()
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
def test_npu_top_k_graph_replay_reads_updated_limits() -> None:
    from sglang_omni.models.qwen3_tts.npu_sampling import sample_top_k_npu

    scores = torch.zeros((16, 50), device="npu")
    indices = torch.arange(50, device="npu").expand(16, -1).contiguous()
    top_ks = torch.full((16,), 50, device="npu", dtype=torch.long)
    seeds = torch.arange(16, device="npu")
    positions = torch.arange(16, device="npu")
    for _ in range(2):
        sample_top_k_npu(scores, indices, top_ks, seeds, positions)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = sample_top_k_npu(scores, indices, top_ks, seeds, positions)
    for limit in (1, 17, 50, 2):
        top_ks.fill_(limit)
        positions.add_(1)
        scores.copy_(torch.randn_like(scores).sort(descending=True)[0])
        graph.replay()
        masked = scores.cpu().masked_fill(
            torch.arange(50)[None] >= limit, -float("inf")
        )
        expected = sampling_kernels.seeded_gumbel_argmax_float32(
            masked.softmax(dim=-1).log(), seeds.cpu(), positions.cpu()
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_npu_sampling_dispatch_does_not_capture_cpu() -> None:
    sampled = sampling_kernels.sample_from_logprobs_with_seed_npu(
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )

    assert sampled is None


def test_triton_kernel_is_disabled_on_npu(monkeypatch) -> None:
    monkeypatch.setattr(sampling_kernels, "triton", object())
    monkeypatch.setattr(sampling_kernels.current_platform, "is_npu", lambda: True)

    assert not sampling_kernels.has_triton_runtime()


def test_sorted_sampler_uses_float32_path_for_npu(monkeypatch) -> None:
    monkeypatch.setattr(
        sampling_kernels,
        "all_tensors_on_npu",
        lambda *tensors: True,
    )

    sampled = sampling_kernels.sample_from_sorted_logprobs_with_seed_small_k(
        torch.tensor([[-100.0, 0.0]], dtype=torch.float32),
        torch.tensor([[7, 9]], dtype=torch.long),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1_707_985_137], dtype=torch.int64),
    )

    assert sampled is not None
    assert sampled.tolist() == [9]
