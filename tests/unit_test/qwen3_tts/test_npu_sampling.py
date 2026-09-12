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

    actual = sampling_kernels._murmur_hash32_pytorch(seeds, positions, 4)
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

    first = sampling_kernels._seeded_gumbel_argmax_float32(logprobs, seeds, positions)
    second = sampling_kernels._seeded_gumbel_argmax_float32(logprobs, seeds, positions)

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
    sampling_kernels._seeded_gumbel_argmax_float32(
        torch.tensor([[0.0, -1.0]], dtype=torch.float32),
        torch.tensor([5], dtype=torch.int64),
        torch.tensor([7], dtype=torch.int64),
    )

    assert observed_dtypes == [torch.float32, torch.float32]


def test_float32_seeded_sampling_caps_maximum_hash_uniform() -> None:
    seeds = torch.tensor([0], dtype=torch.int64)
    positions = torch.tensor([1_707_985_137], dtype=torch.int64)
    hashes = sampling_kernels._murmur_hash32_pytorch(seeds, positions, 2)
    assert hashes[0, 0].item() == _UINT32_MASK

    sampled = sampling_kernels._seeded_gumbel_argmax_float32(
        torch.tensor([[-100.0, 0.0]], dtype=torch.float32), seeds, positions
    )

    assert sampled.item() == 1


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")
def test_npu_murmur_hash_and_float32_gumbel_execute_on_device() -> None:
    device = torch.device("npu:0")
    seeds = torch.tensor([0, 17, -1], device=device, dtype=torch.int64)
    positions = torch.tensor([1_707_985_137, 3, 9], device=device, dtype=torch.int64)

    hashes = sampling_kernels._murmur_hash32_pytorch(seeds, positions, 4)
    sampled = sampling_kernels._seeded_gumbel_argmax_float32(
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
def test_sorted_seeded_sampler_executes_float32_path_on_npu() -> None:
    device = torch.device("npu:0")
    sampled = sampling_kernels.sample_from_sorted_logprobs_with_seed_small_k(
        torch.tensor([[-100.0, 0.0]], device=device, dtype=torch.float32),
        torch.tensor([[7, 9]], device=device, dtype=torch.long),
        torch.tensor([0], device=device, dtype=torch.int64),
        torch.tensor([1_707_985_137], device=device, dtype=torch.int64),
    )
    torch.npu.synchronize(device)

    assert sampled is not None
    assert sampled.device.type == "npu"
    assert sampled.cpu().tolist() == [9]


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

    assert not sampling_kernels._has_triton_runtime()


def test_sorted_sampler_uses_float32_path_for_npu(monkeypatch) -> None:
    monkeypatch.setattr(
        sampling_kernels,
        "_all_tensors_on_npu",
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
