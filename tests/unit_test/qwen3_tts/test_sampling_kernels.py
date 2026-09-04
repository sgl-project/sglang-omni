# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS optional sampling kernels."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.qwen3_tts import sampling_kernels as sampling_kernels_module
from sglang_omni.models.qwen3_tts import sglang_model as sglang_model_module
from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_logits_with_seed_top_k_top_p,
    sample_from_sorted_logprobs_with_seed_small_k,
)
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Qwen3-TTS sampling kernel needs CUDA"
    ),
]


@pytest.mark.parametrize("batch_size,num_cols", [(1, 1), (3, 2), (7, 30), (16, 64)])
def test_seeded_small_k_sampler_matches_sglang_multinomial(
    batch_size: int, num_cols: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(batch_size * 100 + num_cols)
    probs = torch.rand(
        batch_size,
        num_cols,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    probs = probs / probs.sum(dim=1, keepdim=True)
    sorted_idx = torch.randint(
        0,
        8192,
        (batch_size, num_cols),
        generator=generator,
        device="cuda",
        dtype=torch.long,
    )
    seeds = torch.arange(17, 17 + batch_size, device="cuda", dtype=torch.long)
    positions = torch.arange(3, 3 + batch_size, device="cuda", dtype=torch.long)

    logprobs = probs.log()
    sampled = sample_from_sorted_logprobs_with_seed_small_k(
        logprobs, sorted_idx, seeds, positions
    )
    assert sampled is not None

    sampled_rank = multinomial_with_seed(logprobs, seeds, positions).view(-1, 1)
    expected = sorted_idx.gather(1, sampled_rank).view(-1)
    assert torch.equal(sampled, expected)


def test_seeded_small_k_sampler_matches_sglang_at_the_uint32_max_hash() -> None:
    logprobs = torch.tensor([[-100.0, 0.0]], device="cuda")
    sorted_idx = torch.tensor([[7, 9]], device="cuda", dtype=torch.long)
    seeds = torch.tensor([0], device="cuda", dtype=torch.long)
    positions = torch.tensor([1_707_985_137], device="cuda", dtype=torch.long)

    hashes = murmur_hash32(
        seeds.to(torch.uint64), positions, torch.arange(2, device="cuda")
    )
    expected_rank = multinomial_with_seed(logprobs, seeds, positions).view(-1)
    sampled = sample_from_sorted_logprobs_with_seed_small_k(
        logprobs, sorted_idx, seeds, positions
    )

    assert hashes[0, 0].item() == torch.iinfo(torch.uint32).max
    assert expected_rank.item() == 1
    assert sampled.tolist() == [9]


@pytest.mark.parametrize(
    ("seed_value", "other_logprob", "endpoint_logprob"),
    [
        pytest.param(
            0x804B40E3,
            -100.0,
            0.0,
            id="hash-zero",
        ),
        pytest.param(
            0x572AB199,
            0.0,
            -100.0,
            id="hash-uint32-max",
        ),
        pytest.param(
            0,
            -100.0,
            0.0,
            id="seed-zero-pos-endpoint",
        ),
    ],
)
def test_seeded_small_k_sampler_matches_reference_at_hash_endpoints(
    seed_value: int,
    other_logprob: float,
    endpoint_logprob: float,
) -> None:
    """Keep hash endpoints bit-identical to SGLang multinomial_with_seed."""
    num_cols = 50
    logprobs = torch.full(
        (1, num_cols),
        other_logprob,
        device="cuda",
        dtype=torch.float32,
    )
    logprobs[:, -1] = endpoint_logprob
    sorted_idx = torch.arange(num_cols, device="cuda", dtype=torch.long).view(1, -1)
    seeds = torch.tensor([seed_value], device="cuda", dtype=torch.long)
    # Use a position that can hit uint32 max for seed 0.
    positions = torch.tensor(
        [1_707_985_137 if seed_value == 0 else 0],
        device="cuda",
        dtype=torch.long,
    )

    expected_rank = multinomial_with_seed(logprobs, seeds, positions)
    expected = sorted_idx.gather(1, expected_rank).view(-1)
    actual = sample_from_sorted_logprobs_with_seed_small_k(
        logprobs,
        sorted_idx,
        seeds,
        positions,
    )

    assert actual is not None
    assert torch.equal(actual, expected)


def test_fused_raw_logit_sampler_matches_reference_at_uint32_max_hash() -> None:
    """The raw-logit graph kernel must match SGLang multinomial at hash endpoints."""
    max_top_k = 50
    logits = torch.full((1, 2048), -200.0, device="cuda", dtype=torch.bfloat16)
    logits[0, : max_top_k - 1] = torch.arange(
        0,
        -(max_top_k - 1),
        -1,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    logits[0, max_top_k - 1] = -100.0
    temperatures = torch.ones((1,), device="cuda", dtype=torch.float32)
    top_ks = torch.tensor([max_top_k], device="cuda", dtype=torch.long)
    top_ps = torch.ones((1,), device="cuda", dtype=torch.float32)
    seeds = torch.tensor([0x572AB199], device="cuda", dtype=torch.long)
    positions = torch.zeros((1,), device="cuda", dtype=torch.long)

    sorted_scores, sorted_idx = torch.topk(logits.float(), max_top_k, dim=-1)
    sorted_logprobs = torch.log(torch.softmax(sorted_scores, dim=-1))
    expected_rank = multinomial_with_seed(sorted_logprobs, seeds, positions)
    expected = sorted_idx.gather(1, expected_rank).view(-1)

    actual = sample_from_logits_with_seed_top_k_top_p(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        max_top_k=max_top_k,
        has_top_p=False,
    )

    assert actual is not None
    assert torch.equal(actual, expected)


def test_seeded_small_k_sampler_falls_back_for_cpu() -> None:
    logprobs = torch.zeros((1, 2), dtype=torch.float32)
    sorted_idx = torch.arange(2, dtype=torch.long).view(1, 2)
    seeds = torch.ones((1,), dtype=torch.long)
    positions = torch.zeros((1,), dtype=torch.long)

    assert (
        sample_from_sorted_logprobs_with_seed_small_k(
            logprobs, sorted_idx, seeds, positions
        )
        is None
    )


def _build_sampling_talker(
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    *,
    max_top_k: int,
) -> Qwen3TTSTalker:
    """Build only the fields used by the production sampled-token reference."""
    talker = object.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(num_code_groups=16)
    talker._sub_temperature_tensor = temperatures
    talker._sub_top_k_tensor = top_ks
    talker._sub_top_p_tensor = top_ps
    talker._sub_sampling_seed_tensor = seeds
    talker._sub_sampled_max_top_k = max_top_k
    talker._sub_sampled_has_top_p = bool(((top_ps > 0.0) & (top_ps < 1.0)).any().item())
    talker._sub_sampled_has_unbounded_top_k = False
    return talker


def _production_seeded_tokens(
    talker: Qwen3TTSTalker,
    logits: torch.Tensor,
    *,
    layer_idx: int,
    semantic_positions: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    return talker._sample_subtalker_token_seeded(
        logits,
        layer_idx,
        row_indices=rows,
        semantic_positions=semantic_positions,
    )


def _reference_seeded_tokens(
    talker: Qwen3TTSTalker,
    logits: torch.Tensor,
    *,
    layer_idx: int,
    semantic_positions: torch.Tensor,
) -> torch.Tensor:
    with mock.patch.object(
        sglang_model_module,
        "sample_from_logits_with_seed_top_k_top_p",
        return_value=None,
    ):
        return _production_seeded_tokens(
            talker,
            logits,
            layer_idx=layer_idx,
            semantic_positions=semantic_positions,
        )


def _fused_seeded_tokens(
    talker: Qwen3TTSTalker,
    logits: torch.Tensor,
    *,
    layer_idx: int,
    semantic_positions: torch.Tensor,
) -> torch.Tensor:
    sub_positions = (
        semantic_positions * max(int(talker.config.num_code_groups) - 1, 1)
        + layer_idx
        + 1
    )
    sampled = sample_from_logits_with_seed_top_k_top_p(
        logits,
        talker._sub_temperature_tensor.clamp_min(1e-5),
        talker._sub_top_k_tensor,
        talker._sub_top_p_tensor,
        talker._sub_sampling_seed_tensor,
        sub_positions,
        max_top_k=talker._sub_sampled_max_top_k,
        has_top_p=talker._sub_sampled_has_top_p,
    )
    assert sampled is not None
    return sampled


@pytest.mark.parametrize(
    "batch_size,max_top_k,top_p",
    [
        (1, 4, 1.0),
        (4, 16, 0.8),
        (4, 32, 1.0),
        (8, 32, 0.95),
        (4, 50, 1.0),
        (8, 64, 0.8),
        (4, 128, 0.95),
        (4, 256, 0.5),
        (8, 512, 0.8),
        (1, 1024, 0.95),
    ],
)
def test_fused_raw_logit_sampler_matches_reference(
    batch_size: int,
    max_top_k: int,
    top_p: float,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        100_000 + 100 * batch_size + max_top_k
    )
    logits = torch.randn(
        batch_size,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    temperatures = torch.tensor(
        [0.05 + 0.2 * (row % 5) for row in range(batch_size)],
        device="cuda",
        dtype=torch.float32,
    )
    top_ks = torch.tensor(
        [max(1, max_top_k - (row % min(max_top_k, 5))) for row in range(batch_size)],
        device="cuda",
        dtype=torch.long,
    )
    top_ps = torch.full((batch_size,), top_p, device="cuda", dtype=torch.float32)
    seeds = torch.arange(500, 500 + batch_size, device="cuda", dtype=torch.long)
    positions = torch.arange(17, 17 + batch_size, device="cuda", dtype=torch.long)
    talker = _build_sampling_talker(
        temperatures,
        top_ks,
        top_ps,
        seeds,
        max_top_k=max_top_k,
    )

    expected = _reference_seeded_tokens(
        talker,
        logits,
        layer_idx=3,
        semantic_positions=positions,
    )
    actual = _fused_seeded_tokens(
        talker,
        logits,
        layer_idx=3,
        semantic_positions=positions,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("max_top_k", [4, 8, 16, 32, 50, 64, 128, 256, 512, 1024])
def test_fused_raw_logit_sampler_matches_reference_for_equal_logits(
    max_top_k: int,
) -> None:
    logits = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
    temperatures = torch.ones((1,), device="cuda", dtype=torch.float32)
    top_ks = torch.tensor([max_top_k], device="cuda", dtype=torch.long)
    top_ps = torch.ones((1,), device="cuda", dtype=torch.float32)
    positions = torch.tensor([9], device="cuda", dtype=torch.long)

    for seed_value in range(32):
        seeds = torch.tensor([seed_value], device="cuda", dtype=torch.long)
        talker = _build_sampling_talker(
            temperatures,
            top_ks,
            top_ps,
            seeds,
            max_top_k=max_top_k,
        )
        expected = _reference_seeded_tokens(
            talker,
            logits,
            layer_idx=2,
            semantic_positions=positions,
        )
        actual = _fused_seeded_tokens(
            talker,
            logits,
            layer_idx=2,
            semantic_positions=positions,
        )
        assert torch.equal(actual, expected), f"seed={seed_value}"


@pytest.mark.parametrize("max_top_k", [4, 32, 50, 128, 1024])
@pytest.mark.parametrize("top_p", [0.5, 0.8, 0.95, 1.0])
def test_fused_raw_logit_sampler_matches_reference_for_threshold_ties(
    max_top_k: int,
    top_p: float,
) -> None:
    """Repeated BF16 values exercise top-k threshold and sort tie behavior."""
    batch_size = 4
    values = (torch.arange(2048, device="cuda") % 23).to(torch.float32)
    logits = torch.stack([values.roll(37 * row) for row in range(batch_size)]).to(
        torch.bfloat16
    )
    temperatures = torch.tensor(
        [0.05, 0.7, 1.0, 1.5], device="cuda", dtype=torch.float32
    )
    top_ks = torch.tensor(
        [1, min(2, max_top_k), max_top_k - 1, max_top_k],
        device="cuda",
        dtype=torch.long,
    )
    top_ps = torch.full((batch_size,), top_p, device="cuda", dtype=torch.float32)
    positions = torch.arange(80, 80 + batch_size, device="cuda", dtype=torch.long)

    for seed_offset in range(8):
        seeds = torch.arange(
            900 + seed_offset * batch_size,
            900 + (seed_offset + 1) * batch_size,
            device="cuda",
            dtype=torch.long,
        )
        talker = _build_sampling_talker(
            temperatures,
            top_ks,
            top_ps,
            seeds,
            max_top_k=max_top_k,
        )
        expected = _reference_seeded_tokens(
            talker,
            logits,
            layer_idx=5,
            semantic_positions=positions,
        )
        actual = _fused_seeded_tokens(
            talker,
            logits,
            layer_idx=5,
            semantic_positions=positions,
        )
        assert torch.equal(actual, expected), f"seed_offset={seed_offset}"


@pytest.mark.parametrize(
    "max_top_k,positive_zero_pattern",
    [
        (32, "alternating"),
        (50, "alternating"),
        (1024, "alternating"),
        (32, "leading"),
    ],
)
def test_fused_raw_logit_sampler_matches_reference_for_signed_zero_order(
    max_top_k: int,
    positive_zero_pattern: str,
) -> None:
    """CUDA top-k's signed-zero selection order is part of the seed contract."""
    batch_size = 4
    if positive_zero_pattern == "alternating":
        signs = torch.ones((batch_size, 2048), device="cuda", dtype=torch.float32)
        signs[:, ::2] = -1.0
    else:
        signs = -torch.ones((batch_size, 2048), device="cuda", dtype=torch.float32)
        signs[:, :16] = 1.0
    logits = torch.copysign(torch.zeros_like(signs), signs).to(torch.bfloat16)
    temperatures = torch.ones((batch_size,), device="cuda", dtype=torch.float32)
    top_ks = torch.full((batch_size,), max_top_k, device="cuda", dtype=torch.long)
    top_ps = torch.ones((batch_size,), device="cuda", dtype=torch.float32)
    positions = torch.arange(140, 140 + batch_size, device="cuda", dtype=torch.long)

    for seed_offset in range(32):
        seeds = torch.arange(
            20_000 + seed_offset * batch_size,
            20_000 + (seed_offset + 1) * batch_size,
            device="cuda",
            dtype=torch.long,
        )
        talker = _build_sampling_talker(
            temperatures,
            top_ks,
            top_ps,
            seeds,
            max_top_k=max_top_k,
        )
        expected = _reference_seeded_tokens(
            talker,
            logits,
            layer_idx=6,
            semantic_positions=positions,
        )
        actual = _fused_seeded_tokens(
            talker,
            logits,
            layer_idx=6,
            semantic_positions=positions,
        )
        assert torch.equal(actual, expected), f"seed_offset={seed_offset}"


def test_fused_raw_logit_sampler_captures_without_reference_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = 4
    logits = torch.randn(
        batch_size,
        2048,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    temperatures = torch.tensor(
        [0.7, 0.8, 0.9, 1.0], device="cuda", dtype=torch.float32
    )
    top_ks = torch.tensor([32, 31, 30, 29], device="cuda", dtype=torch.long)
    top_ps = torch.full((batch_size,), 0.8, device="cuda", dtype=torch.float32)
    seeds = torch.arange(700, 700 + batch_size, device="cuda", dtype=torch.long)
    positions = torch.arange(30, 30 + batch_size, device="cuda", dtype=torch.long)
    talker = _build_sampling_talker(
        temperatures,
        top_ks,
        top_ps,
        seeds,
        max_top_k=32,
    )

    expected = _reference_seeded_tokens(
        talker,
        logits,
        layer_idx=4,
        semantic_positions=positions,
    )
    _fused_seeded_tokens(
        talker,
        logits,
        layer_idx=4,
        semantic_positions=positions,
    )

    def fail_top_k(*args, **kwargs):
        del args, kwargs
        raise AssertionError("captured fused sampler must not call torch.topk")

    monkeypatch.setattr(torch, "topk", fail_top_k)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = _production_seeded_tokens(
            talker,
            logits,
            layer_idx=4,
            semantic_positions=positions,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured, expected)


def test_fused_raw_logit_sampler_capture_falls_back_without_triton_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable Triton primitive must keep graph capture on the reference."""
    batch_size = 4
    logits = torch.randn(
        batch_size,
        2048,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    temperatures = torch.tensor(
        [0.7, 0.8, 0.9, 1.0], device="cuda", dtype=torch.float32
    )
    top_ks = torch.tensor([32, 31, 30, 29], device="cuda", dtype=torch.long)
    top_ps = torch.full((batch_size,), 0.8, device="cuda", dtype=torch.float32)
    seeds = torch.arange(800, 800 + batch_size, device="cuda", dtype=torch.long)
    positions = torch.arange(40, 40 + batch_size, device="cuda", dtype=torch.long)
    talker = _build_sampling_talker(
        temperatures,
        top_ks,
        top_ps,
        seeds,
        max_top_k=32,
    )
    expected = _reference_seeded_tokens(
        talker,
        logits,
        layer_idx=5,
        semantic_positions=positions,
    )

    top_k_calls = []
    original_top_k = torch.topk

    def record_top_k(*args, **kwargs):
        top_k_calls.append(1)
        return original_top_k(*args, **kwargs)

    monkeypatch.setattr(sampling_kernels_module, "_TRITON_GATHER_SUPPORTED", False)
    monkeypatch.setattr(torch, "topk", record_top_k)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = _production_seeded_tokens(
            talker,
            logits,
            layer_idx=5,
            semantic_positions=positions,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert top_k_calls
    assert torch.equal(captured, expected)


def test_fused_raw_logit_sampler_falls_back_for_unproven_shapes() -> None:
    logits = torch.zeros((1, 2048), dtype=torch.bfloat16)
    temperatures = torch.ones((1,), dtype=torch.float32)
    top_ks = torch.ones((1,), dtype=torch.long)
    top_ps = torch.ones((1,), dtype=torch.float32)
    seeds = torch.ones((1,), dtype=torch.long)
    positions = torch.zeros((1,), dtype=torch.long)

    assert (
        sample_from_logits_with_seed_top_k_top_p(
            logits,
            temperatures,
            top_ks,
            top_ps,
            seeds,
            positions,
            max_top_k=32,
            has_top_p=False,
        )
        is None
    )


def test_fused_raw_logit_sampler_falls_back_for_unsupported_cuda_inputs() -> None:
    logits = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
    temperatures = torch.ones((1,), device="cuda", dtype=torch.float32)
    top_ks = torch.ones((1,), device="cuda", dtype=torch.long)
    top_ps = torch.ones((1,), device="cuda", dtype=torch.float32)
    seeds = torch.ones((1,), device="cuda", dtype=torch.long)
    positions = torch.zeros((1,), device="cuda", dtype=torch.long)

    assert (
        sample_from_logits_with_seed_top_k_top_p(
            logits,
            temperatures,
            top_ks,
            top_ps,
            seeds,
            positions,
            max_top_k=1025,
            has_top_p=False,
        )
        is None
    )
    assert (
        sample_from_logits_with_seed_top_k_top_p(
            torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)[:, ::2],
            temperatures,
            top_ks,
            top_ps,
            seeds,
            positions,
            max_top_k=32,
            has_top_p=False,
        )
        is None
    )


def test_fused_raw_logit_sampler_falls_back_without_triton_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
    temperatures = torch.ones((1,), device="cuda", dtype=torch.float32)
    top_ks = torch.ones((1,), device="cuda", dtype=torch.long)
    top_ps = torch.ones((1,), device="cuda", dtype=torch.float32)
    seeds = torch.ones((1,), device="cuda", dtype=torch.long)
    positions = torch.zeros((1,), device="cuda", dtype=torch.long)
    monkeypatch.setattr(sampling_kernels_module, "_TRITON_GATHER_SUPPORTED", False)

    assert (
        sample_from_logits_with_seed_top_k_top_p(
            logits,
            temperatures,
            top_ks,
            top_ps,
            seeds,
            positions,
            max_top_k=32,
            has_top_p=False,
        )
        is None
    )
