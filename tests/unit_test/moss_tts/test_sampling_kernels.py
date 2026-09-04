# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for MOSS-TTS sampling kernels."""

from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.moss_tts.sampling_kernels import (
    multinomial_with_seed_and_token_ids,
    seeded_gumbel_argmax,
)

pytestmark = pytest.mark.accelerator

_UINT32_MAX_HASH_POSITION = 1_707_985_137


@pytest.mark.parametrize("equal_scores", [False, True], ids=["random", "tied"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_production_shape(
    equal_scores: bool,
) -> None:
    device = torch.device("cuda")
    rows, vocab = 32, 1025
    scores = (
        torch.zeros(rows, vocab, device=device, dtype=torch.float32)
        if equal_scores
        else torch.randn(rows, vocab, device=device, dtype=torch.float32)
    )
    seeds = torch.tensor([20260720], device=device, dtype=torch.long).expand(rows)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows, device=device, dtype=torch.long)

    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert seeds.stride(0) == 0
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_uint32_max_hash() -> None:
    device = torch.device("cuda")
    scores = torch.tensor([[-100.0, 0.0]], device=device)
    seeds = torch.tensor([0], device=device, dtype=torch.long)
    # This position makes MurmurHash(seed=0, position, token_id=0) UINT32_MAX.
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
    )
    output = torch.empty(1, device=device, dtype=torch.long)

    hashes = murmur_hash32(
        seeds.to(torch.uint64),
        positions,
        torch.arange(scores.shape[1], device=device),
    )
    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert hashes[0, 0].item() == torch.iinfo(torch.uint32).max
    assert expected.item() == 1
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_id_sampler_matches_sglang_at_the_uint32_max_hash() -> None:
    device = torch.device("cuda")
    scores = torch.tensor([[-100.0, 0.0]], device=device)
    seeds = torch.tensor([0], device=device, dtype=torch.long)
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
    )
    token_ids = torch.arange(scores.shape[1], device=device)

    hashes = murmur_hash32(seeds.to(torch.uint64), positions, token_ids)
    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = multinomial_with_seed_and_token_ids(scores, seeds, positions, token_ids)

    assert hashes[0, 0].item() == torch.iinfo(torch.uint32).max
    assert expected.item() == 1
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_rejects_strided_output() -> None:
    device = torch.device("cuda")
    rows, vocab = 2, 16
    scores = torch.randn(rows, vocab, device=device, dtype=torch.float32)
    seeds = torch.arange(rows, device=device, dtype=torch.long)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows * 2, device=device, dtype=torch.long)[::2]

    assert output.stride(0) == 2
    with pytest.raises(ValueError, match="output must have stride 1"):
        seeded_gumbel_argmax(scores, seeds, positions, output)


def _fused_case(vocab: int, temp: float, top_p: float, top_k: int, tie: bool):
    return pytest.param(
        vocab,
        temp,
        top_p,
        top_k,
        tie,
        id=f"v{vocab}-t{temp}-p{top_p}-k{top_k}-{'tied' if tie else 'random'}",
    )


@pytest.mark.parametrize(
    "vocab,temp,top_p,top_k,tie",
    [
        _fused_case(1025, 1.7, 0.8, 25, False),
        _fused_case(1025, 1.7, 0.8, 25, True),
        _fused_case(1025, 1.0, 0.9, 64, True),
        _fused_case(1025, 1.2, 0.5, 1, False),
        _fused_case(1025, 1.7, 0.001, 25, False),
        _fused_case(1025, 1.7, 0.999, 25, False),
        _fused_case(1025, 0.0, 0.8, 25, False),
        _fused_case(1024, 1.7, 0.8, 25, False),
        _fused_case(1024, 1.7, 0.8, 25, True),
        _fused_case(1024, 1.0, 0.9, 64, True),
        _fused_case(1024, 1.2, 0.5, 1, False),
        _fused_case(1024, 0.0, 0.8, 25, False),
        _fused_case(2, 1.0, 1.0, 50, False),
        _fused_case(2, 1.3, 0.7, 0, False),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_matches_branchless(
    vocab: int, temp: float, top_p: float, top_k: int, tie: bool
) -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20260821)
    for rows in (1, 4, 16):
        for _ in range(20):
            logits = torch.randn(rows, vocab, device=device, generator=gen) * 4.0
            # Note (Jiaxin Deng): bf16 round-trip manufactures score ties so the
            # stable-sort tie ordering is exercised, not just clean floats.
            logits = logits.to(torch.bfloat16).float()
            if tie:
                logits = (logits * 4).round() / 4.0
            params = dict(
                temperature=torch.full((rows,), temp, device=device),
                top_p=torch.full((rows,), top_p, device=device),
                top_k=torch.full((rows,), top_k, device=device, dtype=torch.long),
                seeds=torch.randint(
                    0, 2**62, (rows,), device=device, dtype=torch.long, generator=gen
                ),
                positions=torch.randint(
                    0, 10000, (rows,), device=device, dtype=torch.long, generator=gen
                ),
            )
            expected = sample_seeded_branchless(logits, **params)
            actual = sample_seeded_fused(logits, **params)
            assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_mixed_rows_and_greedy_fallback() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7)
    rows, vocab = 8, 1025
    logits = (
        (torch.randn(rows, vocab, device=device, generator=gen) * 4.0)
        .to(torch.bfloat16)
        .float()
    )
    params = dict(
        temperature=torch.tensor(
            [1.7, 0.0, 1.0, 0.0, 1.2, 1.7, 0.5, 1.0], device=device
        ),
        top_p=torch.tensor([0.8, 0.8, 1.0, 0.0, 0.5, 0.9, 0.2, 0.7], device=device),
        top_k=torch.tensor([25, 25, 0, 50, 1, 64, 7, 2000], device=device),
        seeds=torch.randint(
            0, 2**62, (rows,), device=device, dtype=torch.long, generator=gen
        ),
        positions=torch.arange(rows, device=device, dtype=torch.long),
    )
    expected = sample_seeded_branchless(logits, **params)
    actual = sample_seeded_fused(logits, **params)
    assert torch.equal(expected, actual)


def test_sample_seeded_fused_rejects_large_vocab() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        MAX_FUSED_SAMPLE_VOCAB,
        sample_seeded_fused,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    rows = 2
    vocab = MAX_FUSED_SAMPLE_VOCAB + 1
    with pytest.raises(ValueError, match="vocab"):
        sample_seeded_fused(
            torch.randn(rows, vocab, device=device),
            temperature=torch.ones(rows, device=device),
            top_p=torch.ones(rows, device=device),
            top_k=torch.full((rows,), 25, device=device, dtype=torch.long),
            seeds=torch.zeros(rows, device=device, dtype=torch.long),
            positions=torch.arange(rows, device=device, dtype=torch.long),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_nucleus_knife_edges() -> None:
    """Away from fp32 knife edges the mask matches bit-for-bit; exactly at a
    cumulative-probability boundary the kept set may differ by the boundary
    token only, so the sample must stay inside the top-k prefix."""
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    vocab, k = 1025, 25
    gen = torch.Generator(device=device).manual_seed(11)
    logits = (torch.randn(1, vocab, device=device, generator=gen) * 4.0).float()
    scores = logits.clone()
    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    kth = sorted_scores[0, k - 1]
    masked = sorted_scores.masked_fill(sorted_scores < kth, float("-inf"))
    cum = torch.cumsum(torch.softmax(masked, dim=-1), dim=-1)[0]
    topk_ids = set(sorted_idx[0, :k].tolist())

    for j in (3, 10, 20):
        edge = float(cum[j].item())
        for p in (
            (edge + float(cum[j + 1].item())) / 2.0,  # strictly between edges
            edge,  # exactly at the boundary
            float(torch.nextafter(cum[j], cum[j] + 1).item()),
        ):
            params = dict(
                temperature=torch.ones(1, device=device),
                top_p=torch.full((1,), p, device=device),
                top_k=torch.full((1,), k, device=device, dtype=torch.long),
                seeds=torch.full((1,), 42, device=device, dtype=torch.long),
                positions=torch.full((1,), 7, device=device, dtype=torch.long),
            )
            a = sample_seeded_branchless(logits, **params)
            b = sample_seeded_fused(logits, **params)
            assert int(b.item()) in topk_ids
            if p != edge:
                assert torch.equal(a, b), f"j={j} p={p}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_signed_zero_and_nonfinite() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    vocab = 64
    logits = torch.full((3, vocab), -5.0, device=device)
    logits[0, 3], logits[0, 7] = 0.0, -0.0  # numerically tied signed zeros
    logits[1, 5] = float("-inf")
    logits[2, :] = float("-inf")  # fully masked row -> greedy fallback
    params = dict(
        temperature=torch.tensor([1.0, 1.0, 1.0], device=device),
        top_p=torch.tensor([0.9, 0.9, 0.9], device=device),
        top_k=torch.tensor([8, 8, 8], device=device, dtype=torch.long),
        seeds=torch.tensor([1, 2, 3], device=device, dtype=torch.long),
        positions=torch.tensor([0, 1, 2], device=device, dtype=torch.long),
    )
    a = sample_seeded_branchless(logits, **params)
    b = sample_seeded_fused(logits, **params)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_hash_endpoint() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    logits = torch.tensor([[-100.0, 0.0]], device=device)
    params = dict(
        temperature=torch.ones(1, device=device),
        top_p=torch.ones(1, device=device),
        top_k=torch.zeros(1, device=device, dtype=torch.long),
        seeds=torch.zeros(1, device=device, dtype=torch.long),
        # MurmurHash(seed=0, position, token_id=0) == UINT32_MAX here.
        positions=torch.tensor(
            [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
        ),
    )
    a = sample_seeded_branchless(logits, **params)
    b = sample_seeded_fused(logits, **params)
    assert torch.equal(a, b)
    assert a.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_input_hardening() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    empty = sample_seeded_fused(
        torch.empty(0, 1025, device=device),
        temperature=torch.empty(0, device=device),
        top_p=torch.empty(0, device=device),
        top_k=torch.empty(0, device=device, dtype=torch.long),
        seeds=torch.empty(0, device=device, dtype=torch.long),
        positions=torch.empty(0, device=device, dtype=torch.long),
    )
    assert empty.shape == (0,) and empty.dtype == torch.int64

    with pytest.raises(TypeError, match="integer"):
        sample_seeded_fused(
            torch.randn(2, 64, device=device),
            temperature=torch.ones(2, device=device),
            top_p=torch.ones(2, device=device),
            top_k=torch.full((2,), 8.0, device=device),
            seeds=torch.zeros(2, device=device, dtype=torch.long),
            positions=torch.arange(2, device=device, dtype=torch.long),
        )

    gen = torch.Generator(device=device).manual_seed(5)
    logits = torch.randn(4, 1025, device=device, generator=gen)
    strided = torch.zeros(8, device=device, dtype=torch.long)[::2] + 25
    contig = dict(
        temperature=torch.full((4,), 1.7, device=device),
        top_p=torch.full((4,), 0.8, device=device),
        top_k=torch.full((4,), 25, device=device, dtype=torch.long),
        seeds=torch.arange(4, device=device, dtype=torch.long),
        positions=torch.arange(4, device=device, dtype=torch.long),
    )
    a = sample_seeded_fused(logits, **contig)
    b = sample_seeded_fused(logits, **{**contig, "top_k": strided})
    c = sample_seeded_branchless(logits, **contig)
    assert torch.equal(a, b)
    assert torch.equal(a, c)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_masked_lane_hash_endpoint() -> None:
    """A top-k-masked token whose Gumbel hash hits UINT32_MAX yields
    a capped gumbel, so the masked lane stays at -inf instead of the NaN the
    uncapped endpoint used to produce. The fused kernel must follow the baseline
    either way (parity, not top-k membership, is the contract)."""
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    vocab = 64
    logits = torch.full((1, vocab), 0.0, device=device)
    logits[0, 0] = -100.0  # token 0: excluded by top-k, hash endpoint target
    params = dict(
        temperature=torch.ones(1, device=device),
        top_p=torch.ones(1, device=device),
        top_k=torch.full((1,), 8, device=device, dtype=torch.long),
        seeds=torch.zeros(1, device=device, dtype=torch.long),
        positions=torch.tensor(
            [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
        ),
    )
    a = sample_seeded_branchless(logits, **params)
    b = sample_seeded_fused(logits, **params)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_signed_zero_orderings_and_nonfinite() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    vocab = 64
    base = torch.full((vocab,), -5.0, device=device)
    r0 = base.clone()
    r0[3], r0[7] = 0.0, -0.0  # +0 before -0
    r1 = base.clone()
    r1[3], r1[7] = -0.0, 0.0  # -0 before +0 (reversed)
    r2 = base.clone()
    r2[0], r2[1] = -0.0, 0.0  # adjacent reversed
    r3 = base.clone()
    r3[5] = float("inf")  # +inf logit
    r4 = base.clone()
    r4[9] = float("nan")  # NaN logit
    logits = torch.stack([r0, r1, r2, r3, r4])
    n = logits.shape[0]
    for seed in (1, 9, 1234):
        params = dict(
            temperature=torch.ones(n, device=device),
            top_p=torch.full((n,), 0.9, device=device),
            top_k=torch.full((n,), 8, device=device, dtype=torch.long),
            seeds=torch.full((n,), seed, device=device, dtype=torch.long),
            positions=torch.arange(n, device=device, dtype=torch.long),
        )
        a = sample_seeded_branchless(logits, **params)
        b = sample_seeded_fused(logits, **params)
        assert torch.equal(a, b), f"seed={seed}: {a.tolist()} vs {b.tolist()}"


def test_sample_seeded_fused_rejects_complex_top_k() -> None:
    from sglang_omni.models.moss_tts.sampling_kernels import sample_seeded_fused

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    with pytest.raises(TypeError, match="integer"):
        sample_seeded_fused(
            torch.randn(2, 64, device=device),
            temperature=torch.ones(2, device=device),
            top_p=torch.ones(2, device=device),
            top_k=torch.full((2,), 8, device=device, dtype=torch.complex64),
            seeds=torch.zeros(2, device=device, dtype=torch.long),
            positions=torch.arange(2, device=device, dtype=torch.long),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_signed_zero_nucleus_boundary() -> None:
    """top_p=0.2 keeps only the FIRST sorted zero, so which of +0.0/-0.0 sorts
    first decides the nucleus mask; fused must match baseline for both input
    orders."""
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    vocab = 64
    for first, second in ((0.0, -0.0), (-0.0, 0.0)):
        logits = torch.full((1, vocab), -30.0, device=device)
        logits[0, 11] = first
        logits[0, 29] = second
        for seed in range(20):
            params = dict(
                temperature=torch.ones(1, device=device),
                top_p=torch.full((1,), 0.2, device=device),
                top_k=torch.full((1,), 8, device=device, dtype=torch.long),
                seeds=torch.full((1,), seed, device=device, dtype=torch.long),
                positions=torch.full((1,), 3, device=device, dtype=torch.long),
            )
            a = sample_seeded_branchless(logits, **params)
            b = sample_seeded_fused(logits, **params)
            assert torch.equal(
                a, b
            ), f"order=({first},{second}) seed={seed}: {a.item()} vs {b.item()}"


@pytest.mark.parametrize("temp", [0.0, 1.0], ids=["greedy", "sampled"])
@pytest.mark.parametrize("kind", ["one_nan", "all_nan", "nan_with_inf"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sample_seeded_fused_nan_rows_match_baseline(temp: float, kind: str) -> None:
    """A NaN row falls back to torch.argmax in the baseline, which ranks NaN
    above every number; the fused kernel must land on the same token id and
    never on the reduction sentinel."""
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    vocab = 64
    gen = torch.Generator(device=device).manual_seed(20260826)
    logits = torch.randn(1, vocab, device=device, generator=gen)
    if kind == "one_nan":
        logits[0, 11] = float("nan")
    elif kind == "all_nan":
        logits[0, :] = float("nan")
    else:
        logits[0, 11] = float("nan")
        logits[0, 3] = float("inf")
    params = dict(
        temperature=torch.full((1,), temp, device=device),
        top_p=torch.full((1,), 0.8, device=device),
        top_k=torch.full((1,), 25, device=device, dtype=torch.long),
        seeds=torch.full((1,), 20260826, device=device, dtype=torch.long),
        positions=torch.full((1,), 5, device=device, dtype=torch.long),
    )
    expected = sample_seeded_branchless(logits, **params)
    actual = sample_seeded_fused(logits, **params)

    assert torch.equal(expected, actual)
    assert int(actual.item()) == int(torch.argmax(logits, dim=-1).item())
    assert 0 <= int(actual.item()) < vocab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tied_two_token_head_agrees_across_sampling_paths() -> None:
    """The production stop head is vocab=2, where a nucleus cut keeps a single
    lane, so the tie order alone decides continue versus stop. All three paths
    that serve it must agree: the graphed fused kernel, the large-vocab
    branchless fallback, and the eager `_sample_tokens` path taken when an audio
    repetition penalty is set or the batch outgrows the frame graph."""
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.sampling_kernels import (
        sample_seeded_branchless,
        sample_seeded_fused,
    )

    device = torch.device("cuda")
    logits = torch.full((1, 2), 2.0, device=device)
    _, order = torch.sort(logits, descending=True, dim=-1, stable=True)
    assert order[0].tolist() == [0, 1]

    for seed in range(16):
        params = dict(
            temperature=torch.ones(1, device=device),
            top_p=torch.full((1,), 0.3, device=device),
            top_k=torch.zeros(1, device=device, dtype=torch.long),
            seeds=torch.full((1,), seed, device=device, dtype=torch.long),
            positions=torch.full((1,), 9, device=device, dtype=torch.long),
        )
        fused = sample_seeded_fused(logits, **params)
        branchless = sample_seeded_branchless(logits, **params)
        eager = MossTTSModelRunner._sample_tokens(logits, **params).view(-1)
        assert int(fused.item()) == 0
        assert torch.equal(fused, branchless), f"branchless seed={seed}"
        assert torch.equal(fused, eager), f"eager seed={seed}"
