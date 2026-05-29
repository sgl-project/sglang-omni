# SPDX-License-Identifier: Apache-2.0
"""Parity + CG-capture tests for batched_step.

Sampling parity uses ``top_k=1`` to force greedy choices; stochastic
multinomial parity is not asserted because per-row and batched modes draw
in different orders. State-machine parity is exact.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.sampler import (
    K_MAX,
    STOP_CODE,
    HiggsBatchedSamplerState,
    batched_step,
    step,
)
from sglang_omni.models.higgs_tts.utils import EOC_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# top_k=1 forces greedy (only argmax stays finite after the filter), matching
# sglang's normalization of ``SamplingParams(temperature=0)`` to ``top_k=1``.
GREEDY_TOP_K = 1
N = 8
V = 1026


def _peaky_logits(target_codes_BN: torch.Tensor) -> torch.Tensor:
    """Build logits whose argmax along ``V`` equals ``target_codes_BN``."""
    B, N_ = target_codes_BN.shape
    logits = torch.full((B, N_, V), -10.0, device=target_codes_BN.device)
    logits.scatter_(-1, target_codes_BN.unsqueeze(-1), 10.0)
    return logits


def _run_per_row(
    logits_BNV: torch.Tensor,
    pool: HiggsBatchedSamplerState,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Per-row :func:`step` over each row; returns same ``[B, N]`` as batched."""
    B = logits_BNV.shape[0]
    codes_out = torch.empty((B, N), dtype=torch.long, device=logits_BNV.device)
    for b in range(B):
        row = int(row_indices[b].item())
        state = pool.view_row(row)
        codes_b = step(logits_BNV[b], state, top_k=GREEDY_TOP_K)
        pool.write_row(row, state)
        codes_out[b] = codes_b
    return codes_out


def _snapshot_pool(pool: HiggsBatchedSamplerState) -> dict:
    """Snapshot pool tensors for cross-mode equality checks."""
    return {
        "delay_count": pool.delay_count.clone(),
        "eoc_countdown": pool.eoc_countdown.clone(),
        "generation_done": pool.generation_done.clone(),
        "last_codes": pool.last_codes.clone(),
    }


def _assert_pools_equal(a: dict, b: dict) -> None:
    for key in a:
        assert torch.equal(
            a[key], b[key]
        ), f"mismatch on {key}\n a={a[key]}\n b={b[key]}"


# ---------------------------------------------------------------------------
# Parity: delay window
# ---------------------------------------------------------------------------


def test_batched_matches_per_row_delay_window():
    """First N steps must force codebooks > delay_count to BOC."""
    B = 3
    pool_pr = HiggsBatchedSamplerState(B, N, device=DEVICE)
    pool_bt = HiggsBatchedSamplerState(B, N, device=DEVICE)
    row_indices = torch.arange(B, device=DEVICE)
    temp_t = torch.full((B,), 1.0, device=DEVICE)
    top_k_buf = torch.full((B,), GREEDY_TOP_K, dtype=torch.long, device=DEVICE)

    torch.manual_seed(0)
    for t in range(N + 2):
        target = torch.randint(0, V, (B, N), device=DEVICE)
        logits = _peaky_logits(target)

        codes_pr = _run_per_row(logits, pool_pr, row_indices)
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t, top_k_buf=top_k_buf
        )

        assert torch.equal(codes_pr, codes_bt), f"codes mismatch at t={t}"
        _assert_pools_equal(_snapshot_pool(pool_pr), _snapshot_pool(pool_bt))


# ---------------------------------------------------------------------------
# Parity: EOC + wind-down + generation_done flag
# ---------------------------------------------------------------------------


def test_batched_matches_per_row_eoc_winddown():
    """After delay, fire cb0=EOC and verify wind-down + done flag match."""
    B = 2
    pool_pr = HiggsBatchedSamplerState(B, N, device=DEVICE)
    pool_bt = HiggsBatchedSamplerState(B, N, device=DEVICE)
    row_indices = torch.arange(B, device=DEVICE)
    temp_t = torch.full((B,), 1.0, device=DEVICE)
    top_k_buf = torch.full((B,), GREEDY_TOP_K, dtype=torch.long, device=DEVICE)

    # Phase 1: fill delay window (N steps) with arbitrary codes.
    torch.manual_seed(1)
    for _ in range(N):
        target = torch.randint(0, V - 2, (B, N), device=DEVICE)
        logits = _peaky_logits(target)
        codes_pr = _run_per_row(logits, pool_pr, row_indices)
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t, top_k_buf=top_k_buf
        )
        assert torch.equal(codes_pr, codes_bt)

    # Phase 2: cb0 emits EOC; rest of codebooks any value.
    target = torch.randint(0, V - 2, (B, N), device=DEVICE)
    target[:, 0] = EOC_ID
    logits = _peaky_logits(target)
    codes_pr = _run_per_row(logits, pool_pr, row_indices)
    codes_bt = batched_step(
        logits, pool_bt, row_indices, temperature=temp_t, top_k_buf=top_k_buf
    )
    assert torch.equal(codes_pr, codes_bt)
    _assert_pools_equal(_snapshot_pool(pool_pr), _snapshot_pool(pool_bt))
    # eoc_countdown should now be N-2 on both rows.
    assert torch.equal(
        pool_pr.eoc_countdown,
        torch.full_like(pool_pr.eoc_countdown, N - 2),
    )

    # Phase 3: wind down through N-2 more steps until done.
    for k in range(N - 2):
        target = torch.randint(0, V - 2, (B, N), device=DEVICE)
        logits = _peaky_logits(target)
        codes_pr = _run_per_row(logits, pool_pr, row_indices)
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t, top_k_buf=top_k_buf
        )
        assert torch.equal(codes_pr, codes_bt), f"mismatch at wind-down step {k}"
        _assert_pools_equal(_snapshot_pool(pool_pr), _snapshot_pool(pool_bt))

    assert bool(pool_pr.generation_done.all().item())
    assert bool(pool_bt.generation_done.all().item())


# ---------------------------------------------------------------------------
# Done rows: subsequent calls return STOP and leave state untouched
# ---------------------------------------------------------------------------


def test_batched_done_row_returns_stop_and_freezes_state():
    """A row already marked generation_done must return STOP and not mutate."""
    pool = HiggsBatchedSamplerState(2, N, device=DEVICE)
    pool.generation_done[0] = True
    pool.delay_count[0] = 42  # arbitrary sentinel to verify no overwrite
    pool.eoc_countdown[0] = 7
    pool.last_codes[0] = torch.arange(N, device=DEVICE)

    row_indices = torch.tensor([0, 1], device=DEVICE)
    temp_t = torch.full((2,), 1.0, device=DEVICE)
    top_k_buf = torch.full((2,), GREEDY_TOP_K, dtype=torch.long, device=DEVICE)
    target = torch.randint(0, V - 2, (2, N), device=DEVICE)
    logits = _peaky_logits(target)

    codes = batched_step(
        logits, pool, row_indices, temperature=temp_t, top_k_buf=top_k_buf
    )

    # Row 0 must return STOP and have unchanged state.
    assert torch.equal(
        codes[0], torch.full((N,), STOP_CODE, device=DEVICE, dtype=torch.long)
    )
    assert int(pool.delay_count[0].item()) == 42
    assert int(pool.eoc_countdown[0].item()) == 7
    assert torch.equal(
        pool.last_codes[0], torch.arange(N, device=DEVICE, dtype=torch.long)
    )

    # Row 1 should have advanced (delay window).
    assert int(pool.delay_count[1].item()) == 1


# ---------------------------------------------------------------------------
# Mixed batch: each row in a different phase, still parity
# ---------------------------------------------------------------------------


def test_batched_matches_per_row_mixed_phases():
    """One row mid-delay, one mid-winddown, one fresh — batched == per-row."""
    pool_pr = HiggsBatchedSamplerState(3, N, device=DEVICE)
    pool_bt = HiggsBatchedSamplerState(3, N, device=DEVICE)

    # Row 0: fresh.
    # Row 1: mid-delay (delay_count = N//2).
    for pool in (pool_pr, pool_bt):
        pool.delay_count[1] = N // 2
        pool.last_codes[1] = torch.arange(N, device=DEVICE)
    # Row 2: mid-winddown.
    for pool in (pool_pr, pool_bt):
        pool.delay_count[2] = N
        pool.eoc_countdown[2] = N - 4
        pool.last_codes[2] = torch.arange(N, device=DEVICE) + 10

    row_indices = torch.arange(3, device=DEVICE)
    temp_t = torch.full((3,), 1.0, device=DEVICE)
    top_k_buf = torch.full((3,), GREEDY_TOP_K, dtype=torch.long, device=DEVICE)

    torch.manual_seed(2)
    for t in range(N + 2):
        target = torch.randint(0, V - 2, (3, N), device=DEVICE)
        logits = _peaky_logits(target)
        codes_pr = _run_per_row(logits, pool_pr, row_indices)
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t, top_k_buf=top_k_buf
        )
        assert torch.equal(codes_pr, codes_bt), f"mixed-phase mismatch at t={t}"
        _assert_pools_equal(_snapshot_pool(pool_pr), _snapshot_pool(pool_bt))


# ---------------------------------------------------------------------------
# Per-row top_k regression
# ---------------------------------------------------------------------------


def test_batched_step_mixed_top_k_per_row_filter():
    """Regression: eager path must accept heterogeneous top_k per row.
    Row 0 uses ``K_MAX`` (no-op filter, mirrors ``top_k=None``); row 1
    uses ``top_k=5``. Row 0 logits have 10 strong cols, row 1 has 5;
    each row's samples must land in its own strong-set.
    """
    if not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    torch.manual_seed(0)
    B = 2
    pool = HiggsBatchedSamplerState(B, N, device="cuda")
    pool.delay_count.fill_(N)
    row_indices = torch.arange(B, device="cuda")
    temp = torch.full((B,), 1.0, device="cuda")

    row0_strong = torch.tensor(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29], device="cuda", dtype=torch.long
    )  # 10 cols, top_k=K_MAX keeps all
    row1_strong = torch.tensor(
        [101, 103, 107, 109, 113], device="cuda", dtype=torch.long
    )  # 5 cols, top_k=5 keeps all of these

    logits = torch.full((B, N, V), -1e9, device="cuda")
    logits[0, :, row0_strong] = 5.0
    logits[1, :, row1_strong] = 5.0

    top_k_buf = torch.tensor([K_MAX, 5], dtype=torch.long, device="cuda")

    # Must not raise — old code checked uniformity of top_k across rows.
    codes = batched_step(
        logits.contiguous(),
        pool,
        row_indices,
        temperature=temp,
        top_k_buf=top_k_buf,
    )

    row0_allowed = set(row0_strong.tolist())
    row1_allowed = set(row1_strong.tolist())
    for cb in range(N):
        assert int(codes[0, cb].item()) in row0_allowed, (
            f"row 0 cb {cb} sampled {int(codes[0, cb].item())} "
            f"outside its own strong-set {row0_allowed}"
        )
        assert int(codes[1, cb].item()) in row1_allowed, (
            f"row 1 cb {cb} sampled {int(codes[1, cb].item())} "
            f"outside its own strong-set {row1_allowed}"
        )


# ---------------------------------------------------------------------------
# Greedy short-circuit determinism (T4): temperature=0 / top_k=1 -> argmax,
# RNG-free and reproducible (the batched sampler used to always go through
# multinomial, making temperature=0 decode non-deterministic run-to-run).
# ---------------------------------------------------------------------------


def _tie_logits(B: int, device: str) -> torch.Tensor:
    """Logits with an EXACT two-way tie for the max in every (row, codebook),
    so multinomial would break the tie randomly but argmax is deterministic."""
    logits = torch.full((B, N, V), -10.0, device=device)
    logits[..., 5] = 9.0
    logits[..., 7] = 9.0  # exact tie with index 5
    return logits


def test_batched_greedy_temperature_zero_is_deterministic_argmax():
    from sglang_omni.models.higgs_tts.sampler import _sample_independent_batched

    B = 4
    logits = _tie_logits(B, DEVICE)
    temperature = torch.zeros(B, device=DEVICE)
    expected = logits.argmax(dim=-1)
    outs = [
        _sample_independent_batched(logits, temperature=temperature, top_p=None)
        for _ in range(100)
    ]
    for o in outs:
        assert torch.equal(o, expected), "temperature=0 must be deterministic argmax"


def test_batched_greedy_top_k_one_is_argmax():
    from sglang_omni.models.higgs_tts.sampler import _sample_independent_batched

    B = 4
    logits = _tie_logits(B, DEVICE)
    temperature = torch.full((B,), 1.0, device=DEVICE)  # NOT temp-greedy
    top_k_buf = torch.ones(B, dtype=torch.long, device=DEVICE)  # top_k == 1
    expected = logits.argmax(dim=-1)
    outs = [
        _sample_independent_batched(
            logits, temperature=temperature, top_p=None, top_k_buf=top_k_buf
        )
        for _ in range(50)
    ]
    for o in outs:
        assert torch.equal(o, expected), "top_k=1 must collapse to argmax"


def test_batched_mixed_greedy_rows_deterministic_sampled_rows_free():
    from sglang_omni.models.higgs_tts.sampler import _sample_independent_batched

    B = 4
    logits = _tie_logits(B, DEVICE)
    # rows 0 & 2 greedy (temp 0); rows 1 & 3 stochastic (temp 1)
    temperature = torch.tensor([0.0, 1.0, 0.0, 1.0], device=DEVICE)
    expected = logits.argmax(dim=-1)
    for _ in range(50):
        o = _sample_independent_batched(logits, temperature=temperature, top_p=None)
        assert torch.equal(o[0], expected[0])
        assert torch.equal(o[2], expected[2])
        # stochastic rows still pick a tied-max token (5 or 7), never a -10 one
        for b in (1, 3):
            assert set(o[b].tolist()) <= {5, 7}
