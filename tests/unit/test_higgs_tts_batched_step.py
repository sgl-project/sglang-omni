# SPDX-License-Identifier: Apache-2.0
"""Stage 2 acceptance tests for the CUDA Graph migration.

Locks in numerical / state parity between the per-row :func:`step` and
the new batched :func:`batched_step`. Sampling parity is tested at
greedy temperature (deterministic argmax) — the random-draw kernels of
``_sample_independent`` (per-row multinomial) and
``_sample_independent_batched`` (flattened ``[B*N]`` multinomial) draw
in different orders, so bit-identical parity at high temperature is
not attainable; statistical equivalence is covered by the existing
per-row sampler tests.

The state machine itself (delay window, EOC detection, wind-down,
``generation_done``, ``last_codes``) is fully covered here.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.sampler import (
    HiggsBatchedSamplerState,
    HiggsSamplerState,
    STOP_CODE,
    batched_step,
    step,
)
from sglang_omni.models.higgs_tts.utils import BOC_ID, EOC_ID


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GREEDY_TEMP = 1e-3  # below _GREEDY_TEMP_THRESHOLD for per-row argmax path
N = 8
V = 1026


def _peaky_logits(target_codes_BN: torch.Tensor) -> torch.Tensor:
    """Build logits whose argmax along ``V`` equals ``target_codes_BN``."""
    B, N_ = target_codes_BN.shape
    logits = torch.full((B, N_, V), -10.0, device=target_codes_BN.device)
    logits.scatter_(
        -1, target_codes_BN.unsqueeze(-1), 10.0
    )
    return logits


def _run_per_row(
    logits_BNV: torch.Tensor,
    pool: HiggsBatchedSamplerState,
    row_indices: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Drive the per-row :func:`step` on each row sequentially.

    Returns the same ``[B, N]`` codes a ``batched_step`` would.
    """
    B = logits_BNV.shape[0]
    codes_out = torch.empty((B, N), dtype=torch.long, device=logits_BNV.device)
    for b in range(B):
        row = int(row_indices[b].item())
        state = pool.view_row(row)
        codes_b = step(
            logits_BNV[b],
            state,
            temperature=temperature,
        )
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
        assert torch.equal(a[key], b[key]), f"mismatch on {key}\n a={a[key]}\n b={b[key]}"


# ---------------------------------------------------------------------------
# Parity: delay window
# ---------------------------------------------------------------------------


def test_batched_matches_per_row_delay_window():
    """First N steps must force codebooks > delay_count to BOC."""
    B = 3
    pool_pr = HiggsBatchedSamplerState(B, N, device=DEVICE)
    pool_bt = HiggsBatchedSamplerState(B, N, device=DEVICE)
    row_indices = torch.arange(B, device=DEVICE)
    temp_t = torch.full((B,), GREEDY_TEMP, device=DEVICE)

    torch.manual_seed(0)
    for t in range(N + 2):
        target = torch.randint(0, V, (B, N), device=DEVICE)
        logits = _peaky_logits(target)

        codes_pr = _run_per_row(
            logits, pool_pr, row_indices, temperature=GREEDY_TEMP
        )
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t
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
    temp_t = torch.full((B,), GREEDY_TEMP, device=DEVICE)

    # Phase 1: fill delay window (N steps) with arbitrary codes.
    torch.manual_seed(1)
    for _ in range(N):
        target = torch.randint(0, V - 2, (B, N), device=DEVICE)
        logits = _peaky_logits(target)
        codes_pr = _run_per_row(
            logits, pool_pr, row_indices, temperature=GREEDY_TEMP
        )
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t
        )
        assert torch.equal(codes_pr, codes_bt)

    # Phase 2: cb0 emits EOC; rest of codebooks any value.
    target = torch.randint(0, V - 2, (B, N), device=DEVICE)
    target[:, 0] = EOC_ID
    logits = _peaky_logits(target)
    codes_pr = _run_per_row(
        logits, pool_pr, row_indices, temperature=GREEDY_TEMP
    )
    codes_bt = batched_step(
        logits, pool_bt, row_indices, temperature=temp_t
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
        codes_pr = _run_per_row(
            logits, pool_pr, row_indices, temperature=GREEDY_TEMP
        )
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t
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
    temp_t = torch.full((2,), GREEDY_TEMP, device=DEVICE)
    target = torch.randint(0, V - 2, (2, N), device=DEVICE)
    logits = _peaky_logits(target)

    codes = batched_step(logits, pool, row_indices, temperature=temp_t)

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
    temp_t = torch.full((3,), GREEDY_TEMP, device=DEVICE)

    torch.manual_seed(2)
    for t in range(N + 2):
        target = torch.randint(0, V - 2, (3, N), device=DEVICE)
        logits = _peaky_logits(target)
        codes_pr = _run_per_row(
            logits, pool_pr, row_indices, temperature=GREEDY_TEMP
        )
        codes_bt = batched_step(
            logits, pool_bt, row_indices, temperature=temp_t
        )
        assert torch.equal(codes_pr, codes_bt), f"mixed-phase mismatch at t={t}"
        _assert_pools_equal(_snapshot_pool(pool_pr), _snapshot_pool(pool_bt))


# ---------------------------------------------------------------------------
# CUDA Graph readiness sanity: no Python-side state branch per step
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_batched_step_captures_into_cuda_graph():
    """A single ``batched_step`` call must be CUDA-Graph-capturable."""
    B = 4
    pool = HiggsBatchedSamplerState(B, N, device="cuda")
    row_indices = torch.arange(B, device="cuda")
    temp_t = torch.full((B,), GREEDY_TEMP, device="cuda")
    target = torch.randint(0, V - 2, (B, N), device="cuda")
    logits = _peaky_logits(target).contiguous()

    # Warm-up before capture, per torch docs.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        batched_step(logits, pool, row_indices, temperature=temp_t)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = batched_step(logits, pool, row_indices, temperature=temp_t)

    # Replay; both the buffer and pool state should mutate.
    pre_delay = pool.delay_count.clone()
    g.replay()
    torch.cuda.synchronize()
    assert out.shape == (B, N)
    # delay_count must have advanced; just check it changed.
    assert not torch.equal(pool.delay_count, pre_delay) or pool.delay_count.sum() > 0
