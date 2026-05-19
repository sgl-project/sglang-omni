# SPDX-License-Identifier: Apache-2.0
"""Multi-codebook sampler state machine for Higgs TTS.

Pure torch / pure Python so it can be unit-tested in isolation from sglang.

Per-request algorithm each step (codebook logits ``[N, V]`` in, codes
``[N]`` out):

1. If ``generation_done``: return ``[-1, ..., -1]`` (stop signal).
2. Sample ``N`` codebooks independently from the logits (temperature / top-k /
   top-p / multinomial; or argmax when temperature <= 0).
3. **Delay window** (``delay_count < N``): force codebooks at indices
   ``> delay_count`` to :data:`BOC_ID`. Increment ``delay_count``.
4. **Wind-down** (``eoc_countdown is not None``): free sampling, decrement.
   When the counter hits 0, set ``generation_done``.
5. **EOC detection**: if codebook-0's sampled code equals :data:`EOC_ID`,
   start wind-down (``eoc_countdown = N - 2``); for ``N <= 2`` mark done
   immediately.
6. Update ``last_codes`` unless ``generation_done`` was just set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.higgs_tts.utils import BOC_ID, EOC_ID

# Sentinel returned by ``step`` after ``generation_done``; engine treats as stop.
STOP_CODE = -1


@dataclass
class HiggsSamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Batched (CUDA-Graph-compatible) sampler state
# ---------------------------------------------------------------------------


class HiggsBatchedSamplerState:
    """Per-request sampler state stored as ``[max_bs, ...]`` GPU tensors.

    This is the storage half of the CUDA Graph migration (Stage 1). The
    sampler itself still runs the Python state machine in :func:`step`
    on a per-row :class:`HiggsSamplerState`; Stage 2 vectorises the step
    so it operates on this batched state directly.

    Per-row meaning (matches :class:`HiggsSamplerState`):

    - ``delay_count[i]``: how many AR steps row ``i`` has produced so far.
      While ``delay_count < num_codebooks`` we're in the delay window.
    - ``eoc_countdown[i]``: ``-1`` when cb0 hasn't emitted EOC yet, else
      remaining wind-down steps. Once it hits ``0`` we set
      ``generation_done[i] = True``.
    - ``generation_done[i]``: terminal flag; the model runner reads this
      back each step and sets ``Req.finished_reason``.
    - ``last_codes[i]``: last sampled multi-codebook row, used by the
      model's decode-step input overlay.
    """

    def __init__(
        self,
        max_batch_size: int,
        num_codebooks: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.max_batch_size = int(max_batch_size)
        self.num_codebooks = int(num_codebooks)
        self.device = torch.device(device)
        self.delay_count = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )
        self.eoc_countdown = torch.full(
            (self.max_batch_size,), -1, dtype=torch.int32, device=self.device
        )
        self.generation_done = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device=self.device
        )
        self.last_codes = torch.zeros(
            self.max_batch_size,
            self.num_codebooks,
            dtype=torch.long,
            device=self.device,
        )

    def reset_row(self, row: int) -> None:
        """Wipe row ``row`` back to its initial state.

        Called when a slot is acquired for a new request (so a previously
        finished or aborted request can't leave stale flags behind).
        """
        self.delay_count[row] = 0
        self.eoc_countdown[row] = -1
        self.generation_done[row] = False
        self.last_codes[row].zero_()

    def view_row(self, row: int) -> HiggsSamplerState:
        """Materialise row ``row`` as a per-request :class:`HiggsSamplerState`.

        Stage 1 transitional helper: the existing :func:`step` is per-row,
        so we read out one row's tensors as Python scalars + a 1-D tensor,
        run the step, then call :meth:`write_row` to commit changes. Stage
        2 replaces this with a true batched step that mutates the pool
        tensors in place.

        ``last_codes`` is returned as ``None`` while ``delay_count == 0``
        (i.e. the row hasn't produced any AR steps yet) to match the old
        per-request dict's "freshly constructed" semantics. The model
        runner uses that signal to fall back to text-only embed at decode
        time for never-sampled rows.
        """
        delay = int(self.delay_count[row].item())
        eoc = int(self.eoc_countdown[row].item())
        return HiggsSamplerState(
            num_codebooks=self.num_codebooks,
            delay_count=delay,
            eoc_countdown=None if eoc < 0 else eoc,
            generation_done=bool(self.generation_done[row].item()),
            last_codes=None if delay == 0 else self.last_codes[row],
        )

    def write_row(self, row: int, state: HiggsSamplerState) -> None:
        """Commit a per-row :class:`HiggsSamplerState` back to the pool."""
        self.delay_count[row] = state.delay_count
        self.eoc_countdown[row] = (
            -1 if state.eoc_countdown is None else state.eoc_countdown
        )
        self.generation_done[row] = state.generation_done
        if state.last_codes is not None:
            self.last_codes[row].copy_(state.last_codes.to(self.last_codes.dtype))


_GREEDY_TEMP_THRESHOLD = 1e-5


def _sample_independent(
    logits_NV: torch.Tensor,
    *,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> torch.Tensor:
    # Short-circuit greedy to dodge the inf/NaN from logits / tiny_temperature.
    if temperature <= _GREEDY_TEMP_THRESHOLD:
        return logits_NV.argmax(dim=-1)

    logits = logits_NV / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        # Shift right + force-keep top token so the highest-prob token never gets cut.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)

    probs = logits.softmax(dim=-1)
    return probs.multinomial(num_samples=1).squeeze(-1)


def step(
    logits_NV: torch.Tensor,
    state: HiggsSamplerState,
    *,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    boc_id: int = BOC_ID,
    eoc_id: int = EOC_ID,
) -> torch.Tensor:
    """Run one AR step of the multi-codebook sampler.

    Mutates ``state`` in place.

    Args:
        logits_NV: Model logits for this step, shape ``[N, V_codebook]``.
        state: Per-request :class:`HiggsSamplerState`. Must have
            ``state.num_codebooks == N``.

    Returns:
        Sampled codes of shape ``[N]``. If the request has already finished,
        returns a tensor of :data:`STOP_CODE` (``-1``) sentinels.
    """
    N = state.num_codebooks
    if logits_NV.ndim != 2 or logits_NV.shape[0] != N:
        raise ValueError(
            f"logits shape {tuple(logits_NV.shape)} incompatible with num_codebooks={N}"
        )

    if state.generation_done:
        return torch.full((N,), STOP_CODE, dtype=torch.long, device=logits_NV.device)

    codes_N = _sample_independent(
        logits_NV,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ).to(torch.long)

    if state.delay_count < N:
        next_cb = state.delay_count + 1
        if next_cb < N:
            codes_N[next_cb:] = boc_id
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_N[0].item()) == eoc_id:
        if N <= 2:
            state.generation_done = True
        else:
            state.eoc_countdown = N - 2

    if not state.generation_done:
        state.last_codes = codes_N.clone()

    return codes_N


# ---------------------------------------------------------------------------
# Batched (CUDA-Graph-friendly) sampler step — Stage 2
# ---------------------------------------------------------------------------


def _sample_independent_batched(
    logits_BNV: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor | None,
    top_k: int | None,
) -> torch.Tensor:
    """Batched analogue of :func:`_sample_independent`.

    Operates on ``logits_BNV`` of shape ``[B, N, V]`` and returns codes
    ``[B, N]``. ``temperature`` is per-row ``[B]``; ``top_p`` is
    per-row ``[B]`` or ``None``; ``top_k`` is a scalar (uniform across
    the batch) or ``None`` for "no top-k". Heterogeneous ``top_k``
    across the batch is intentionally not supported in Stage 2 — every
    Higgs request uses the same value in practice, and a uniform
    ``topk(...)`` call keeps the kernel sequence CUDA-Graph stable.

    No Python control flow (other than the static ``top_k`` / ``top_p``
    None checks evaluated once at trace time), so this body is safe to
    capture into a CUDA Graph.
    """
    B, N, V = logits_BNV.shape
    # Per-row temperature scaling. We do NOT short-circuit greedy
    # because the resulting Python branch would break graph capture;
    # callers wanting deterministic argmax should pass temperature very
    # low (e.g. 1e-3) — the softmax then concentrates ~all probability
    # on the argmax token.
    safe_temp = temperature.clamp(min=_GREEDY_TEMP_THRESHOLD).view(B, 1, 1)
    logits = logits_BNV / safe_temp

    if top_k is not None and top_k > 0:
        k = min(int(top_k), V)
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p is not None:
        # top_p as per-row [B] tensor; broadcast over (N, V)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        thresh = top_p.view(B, 1, 1)
        remove = cum_probs > thresh
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)

    probs = logits.softmax(dim=-1)
    # multinomial wants 2-D input; collapse the [B, N] rows then reshape.
    codes_flat = probs.reshape(B * N, V).multinomial(num_samples=1).squeeze(-1)
    return codes_flat.view(B, N).to(torch.long)


def batched_step(
    logits_BNV: torch.Tensor,
    state: HiggsBatchedSamplerState,
    row_indices: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor | None = None,
    top_k: int | None = None,
    boc_id: int = BOC_ID,
    eoc_id: int = EOC_ID,
) -> torch.Tensor:
    """Run one AR step for a batch of requests, mutating ``state`` in place.

    Stage 2 of the CUDA Graph migration: vectorised replacement for
    per-row :func:`step`. Identical state machine, expressed entirely
    as tensor ops + ``torch.where`` so the kernel sequence is fixed
    and capturable.

    Args:
        logits_BNV: ``[B, N, V]`` model logits for this AR step.
        state:      :class:`HiggsBatchedSamplerState` pool (``max_bs`` rows).
        row_indices: ``[B]`` int64 mapping batch index → pool row.
        temperature: ``[B]`` float per-row sampling temperature.
        top_p:      ``[B]`` float per-row top-p (or ``None``).
        top_k:      Scalar uniform top-k across the batch (or ``None``).

    Returns:
        ``[B, N]`` int64 sampled codes. Rows whose ``generation_done`` was
        ``True`` on entry produce :data:`STOP_CODE` sentinels; their
        state is left unchanged.
    """
    B, N, V = logits_BNV.shape
    device = logits_BNV.device

    # ----- Gather per-row state ---------------------------------------
    delay_count = state.delay_count[row_indices].to(torch.long)
    eoc_countdown = state.eoc_countdown[row_indices].to(torch.long)
    generation_done = state.generation_done[row_indices]

    # ----- Independent sampling ---------------------------------------
    codes_BN = _sample_independent_batched(
        logits_BNV,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # ----- Delay window: force later codebooks to BOC_ID ---------------
    # Mask matches step()'s ``if delay_count + 1 < N: codes[delay_count+1:] = BOC``:
    # any cb index > delay_count (1-indexed past current frontier) gets BOC.
    cb_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    in_delay = (delay_count < N).unsqueeze(-1)
    delay_mask = in_delay & (cb_idx > delay_count.unsqueeze(-1))
    codes_BN = torch.where(
        delay_mask, torch.full_like(codes_BN, boc_id), codes_BN
    )

    # ----- State machine (all torch ops, no Python branches) -----------
    active = ~generation_done                         # [B]
    in_delay_active = active & (delay_count < N)
    in_winddown_active = active & (eoc_countdown >= 0) & (~in_delay_active)
    cb0_eoc_now_active = (
        active
        & (~in_delay_active)
        & (~in_winddown_active)
        & (codes_BN[:, 0] == eoc_id)
    )

    new_delay_count = torch.where(
        in_delay_active, delay_count + 1, delay_count
    ).to(state.delay_count.dtype)

    # ``N`` is a static module dimension (fixed at model init) so a
    # Python branch on it does NOT break CUDA Graph capture — both
    # arms produce the same kernel sequence each time the graph runs.
    if N > 2:
        new_eoc_countdown = torch.where(
            cb0_eoc_now_active,
            torch.full_like(eoc_countdown, N - 2),
            torch.where(in_winddown_active, eoc_countdown - 1, eoc_countdown),
        )
        done_this_step = in_winddown_active & (new_eoc_countdown <= 0)
    else:
        # N <= 2: per-row step() sets generation_done without writing
        # eoc_countdown, so we mirror that exactly to keep states equal.
        new_eoc_countdown = torch.where(
            in_winddown_active, eoc_countdown - 1, eoc_countdown
        )
        done_this_step = cb0_eoc_now_active | (
            in_winddown_active & (new_eoc_countdown <= 0)
        )
    new_generation_done = generation_done | done_this_step

    new_eoc_countdown = new_eoc_countdown.to(state.eoc_countdown.dtype)

    # ----- last_codes update: only when active and not just-finished ---
    update_codes = (active & (~done_this_step)).unsqueeze(-1)
    prev_last = state.last_codes[row_indices]
    new_last_codes = torch.where(update_codes, codes_BN, prev_last)

    # ----- Scatter back to pool ----------------------------------------
    state.delay_count[row_indices] = new_delay_count
    state.eoc_countdown[row_indices] = new_eoc_countdown
    state.generation_done[row_indices] = new_generation_done
    state.last_codes[row_indices] = new_last_codes

    # ----- Return codes (STOP for rows already done at entry) ----------
    stop = torch.full_like(codes_BN, STOP_CODE)
    return torch.where(generation_done.unsqueeze(-1), stop, codes_BN)


__all__ = [
    "STOP_CODE",
    "HiggsBatchedSamplerState",
    "HiggsSamplerState",
    "batched_step",
    "step",
]
