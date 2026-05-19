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


__all__ = [
    "STOP_CODE",
    "HiggsBatchedSamplerState",
    "HiggsSamplerState",
    "step",
]
