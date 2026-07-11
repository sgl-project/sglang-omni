# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS multi-codebook sampler — two parallel implementations of
the same delay/EOC state machine:

- ``step`` / ``HiggsSamplerState``: per-row, Python control flow.
  Reference / test oracle.
- ``batched_step`` / ``batched_step_direct`` / ``HiggsBatchedSamplerState``:
  batched, ``torch.where``-vectorised, CUDA-Graph-friendly. Production.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sgl_kernel import top_k_renorm_prob as _fused_top_k_renorm
from sgl_kernel import top_p_renorm_prob as _fused_top_p_renorm
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.higgs_tts.utils import BOC_ID, EOC_ID

# Sentinel seed for rows with no user seed: keeps the legacy unseeded
# torch.multinomial path, so unseeded decode is byte-identical to before.
NO_SEED = -1

# Sentinel returned by ``step`` after ``generation_done``; engine treats as stop.
STOP_CODE = -1

# CG-baked top-k upper bound = full codec vocab, so the default value is a no-op filter.
K_MAX = 1026


@dataclass
class HiggsSamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None


class HiggsBatchedSamplerState:
    """Per-request sampler state stored as ``[max_bs, ...]`` GPU tensors.

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
        # Per-request seed (``NO_SEED`` = unseeded) and monotonic AR step, used
        # to seed each ``(step, codebook)`` draw reproducibly.
        self.seeds = torch.full(
            (self.max_batch_size,), NO_SEED, dtype=torch.long, device=self.device
        )
        self.step_count = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=self.device
        )

    def reset_row(self, row: int) -> None:
        """Wipe row ``row`` so the next owner can't read stale state."""
        self.delay_count[row] = 0
        self.eoc_countdown[row] = -1
        self.generation_done[row] = False
        self.last_codes[row].zero_()
        self.seeds[row] = NO_SEED
        self.step_count[row] = 0

    def view_row(self, row: int) -> HiggsSamplerState:
        """Materialise row ``row`` as a per-request :class:`HiggsSamplerState`.
        ``last_codes`` is ``None`` while ``delay_count == 0`` (never sampled).
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


def _sample_independent_batched(
    logits_BNV: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor | None,
    top_k_buf: torch.Tensor | None = None,
    seeds_B: torch.Tensor | None = None,
    step_B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched ``[B, N, V] → [B, N]`` sampler.

    Greedy rows short-circuit to ``argmax`` over the raw logits — mirroring the
    per-row :func:`_sample_independent` — so they are RNG-free and reproducible.
    A row is greedy when ``temperature <= _GREEDY_TEMP_THRESHOLD`` (or
    ``top_k == 1``). Without this, multinomial on the near-one-hot distribution
    that ``temperature≈0`` produces breaks near-ties differently run-to-run,
    making ``temperature=0`` decode non-deterministic. The selection is
    branchless (compute both, then ``torch.where``) because this runs inside the
    captured CUDA graph, where data-dependent host control flow is illegal.
    """
    B, N, V = logits_BNV.shape

    # Per-row greedy mask (broadcast over codebooks). argmax over RAW logits,
    # exactly as _sample_independent does.
    greedy_B1 = (temperature <= _GREEDY_TEMP_THRESHOLD).view(B, 1)
    if top_k_buf is not None:
        greedy_B1 = greedy_B1 | (top_k_buf == 1).view(B, 1)
    argmax_BN = logits_BNV.argmax(dim=-1)

    safe_temp = temperature.clamp(min=_GREEDY_TEMP_THRESHOLD).view(B, 1, 1)
    logits = logits_BNV / safe_temp

    # PR-D: fused top-k/top-p renormalization replaces full-vocab torch.sort +
    # logit masking. Numerically equivalent to the sort path (max prob diff ~5e-7,
    # identical support across temp/top_k/top_p sweeps); only differs from the prior
    # code at an exact cumsum==top_p boundary, where it uses the standard nucleus
    # convention. Inputs MUST be contiguous fp32 for the flashinfer renorm kernels.
    probs = logits.float().softmax(dim=-1).reshape(B * N, V).contiguous()
    if top_k_buf is not None:
        tk = (
            top_k_buf.view(B, 1)
            .expand(B, N)
            .reshape(B * N)
            .clamp(min=1, max=V)
            .to(torch.int32)
            .contiguous()
        )
        probs = _fused_top_k_renorm(probs, tk)
    if top_p is not None:
        tp = top_p.view(B, 1).expand(B, N).reshape(B * N).to(torch.float32).contiguous()
        probs = _fused_top_p_renorm(probs, tp)

    codes_flat = probs.multinomial(num_samples=1).squeeze(-1)
    if seeds_B is not None:
        # Seeded rows draw deterministically from (seed, step*N + codebook);
        # unseeded rows (seed == NO_SEED) keep the torch.multinomial draw above.
        cb = torch.arange(N, device=logits_BNV.device).view(1, N).expand(B, N)
        positions = (step_B.view(B, 1) * N + cb).reshape(B * N)
        seeds_flat = seeds_B.clamp_min(0).view(B, 1).expand(B, N).reshape(B * N)
        seeded_flat = multinomial_with_seed(
            torch.log(probs), seeds_flat, positions
        ).squeeze(-1)
        has_seed = (seeds_B >= 0).view(B, 1).expand(B, N).reshape(B * N)
        codes_flat = torch.where(has_seed, seeded_flat, codes_flat)
    sampled_BN = codes_flat.view(B, N)

    return torch.where(greedy_B1, argmax_BN, sampled_BN).to(torch.long)


def selected_token_logprobs(
    logits_BNV: torch.Tensor,
    codes_BN: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Selected-action log-prob ``[B, N]`` of the sampled codes"""
    B = logits_BNV.shape[0]
    logits = logits_BNV.float()

    greedy_B1 = (temperature <= _GREEDY_TEMP_THRESHOLD).view(B, 1)
    if top_k_buf is not None:
        greedy_B1 = greedy_B1 | (top_k_buf == 1).view(B, 1)

    safe_temp = temperature.clamp(min=_GREEDY_TEMP_THRESHOLD).view(B, 1, 1)
    eff_temp = torch.where(
        greedy_B1.unsqueeze(-1), torch.ones_like(safe_temp), safe_temp
    )
    logprobs_full = torch.log_softmax(logits / eff_temp, dim=-1)
    return logprobs_full.gather(-1, codes_BN.long().unsqueeze(-1)).squeeze(-1)


def batched_step(
    logits_BNV: torch.Tensor,
    state: HiggsBatchedSamplerState,
    row_indices: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor | None = None,
    top_k_buf: torch.Tensor | None = None,
    boc_id: int = BOC_ID,
    eoc_id: int = EOC_ID,
) -> torch.Tensor:
    """Eager-path wrapper: gather pool state by ``row_indices``, call
    :func:`batched_step_direct`, scatter the new state back. Done rows
    return :data:`STOP_CODE` with state untouched.

    Returns ``out_codes``.
    """
    delay_count = state.delay_count[row_indices]
    eoc_countdown = state.eoc_countdown[row_indices]
    generation_done = state.generation_done[row_indices]
    last_codes = state.last_codes[row_indices]
    seeds = state.seeds[row_indices]
    step_count = state.step_count[row_indices]

    (
        out_codes,
        new_delay_count,
        new_eoc_countdown,
        new_generation_done,
        new_last_codes,
        new_step_count,
    ) = batched_step_direct(
        logits_BNV,
        delay_count,
        eoc_countdown,
        generation_done,
        last_codes,
        temperature=temperature,
        top_p=top_p,
        top_k_buf=top_k_buf,
        seeds=seeds,
        step_count=step_count,
        boc_id=boc_id,
        eoc_id=eoc_id,
    )

    state.delay_count[row_indices] = new_delay_count.to(state.delay_count.dtype)
    state.eoc_countdown[row_indices] = new_eoc_countdown.to(state.eoc_countdown.dtype)
    state.generation_done[row_indices] = new_generation_done
    state.last_codes[row_indices] = new_last_codes
    state.step_count[row_indices] = new_step_count

    return out_codes


def batched_step_direct(
    logits_BNV: torch.Tensor,
    delay_count: torch.Tensor,
    eoc_countdown: torch.Tensor,
    generation_done: torch.Tensor,
    last_codes: torch.Tensor,
    *,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    step_count: torch.Tensor,
    top_p: torch.Tensor | None = None,
    top_k_buf: torch.Tensor | None = None,
    boc_id: int = BOC_ID,
    eoc_id: int = EOC_ID,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """CG-friendly state machine: state in/out as direct ``[B, ...]`` tensors,
    no ``state``/``row_indices`` indirection. Caller persists the returned
    new state. See :func:`batched_step` for arg semantics.

    ``seeds``/``step_count`` (both ``[B]``) make seeded rows reproducible; the
    returned ``new_step_count`` advances active rows for the next step.
    """
    B, N, _ = logits_BNV.shape
    device = logits_BNV.device

    delay_count = delay_count.to(torch.long)
    eoc_countdown = eoc_countdown.to(torch.long)

    codes_BN = _sample_independent_batched(
        logits_BNV,
        temperature=temperature,
        top_p=top_p,
        top_k_buf=top_k_buf,
        seeds_B=seeds,
        step_B=step_count,
    )
    cb_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    in_delay = (delay_count < N).unsqueeze(-1)
    delay_mask = in_delay & (cb_idx > delay_count.unsqueeze(-1))
    codes_BN = torch.where(delay_mask, torch.full_like(codes_BN, boc_id), codes_BN)

    active = ~generation_done
    in_delay_active = active & (delay_count < N)
    in_winddown_active = active & (eoc_countdown >= 0) & (~in_delay_active)
    cb0_eoc_now_active = (
        active & (~in_delay_active) & (~in_winddown_active) & (codes_BN[:, 0] == eoc_id)
    )

    new_delay_count = torch.where(in_delay_active, delay_count + 1, delay_count)

    if N > 2:
        new_eoc_countdown = torch.where(
            cb0_eoc_now_active,
            torch.full_like(eoc_countdown, N - 2),
            torch.where(in_winddown_active, eoc_countdown - 1, eoc_countdown),
        )
        done_this_step = in_winddown_active & (new_eoc_countdown <= 0)
    else:
        new_eoc_countdown = torch.where(
            in_winddown_active, eoc_countdown - 1, eoc_countdown
        )
        done_this_step = cb0_eoc_now_active | (
            in_winddown_active & (new_eoc_countdown <= 0)
        )
    new_generation_done = generation_done | done_this_step

    update_codes = (active & (~done_this_step)).unsqueeze(-1)
    new_last_codes = torch.where(update_codes, codes_BN, last_codes)

    new_step_count = step_count + active.to(step_count.dtype)

    stop = torch.full_like(codes_BN, STOP_CODE)
    out_codes = torch.where(generation_done.unsqueeze(-1), stop, codes_BN)
    return (
        out_codes,
        new_delay_count,
        new_eoc_countdown,
        new_generation_done,
        new_last_codes,
        new_step_count,
    )


__all__ = [
    "K_MAX",
    "STOP_CODE",
    "HiggsBatchedSamplerState",
    "HiggsSamplerState",
    "batched_step",
    "batched_step_direct",
    "selected_token_logprobs",
    "step",
]
