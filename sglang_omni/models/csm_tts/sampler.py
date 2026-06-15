# SPDX-License-Identifier: Apache-2.0
# Sampling / EOS semantics follow HuggingFace Transformers `transformers/models/csm` (Apache-2.0).
"""Branchless batched sampling + EOS state machine for CSM.

Simpler than higgs by design: CSM has no delay pattern and no EOC wind-down,
so the three-phase higgs machine collapses to one transition
(``eos_now = all(codes[:, :31] == 0)``; done rows freeze and emit
``STOP_CODE``). Dual sampling param sets: cb0 (rides SGLang's sampling_info)
and depth (CSM-private, from ``req._omni_data``).


"""

from __future__ import annotations

import torch

from sglang_omni.models.csm_tts.utils import (
    CODEBOOK_EOS,
    K_MAX,
    NUM_CODEBOOKS,
    STOP_CODE,
)

__all__ = [
    "CsmBatchedSamplerState",
    "K_MAX",
    "STAGING_WIDTH",
    "frame_finalize_direct",
    "sample_codes_batched",
    "step_reference",
]

# Greedy short-circuit threshold (higgs sampler.py parity).
_GREEDY_TEMP_THRESHOLD = 1e-5

# Packed D2H staging row layout: ``[c0..c31 | was_done |
# generation_done]`` int64 — model.py allocates ``_cg_collect_staging
# [P, STAGING_WIDTH]``; one blocking D2H per decode step pulls everything.
STAGING_WIDTH = NUM_CODEBOOKS + 2


class CsmBatchedSamplerState:
    """Pool-row sampler state surviving across decode steps (higgs row
    discipline: rows are acquired/released by rid while batch composition
    changes under a fixed-shape CUDA graph).

    Per row (pool size ``P``; last row = padding row):

    - ``last_codes``: int64 ``[P, 32]`` — previous frame, feeds
      ``CsmFrameEmbedding`` on the next decode step.
    - ``generation_done``: bool ``[P]`` — EOS latched; row frozen.
    - ``frames_emitted``: int32 ``[P]`` — frames appended so far.

    No ``delay_count`` / ``eoc_countdown`` — CSM has neither.
    """

    def __init__(self, pool_size: int, device: torch.device | str = "cuda") -> None:
        """Allocate ``last_codes [P, 32]`` int64, ``generation_done [P]`` bool,
        ``frames_emitted [P]`` int32 on ``device``."""
        self.pool_size = int(pool_size)
        self.device = torch.device(device)
        self.last_codes = torch.zeros(
            self.pool_size, NUM_CODEBOOKS, dtype=torch.int64, device=self.device
        )
        self.generation_done = torch.zeros(
            self.pool_size, dtype=torch.bool, device=self.device
        )
        self.frames_emitted = torch.zeros(
            self.pool_size, dtype=torch.int32, device=self.device
        )

    def reset_row(self, row: int) -> None:
        """Reset one pool row to the fresh-request state
        (``last_codes=0``, ``generation_done=False``, ``frames_emitted=0``)."""
        self.last_codes[row].zero_()
        self.generation_done[row] = False
        self.frames_emitted[row] = 0


def sample_codes_batched(
    logits_BV_fp32: torch.Tensor,
    temperature_B: torch.Tensor,
    top_k_buf_B: torch.Tensor,
    top_p_B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Branchless per-row greedy/top-k sampling (higgs
    ``_sample_independent_batched`` with ``K_MAX=2051``).

    Greedy rows (``temperature <= 1e-5`` or ``top_k == 1``) short-circuit
    branchlessly to argmax over RAW logits; sampled rows use a fixed-shape
    ``topk(K_MAX)`` + per-row k-th-value gather (CG-deterministic at temp=0).
    Used for BOTH cb0 and every one of the 31 depth steps.

    Args:
        logits_BV_fp32: fp32 ``[B, 2051]`` (callers cast before this).
        temperature_B: fp32 ``[B]``.
        top_k_buf_B: int64 ``[B]`` (neutral rows carry ``K_MAX``).
        top_p_B: optional fp32 ``[B]``.

    Returns:
        int64 ``[B]`` sampled codes in ``[0, 2050]``.
    """
    B = logits_BV_fp32.shape[0]

    # Per-row greedy mask; argmax over RAW logits exactly like the higgs
    # per-row reference. Selection is branchless (compute both, torch.where)
    # because this runs inside the captured CUDA graph.
    greedy_B = (temperature_B <= _GREEDY_TEMP_THRESHOLD) | (top_k_buf_B == 1)
    argmax_B = logits_BV_fp32.argmax(dim=-1)

    safe_temp = temperature_B.clamp(min=_GREEDY_TEMP_THRESHOLD).view(B, 1)
    logits = logits_BV_fp32 / safe_temp

    # Fixed-shape top-k: always topk(full width) + per-row k-th-value gather.
    # K_MAX (= 2051) in production; clamped to the logits width so tiny-config
    # tests run the same code path. Still
    # shape-static under CUDA-graph capture (V is fixed per graph).
    k_width = min(K_MAX, int(logits_BV_fp32.shape[-1]))
    top_vals = logits.topk(k_width, dim=-1).values
    k_idx = top_k_buf_B.view(B, 1).clamp(min=1, max=k_width) - 1
    kth = top_vals.gather(-1, k_idx)
    logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p_B is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p_B.view(B, 1)
        # Shift right + force-keep top token so the highest-prob token never
        # gets cut.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)

    probs = logits.softmax(dim=-1)
    sampled_B = probs.multinomial(num_samples=1).squeeze(-1)

    return torch.where(greedy_B, argmax_B, sampled_B).to(torch.long)


def frame_finalize_direct(
    codes_B32: torch.Tensor,
    generation_done_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The branchless EOS machine (single transition).

    ``eos_now = (codes[:, :31] == CODEBOOK_EOS).all(-1)`` — cb31 EXCLUDED,
    HF-exact (stop(31)/trim(32) inconsistency is mirrored downstream);
    ``new_done = done | eos_now``; ``out_codes = where(was_done, STOP_CODE,
    codes)``; done rows freeze. Runs on decode steps AND on the
    prefill-sampled frame 0 (frame-0 EOS is legal + empirically real).

    Args:
        codes_B32: int64 ``[B, 32]`` freshly sampled frame.
        generation_done_B: bool ``[B]`` done flags BEFORE this frame.

    Returns:
        ``(out_codes_B32 int64 [B, 32], new_done_B bool [B],
        was_done_B bool [B])``.
    """
    # Clone: callers scatter new_done back into the same CG buffer this view
    # came from; was_done must survive that write.
    was_done_B = generation_done_B.clone()
    eos_now_B = (codes_B32[:, : NUM_CODEBOOKS - 1] == CODEBOOK_EOS).all(dim=-1)
    new_done_B = was_done_B | eos_now_B
    stop = torch.full_like(codes_B32, STOP_CODE)
    out_codes_B32 = torch.where(was_done_B.unsqueeze(-1), stop, codes_B32)
    return out_codes_B32, new_done_B, was_done_B


def step_reference(
    codes_32: torch.Tensor,
    generation_done: bool,
) -> tuple[torch.Tensor, bool, bool]:
    """Per-row eager mirror of :func:`frame_finalize_direct` for parity tests
    (higgs pattern: tests assert per-row vs batched equality across
    running / eos_now / done-freeze / mixed-batch phases).

    Args:
        codes_32: int64 ``[32]``.
        generation_done: row done flag before this frame.

    Returns:
        ``(out_codes_32 int64 [32], new_done bool, was_done bool)``.
    """
    was_done = bool(generation_done)
    if was_done:
        return torch.full_like(codes_32, STOP_CODE), True, True
    eos_now = bool((codes_32[: NUM_CODEBOOKS - 1] == CODEBOOK_EOS).all().item())
    return codes_32.clone(), eos_now, False
