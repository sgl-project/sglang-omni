# SPDX-License-Identifier: Apache-2.0
"""Branchless batched sampling + EOS state machine for CSM.

EOS is all(codes[:, :31] == 0); codebook 31 is excluded. Done rows emit STOP_CODE.
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
    "finalize_frame",
    "finalize_frame_reference",
    "sample_codes_batched",
]

_GREEDY_TEMP_THRESHOLD = 1e-5

STAGING_WIDTH = NUM_CODEBOOKS + 2  # codes + was_done + generation_done


class CsmBatchedSamplerState:
    """Per-row sampler state across decode steps. Last row is padding."""

    def __init__(self, pool_size: int, device: torch.device | str = "cuda") -> None:
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
        self.last_codes[row].zero_()
        self.generation_done[row] = False
        self.frames_emitted[row] = 0


def sample_codes_batched(
    logits_BV_fp32: torch.Tensor,
    temperature_B: torch.Tensor,
    top_k_buf_B: torch.Tensor,
    top_p_B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row greedy / top-k / optional nucleus sampling."""
    B = logits_BV_fp32.shape[0]

    greedy_B = (temperature_B <= _GREEDY_TEMP_THRESHOLD) | (top_k_buf_B == 1)
    argmax_B = logits_BV_fp32.argmax(dim=-1)

    safe_temp = temperature_B.clamp(min=_GREEDY_TEMP_THRESHOLD).view(B, 1)
    logits = logits_BV_fp32 / safe_temp

    # Full-width top-k, then gather the per-row k-th value.
    k_width = min(K_MAX, int(logits_BV_fp32.shape[-1]))
    top_vals = logits.topk(k_width, dim=-1).values
    k_idx = top_k_buf_B.view(B, 1).clamp(min=1, max=k_width) - 1
    kth = top_vals.gather(-1, k_idx)
    logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p_B is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p_B.view(B, 1)
        # Keep the top token so nucleus never empties the row.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)

    probs = logits.softmax(dim=-1)
    sampled_B = probs.multinomial(num_samples=1).squeeze(-1)

    return torch.where(greedy_B, argmax_B, sampled_B).to(torch.long)


def finalize_frame(
    codes_B32: torch.Tensor,
    generation_done_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latch EOS when codebooks 0..30 are all zero. Done rows emit STOP_CODE."""
    was_done_B = generation_done_B.clone()
    eos_now_B = (codes_B32[:, : NUM_CODEBOOKS - 1] == CODEBOOK_EOS).all(dim=-1)
    new_done_B = was_done_B | eos_now_B
    stop = torch.full_like(codes_B32, STOP_CODE)
    out_codes_B32 = torch.where(was_done_B.unsqueeze(-1), stop, codes_B32)
    return out_codes_B32, new_done_B, was_done_B


def finalize_frame_reference(
    codes_32: torch.Tensor,
    generation_done: bool,
) -> tuple[torch.Tensor, bool, bool]:
    """Per-row mirror of :func:`finalize_frame` for tests."""
    was_done = bool(generation_done)
    if was_done:
        return torch.full_like(codes_32, STOP_CODE), True, True
    eos_now = bool((codes_32[: NUM_CODEBOOKS - 1] == CODEBOOK_EOS).all().item())
    return codes_32.clone(), eos_now, False
