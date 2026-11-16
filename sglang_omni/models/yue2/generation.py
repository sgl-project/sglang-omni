# SPDX-License-Identifier: Apache-2.0
"""YuE2 logit shaping: phase vocabulary, end gating and windowed repetition penalty."""

from functools import cache
from typing import Literal

import torch

from sglang_omni.models.yue2.request import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    MUSIC_END,
    TEXT_END,
    SamplingConfig,
)


@cache
def vocabulary_bias(
    phase: Literal["abc", "semantic"],
    vocabulary_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Zero on the tokens a phase may emit and -inf elsewhere."""
    bias = torch.full((vocabulary_size,), -torch.inf, dtype=dtype, device=device)
    if phase == "abc":
        bias[:TEXT_END] = 0
        bias[ABC_END] = 0
    else:
        bias[CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
        bias[MUSIC_END] = 0
    return bias


def masked_scores(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    recent_tokens: torch.Tensor,
    allow_end: torch.Tensor,
    phase: Literal["abc", "semantic"],
    legacy_off: bool,
) -> torch.Tensor:
    """Mask the phase vocabulary and penalize tokens in each row's recent window.

    Args:
        logits: [rows, vocabulary] next-token logits.
        recent_tokens: [rows, window] recently sampled tokens; -1 marks an empty slot.
        allow_end: [rows] whether each row may emit its phase's end token.
    """
    dtype = torch.bfloat16 if legacy_off else torch.float32
    vocabulary_size = logits.shape[-1]
    scores = logits.to(dtype) + vocabulary_bias(
        phase, vocabulary_size, dtype, logits.device
    )
    end_token = ABC_END if phase == "abc" else MUSIC_END
    scores[:, end_token] = torch.where(allow_end, scores[:, end_token], -torch.inf)
    if sampling.repetition_penalty != 1.0:
        filled = recent_tokens >= 0
        # note (ruiling): empty slots point at the last id, which neither phase may emit, so rewriting it is a no-op.
        tokens = torch.where(filled, recent_tokens, vocabulary_size - 1)
        same_token = tokens[:, :, None] == tokens[:, None, :]
        counts = (same_token & filled[:, None, :]).sum(-1) * filled
        penalty = sampling.repetition_penalty ** counts.to(dtype)
        current = scores.gather(-1, tokens)
        scores.scatter_(
            -1,
            tokens,
            torch.where(current < 0, current * penalty, current / penalty),
        )
    else:
        pass
    return scores
