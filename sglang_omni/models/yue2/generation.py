# SPDX-License-Identifier: Apache-2.0
"""YuE2 token filtering: phase vocabulary, end gating, windowed penalty, top-k/top-p."""

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


def filter_scores(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    frequencies: torch.Tensor,
    allow_end: torch.Tensor,
    phase: Literal["abc", "semantic"],
    legacy_off: bool,
) -> torch.Tensor:
    scores = logits.clone() if legacy_off else logits.float().clone()
    end_token = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[..., :TEXT_END] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end_token] = 0
    scores = scores + allowed
    scores[..., end_token] = torch.where(allow_end, scores[..., end_token], -torch.inf)
    if sampling.repetition_penalty != 1.0:
        penalty = sampling.repetition_penalty**frequencies
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    else:
        pass
    if sampling.temperature == 0:
        return scores
    else:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., : 3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = scores.scatter(-1, indices, values)
    else:
        pass
    return scores
