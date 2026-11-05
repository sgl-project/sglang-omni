# SPDX-License-Identifier: Apache-2.0
"""Request-local autoregressive sampling for YuE2."""

from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.models.yue2.layers import TokenCache
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    MUSIC_END,
    TEXT_END,
    CancellationCheck,
    SamplingConfig,
    check_cancelled,
)


@dataclass(kw_only=True)
class TokenResult:
    tokens: list[int]
    truncated: bool


def sampling_scores(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    history: list[int],
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
    if len(history) < sampling.min_tokens:
        scores[..., end_token] = -torch.inf
    else:
        pass
    if sampling.repetition_penalty != 1.0 and history:
        recent = torch.tensor(
            [history[-sampling.penalty_window :]], device=scores.device
        )
        frequencies = torch.zeros_like(scores)
        frequencies.scatter_add_(
            -1, recent, torch.ones_like(recent, dtype=scores.dtype)
        )
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


@torch.inference_mode()
def generate_tokens(
    model: Yue2Model,
    prefix: list[int],
    sampling: SamplingConfig,
    seed: int,
    phase: Literal["abc", "semantic"],
    cancelled: CancellationCheck,
    *,
    negative: list[int] | None = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
) -> TokenResult:
    if cfg_scale == 1:
        prefixes = [prefix]
    else:
        assert negative is not None
        prefixes = [prefix, negative]
    if any(
        len(tokens) + sampling.max_tokens > model.config.max_position_embeddings
        for tokens in prefixes
    ):
        raise ValueError(
            "YuE2 request: prefix and token budget exceed the model context"
        )
    else:
        pass
    check_cancelled(cancelled)
    device = model.model.embed_tokens.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    positive_cache = TokenCache(capacity=len(prefix) + sampling.max_tokens)
    conditional = model(torch.tensor([prefix], device=device), positive_cache)
    negative_cache = None
    unconditional = None
    if cfg_scale != 1:
        assert negative is not None
        negative_cache = TokenCache(capacity=len(negative) + sampling.max_tokens)
        unconditional = model(torch.tensor([negative], device=device), negative_cache)
    else:
        pass
    history: list[int] = []
    end_token = ABC_END if phase == "abc" else MUSIC_END
    for step in range(sampling.max_tokens):
        check_cancelled(cancelled)
        if unconditional is None:
            logits = conditional
        else:
            logits = unconditional + cfg_scale * (conditional - unconditional)
        scores = sampling_scores(logits, sampling, history, phase, legacy_off)
        if sampling.temperature == 0:
            if not torch.isfinite(scores.max()):
                raise FloatingPointError("YuE2 produced no finite selectable logits")
            else:
                next_token = scores.argmax(-1, keepdim=True)
        else:
            next_token = torch.multinomial(scores.softmax(-1), 1, generator=generator)
        token = int(next_token.item())
        if token == end_token:
            return TokenResult(tokens=history, truncated=False)
        else:
            history.append(token)
        if step + 1 < sampling.max_tokens:
            conditional = model(next_token, positive_cache)
            if negative_cache is not None:
                unconditional = model(next_token, negative_cache)
            else:
                pass
        else:
            pass
    return TokenResult(tokens=history, truncated=True)
