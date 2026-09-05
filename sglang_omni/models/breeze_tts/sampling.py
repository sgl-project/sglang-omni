# SPDX-License-Identifier: Apache-2.0
"""Breeze sampling with request-owned RNG and guidance at every codebook."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 750
    cfg_scale: float = 1.0
    seed: int = 0


def apply_cfg(logits: torch.Tensor, scale: float) -> torch.Tensor:
    """Rows are [conditional, unconditional]; return a single guided row."""
    if logits.shape[0] != 2:
        raise ValueError("Breeze CFG requires exactly two branch rows")
    if scale == 1.0:
        return logits[:1].float()
    if scale == 0.0:
        return logits[1:].float()
    cond, uncond = logits.float().split(1)
    return uncond + scale * (cond - uncond)


def sample_logits(
    logits: torch.Tensor,
    params: SamplingConfig,
    generator: torch.Generator,
    *,
    history: Sequence[int] = (),
    codebook_size: int = 2048,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    scores = logits.float().clone()
    if history and params.repetition_penalty != 1.0:
        ids = torch.tensor(history, device=scores.device, dtype=torch.long)
        selected = scores[:, ids]
        scores[:, ids] = torch.where(
            selected < 0,
            selected * params.repetition_penalty,
            selected / params.repetition_penalty,
        )
    # Reserved audio IDs are not codec codes. Only the backbone's extra EOS
    # class is allowed; depth heads must always sample an actual codec code.
    eos = None if eos_token_id is None else scores[:, eos_token_id].clone()
    scores[:, codebook_size:] = -torch.inf
    if eos is not None:
        scores[:, eos_token_id] = eos
    if params.temperature == 0:
        return scores.argmax(dim=-1)
    scores /= params.temperature
    if 0 < params.top_k < scores.shape[-1]:
        threshold = scores.topk(params.top_k, dim=-1).values[:, -1:]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if params.top_p < 1.0:
        ordered, indices = scores.sort(dim=-1, descending=True)
        remove = ordered.softmax(-1).cumsum(-1) > params.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores.masked_fill_(
            torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf
        )
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)
