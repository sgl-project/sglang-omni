# SPDX-License-Identifier: Apache-2.0
"""Dual-channel generation state and per-request sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

MODALITY_PAD = 151667
SOSP = 151646
EOSP = 16384
AUDIO_PAD = 512
TEXT_ENDOFTEXT = 151643
IM_END = 151645
AUDIO_FORBIDDEN_FROM = 16385
TEXT_VOCAB = 151680
AUDIO_VOCAB = 16512
MODE_TEXT: Literal[0] = 0
MODE_AUDIO: Literal[1] = 1
Mode = Literal[0, 1]


@dataclass(frozen=True, kw_only=True)
class MossSamplingParams:
    """Sampling policy shared by the text and audio channels."""

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0


def initial_mode(prompt_rows: torch.Tensor) -> Mode:
    """The audio padding check takes precedence over modality padding."""
    text_token, audio_token = prompt_rows[-1].tolist()
    if audio_token == AUDIO_PAD:
        return MODE_TEXT
    elif text_token == MODALITY_PAD:
        return MODE_AUDIO
    else:
        return MODE_TEXT


def next_mode(last_row: tuple[int, int] | torch.Tensor, current_mode: Mode) -> Mode:
    """Transition from the last committed row before sampling the next one."""
    if current_mode == MODE_TEXT and int(last_row[0]) == SOSP:
        current_mode = MODE_AUDIO
    else:
        pass
    if current_mode == MODE_AUDIO and int(last_row[1]) == EOSP:
        return MODE_TEXT
    else:
        return current_mode


def repetition_penalty_scores(
    logits: torch.Tensor, history: torch.Tensor, penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or history.numel() == 0:
        return logits
    else:
        scores = logits.clone()
        seen = torch.unique(history)
        seen_scores = scores[seen]
        scores[seen] = torch.where(
            seen_scores < 0, seen_scores * penalty, seen_scores / penalty
        )
        return scores


def apply_warpers(logits: torch.Tensor, params: MossSamplingParams) -> torch.Tensor:
    """Apply temperature, top-k, then the ascending nucleus cutoff."""
    scores = logits / params.temperature
    if params.top_k > 0 and params.top_k < scores.shape[-1]:
        threshold = torch.topk(scores, params.top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    else:
        pass
    if params.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probabilities = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative_probabilities <= 1 - params.top_p
        remove[-1] = False
        scores = scores.masked_fill(
            remove.scatter(0, sorted_indices, remove), float("-inf")
        )
    else:
        pass
    return scores


def sample_channel(
    scores: torch.Tensor, do_sample: bool, generator: torch.Generator | None = None
) -> int:
    if do_sample:
        return int(torch.multinomial(scores.softmax(dim=-1), 1, generator=generator))
    else:
        return int(scores.argmax())


def sample_row(
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    params: MossSamplingParams,
    text_history: torch.Tensor,
    audio_history: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Keep float upcasting, audio constraints, penalties, and RNG draw order."""
    text_scores = text_logits.detach().float().clone()
    audio_scores = audio_logits.detach().float().clone()
    audio_scores[AUDIO_FORBIDDEN_FROM:] = float("-inf")
    text_scores = repetition_penalty_scores(
        text_scores, text_history, params.repetition_penalty
    )
    audio_scores = repetition_penalty_scores(
        audio_scores, audio_history, params.repetition_penalty
    )
    if params.do_sample:
        text_scores = apply_warpers(text_scores, params)
        audio_scores = apply_warpers(audio_scores, params)
    else:
        pass
    return (
        sample_channel(text_scores, params.do_sample, generator),
        sample_channel(audio_scores, params.do_sample, generator),
    )


def finalize_row(text_token: int, audio_token: int, mode: Mode) -> tuple[int, int]:
    if mode == MODE_AUDIO:
        return MODALITY_PAD, audio_token
    else:
        return text_token, audio_token


def row_selected_token(row: tuple[int, int]) -> int:
    return row[1] if row[0] == MODALITY_PAD else row[0]


def channel_histories(
    prompt_rows: torch.Tensor, output_rows: list[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_rows = torch.tensor(output_rows, dtype=torch.long).reshape(-1, 2)
    full_history = torch.cat([prompt_rows.to(torch.long), generated_rows], dim=0)
    return full_history[:, 0].contiguous(), full_history[:, 1].contiguous()
