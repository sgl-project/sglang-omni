# SPDX-License-Identifier: Apache-2.0
"""Official-compatible MiMo input-channel and prompt construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

GROUP_SIZE = 4
AUDIO_CHANNELS = 8
SPEECH_EMPTY_IDS = (1024, 1024, 128, 128, 128, 128, 128, 128)

SOSP = "<|sosp|>"
EOSP = "<|eosp|>"
EMPTY = "<|empty|>"


@dataclass(frozen=True)
class MiMoPrompt:
    """Both the official nine-channel representation and SGLang projection."""

    input_ids: list[int]
    audio_codes: torch.Tensor
    official_input_ids: torch.Tensor
    audio_start: int
    audio_end: int


def _token_id(tokenizer: Any, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
    elif hasattr(tokenizer, "token_to_id"):
        token_id = tokenizer.token_to_id(token)
    else:
        raise TypeError("MiMo tokenizer does not expose token ID lookup")
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"MiMo tokenizer is missing required token {token!r}")
    return int(token_id)


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "ids"):
            return [int(token_id) for token_id in encoded.ids]
        return [int(token_id) for token_id in encoded]
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = (
        encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    )
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.reshape(-1).tolist()
    elif input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _expand_text_ids(token_ids: Sequence[int], group_size: int) -> torch.Tensor:
    """Expand each global token to `[token, -100, ...]` timesteps."""

    expanded = torch.full(
        (len(token_ids) * group_size,),
        -100,
        dtype=torch.int64,
    )
    expanded[::group_size] = torch.tensor(token_ids, dtype=torch.int64)
    return expanded


def _speech_empty_rows(length: int, speech_empty_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(speech_empty_ids, dtype=torch.int64)[:, None].expand(-1, length)


def build_text_segment(
    tokenizer: Any,
    text: str,
    *,
    group_size: int = GROUP_SIZE,
    speech_empty_ids: Sequence[int] = SPEECH_EMPTY_IDS,
) -> torch.Tensor:
    """Build an official `[1 + audio_channels, T]` text segment."""

    text_row = _expand_text_ids(_encode_literal(tokenizer, text), group_size)
    return torch.cat(
        [text_row[None, :], _speech_empty_rows(text_row.numel(), speech_empty_ids)],
        dim=0,
    )


def build_audio_segment(
    tokenizer: Any,
    audio_codes: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    speech_empty_ids: Sequence[int] = SPEECH_EMPTY_IDS,
) -> torch.Tensor:
    """Build the official `<|sosp|> codes <|eosp|>` nine-channel segment."""

    codes = torch.as_tensor(audio_codes, dtype=torch.int64)
    audio_channels = len(speech_empty_ids)
    if codes.ndim != 2 or codes.shape[1] != audio_channels:
        raise ValueError(
            "MiMo audio codes must have shape "
            f"[frames, {audio_channels}], got {tuple(codes.shape)}"
        )
    if codes.shape[0] == 0:
        raise ValueError("MiMo audio codes must contain at least one frame")
    if codes.shape[0] % group_size:
        raise ValueError(
            f"MiMo audio-code frames must be divisible by group_size={group_size}, "
            f"got {codes.shape[0]}"
        )

    audio_frames = codes.T.contiguous()
    group_count = codes.shape[0] // group_size
    text_tokens = [
        _token_id(tokenizer, SOSP),
        *([_token_id(tokenizer, EMPTY)] * group_count),
        _token_id(tokenizer, EOSP),
    ]
    text_row = _expand_text_ids(text_tokens, group_size)
    boundary = _speech_empty_rows(group_size, speech_empty_ids)
    speech_rows = torch.cat([boundary, audio_frames, boundary], dim=1)
    return torch.cat([text_row[None, :], speech_rows], dim=0)


def pad_audio_codes(
    audio_codes: torch.Tensor, *, group_size: int = GROUP_SIZE
) -> torch.Tensor:
    """Repeat the last official tokenizer frame to complete a patch group."""

    codes = torch.as_tensor(audio_codes, dtype=torch.int64)
    if codes.ndim != 2 or codes.shape[0] == 0:
        raise ValueError("MiMo audio codes must be a non-empty rank-2 tensor")
    remainder = codes.shape[0] % group_size
    if remainder == 0:
        return codes
    return torch.cat(
        [codes, codes[-1:].expand(group_size - remainder, -1)],
        dim=0,
    )


def build_audio_understanding_prompt(
    tokenizer: Any,
    audio_codes: torch.Tensor,
    instruction: str,
    *,
    thinking: bool = False,
    group_size: int = GROUP_SIZE,
    speech_empty_ids: Sequence[int] = SPEECH_EMPTY_IDS,
) -> MiMoPrompt:
    """Build the official SFT audio-understanding prompt and SGLang projection."""

    if not instruction or not instruction.strip():
        raise ValueError("MiMo audio-understanding instruction must not be empty")
    codes = pad_audio_codes(audio_codes, group_size=group_size)
    segments = [
        build_text_segment(
            tokenizer,
            "<|im_start|>user\n",
            group_size=group_size,
            speech_empty_ids=speech_empty_ids,
        ),
        build_audio_segment(
            tokenizer,
            codes,
            group_size=group_size,
            speech_empty_ids=speech_empty_ids,
        ),
        build_text_segment(
            tokenizer,
            instruction,
            group_size=group_size,
            speech_empty_ids=speech_empty_ids,
        ),
        build_text_segment(
            tokenizer,
            "<|im_end|>\n",
            group_size=group_size,
            speech_empty_ids=speech_empty_ids,
        ),
        build_text_segment(
            tokenizer,
            "<|im_start|>assistant\n",
            group_size=group_size,
            speech_empty_ids=speech_empty_ids,
        ),
        build_text_segment(
            tokenizer,
            "<think>\n" if thinking else "<think>\n\n</think>\n",
            group_size=group_size,
            speech_empty_ids=speech_empty_ids,
        ),
    ]
    official = torch.cat(segments, dim=1)
    input_ids = [int(token_id) for token_id in official[0, ::group_size].tolist()]
    audio_start = input_ids.index(_token_id(tokenizer, SOSP)) + 1
    audio_end = input_ids.index(_token_id(tokenizer, EOSP), audio_start)
    if audio_end - audio_start != codes.shape[0] // group_size:
        raise AssertionError("MiMo prompt audio placeholders do not match code groups")
    return MiMoPrompt(
        input_ids=input_ids,
        audio_codes=codes,
        official_input_ids=official,
        audio_start=audio_start,
        audio_end=audio_end,
    )


__all__ = [
    "AUDIO_CHANNELS",
    "GROUP_SIZE",
    "SPEECH_EMPTY_IDS",
    "MiMoPrompt",
    "build_audio_segment",
    "build_audio_understanding_prompt",
    "build_text_segment",
    "pad_audio_codes",
]
