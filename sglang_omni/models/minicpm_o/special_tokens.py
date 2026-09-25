# SPDX-License-Identifier: Apache-2.0
"""Required token ids for MiniCPM-o native duplex generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_SPECIAL_TOKENS = (
    "<unit>",
    "</unit>",
    "<image>",
    "</image>",
    "<slice>",
    "</slice>",
    "<|listen|>",
    "<|speak|>",
    "<|tts_bos|>",
    "<|tts_eos|>",
    "<|chunk_eos|>",
    "<|chunk_tts_eos|>",
    "<|turn_eos|>",
    "<|tts_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
)


@dataclass(frozen=True)
class MiniCPMOSpecialTokenIds:
    unit: int
    unit_end: int
    image_start: int
    image_end: int
    slice_start: int
    slice_end: int
    listen: int
    speak: int
    tts_bos: int
    tts_eos: int
    chunk_eos: int
    chunk_tts_eos: int
    turn_eos: int
    tts_pad: int
    audio_start: int
    audio_end: int
    chunk_terminators: frozenset[int]
    turn_terminators: frozenset[int]
    forbidden: frozenset[int]


def resolve_special_token_ids(
    tokenizer: Any, bad_token_ids: tuple[int, ...] = ()
) -> MiniCPMOSpecialTokenIds:
    """Resolve the duplex vocabulary and reject incomplete tokenizers."""

    unk_token_id = tokenizer.unk_token_id
    resolved: list[int] = []
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk_token_id:
            raise ValueError(f"MiniCPM-o tokenizer is missing required token {token!r}")
        else:
            pass
        resolved.append(int(token_id))

    (
        unit,
        unit_end,
        image_start,
        image_end,
        slice_start,
        slice_end,
        listen,
        speak,
        tts_bos,
        tts_eos,
        chunk_eos,
        chunk_tts_eos,
        turn_eos,
        tts_pad,
        audio_start,
        audio_end,
    ) = resolved
    return MiniCPMOSpecialTokenIds(
        unit=unit,
        unit_end=unit_end,
        image_start=image_start,
        image_end=image_end,
        slice_start=slice_start,
        slice_end=slice_end,
        listen=listen,
        speak=speak,
        tts_bos=tts_bos,
        tts_eos=tts_eos,
        chunk_eos=chunk_eos,
        chunk_tts_eos=chunk_tts_eos,
        turn_eos=turn_eos,
        tts_pad=tts_pad,
        audio_start=audio_start,
        audio_end=audio_end,
        chunk_terminators=frozenset({listen, chunk_eos, chunk_tts_eos}),
        turn_terminators=frozenset({turn_eos}),
        forbidden=frozenset(
            {
                tts_pad,
                *(int(token_id) for token_id in bad_token_ids),
            }
        ),
    )


__all__ = [
    "REQUIRED_SPECIAL_TOKENS",
    "MiniCPMOSpecialTokenIds",
    "resolve_special_token_ids",
]
