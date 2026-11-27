# SPDX-License-Identifier: Apache-2.0
"""Required token ids for MiniCPM-o native duplex generation."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

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
DELEGATE_SPECIAL_TOKENS = ("<delegate>", "</delegate>", "<backend>", "</backend>")


@dataclass(frozen=True)
class DelegateTokenIds:
    """The spans of a checkpoint that hands tasks to the client: it writes a task inside a delegate span and reads the task's result inside a backend span."""

    delegate_start: int
    delegate_end: int
    backend_start: int
    backend_end: int


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
    # None for a tokenizer without the delegate tokens.
    delegate_tokens: DelegateTokenIds | None


def token_id_of(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    """The id of token; None when the tokenizer does not have it."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        return None
    else:
        return int(token_id)


def resolve_special_token_ids(
    tokenizer: PreTrainedTokenizerBase, bad_token_ids: tuple[int, ...] = ()
) -> MiniCPMOSpecialTokenIds:
    """Resolve the duplex vocabulary and reject incomplete tokenizers."""

    resolved: list[int] = []
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = token_id_of(tokenizer, token)
        if token_id is None:
            raise ValueError(f"MiniCPM-o tokenizer is missing required token {token!r}")
        else:
            resolved.append(token_id)
    delegate_token_ids = [
        token_id_of(tokenizer, token) for token in DELEGATE_SPECIAL_TOKENS
    ]
    if None in delegate_token_ids:
        delegate_tokens = None
    else:
        delegate_start, delegate_end, backend_start, backend_end = delegate_token_ids
        delegate_tokens = DelegateTokenIds(
            delegate_start=delegate_start,
            delegate_end=delegate_end,
            backend_start=backend_start,
            backend_end=backend_end,
        )

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
        delegate_tokens=delegate_tokens,
    )


__all__ = [
    "DELEGATE_SPECIAL_TOKENS",
    "REQUIRED_SPECIAL_TOKENS",
    "DelegateTokenIds",
    "MiniCPMOSpecialTokenIds",
    "resolve_special_token_ids",
]
