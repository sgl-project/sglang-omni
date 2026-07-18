# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


def moss_tts_local_special_token_defaults(
    audio_vocab_size: int = 1024,
) -> tuple[tuple[str, int], ...]:
    """Default special-token ids for MOSS-TTS-Local-Transformer-v1.5.

    These differ from the MOSS Delay family: the Local release introduces
    dedicated ``<|audio_start|>``/``<|audio_end|>`` tokens and reuses the
    Qwen vision/video pad ids as the user/assistant audio slot tokens.
    """
    return (
        ("audio_start_token_id", 151669),
        ("audio_end_token_id", 151670),
        ("audio_user_slot_token_id", 151654),
        ("audio_assistant_slot_token_id", 151656),
        ("audio_assistant_gen_slot_token_id", 151656),
        ("audio_pad_token_id", int(audio_vocab_size)),
        ("audio_pad_code", int(audio_vocab_size)),
        ("im_start_token_id", 151644),
        ("im_end_token_id", 151645),
        ("pad_token_id", 151643),
    )


@dataclass
class MossTTSLocalState(DeclarativeStateBase):
    """Per-request state for MOSS-TTS Local generation."""

    sample_rate: int = wire(48000, codec="int_or")
    text: str = wire("", codec="str")
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    instructions: str | None = None
    token_count: int | None = wire(None, codec="opt_int")
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    audio_codes: Any | None = wire(None, codec="tensor_cpu")
