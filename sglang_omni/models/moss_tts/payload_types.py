# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.pipeline_state import wire


@dataclass(frozen=True)
class ChannelSampling:
    """Sampling defaults for one MOSS-TTS channel group."""

    temperature: float
    top_p: float
    top_k: int


TEXT_SAMPLING = ChannelSampling(temperature=1.5, top_p=1.0, top_k=50)
AUDIO_SAMPLING = ChannelSampling(temperature=1.7, top_p=0.8, top_k=25)
AUDIO_REPETITION_PENALTY = 1.0


def moss_tts_special_token_defaults(
    audio_vocab_size: int = 1024,
) -> tuple[tuple[str, int], ...]:
    """Default MOSS-TTS special-token ids, shared by model- and processor-side
    config normalization. ``audio_pad_code`` follows ``audio_vocab_size``."""
    return (
        ("audio_start_token_id", 151652),
        ("audio_end_token_id", 151653),
        ("audio_assistant_gen_slot_token_id", 151656),
        ("audio_assistant_delay_slot_token_id", 151662),
        ("audio_pad_code", int(audio_vocab_size)),
        ("im_start_token_id", 151644),
        ("im_end_token_id", 151645),
        ("pad_token_id", 151643),
    )


def resolve_moss_audio_pad_code(config: Any) -> int:
    value = getattr(config, "audio_pad_code", None)
    if value is not None:
        return int(value)
    audio_vocab_size = int(getattr(config, "audio_vocab_size", 1024) or 1024)
    return dict(moss_tts_special_token_defaults(audio_vocab_size))["audio_pad_code"]


@dataclass
class MossTTSState(DeclarativeStateBase):
    """Per-request state for MOSS-TTS Delay generation."""

    sample_rate: int = wire(24000, codec="int_or")
    text: str = wire("", codec="str")
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    instructions: str | None = None
    token_count: int | None = wire(None, codec="opt_int")
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    delayed_audio_codes: Any | None = wire(None, codec="tensor_cpu")
    assistant_start_length: int = wire(0, emit="truthy", codec="int")


def load_moss_tts_state(payload: StagePayload) -> MossTTSState:
    return _load_pipeline_state(payload, MossTTSState)


def store_moss_tts_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)
