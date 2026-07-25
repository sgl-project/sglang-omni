# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.pipeline_state import wire

MING_TTS_SAMPLE_RATE = 44100
MING_TTS_DEFAULT_MAX_DECODE_STEPS = 200


@dataclass
class MingTTSState(DeclarativeStateBase):
    """Per-request state for Ming-Omni-TTS generation.

    Tensor fields transport as float32 {name}_bytes/_shape/_dtype triples via
    the shared typed_tensor codec and restore to CPU float32 tensors.
    """

    sample_rate: int = MING_TTS_SAMPLE_RATE

    # -- From preprocessing ------------------------------------------------
    text: str = ""
    prompt: str | None = None
    instructions: str | None = None
    language: str | None = None
    voice: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    input_ids: list[int] | None = wire(None, codec="list")
    max_decode_steps: int = wire(MING_TTS_DEFAULT_MAX_DECODE_STEPS, codec="int_or")
    cfg: float = wire(2.0, codec="float")
    sigma: float = wire(0.25, codec="float")
    temperature: float = wire(0.0, codec="float")

    # -- From reference encode ---------------------------------------------
    prompt_text: str | None = None
    spk_token_positions: list[int] | None = wire(None, codec="list")
    spk_injection_positions: list[int] | None = wire(None, codec="list")
    audio_token_position: int | None = wire(None, codec="opt_int")
    prompt_latent_start_position: int | None = wire(None, codec="opt_int")
    prompt_latent_token_count: int = wire(0, emit="truthy", codec="int")
    spk_emb: Any | None = wire(None, codec="typed_tensor")
    prompt_latent: Any | None = wire(None, codec="typed_tensor")

    # -- From TTS engine ---------------------------------------------------
    generated_latents: Any | None = wire(None, codec="typed_tensor")
    generated_last_chunk: list[bool] | None = wire(None, codec="list")
    stop_step: int | None = wire(None, codec="opt_int")
    finish_reason: str | None = None

    # -- From audio decode -------------------------------------------------
    duration_s: float | None = None
    audio_decode_time_s: float = wire(0.0, emit="truthy", codec="float")


def load_ming_tts_state(payload: StagePayload) -> MingTTSState:
    return _load_pipeline_state(payload, MingTTSState)


def store_ming_tts_state(payload: StagePayload, state: MingTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


__all__ = [
    "MING_TTS_DEFAULT_MAX_DECODE_STEPS",
    "MING_TTS_SAMPLE_RATE",
    "MingTTSState",
    "load_ming_tts_state",
    "store_ming_tts_state",
]
