# SPDX-License-Identifier: Apache-2.0
"""Per-request state for the dots.tts pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.pipeline_state import wire

DOTS_TTS_SAMPLE_RATE = 48000
DOTS_TTS_DEFAULT_NUM_STEPS = 4
DOTS_TTS_DEFAULT_GUIDANCE_SCALE = 1.2
DOTS_TTS_DEFAULT_SPEAKER_SCALE = 1.5
DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES = 500


@dataclass
class DotsTtsState(DeclarativeStateBase):
    """Per-request state for dots.tts generation.

    Tensor fields transport as float32 ``{name}_bytes/_shape/_dtype`` triples
    through the shared typed_tensor codec and restore to CPU float32 tensors.
    """

    sample_rate: int = DOTS_TTS_SAMPLE_RATE

    # -- From preprocessing ------------------------------------------------
    text: str = wire("", emit="truthy", codec="str")
    prompt_text: str = wire("", emit="truthy", codec="str")
    ref_audio_path: str | None = None
    num_steps: int = wire(DOTS_TTS_DEFAULT_NUM_STEPS, codec="int_or")
    guidance_scale: float = wire(DOTS_TTS_DEFAULT_GUIDANCE_SCALE, codec="float")
    speaker_scale: float = wire(DOTS_TTS_DEFAULT_SPEAKER_SCALE, codec="float")
    seed: int | None = wire(None, codec="opt_int")
    max_audio_patches: int = wire(DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES, codec="int_or")

    # -- From reference encode ---------------------------------------------
    generation_schedule_ids: list[int] | None = wire(None, codec="list")
    # note (chenyang): raw (unnormalized) AudioVAE latents for the prompt audio,
    # already truncated by one patch so the AR loop regenerates the prompt tail.
    prompt_latent: Any | None = wire(None, codec="typed_tensor")
    spk_emb: Any | None = wire(None, codec="typed_tensor")
    prompt_patch_count: int = wire(0, codec="int_or")

    # -- From TTS engine ---------------------------------------------------
    generated_latents: Any | None = wire(None, codec="typed_tensor")
    finish_reason: str | None = None

    # -- From audio decode -------------------------------------------------
    waveform: Any | None = wire(None, codec="typed_tensor")
    duration_s: float | None = None
    audio_decode_time_s: float = wire(0.0, emit="truthy", codec="float")


def load_dots_tts_state(payload: StagePayload) -> DotsTtsState:
    return _load_pipeline_state(payload, DotsTtsState)


def store_dots_tts_state(payload: StagePayload, state: DotsTtsState) -> StagePayload:
    return _store_pipeline_state(payload, state)


__all__ = [
    "DOTS_TTS_DEFAULT_GUIDANCE_SCALE",
    "DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES",
    "DOTS_TTS_DEFAULT_NUM_STEPS",
    "DOTS_TTS_DEFAULT_SPEAKER_SCALE",
    "DOTS_TTS_SAMPLE_RATE",
    "DotsTtsState",
    "load_dots_tts_state",
    "store_dots_tts_state",
]
