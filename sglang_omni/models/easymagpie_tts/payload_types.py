# SPDX-License-Identifier: Apache-2.0
"""Cross-stage state for EasyMagpie TTS."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

EASYMAGPIE_SAMPLE_RATE = 22050
DEFAULT_VOICE = "eng"
DEFAULT_CONTEXT_TEXT = "[EN]"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 80
# Width of the decode graph's top-k sampling kernels.
MAX_TOP_K = 80
# Text tokens per request the on-GPU decode state holds.
MAX_TEXT_TOKENS = 8192
DEFAULT_MAX_NEW_FRAMES = 2048


@dataclass
class EasyMagpieTTSState(DeclarativeStateBase):
    sample_rate: int = wire(EASYMAGPIE_SAMPLE_RATE, codec="int")
    text: str = wire("", codec="str")
    voice: str = wire(DEFAULT_VOICE, codec="str_or")
    context_text: str = wire(DEFAULT_CONTEXT_TEXT, codec="str_or")
    temperature: float = wire(DEFAULT_TEMPERATURE, codec="float")
    top_k: int = wire(DEFAULT_TOP_K, codec="int")
    max_new_frames: int = wire(DEFAULT_MAX_NEW_FRAMES, codec="int")
    seed: int | None = wire(None, codec="opt_int")

    phoneme_delay: int = wire(3, codec="int")
    speech_delay: int = wire(5, codec="int")
    text_prefill_num: int = wire(4, codec="int")
    # First decode step whose frame is streamed as audio.
    audio_emit_delay: int = wire(5, codec="int")
    text_token_ids: list[int] = wire(default_factory=list, codec="list")
    context_token_ids: list[int] = wire(default_factory=list, codec="list")
    speaker_frames: int = wire(0, codec="int")

    audio_codes: torch.Tensor | None = wire(None, codec="tensor_cpu")


__all__ = [
    "DEFAULT_CONTEXT_TEXT",
    "DEFAULT_MAX_NEW_FRAMES",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_VOICE",
    "EASYMAGPIE_SAMPLE_RATE",
    "MAX_TEXT_TOKENS",
    "MAX_TOP_K",
    "EasyMagpieTTSState",
]
