# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Chatterbox-Turbo TTS."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class ChatterboxState(DeclarativeStateBase):
    """Cross-stage state for the preprocessing -> tts_engine -> vocoder pipeline."""

    sample_rate: int = 24000

    # Preprocessing -> tts_engine (T3 conditioning and text input).
    text_tokens: list[int] = wire(default_factory=list, codec="list")
    speaker_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    cond_prompt_speech_tokens: list[int] = wire(default_factory=list, codec="list")

    # Generation params.
    max_new_tokens: int = 604
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    seed: int | None = None

    # tts_engine -> vocoder.
    speech_tokens: list[int] = wire(default_factory=list, codec="list")
    finish_reason: str | None = None

    # Vocoder -> response.
    audio_samples: torch.Tensor | None = wire(None, codec="typed_tensor")
