# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Nano pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class MossTTSNanoState(DeclarativeStateBase):
    """Per-request state for MOSS-TTS-Nano generation."""

    sample_rate: int = wire(48000, codec="int_or")
    text: str = wire("", codec="str")
    ref_audio: Any | None = None
    ref_text: str | None = None
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    audio_codes: Any | None = wire(None, codec="tensor_cpu")
    prompt_tokens: int = wire(0, codec="int_or")
    completion_tokens: int = wire(0, codec="int_or")
    engine_time_s: float = wire(0.0, codec="float")
