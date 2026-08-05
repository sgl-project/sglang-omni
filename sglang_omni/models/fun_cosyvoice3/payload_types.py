# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class FunCosyVoice3State(DeclarativeStateBase):
    """Per-request state for Fun-CosyVoice3 generation."""

    sample_rate: int = wire(24000, codec="int")

    text: str = wire("", codec="str")
    language: str = wire("auto", codec="str_or")
    instructions: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    stream: bool = wire(False, codec="bool")
    speed: float = wire(1.0, codec="float")
    seed: int | None = None
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    audio_codes: Any | None = wire(None, codec="tensor_list")
    audio_samples: Any | None = wire(None, codec="tensor_list")
