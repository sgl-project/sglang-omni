# SPDX-License-Identifier: Apache-2.0
"""Per-request state for Audar-TTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class AudarTTSState(DeclarativeStateBase):
    target_text: str = wire("", emit="truthy", codec="str")
    reference_text: str = wire("", emit="truthy", codec="str")
    reference_audio: dict[str, object] | None = None
    prompt: str | None = None
    audio_codes: list[int] | None = None
    generation_kwargs: Mapping[str, object] = wire(default_factory=dict, codec="dict")
    sample_rate: int = wire(24000, codec="int_or")
