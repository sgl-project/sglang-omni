"""Pipeline state definition for Voxtral TTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class VoxtralTTSState(DeclarativeStateBase):
    """Per-request pipeline state for Voxtral TTS."""

    input_ids: list[int] | None = None
    voice: str | None = None

    max_new_tokens: int = 4096

    # Generation output: list of [num_codebooks] tensors, one per frame.
    audio_codes: Any | None = wire(None, codec="typed_tensor")

    # Vocoder output
    audio_samples: Any | None = wire(None, codec="tensor_list")
