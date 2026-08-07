# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Realtime pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

SAMPLE_RATE = 24000
N_CODEBOOKS = 16
AUDIO_PAD_TOKEN = 1024
AUDIO_BOS_TOKEN = 1025
AUDIO_EOS_TOKEN = 1026
TEXT_PAD_TOKEN = 151655
REFERENCE_AUDIO_PAD_TOKEN = 151654
PREFILL_TEXT_TOKENS = 12


@dataclass
class MossTTSRealtimeState(DeclarativeStateBase):
    """Per-request MOSS-TTS-Realtime state."""

    sample_rate: int = wire(SAMPLE_RATE, codec="int_or")
    text: str = wire("", codec="str")
    ref_audio: Any | None = None
    generation_kwargs: dict[str, Any] = wire(  # noqa: RUF009
        default_factory=dict, codec="dict"
    )
    audio_codes: Any | None = wire(None, codec="tensor_cpu")  # noqa: RUF009


__all__ = [
    "AUDIO_BOS_TOKEN",
    "AUDIO_EOS_TOKEN",
    "AUDIO_PAD_TOKEN",
    "N_CODEBOOKS",
    "PREFILL_TEXT_TOKENS",
    "REFERENCE_AUDIO_PAD_TOKEN",
    "SAMPLE_RATE",
    "TEXT_PAD_TOKEN",
    "MossTTSRealtimeState",
]
