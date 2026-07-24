# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 cross-stage payload contract (frozen, append-only).

Carried inside ``StagePayload.data`` as a plain dict. Serialization is derived
from the field wire metadata by :class:`DeclarativeStateBase`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

ZONOS2_SAMPLE_RATE = 44100
N_CODEBOOKS = 9
# 9 audio codebook columns + 1 text column.
FRAME_WIDTH = 10


@dataclass
class Zonos2State(DeclarativeStateBase):
    """Per-request state threaded through the ZONOS2 pipeline."""

    sample_rate: int = wire(ZONOS2_SAMPLE_RATE, codec="int")

    # request inputs (preprocessing)
    text: str = wire("", codec="str")
    ref_audio: Any | None = None  # path / bytes / data-uri for voice cloning
    ref_text: str | None = None
    language: str | None = None
    speaking_rate: float | None = None
    conditioning: dict[str, Any] = wire(
        default_factory=dict, emit="truthy", codec="dict"
    )

    # preprocessing output
    # (T, FRAME_WIDTH) rows: audio cols hold audio_pad_id, last col holds text/conditioning ids.
    input_ids: Any | None = wire(None, codec="tensor_cpu")
    speaker_token_positions: list[int] = wire(default_factory=lambda: [0], codec="list")

    # speaker_encode output
    speaker_emb: Any | None = wire(None, codec="tensor_cpu")  # (2048,) f32 CPU, or None
    speaker_fingerprint: str | None = None  # stable hash for radix extra_key

    # tts_engine output
    audio_codes: Any | None = wire(
        None, codec="tensor_cpu"
    )  # delayed (T, 9), pre-shear
    eos_frame: int | None = wire(None, codec="opt_int")

    # bookkeeping
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")


__all__ = ["Zonos2State", "ZONOS2_SAMPLE_RATE", "N_CODEBOOKS", "FRAME_WIDTH"]
