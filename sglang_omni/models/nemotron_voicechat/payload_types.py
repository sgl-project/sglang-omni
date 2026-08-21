# SPDX-License-Identifier: Apache-2.0
"""Typed state passed through the frame-locked VoiceChat pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050
FRAME_SAMPLES = 1_280
OUTPUT_FRAME_SAMPLES = 1_764

VoiceChatEvent = Literal["audio_frame", "session_close"]


@dataclass(slots=True)
class VoiceChatFrameState:
    event: VoiceChatEvent
    session_id: str
    frame_index: int | None = None
    pcm16: str | None = None
    instructions: str | None = None
    acoustic_embedding: Any = None
    text_token: int | None = None
    text_delta: str | None = None
    function_token: int | None = None
    audio_codes: list[int] | None = None
    audio_data: Any = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: Any) -> VoiceChatFrameState:
        if not isinstance(data, dict):
            raise TypeError("VoiceChat stage payload data must be an object")
        # The coordinator wraps a pipeline's user input at the entry stage so
        # generic preprocessors can distinguish it from stage-produced data.
        # Downstream VoiceChat stages receive the unwrapped state directly.
        if set(data) == {"raw_inputs"} and isinstance(data["raw_inputs"], dict):
            data = data["raw_inputs"]
        event = data.get("event")
        if event not in ("audio_frame", "session_close"):
            raise ValueError(f"Unsupported VoiceChat event: {event!r}")
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("VoiceChat event requires a non-empty session_id")
        state = cls(
            event=event,
            session_id=session_id,
            frame_index=data.get("frame_index"),
            pcm16=data.get("pcm16"),
            instructions=data.get("instructions"),
            acoustic_embedding=data.get("acoustic_embedding"),
            text_token=data.get("text_token"),
            text_delta=data.get("text_delta"),
            function_token=data.get("function_token"),
            audio_codes=data.get("audio_codes"),
            audio_data=data.get("audio_data"),
            timings_ms=dict(data.get("timings_ms") or {}),
        )
        state.validate()
        return state

    def validate(self) -> None:
        if self.event == "session_close":
            return
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("VoiceChat audio_frame requires frame_index >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "event": self.event,
                "session_id": self.session_id,
                "frame_index": self.frame_index,
                "pcm16": self.pcm16,
                "instructions": self.instructions,
                "acoustic_embedding": self.acoustic_embedding,
                "text_token": self.text_token,
                "text_delta": self.text_delta,
                "function_token": self.function_token,
                "audio_codes": self.audio_codes,
                "audio_data": self.audio_data,
                "timings_ms": self.timings_ms,
            }.items()
            if value is not None
        }
