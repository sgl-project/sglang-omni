# SPDX-License-Identifier: Apache-2.0
"""Backend abstraction for realtime turn responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    accepts_audio_input: bool = False
    accepts_video_input: bool = False
    returns_text: bool = True
    returns_audio: bool = False
    supports_cancel: bool = True


@dataclass
class TurnContext:
    session_id: str
    history: list[dict[str, str]]
    instructions: str | None
    user_text: str | None
    user_audio: Any | None
    user_audio_sample_rate: int | None
    recent_video: Any | None
    recent_video_fps: float | None


@dataclass
class ResponseEvent:
    type: str
    response_id: str
    text: str | None = None
    audio: Any | None = None
    sample_rate: int | None = None
    finish_reason: str | None = None
    error: str | None = None


class ResponseBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    async def stream_response(
        self,
        turn: TurnContext,
    ) -> AsyncIterator[ResponseEvent]: ...

    async def cancel(self, response_id: str) -> None: ...
