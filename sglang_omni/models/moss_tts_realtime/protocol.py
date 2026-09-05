# SPDX-License-Identifier: Apache-2.0
"""Wire schemas for MOSS-TTS-Realtime speech WebSocket sessions."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sglang_omni.serve.protocol import SpeechReference


class _StrictWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MossTTSRealtimeSpeechSessionConfig(_StrictWireModel):
    """Strict public configuration for the realtime speech endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
    )

    model: str | None = None
    voice: str = Field(
        default="default",
        validation_alias=AliasChoices("voice", "speaker"),
    )
    response_format: Literal["pcm"] = "pcm"
    stream_audio: Literal[True] = True
    sample_rate: Literal[24000] = 24000
    ref_audio: str | None = None
    ref_text: str | None = None
    references: list[SpeechReference] | None = None
    language: str | None = None
    instructions: str | None = None
    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=0.8, ge=0.0)
    top_p: float = Field(default=0.6, gt=0.0, le=1.0)
    top_k: int = Field(default=30, gt=0)
    repetition_penalty: float = Field(default=1.1, gt=0.0)
    repetition_window: int = Field(default=50, gt=0)
    seed: int | None = Field(default=None, ge=0)
    stage_params: dict[str, dict[str, Any]] | None = None

    @field_validator(
        "model",
        "voice",
        "ref_audio",
        "ref_text",
        "language",
        "instructions",
    )
    @classmethod
    def _non_empty_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value


class MossTTSRealtimeTurnUser(_StrictWireModel):
    """Optional full-fidelity user context preceding assistant speech."""

    text: str | None = None
    audio: str | None = None

    @model_validator(mode="after")
    def _require_complete_pair(self) -> "MossTTSRealtimeTurnUser":
        has_text = self.text is not None
        has_audio = self.audio is not None
        if has_text != has_audio:
            raise ValueError("user context requires both text and audio")
        if not has_text:
            raise ValueError("user context must not be empty")
        assert self.text is not None and self.audio is not None
        if not self.text.strip():
            raise ValueError("user text must not be empty")
        if not self.audio.strip():
            raise ValueError("user audio must not be empty")
        return self


class MossTTSRealtimeClientEvent(_StrictWireModel):
    type: str


class MossTTSRealtimeTurnStart(MossTTSRealtimeClientEvent):
    type: Literal["turn.start"]
    turn_id: str
    user: MossTTSRealtimeTurnUser | None = None

    @field_validator("turn_id")
    @classmethod
    def _turn_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("turn_id must not be empty")
        return value


class MossTTSRealtimeInputText(MossTTSRealtimeClientEvent):
    type: Literal["input.text"]
    turn_id: str
    seq_no: int = Field(ge=0)
    text: str = Field(min_length=1)


class MossTTSRealtimeInputTokens(MossTTSRealtimeClientEvent):
    type: Literal["input.tokens"]
    turn_id: str
    seq_no: int = Field(ge=0)
    token_ids: list[int] = Field(min_length=1)

    @field_validator("token_ids")
    @classmethod
    def _token_ids_are_non_negative(cls, token_ids: list[int]) -> list[int]:
        for token_id in token_ids:
            if token_id < 0:
                raise ValueError("token_ids entries must be non-negative")
        return token_ids


class MossTTSRealtimeInputDone(MossTTSRealtimeClientEvent):
    type: Literal["input.done"]
    turn_id: str
    seq_no: int = Field(ge=0)


class MossTTSRealtimeTurnCancel(MossTTSRealtimeClientEvent):
    type: Literal["turn.cancel"]
    turn_id: str


class MossTTSRealtimeSessionClose(MossTTSRealtimeClientEvent):
    type: Literal["session.close"]


MossTTSRealtimeInputEvent = (
    MossTTSRealtimeInputText | MossTTSRealtimeInputTokens | MossTTSRealtimeInputDone
)

MOSS_TTS_REALTIME_CLIENT_EVENT_TYPES: dict[str, type[MossTTSRealtimeClientEvent]] = {
    "turn.start": MossTTSRealtimeTurnStart,
    "input.text": MossTTSRealtimeInputText,
    "input.tokens": MossTTSRealtimeInputTokens,
    "input.done": MossTTSRealtimeInputDone,
    "turn.cancel": MossTTSRealtimeTurnCancel,
    "session.close": MossTTSRealtimeSessionClose,
}


def speech_websocket_session_config_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the flat config object accepted by the realtime endpoint."""

    raw_config = payload.get("session")
    if raw_config is None:
        return {key: value for key, value in payload.items() if key != "type"}
    if not isinstance(raw_config, dict):
        raise TypeError("session.config session must be an object")
    return dict(raw_config)


def parse_moss_tts_realtime_client_event(
    payload: dict[str, Any],
) -> MossTTSRealtimeClientEvent | None:
    event_type = payload.get("type")
    if not isinstance(event_type, str):
        return None
    event_cls = MOSS_TTS_REALTIME_CLIENT_EVENT_TYPES.get(event_type)
    if event_cls is None:
        return None
    return event_cls.model_validate(payload)


def moss_tts_realtime_event_fingerprint(
    event: MossTTSRealtimeClientEvent,
) -> str:
    """Return a stable digest for exact retry detection within one turn."""

    payload = event.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(
        encoded,
        digest_size=16,
        person=b"moss-tts-ws-v1",
    ).hexdigest()


__all__ = [
    "MOSS_TTS_REALTIME_CLIENT_EVENT_TYPES",
    "MossTTSRealtimeClientEvent",
    "MossTTSRealtimeInputDone",
    "MossTTSRealtimeInputEvent",
    "MossTTSRealtimeInputText",
    "MossTTSRealtimeInputTokens",
    "MossTTSRealtimeSessionClose",
    "MossTTSRealtimeSpeechSessionConfig",
    "MossTTSRealtimeTurnCancel",
    "MossTTSRealtimeTurnStart",
    "MossTTSRealtimeTurnUser",
    "moss_tts_realtime_event_fingerprint",
    "parse_moss_tts_realtime_client_event",
    "speech_websocket_session_config_payload",
]
