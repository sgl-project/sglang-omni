from __future__ import annotations

import logging

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.config import AudioChunkingConfig, RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.semantic_vad import SemanticEOUModel
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)

logger = logging.getLogger(__name__)


class RealtimeSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        audio_chunking: AudioChunkingConfig,
        supports_audio_output: bool = False,
        transcription_config: RealtimeTranscriptionConfig | None = None,
        smart_turn_model: SemanticEOUModel | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.supports_audio_output = supports_audio_output
        self.transcription_config = transcription_config
        self.audio_chunking = audio_chunking
        self.smart_turn_model = smart_turn_model
        self.sessions: dict[str, RealtimeSession | RealtimeTranscriptionSession] = {}

    def open(
        self, websocket: WebSocket, *, intent: str = "conversation"
    ) -> RealtimeSession | RealtimeTranscriptionSession:
        normalized_intent = intent.strip().casefold()
        if normalized_intent == "conversation":
            session: RealtimeSession | RealtimeTranscriptionSession = RealtimeSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                supports_audio_output=self.supports_audio_output,
                smart_turn_model=self.smart_turn_model,
            )
        elif normalized_intent == "transcription":
            if self.transcription_config is None:
                raise ValueError(
                    "This pipeline does not support realtime transcription."
                )
            session = RealtimeTranscriptionSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                capability=self.transcription_config,
                audio_chunking=self.audio_chunking,
                strategy=self.transcription_config.strategy_cls(),
            )
        else:
            raise ValueError(
                "Realtime intent must be 'conversation' or 'transcription'."
            )
        self.sessions[session.session_id] = session
        logger.info(
            "Realtime session opened: %s intent=%s",
            session.session_id,
            normalized_intent,
        )
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions[session_id]
        await session.teardown()
        del self.sessions[session_id]
        logger.info(f"Realtime session closed: {session_id}")

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
