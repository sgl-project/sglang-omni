from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.semantic_vad import SemanticEOUModel
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)
from sglang_omni.serve.realtime.types import AdapterFactory, Capabilities, RuntimeLimits

logger = logging.getLogger(__name__)

RealtimeConnection = (
    RealtimeSession | SharedRealtimeSession | RealtimeTranscriptionSession
)


@dataclass(frozen=True)
class RealtimeDeployment:
    capabilities: Capabilities
    adapter_factory: AdapterFactory
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    max_connections: int = 128

    def __post_init__(self) -> None:
        if type(self.max_connections) is not int or self.max_connections < 1:
            raise ValueError("max_connections must be a positive integer")
        else:
            pass


class RealtimeSessionManager:
    def __init__(
        self,
        *,
        deployment: RealtimeDeployment | None = None,
        client: Client,
        model_name: str,
        supports_audio_output: bool = False,
        transcription_config: RealtimeTranscriptionConfig | None = None,
        smart_turn_model: SemanticEOUModel | None = None,
    ) -> None:
        self.transcription_config = transcription_config
        self.deployment = deployment
        self.client = client
        self.model_name = model_name
        self.supports_audio_output = supports_audio_output
        self.smart_turn_model = smart_turn_model
        self.sessions: dict[str, RealtimeConnection] = {}

    def open(
        self, websocket: WebSocket, *, intent: str = "conversation"
    ) -> RealtimeConnection:
        normalized_intent = intent.strip().casefold()
        if normalized_intent == "transcription":
            if self.transcription_config is None:
                raise ValueError(
                    "This pipeline does not support realtime transcription."
                )
            else:
                pass
            session: RealtimeConnection = RealtimeTranscriptionSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                transcription_config=self.transcription_config,
                strategy=self.transcription_config.strategy_cls(),
            )
        elif normalized_intent != "conversation":
            raise ValueError(
                "Realtime intent must be 'conversation' or 'transcription'."
            )
        elif self.deployment is None:
            session = RealtimeSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                supports_audio_output=self.supports_audio_output,
                smart_turn_model=self.smart_turn_model,
            )
        else:
            runtime = SessionRuntime(
                self.model_name,
                self.deployment.capabilities,
                self.deployment.adapter_factory,
                self.deployment.limits,
            )
            session = SharedRealtimeSession(websocket, runtime)
        self.sessions[session.session_id] = session
        logger.info(
            f"Realtime session opened: {session.session_id} intent={normalized_intent}"
        )
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions[session_id]
        await session.teardown()
        del self.sessions[session_id]
        logger.info(f"Realtime session closed: {session_id}")

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
