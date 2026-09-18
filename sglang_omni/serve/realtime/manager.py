from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    InteractionAdapter,
    RuntimeLimits,
    SessionRuntime,
)
from sglang_omni.serve.realtime.semantic_vad import SemanticEOUModel
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)


@dataclass(frozen=True)
class RealtimeDeployment:

    capabilities: Capabilities
    adapter_factory: Callable[[], InteractionAdapter]
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    max_connections: int = 128

    def __post_init__(self) -> None:
        if type(self.max_connections) is not int or self.max_connections < 1:
            raise ValueError("max_connections must be a positive integer")


logger = logging.getLogger(__name__)


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
        if deployment is None:
            deployment = RealtimeDeployment(
                Capabilities(
                    interaction="turn_based",
                    output_rate=24000,
                    output_modalities=(
                        ("audio", "text") if supports_audio_output else ("text",)
                    ),
                ),
                TurnBasedAdapterFactory(
                    client, model_name, supports_audio_output, smart_turn_model
                ),
            )
        self.transcription_config = transcription_config
        self.deployment = deployment
        self.client = client
        self.model_name = model_name
        self.supports_audio_output = supports_audio_output
        self.smart_turn_model = smart_turn_model
        self.sessions: dict[
            str, SharedRealtimeSession | RealtimeTranscriptionSession
        ] = {}

    def open(
        self, websocket: WebSocket, *, intent: str = "conversation"
    ) -> SharedRealtimeSession | RealtimeTranscriptionSession:
        intent = intent.strip().casefold()
        if intent == "transcription":
            if self.transcription_config is None:
                raise ValueError(
                    "This pipeline does not support realtime transcription."
                )
            session = RealtimeTranscriptionSession(
                websocket,
                client=self.client,
                model_name=self.model_name,
                transcription_config=self.transcription_config,
                strategy=self.transcription_config.strategy_cls(),
            )
            self.sessions[session.session_id] = session
            return session
        if intent != "conversation":
            raise ValueError(
                "Realtime intent must be 'conversation' or 'transcription'."
            )
        runtime = SessionRuntime(
            self.model_name,
            self.deployment.capabilities,
            self.deployment.adapter_factory,
            self.deployment.limits,
        )
        session = SharedRealtimeSession(websocket, runtime)
        self.sessions[session.session_id] = session
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions[session_id]
        await session.teardown()
        del self.sessions[session_id]
        logger.info(f"Realtime session closed: {session_id}")

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
