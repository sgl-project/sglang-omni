from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from fastapi import WebSocket

from sglang_omni.audio.semantic_vad import SemanticEOUModel
from sglang_omni.client import Client
from sglang_omni.config.schema import PipelineConfig, RealtimeTranscriptionConfig
from sglang_omni.proto.session import SessionLimits
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.tick_clock import TickClock
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
    session_limits: SessionLimits = field(default_factory=SessionLimits)
    max_connections: int = 128
    # note (Junnan Li): called once with the warmup timeout in seconds, before the server reports ready.
    warmup: Callable[[float], Awaitable[None]] | None = None
    tick_groups: int = 0
    tick_openings_per_unit: bool = False
    tick_follow_arrivals: bool = False
    tick_arrivals_quiet_s: float = 0.0
    tick_growth_margin: float = 0.0
    max_units_in_flight: int = 1

    def __post_init__(self) -> None:
        if type(self.max_connections) is not int or self.max_connections < 1:
            raise ValueError("max_connections must be a positive integer")
        elif self.tick_groups and self.capabilities.interaction != "native":
            raise ValueError("the unit grid needs native full-duplex sessions")
        else:
            pass


def runtime_limits_for_pipeline(
    pipeline_config: PipelineConfig | None,
    runtime_limits: RuntimeLimits | None = None,
) -> RuntimeLimits:
    """Apply pipeline timeouts without changing other runtime limits."""
    resolved_limits = RuntimeLimits() if runtime_limits is None else runtime_limits
    if pipeline_config is None:
        return resolved_limits
    else:
        return replace(
            resolved_limits,
            session_update_timeout_s=pipeline_config.realtime_session_update_timeout_s,
            idle_timeout_s=pipeline_config.realtime_idle_timeout_s,
        )


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
        self.tick_clock = (
            TickClock(
                deployment.capabilities.native_unit_ms,
                growth_margin=deployment.tick_growth_margin,
                groups=deployment.tick_groups,
                openings_per_unit=deployment.tick_openings_per_unit,
                follow_arrivals=deployment.tick_follow_arrivals,
                arrivals_quiet_s=deployment.tick_arrivals_quiet_s,
            )
            if deployment is not None and deployment.tick_groups
            else None
        )

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
                tick_clock=self.tick_clock,
                max_units_in_flight=self.deployment.max_units_in_flight,
            )
            session = SharedRealtimeSession(websocket, runtime)
            if self.tick_clock is not None:
                self.tick_clock.session_count_changed()
            else:
                pass
        self.sessions[session.session_id] = session
        logger.info(
            f"Realtime session opened: {session.session_id} intent={normalized_intent}"
        )
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions[session_id]
        await session.teardown()
        del self.sessions[session_id]
        if self.tick_clock is not None:
            self.tick_clock.session_count_changed()
        else:
            pass
        logger.info(f"Realtime session closed: {session_id}")

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
