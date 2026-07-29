from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket

from sglang_omni.client import Client

if TYPE_CHECKING:
    from sglang_omni.serve.realtime.duplex_session import DuplexRealtimeSession
    from sglang_omni.serve.realtime.session import RealtimeSession

logger = logging.getLogger(__name__)


class RealtimeSessionManager:
    def __init__(
        self, *, client: Client, model_name: str, native_duplex: bool = False
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.native_duplex = native_duplex
        self.sessions: dict[str, RealtimeSession | DuplexRealtimeSession] = {}

    def open(self, websocket: WebSocket) -> RealtimeSession | DuplexRealtimeSession:
        if self.native_duplex:
            if self.sessions:
                raise RuntimeError(
                    "The model-native duplex worker is at capacity; "
                    "retry after the active session closes"
                )
            from sglang_omni.serve.realtime.duplex_session import DuplexRealtimeSession

            session_cls = DuplexRealtimeSession
        else:
            from sglang_omni.serve.realtime.session import RealtimeSession

            session_cls = RealtimeSession
        session = session_cls(
            websocket,
            client=self.client,
            model_name=self.model_name,
        )
        self.sessions[session.session_id] = session
        logger.info(
            "Realtime session opened: %s mode=%s",
            session.session_id,
            "native_duplex" if self.native_duplex else "turn_based",
        )
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        try:
            await session.teardown()
        finally:
            self.sessions.pop(session_id, None)
            logger.info("Realtime session closed: %s", session_id)

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
