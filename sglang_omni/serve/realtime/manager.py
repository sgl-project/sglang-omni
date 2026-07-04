from __future__ import annotations

import logging

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.serve.realtime.session import RealtimeSession

logger = logging.getLogger(__name__)


class RealtimeSessionManager:
    def __init__(self, *, client: Client, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.sessions: dict[str, RealtimeSession] = {}

    def open(self, websocket: WebSocket) -> RealtimeSession:
        session = RealtimeSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
        )
        self.sessions[session.session_id] = session
        logger.info(f"Realtime session opened: {session.session_id}")
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions[session_id]
        await session.teardown()
        del self.sessions[session_id]
        logger.info(f"Realtime session closed: {session_id}")

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
