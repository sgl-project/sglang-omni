# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime API (WebSocket /v1/realtime).

Reference: https://developers.openai.com/api/docs/guides/realtime
"""

from typing import Any

from sglang_omni.serve.realtime.manager import RealtimeSessionManager

__all__ = ["DuplexRealtimeSession", "RealtimeSession", "RealtimeSessionManager"]


def __getattr__(name: str) -> Any:
    if name == "DuplexRealtimeSession":
        from sglang_omni.serve.realtime.duplex_session import DuplexRealtimeSession

        return DuplexRealtimeSession
    if name == "RealtimeSession":
        from sglang_omni.serve.realtime.session import RealtimeSession

        return RealtimeSession
    raise AttributeError(name)
