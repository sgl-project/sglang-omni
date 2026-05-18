# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime API (WebSocket /v1/realtime).

Reference: https://developers.openai.com/api/docs/guides/realtime
"""

from sglang_omni.serve.realtime.manager import RealtimeSessionManager
from sglang_omni.serve.realtime.session import RealtimeSession

__all__ = ["RealtimeSession", "RealtimeSessionManager"]
