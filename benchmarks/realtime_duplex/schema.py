# SPDX-License-Identifier: Apache-2.0
"""Wire-schema bindings for the duplex conversation benchmark.

Client events are validated through the serve layer's own pydantic models
so the harness and the server cannot drift apart. The events module is
loaded directly from its file: importing it through the package chain
would pull the serve runtime (fastapi and the engine client) into what
must stay a CPU-only harness. Server events arrive as plain dicts; this
module names the subset the harness records and the replay verdicts
reason about, and classifies everything else as unknown rather than
dropping it (attempt accounting must stay complete).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
DEFAULT_FRAME_MS = 80
"""Native duplex cadence: 80 ms audio frames, matching #2188."""

DEFAULT_SESSION_CONFIG: dict[str, Any] = {
    "modalities": ["text", "audio"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
}


def frame_bytes(frame_ms: int = DEFAULT_FRAME_MS) -> int:
    total = SAMPLE_RATE * frame_ms // 1000 * BYTES_PER_SAMPLE
    if total <= 0:
        raise ValueError("frame_ms must yield at least one sample")
    return total


def conversation_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/v1/realtime?intent=conversation"


# Server event types the harness understands structurally. Anything else is
# still recorded and counted, but no ordering verdict is derived from it.
SESSION_LIFECYCLE_EVENTS = ("session.created", "session.updated")
INPUT_BUFFER_EVENTS = (
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
)
RESPONSE_EVENTS = (
    "response.created",
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.text.delta",
    "response.audio.done",
    "response.audio_transcript.done",
    "response.text.done",
    "response.done",
)
CONVERSATION_ITEM_EVENTS = (
    "conversation.item.created",
    "conversation.item.truncated",
)
KNOWN_SERVER_EVENT_TYPES = frozenset(
    {"error", *SESSION_LIFECYCLE_EVENTS, *INPUT_BUFFER_EVENTS, *RESPONSE_EVENTS, *CONVERSATION_ITEM_EVENTS}
)

RESPONSE_DELTA_EVENTS = (
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.text.delta",
)


_events_module: Any = None


def events_module() -> Any:
    """The serve layer's event models, loaded without the serve runtime."""
    global _events_module
    if _events_module is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "sglang_omni"
            / "serve"
            / "realtime"
            / "events.py"
        )
        spec = importlib.util.spec_from_file_location(
            "sglang_omni.serve.realtime.events", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load event models from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _events_module = module
    return _events_module


def parse_client_event(raw: dict[str, Any]) -> Any:
    """Parse one client event with the serve layer's own parser; None when
    the conversation protocol would reject it."""
    return events_module().parse_conversation_client_event(raw)


def is_known_server_event(event: dict[str, Any]) -> bool:
    return str(event.get("type", "")) in KNOWN_SERVER_EVENT_TYPES
