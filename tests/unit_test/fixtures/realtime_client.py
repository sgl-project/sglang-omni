"""WebSocket helpers shared by realtime audio and video contract tests."""

import base64

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.schema import JsonObject, JsonValue
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_fakes import (
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2


def build_test_client(
    adapter: ScriptedAdapter,
    *,
    capabilities: Capabilities | None = None,
    limits: RuntimeLimits | None = None,
    max_connections: int = 4,
    is_running: bool = True,
) -> TestClient:
    deployment = RealtimeDeployment(
        capabilities=capabilities or Capabilities(),
        adapter_factory=lambda: adapter,
        limits=limits or RuntimeLimits(),
        max_connections=max_connections,
    )
    app = create_app(
        Client(HealthCoordinator(is_running)),
        model_name=MODEL_NAME,
        realtime_deployment=deployment,
    )
    return TestClient(app)


def send_event(
    websocket: WebSocketTestSession, event_type: str, **fields: JsonValue
) -> None:
    websocket.send_json(
        {"event_id": f"client_{event_type}", "type": event_type, **fields}
    )


def append_audio(
    websocket: WebSocketTestSession,
    pcm: bytes,
    seq: int,
    t_start_ms: float | None = None,
) -> None:
    metadata: JsonObject = {"seq": seq}
    if t_start_ms is not None:
        metadata["t_start_ms"] = t_start_ms
    else:
        pass
    send_event(
        websocket,
        "input_audio_buffer.append",
        audio=base64.b64encode(pcm).decode("ascii"),
        sglang=metadata,
    )


def receive_until(
    websocket: WebSocketTestSession, *event_types: str
) -> list[JsonObject]:
    events: list[JsonObject] = []
    while not events or events[-1]["type"] not in event_types:
        events.append(websocket.receive_json())
    return events
