# SPDX-License-Identifier: Apache-2.0
import base64
import json
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime.session import RealtimeSession

GOOD_FRAME = {"type": "session.update", "session": {"input_audio_format": "pcm16"}}
_OVER_THE_CAP = base64.b64encode(b"\x00" * (61 * 16000 * 2)).decode()


def text(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    return {"type": "websocket.receive", "text": payload}


class FakeVAD:
    def __init__(self, _config: object | None = None) -> None: ...

    def reset(self) -> None: ...

    def process(self, _chunk: bytes) -> list[object]:
        return []


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.events: list[dict[str, Any]] = []
        self._inbox = list(messages)

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))

    async def receive(self) -> dict[str, Any]:
        """Replay queued frames, then disconnect so ``run`` returns."""
        if not self._inbox:
            return {"type": "websocket.disconnect"}
        return self._inbox.pop(0)


def _session(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
) -> tuple[RealtimeSession, RecordingWebSocket]:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    websocket = RecordingWebSocket(messages)
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    return session, websocket


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_frame",
    [
        pytest.param(
            {"type": "websocket.receive", "bytes": b"\x00\x01"}, id="binary-frame"
        ),
        pytest.param(text("not json"), id="malformed-json"),
        pytest.param(text("[1, 2]"), id="non-object-json"),
        pytest.param(text({"type": "session.update"}), id="known-type-invalid-body"),
        # input_audio_buffer.commit is how the OpenAI flow finalizes audio, so a
        # spec-following client reaches this rejection on the happy path.
        pytest.param(text({"type": "input_audio_buffer.commit"}), id="unknown-type"),
        pytest.param(
            text({"type": "session.update", "session": {"input_audio_format": "opus"}}),
            id="format-outside-schema",
        ),
        pytest.param(
            text(
                {
                    "type": "session.update",
                    "session": {"input_audio_format": "g711_ulaw"},
                }
            ),
            id="unsupported-format-in-schema",
        ),
        # SessionConfig allows extras, so a handler still validates client data
        # after dispatch returns.
        pytest.param(text({"type": "session.update", "session": {"id": 123}})),
        pytest.param(
            text({"type": "input_audio_buffer.append", "audio": _OVER_THE_CAP}),
            id="audio-past-the-60s-cap",
        ),
        pytest.param(
            {"type": "websocket.receive", "text": None, "bytes": b"\x00"},
            id="binary-frame-with-null-text",
        ),
    ],
)
async def test_a_bad_frame_is_reported_and_the_session_serves_the_next_one(
    bad_frame: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket = _session(monkeypatch, [bad_frame, text(GOOD_FRAME)])

    await session.run()

    assert [event["type"] for event in websocket.events] == [
        "session.created",
        "error",
        "session.updated",
    ]


@pytest.mark.asyncio
async def test_a_rejected_format_does_not_reach_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The granted format must stay one the PCM16 audio buffer can decode."""
    session, _ = _session(
        monkeypatch,
        [
            text(
                {
                    "type": "session.update",
                    "session": {"input_audio_format": "g711_ulaw"},
                }
            )
        ],
    )

    await session.run()

    assert session.session_object.input_audio_format == "pcm16"
