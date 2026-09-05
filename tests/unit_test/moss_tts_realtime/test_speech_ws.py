# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from sglang_omni.client import GenerateChunk
from sglang_omni.models.moss_tts_realtime import speech_ws as speech_ws_module
from sglang_omni.models.moss_tts_realtime import text_delta
from sglang_omni.models.moss_tts_realtime.protocol import MossTTSRealtimeTurnStart
from sglang_omni.models.moss_tts_realtime.speech_ws import (
    MossTTSRealtimeSpeechWebSocketSession,
    create_moss_tts_realtime_speech_ws_handler,
)
from sglang_omni.proto import InputUpdateMessage
from sglang_omni.serve import create_app


class BoundaryTokenizer:
    vocab_size = 10000

    def __init__(self) -> None:
        self.len_calls = 0

    def __len__(self) -> int:
        self.len_calls += 1
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids: list[int] = []
        index = 0
        while index < len(text):
            pair = text[index : index + 2]
            if pair == "ab":
                ids.append(9000)
                index += 2
                continue
            ids.append(1 + ord(text[index]) % 8999)
            index += 1
        return ids


@pytest.fixture(autouse=True)
def _clear_tokenizer_vocab_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_delta, "_TOKENIZER_VOCAB_SIZE", None)


def _realtime_handler(
    tokenizer: Any | None = None,
    *,
    realtime_input_stage: str | None = None,
) -> Any:
    return create_moss_tts_realtime_speech_ws_handler(
        tokenizer=(tokenizer if tokenizer is not None else BoundaryTokenizer()),
        realtime_input_stage=realtime_input_stage,
    )


class _RealtimeSpeechHandle:
    def __init__(
        self,
        owner: "RealtimeSpeechClient",
        request_id: str,
        *,
        session_id: str,
        turn_id: str,
        input_stage: str,
    ) -> None:
        self.owner = owner
        self.request_id = request_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.input_stage = input_stage
        self._yielded = False
        self._closed = False

    def __aiter__(self) -> "_RealtimeSpeechHandle":
        return self

    async def __anext__(self) -> GenerateChunk:
        if self._yielded or self._closed:
            raise StopAsyncIteration
        done = self.owner._done_events.setdefault(self.request_id, asyncio.Event())
        await done.wait()
        if self.request_id in self.owner.aborted:
            raise StopAsyncIteration
        self._yielded = True
        return GenerateChunk(
            request_id=self.request_id,
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
            finish_reason="stop",
        )

    async def send_input(self, message: InputUpdateMessage) -> None:
        self.owner.updates.append((self.request_id, message))
        if message.input_done:
            self.owner._done_events.setdefault(self.request_id, asyncio.Event()).set()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.owner.abort(self.request_id)


class RealtimeSpeechClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []
        self.realtime_opens: list[dict[str, Any]] = []
        self.updates: list[tuple[str, InputUpdateMessage]] = []
        self.aborted: list[str] = []
        self.closed_sessions: list[tuple[str, tuple[str, ...]]] = []
        self._done_events: dict[str, asyncio.Event] = {}

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def open_realtime(
        self,
        request: Any,
        request_id: str | None = None,
        *,
        session_id: str,
        turn_id: str,
        input_stage: str,
    ) -> _RealtimeSpeechHandle:
        assert request_id is not None
        self.requests.append((request_id, request))
        self.realtime_opens.append(
            {
                "request_id": request_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "input_stage": input_stage,
            }
        )
        return _RealtimeSpeechHandle(
            self,
            request_id,
            session_id=session_id,
            turn_id=turn_id,
            input_stage=input_stage,
        )

    async def abort(self, request_id: str) -> None:
        if request_id not in self.aborted:
            self.aborted.append(request_id)
        self._done_events.setdefault(request_id, asyncio.Event()).set()

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        assert action == "close_realtime_session"
        assert payload is not None
        # Session close targets the engine stage plus the vocoder stage so the
        # session-keyed codec slot is released together with the engine KV.
        assert stages is not None and len(stages) == 2
        assert timeout_s == 30.0
        self.closed_sessions.append((payload["session_id"], tuple(stages)))
        return {"success": True, "message": "closed"}


class _RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def test_turn_started_waits_for_backend_stream_submission() -> None:
    async def _run() -> None:
        websocket = _RecordingWebSocket()
        session = MossTTSRealtimeSpeechWebSocketSession(
            websocket,
            client=object(),
            speech_service=None,
            session_id="session-1",
            tokenizer=BoundaryTokenizer(),
        )
        generation_entered = asyncio.Event()
        release_submission = asyncio.Event()
        release_generation = asyncio.Event()

        async def _fake_generation(turn, event, submitted) -> None:
            del turn, event
            generation_entered.set()
            await release_submission.wait()
            submitted.set_result(None)
            await release_generation.wait()

        session._run_turn_generation = _fake_generation
        handler = asyncio.create_task(
            session._handle_turn_start(
                MossTTSRealtimeTurnStart(type="turn.start", turn_id="turn-0")
            )
        )

        await generation_entered.wait()
        assert websocket.sent == []

        release_submission.set()
        await handler
        assert websocket.sent[0]["type"] == "turn.started"
        assert session.active_turn is not None
        assert session.active_turn.client_started.is_set()

        release_generation.set()
        assert session.active_turn.generation_task is not None
        await session.active_turn.generation_task
        session.active_turn = None

    asyncio.run(_run())


def _realtime_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": "tts",
        "response_format": "pcm",
        "stream_audio": True,
        "sample_rate": 24000,
    }
    config.update(overrides)
    return {"type": "session.config", "session": config}


def _collect_until_turn_done(
    websocket: Any,
) -> tuple[list[dict[str, Any]], list[bytes]]:
    events: list[dict[str, Any]] = []
    audio_frames: list[bytes] = []
    while not any(event.get("type") == "turn.done" for event in events):
        message = websocket.receive()
        if message.get("text") is not None:
            events.append(json.loads(message["text"]))
        elif message.get("bytes") is not None:
            audio_frames.append(message["bytes"])
    return events, audio_frames


def test_realtime_endpoint_requires_session_config_first() -> None:
    client_impl = RealtimeSpeechClient()
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            speech_realtime_handler=_realtime_handler(),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json({"type": "turn.start", "turn_id": "turn"})
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["param"] == "type"
    assert client_impl.requests == []


def test_realtime_endpoint_streams_two_turns_and_preserves_input_ids() -> None:
    client_impl = RealtimeSpeechClient()
    tokenizer = BoundaryTokenizer()
    handler = _realtime_handler(tokenizer)
    assert tokenizer.len_calls == 1
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            architectures=["MossTTSRealtime"],
            speech_realtime_handler=handler,
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config())
        configured = websocket.receive_json()
        assert configured == {
            "type": "session.configured",
            "session_id": configured["session_id"],
            "response_format": "pcm",
            "sample_rate": 24000,
            "stream_audio": True,
            "input_modes": ["text", "tokens"],
            "max_active_turns": 1,
        }

        websocket.send_json({"type": "turn.start", "turn_id": "turn-0"})
        started = websocket.receive_json()
        assert started["type"] == "turn.started"
        assert started["next_seq_no"] == 0

        websocket.send_json(
            {"type": "input.text", "turn_id": "turn-0", "seq_no": 0, "text": "a"}
        )
        assert websocket.receive_json()["next_seq_no"] == 1
        websocket.send_json(
            {
                "type": "input.text",
                "turn_id": "turn-0",
                "seq_no": 1,
                "text": "bcdef",
            }
        )
        assert websocket.receive_json()["next_seq_no"] == 2

        # Exact retries are acknowledged without a second backend update.
        update_count = len(client_impl.updates)
        websocket.send_json(
            {
                "type": "input.text",
                "turn_id": "turn-0",
                "seq_no": 1,
                "text": "bcdef",
            }
        )
        retry_ack = websocket.receive_json()
        assert retry_ack["accepted_seq_no"] == 1
        assert len(client_impl.updates) == update_count

        websocket.send_json({"type": "input.done", "turn_id": "turn-0", "seq_no": 2})
        first_events, first_audio = _collect_until_turn_done(websocket)
        assert any(event["type"] == "input.ack" for event in first_events)
        assert any(event["type"] == "audio.start" for event in first_events)
        assert any(
            event["type"] == "audio.done" and event["error"] is False
            for event in first_events
        )
        assert first_events[-1]["committed"] is True
        assert first_audio

        first_request_id = started["request_id"]
        first_updates = [
            update
            for request_id, update in client_impl.updates
            if request_id == first_request_id
        ]
        emitted_ids = tuple(
            token_id for update in first_updates for token_id in update.token_ids
        )
        assert all(update.request_id == first_request_id for update in first_updates)
        assert all(
            update.session_id == started["session_id"] for update in first_updates
        )
        assert all(update.turn_id == started["turn_id"] for update in first_updates)
        assert emitted_ids == tuple(tokenizer.encode("abcdef"))
        assert sum(update.byte_count for update in first_updates) == 6
        assert first_updates[0].token_ids == ()
        assert first_updates[-1].input_done is True
        first_request = client_impl.requests[0][1]
        assert client_impl.realtime_opens[0] == {
            "request_id": first_request_id,
            "session_id": started["session_id"],
            "turn_id": started["turn_id"],
            "input_stage": "tts_engine",
        }
        assert first_request.metadata == {
            "task": "tts",
            "session_id": started["session_id"],
        }

        websocket.send_json({"type": "turn.start", "turn_id": "turn-1"})
        second_started = websocket.receive_json()
        websocket.send_json(
            {
                "type": "input.tokens",
                "turn_id": "turn-1",
                "seq_no": 0,
                "token_ids": [7, 8, 9],
            }
        )
        assert websocket.receive_json()["next_seq_no"] == 1
        websocket.send_json({"type": "input.done", "turn_id": "turn-1", "seq_no": 1})
        second_events, second_audio = _collect_until_turn_done(websocket)
        assert second_events[-1]["turn_id"] == "turn-1"
        assert second_audio

        second_updates = [
            update
            for request_id, update in client_impl.updates
            if request_id == second_started["request_id"]
        ]
        assert second_updates[0].token_ids == (7, 8, 9)

        websocket.send_json({"type": "session.close"})
        closed = websocket.receive_json()
        assert closed["type"] == "session.closed"

    assert len(client_impl.requests) == 2
    assert len(client_impl.closed_sessions) == 1
    assert client_impl.closed_sessions[0][1] == ("tts_engine", "vocoder")
    assert tokenizer.len_calls == 1


def test_realtime_events_cover_input_tokenize_and_first_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        speech_ws_module,
        "_emit_event",
        lambda **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(speech_ws_module, "realtime_events_active", lambda: True)
    client_impl = RealtimeSpeechClient()
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            speech_realtime_handler=_realtime_handler(),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config())
        configured = websocket.receive_json()
        websocket.send_json({"type": "turn.start", "turn_id": "turn-events"})
        started = websocket.receive_json()
        websocket.send_json(
            {
                "type": "input.text",
                "turn_id": "turn-events",
                "seq_no": 0,
                "text": "abcdef",
            }
        )
        assert websocket.receive_json()["type"] == "input.ack"
        websocket.send_json(
            {"type": "input.done", "turn_id": "turn-events", "seq_no": 1}
        )
        _, audio = _collect_until_turn_done(websocket)

    assert audio
    request_events = [
        event for event in captured if event["request_id"] == started["request_id"]
    ]
    assert [event["event_name"] for event in request_events] == [
        "ws_input_received",
        "text_tokenize_start",
        "text_tokenize_end",
        "ws_input_received",
        "text_tokenize_start",
        "text_tokenize_end",
        "pcm_encode_start",
        "pcm_host_ready",
        "pcm_send_begin",
        "pcm_send_end",
    ]
    assert request_events[0]["metadata"] == {
        "session_id": configured["session_id"],
        "turn_id": "turn-events",
        "turn_index": 0,
        "seq_no": 0,
        "input_type": "input.text",
        "input_done": False,
        "supplied_text_bytes": 6,
        "supplied_token_count": 0,
        "stable_token_count": 0,
    }
    assert request_events[2]["metadata"]["new_stable_token_count"] > 0
    assert request_events[-1]["metadata"]["pcm_bytes"] > 0
    assert request_events[-1]["metadata"]["audio_start_sent"] is True


def test_realtime_custom_input_stage_is_used_for_updates_and_session_close() -> None:
    client_impl = RealtimeSpeechClient()
    custom_stage = "custom_realtime_stage"
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            speech_realtime_handler=_realtime_handler(
                realtime_input_stage=custom_stage
            ),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config())
        configured = websocket.receive_json()
        websocket.send_json({"type": "turn.start", "turn_id": "turn"})
        started = websocket.receive_json()
        websocket.send_json(
            {
                "type": "input.tokens",
                "turn_id": "turn",
                "seq_no": 0,
                "token_ids": [7, 8, 9],
            }
        )
        assert websocket.receive_json()["next_seq_no"] == 1
        websocket.send_json({"type": "input.done", "turn_id": "turn", "seq_no": 1})
        _collect_until_turn_done(websocket)
        websocket.send_json({"type": "session.close"})
        assert websocket.receive_json()["type"] == "session.closed"

    assert configured["session_id"] == client_impl.closed_sessions[0][0]
    assert client_impl.closed_sessions[0][1] == (custom_stage, "vocoder")
    assert client_impl.realtime_opens[0] == {
        "request_id": started["request_id"],
        "session_id": configured["session_id"],
        "turn_id": "turn",
        "input_stage": custom_stage,
    }
    assert client_impl.updates[0][0] == started["request_id"]
    assert client_impl.updates[0][1].request_id == started["request_id"]
    assert client_impl.updates[0][1].session_id == configured["session_id"]
    assert client_impl.updates[0][1].turn_id == "turn"


def test_realtime_messages_before_turn_are_recoverable() -> None:
    client_impl = RealtimeSpeechClient()
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            speech_realtime_handler=_realtime_handler(),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config())
        assert websocket.receive_json()["type"] == "session.configured"

        websocket.send_json(
            {
                "type": "input.tokens",
                "turn_id": "missing",
                "seq_no": 0,
                "token_ids": [1],
            }
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["param"] == "type"

        websocket.send_json({"type": "turn.start", "turn_id": "turn"})
        assert websocket.receive_json()["type"] == "turn.started"
        websocket.send_json({"type": "turn.cancel", "turn_id": "turn"})
        events, _ = _collect_until_turn_done(websocket)
        assert events[-1]["committed"] is False
        assert events[-1]["reason"] == "client_cancelled"


def test_realtime_rejects_mixed_modes_gaps_and_changed_retries() -> None:
    client_impl = RealtimeSpeechClient()
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            speech_realtime_handler=_realtime_handler(),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config())
        websocket.receive_json()
        websocket.send_json({"type": "turn.start", "turn_id": "turn"})
        websocket.receive_json()

        original = {
            "type": "input.text",
            "turn_id": "turn",
            "seq_no": 0,
            "text": "abcdef",
        }
        websocket.send_json(original)
        assert websocket.receive_json()["type"] == "input.ack"

        websocket.send_json(original | {"text": "changed"})
        assert "different content" in websocket.receive_json()["message"]

        websocket.send_json(
            {
                "type": "input.tokens",
                "turn_id": "turn",
                "seq_no": 1,
                "token_ids": [1],
            }
        )
        assert "cannot mix" in websocket.receive_json()["message"]

        websocket.send_json(
            {"type": "input.text", "turn_id": "turn", "seq_no": 2, "text": "x"}
        )
        assert "expected 1, got 2" in websocket.receive_json()["message"]

        websocket.send_json({"type": "input.done", "turn_id": "turn", "seq_no": 1})
        events, _ = _collect_until_turn_done(websocket)
        assert events[-1]["committed"] is True


@pytest.mark.parametrize(
    "overrides, param",
    [
        ({"response_format": "wav"}, "response_format"),
        ({"stream_audio": False}, "stream_audio"),
        ({"sample_rate": 16000}, "sample_rate"),
    ],
)
def test_realtime_config_requires_24khz_streamed_pcm(
    overrides: dict[str, Any], param: str
) -> None:
    client = TestClient(
        create_app(
            RealtimeSpeechClient(),
            model_name="tts",
            speech_realtime_handler=_realtime_handler(),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config(**overrides))
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["param"] == param


@pytest.mark.parametrize("token_ids", [[], [True], [-1], [10000]])
def test_realtime_direct_tokens_are_strictly_validated(token_ids: list[Any]) -> None:
    client_impl = RealtimeSpeechClient()
    client = TestClient(
        create_app(
            client_impl,
            model_name="tts",
            speech_realtime_handler=_realtime_handler(),
        )
    )

    with client.websocket_connect("/v1/audio/speech/realtime") as websocket:
        websocket.send_json(_realtime_config())
        websocket.receive_json()
        websocket.send_json({"type": "turn.start", "turn_id": "turn"})
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "input.tokens",
                "turn_id": "turn",
                "seq_no": 0,
                "token_ids": token_ids,
            }
        )
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert client_impl.updates == []
