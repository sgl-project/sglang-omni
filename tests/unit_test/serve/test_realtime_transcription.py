# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import numpy as np
import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client import CompletionResult, GenerateRequest
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.models.qwen3_asr.streaming import Qwen3ASRStreamingStrategy
from sglang_omni.serve.realtime import transcription_session as session_module
from sglang_omni.serve.realtime import vad as vad_module
from sglang_omni.serve.realtime.events import MAX_TRANSCRIPTION_PROMPT_CHARACTERS
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)
from sglang_omni.serve.realtime.vad import (
    StatelessVAD,
    StreamingVAD,
    VADConfig,
    VADEvent,
)


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class FakeSpeechModel:
    """Scores a frame as speech when it holds any non-zero sample."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def predict(self, frame: np.ndarray, _sample_rate: int) -> float:
        return 1.0 if np.any(frame) else 0.0

    def reset(self) -> None:
        self.reset_calls += 1


def use_fake_speech_model(monkeypatch: pytest.MonkeyPatch) -> FakeSpeechModel:
    """Run the real StatelessVAD over a fake model so Silero is never loaded."""
    model = FakeSpeechModel()
    monkeypatch.setattr(
        session_module, "StatelessVAD", lambda config: StatelessVAD(config, model=model)
    )
    return model


class FakeStrategy:
    def create_state(
        self, *, model_name: str, language: str | None
    ) -> dict[str, str | None]:
        return {"model_name": model_name, "language": language}

    def build_decode_request(self, **_: Any) -> GenerateRequest:
        return GenerateRequest(prompt="audio", stream=False)

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        language: str | None,
        state: object,
    ) -> str:
        assert isinstance(state, dict)
        state["language"] = language
        return generated_text


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []
        self.requests: list[GenerateRequest] = []
        self.aborted: list[str] = []

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        self.requests.append(_request)
        text = self.outputs.pop(0) if self.outputs else f"text-{len(self.calls)}"
        return CompletionResult(request_id=request_id, text=text, language="English")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        self.requests.append(_request)
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
        return CompletionResult(
            request_id=request_id,
            text=f"text-{len(self.calls)}",
            language="English",
        )


class BlockingSecondClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.second_started = asyncio.Event()

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        if len(self.calls) == 2:
            self.second_started.set()
            await asyncio.Event().wait()
        return CompletionResult(
            request_id=request_id,
            text=f"text-{len(self.calls)}",
            language="English",
        )


def make_pcm(seconds: float, amplitude: int = 1000) -> bytes:
    samples = int(16000 * seconds)
    return amplitude.to_bytes(2, "little", signed=True) * samples


def audio_event(pcm: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode(),
    }


async def make_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: list[str] | None = None,
    max_segment_s: float | None = 60.0,
    supports_prompt: bool = False,
) -> tuple[RealtimeTranscriptionSession, RecordingWebSocket, FakeClient]:
    use_fake_speech_model(monkeypatch)
    websocket = RecordingWebSocket()
    client = FakeClient(outputs or [])
    strategy = Qwen3ASRStreamingStrategy() if supports_prompt else FakeStrategy()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-asr",
        transcription_config=RealtimeTranscriptionConfig(
            strategy_cls=type(strategy),
            decode_interval_ms=2000,
            max_segment_s=max_segment_s,
            supports_prompt=supports_prompt,
        ),
        strategy=strategy,
        session_id="sess-test",
    )
    await session.dispatch(
        {
            "type": "session.update",
            "session": {"turn_detection": None},
        }
    )
    return session, websocket, client


@pytest.mark.asyncio
async def test_partial_is_replaced_by_one_final_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(
        monkeypatch, outputs=["hello wor", "hello world"]
    )
    await session.dispatch(audio_event(make_pcm(2.0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if any(event["type"] == "transcription.segment" for event in websocket.events):
            break

    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    hypotheses = [
        event for event in websocket.events if event["type"] == "transcription.segment"
    ]
    assert [(event["text"], event["is_final"]) for event in hypotheses] == [
        ("hello wor", False),
        ("hello world", True),
    ]
    indexes = [event["event_index"] for event in websocket.events]
    assert indexes == sorted(indexes) and len(indexes) == len(set(indexes))
    completed = websocket.events[-1]
    assert completed["type"] == "transcription.completed"
    assert completed["text"] == "hello world"
    assert session.decode_worker_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [(" \nPyTorch, 张三\t", "PyTorch, 张三"), (None, None), ("", None), (" \n", None)],
)
async def test_prompt_updates_normalize_clear_and_retain_omitted_values(
    monkeypatch: pytest.MonkeyPatch, prompt: str | None, expected: str | None
) -> None:
    session, websocket, client = await make_session(monkeypatch, supports_prompt=True)
    try:
        await session.send(
            session_module.TranscriptionSessionCreated(session=session.session_object())
        )
        assert websocket.events[-1]["session"]["prompt"] is None
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "original"}}
        )
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": prompt}}
        )
        assert websocket.events[-1]["type"] == "session.updated"
        assert websocket.events[-1]["session"]["prompt"] == expected
        await session.dispatch(
            {"type": "session.update", "session": {"language": "English"}}
        )
        assert websocket.events[-1]["session"]["prompt"] == expected
        await session.dispatch(audio_event(make_pcm(0.5)))
        await session.dispatch({"type": "transcription.done"})
        assert client.requests[0].extra_params.get("prompt") == expected
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_prompt_limit_counts_unicode_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(monkeypatch, supports_prompt=True)
    prompt = "词" * MAX_TRANSCRIPTION_PROMPT_CHARACTERS
    try:
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": prompt}}
        )
        assert websocket.events[-1]["type"] == "session.updated"
        assert websocket.events[-1]["session"]["prompt"] == prompt
        await session.dispatch(
            {
                "type": "session.update",
                "session": {"prompt": prompt + "词", "language": "French"},
            }
        )
        assert websocket.events[-1]["error"]["code"] == "invalid_event"
        assert session.session_object().prompt == prompt
        assert session.settings.language is None
    finally:
        await session.teardown()


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [123, ["PyTorch"], {"term": "PyTorch"}])
async def test_prompt_rejects_non_string_values(
    monkeypatch: pytest.MonkeyPatch, prompt: int | list[str] | dict[str, str]
) -> None:
    session, websocket, _client = await make_session(monkeypatch, supports_prompt=True)
    try:
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": prompt}}
        )
        assert websocket.events[-1]["error"]["code"] == "invalid_event"
        assert session.session_object().prompt is None
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_unsupported_prompt_rejects_entire_update_and_keeps_legacy_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, client = await make_session(monkeypatch)
    try:
        await session.dispatch(
            {
                "type": "session.update",
                "session": {"prompt": "PyTorch", "language": "French"},
            }
        )
        assert websocket.events[-1]["error"]["code"] == "unsupported_prompt"
        assert session.session_object().prompt is None
        assert session.settings.language is None
        for prompt in (None, "", " \t"):
            await session.dispatch(
                {"type": "session.update", "session": {"prompt": prompt}}
            )
            assert websocket.events[-1]["type"] == "session.updated"
        await session.dispatch(audio_event(make_pcm(0.5)))
        await session.dispatch({"type": "transcription.done"})
        assert len(client.requests) == 1
        assert websocket.events[-1]["type"] == "transcription.completed"
        assert [
            event["error"]["code"]
            for event in websocket.events
            if event["type"] == "error"
        ] == ["unsupported_prompt"]
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_invalid_vad_does_not_apply_prompt_or_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket = vad_session(monkeypatch)
    session.transcription_config = RealtimeTranscriptionConfig(
        strategy_cls=Qwen3ASRStreamingStrategy, server_vad=True, supports_prompt=True
    )
    session.strategy = Qwen3ASRStreamingStrategy()
    try:
        await session.dispatch(
            {
                "type": "session.update",
                "session": {"prompt": "original", "language": "English"},
            }
        )
        before = session.session_object()
        await session.dispatch(
            {
                "type": "session.update",
                "session": {
                    "prompt": "replacement",
                    "language": "French",
                    "turn_detection": {
                        "type": "server_vad",
                        "prefix_padding_ms": 600,
                        "silence_duration_ms": 500,
                    },
                },
            }
        )
        assert websocket.events[-1]["error"]["code"] == "invalid_turn_detection"
        assert session.session_object() == before
    finally:
        await session.teardown()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration_s", [0.5, 1.0])
async def test_active_audio_and_empty_hard_cap_segment_reject_setting_updates(
    monkeypatch: pytest.MonkeyPatch, duration_s: float
) -> None:
    session, websocket, _client = await make_session(
        monkeypatch, supports_prompt=True, max_segment_s=1.0
    )
    try:
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "original"}}
        )
        await session.dispatch(audio_event(make_pcm(duration_s)))
        before = session.session_object()
        assert session.active_segment is not None
        assert session.audio_buffer.is_empty() == (duration_s == 1.0)
        for update in (
            {"prompt": "replacement"},
            {"language": "French"},
            {"turn_detection": None},
        ):
            await session.dispatch({"type": "session.update", "session": update})
            assert websocket.events[-1]["error"]["code"] == "session_active"
            assert session.session_object() == before
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_prompt_update_rejects_idle_vad_audio_before_a_segment_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket = vad_session(monkeypatch)
    session.transcription_config = RealtimeTranscriptionConfig(
        strategy_cls=Qwen3ASRStreamingStrategy, server_vad=True, supports_prompt=True
    )
    session.strategy = Qwen3ASRStreamingStrategy()
    try:
        await session.dispatch(audio_event(make_pcm(0.1, amplitude=0)))
        assert session.active_segment is None
        assert not session.audio_buffer.is_empty()
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "PyTorch"}}
        )
        assert websocket.events[-1]["error"]["code"] == "session_active"
        assert session.session_object().prompt is None
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_commit_closes_empty_hard_cap_segment_without_losing_its_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(
        monkeypatch, supports_prompt=True, max_segment_s=1.0
    )
    client = BlockingClient()
    session.client = client
    try:
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "original"}}
        )
        await session.dispatch(audio_event(make_pcm(1.0)))
        await asyncio.wait_for(client.started.wait(), timeout=5.0)
        assert session.audio_buffer.is_empty()
        assert session.active_segment is not None
        await session.dispatch({"type": "input_audio_buffer.commit"})
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "replacement"}}
        )
        assert websocket.events[-1]["type"] == "session.updated"
        await session.dispatch(audio_event(make_pcm(0.5)))
        client.release.set()
        await session.dispatch({"type": "transcription.done"})
        assert [request.extra_params["prompt"] for request in client.requests] == [
            "original",
            "replacement",
        ]
        assert client.aborted == []
        assert [
            event["text"]
            for event in websocket.events
            if event["type"] == "transcription.segment" and event["is_final"]
        ] == ["text-1", "text-2"]
        assert websocket.events[-1]["text"] == "text-1 text-2"
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_queued_final_retains_prompt_after_next_segment_setting_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(monkeypatch, supports_prompt=True)
    client = BlockingClient()
    session.client = client
    try:
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "original"}}
        )
        await session.dispatch(audio_event(make_pcm(2.0)))
        await asyncio.wait_for(client.started.wait(), timeout=5.0)
        await session.dispatch({"type": "input_audio_buffer.commit"})
        assert session.pending_finals
        await session.dispatch(
            {"type": "session.update", "session": {"prompt": "replacement"}}
        )
        assert websocket.events[-1]["type"] == "session.updated"
        await session.dispatch(audio_event(make_pcm(0.5)))
        await session.dispatch({"type": "input_audio_buffer.commit"})
        client.release.set()
        await session.dispatch({"type": "transcription.done"})
        assert [request.extra_params["prompt"] for request in client.requests] == [
            "original",
            "original",
            "replacement",
        ]
        assert websocket.events[-1]["type"] == "transcription.completed"
    finally:
        await session.teardown()


@pytest.mark.asyncio
async def test_clear_preserves_prompt_without_leaking_into_another_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _first_websocket, first_client = await make_session(
        monkeypatch, supports_prompt=True
    )
    second, _second_websocket, second_client = await make_session(
        monkeypatch, supports_prompt=True
    )
    try:
        await first.dispatch(
            {"type": "session.update", "session": {"prompt": "first vocabulary"}}
        )
        await second.dispatch(
            {"type": "session.update", "session": {"prompt": "second vocabulary"}}
        )
        await first.dispatch(audio_event(make_pcm(0.5)))
        await first.dispatch({"type": "input_audio_buffer.clear"})
        assert first.session_object().prompt == "first vocabulary"
        await first.dispatch(audio_event(make_pcm(0.5)))
        await first.dispatch({"type": "transcription.done"})
        await second.dispatch(audio_event(make_pcm(0.5)))
        await second.dispatch({"type": "transcription.done"})
        assert [
            request.extra_params["prompt"] for request in first_client.requests
        ] == ["first vocabulary"]
        assert [
            request.extra_params["prompt"] for request in second_client.requests
        ] == ["second vocabulary"]
    finally:
        await first.teardown()
        await second.teardown()


@pytest.mark.asyncio
async def test_audio_during_decode_coalesces_to_one_followup_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _websocket, _client = await make_session(monkeypatch)
    client = BlockingClient()
    session.client = client  # type: ignore[assignment]

    await session.dispatch(audio_event(make_pcm(2.0)))
    await client.started.wait()
    await session.dispatch(audio_event(make_pcm(2.0)))
    client.release.set()
    for _ in range(20):
        await asyncio.sleep(0)
        if len(client.calls) == 2:
            break

    assert len(client.calls) == 2
    await session.teardown()


@pytest.mark.asyncio
async def test_events_after_done_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    session, websocket, client = await make_session(monkeypatch)
    await session.dispatch(audio_event(make_pcm(1.0)))
    await session.dispatch({"type": "transcription.done"})
    assert websocket.events[-1]["type"] == "transcription.completed"
    worker = session.decode_worker_task

    for event in (
        {"type": "input_audio_buffer.clear"},
        {"type": "input_audio_buffer.commit"},
        {"type": "transcription.done"},
        audio_event(make_pcm(0.1)),
    ):
        await session.dispatch(event)
        assert websocket.events[-1]["type"] == "error"
        assert websocket.events[-1]["error"]["code"] == "input_already_done"

    # clear in particular must not spawn a fresh worker or a second decode.
    assert session.decode_worker_task is worker
    assert len(client.calls) == 1
    await session.teardown()


@pytest.mark.asyncio
async def test_teardown_aborts_inflight_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(monkeypatch)
    client = BlockingClient()
    session.client = client  # type: ignore[assignment]

    await session.dispatch(audio_event(make_pcm(2.0)))
    await client.started.wait()
    await session.teardown()

    assert client.aborted == [client.calls[0]]
    assert session.decode_worker_task.done()
    assert websocket.client_state == WebSocketState.DISCONNECTED


@pytest.mark.asyncio
async def test_clear_aborts_active_segment_and_session_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad = use_fake_speech_model(monkeypatch)
    websocket = RecordingWebSocket()
    client = BlockingSecondClient()
    strategy = FakeStrategy()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-asr",
        transcription_config=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            decode_interval_ms=2000,
            server_vad=True,
            max_segment_s=30.0,
        ),
        strategy=strategy,
        session_id="sess-clear",
    )

    await session.dispatch(audio_event(make_pcm(2.0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if any(event["type"] == "transcription.segment" for event in websocket.events):
            break
    assert session.active_segment is not None
    cleared_state = session.active_segment.strategy_state

    await session.dispatch(audio_event(make_pcm(2.0)))
    await client.second_started.wait()
    await session.dispatch({"type": "input_audio_buffer.clear"})

    assert client.aborted == [client.calls[1]]
    assert session.audio_buffer.is_empty()
    assert session.active_segment is None
    assert vad.reset_calls == 1
    assert not session.decode_worker_task.done()
    assert not session.pending_finals
    assert not session.final_waiters
    assert websocket.events[-1]["type"] == "input_audio_buffer.cleared"
    assert not any(
        event["type"] == "transcription.segment"
        and event["segment_id"] == 0
        and event["is_final"]
        for event in websocket.events
    )

    await session.dispatch(audio_event(make_pcm(1.0)))
    assert session.active_segment is not None
    assert session.active_segment.strategy_state is not cleared_state
    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    finals = [
        event
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [(event["segment_id"], event["text"]) for event in finals] == [(1, "text-3")]
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == "text-3"


@pytest.mark.asyncio
async def test_silent_final_does_not_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, client = await make_session(monkeypatch)

    await session.dispatch(audio_event(make_pcm(0.5, amplitude=0)))
    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    assert client.calls == []
    assert websocket.events[-2]["type"] == "transcription.segment"
    assert websocket.events[-2]["text"] == ""
    assert websocket.events[-2]["is_final"] is True
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == ""


@pytest.mark.asyncio
async def test_hard_limit_finalizes_in_audio_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(monkeypatch, max_segment_s=1.0)
    await session.dispatch(audio_event(make_pcm(2.25)))
    await session.dispatch({"type": "transcription.done"})

    finals = [
        event
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["segment_id"] for event in finals] == [0, 1, 2]
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == "text-1 text-2 text-3"


@pytest.mark.asyncio
async def test_vad_idle_silence_keeps_buffer_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_speech_model(monkeypatch)
    websocket = RecordingWebSocket()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=FakeClient([]),  # type: ignore[arg-type]
        model_name="qwen3-asr",
        transcription_config=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            decode_interval_ms=2000,
            server_vad=True,
            max_segment_s=1.0,
        ),
        strategy=FakeStrategy(),
        session_id="sess-idle",
    )

    # Server VAD never reports speech, so no segment starts and _queue_final
    # never drains the buffer. Streaming past max_segment_s + 4s of audio
    # must still not raise BufferOverflow.
    for _ in range(8):
        await session.dispatch(audio_event(make_pcm(1.0, amplitude=0)))

    assert session.active_segment is None
    assert not [event for event in websocket.events if event["type"] == "error"]
    assert session.audio_buffer.num_bytes < session.audio_buffer.max_bytes
    await session.teardown()


class ExplodingStrategy(FakeStrategy):
    def create_state(self, **settings: Any) -> object:
        raise RuntimeError("strategy exploded")


@pytest.mark.asyncio
async def test_handler_exception_is_reported_and_session_survives(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    session, websocket, client = await make_session(monkeypatch, outputs=["hello"])
    session.strategy = ExplodingStrategy()

    with caplog.at_level(logging.ERROR):
        await session.dispatch(audio_event(make_pcm(0.5)))

    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "internal_error"
    assert "strategy exploded" in caplog.text

    session.strategy = FakeStrategy()
    await session.dispatch(audio_event(make_pcm(0.5)))
    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == "hello"


@pytest.mark.asyncio
async def test_prefix_padding_must_fit_inside_silence_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(monkeypatch)
    session.transcription_config = RealtimeTranscriptionConfig(
        strategy_cls=FakeStrategy, server_vad=True
    )

    await session.dispatch(
        {
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "server_vad",
                    "prefix_padding_ms": 600,
                    "silence_duration_ms": 500,
                }
            },
        }
    )
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "invalid_turn_detection"
    assert session.vad is None

    await session.dispatch(
        {
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "server_vad",
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": 500,
                }
            },
        }
    )
    assert websocket.events[-1]["type"] == "session.updated"
    assert session.vad is not None
    await session.teardown()


@pytest.mark.asyncio
async def test_vad_settings_reject_negative_padding_and_zero_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await make_session(monkeypatch)
    session.transcription_config = RealtimeTranscriptionConfig(
        strategy_cls=FakeStrategy, server_vad=True
    )

    for turn_detection in (
        {"type": "server_vad", "prefix_padding_ms": -300, "silence_duration_ms": 500},
        {"type": "server_vad", "prefix_padding_ms": 0, "silence_duration_ms": 0},
    ):
        await session.dispatch(
            {"type": "session.update", "session": {"turn_detection": turn_detection}}
        )
        assert websocket.events[-1]["type"] == "error", turn_detection
        assert websocket.events[-1]["error"]["code"] == "invalid_turn_detection"
        assert session.vad is None
    await session.teardown()


class FailOnceStrategy(FakeStrategy):
    def __init__(self) -> None:
        self.failures_left = 1

    def create_state(self, **settings: Any) -> object:
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("onset exploded")
        return super().create_state(**settings)


@pytest.mark.asyncio
async def test_failed_onset_does_not_strand_the_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad = use_fake_speech_model(monkeypatch)
    websocket = RecordingWebSocket()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=FakeClient([]),  # type: ignore[arg-type]
        model_name="qwen3-asr",
        transcription_config=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            decode_interval_ms=2000,
            server_vad=True,
            max_segment_s=30.0,
        ),
        strategy=FailOnceStrategy(),
        session_id="sess-onset",
    )

    # First onset: the VAD reports it, then segment creation fails.
    await session.dispatch(audio_event(make_pcm(0.5)))
    assert websocket.events[-1]["type"] == "error"
    assert session.active_segment is None
    assert vad.reset_calls == 0  # nothing to resync, the VAD holds no turn state

    # Still told that no turn is open, the VAD reports the onset again on the
    # next speech frame and the utterance is transcribed normally.
    await session.dispatch(audio_event(make_pcm(2.0)))
    assert session.active_segment is not None
    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})
    finals = [
        event
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert len(finals) == 1 and finals[0]["text"]
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == finals[0]["text"]


def no_vad_session() -> tuple[RealtimeTranscriptionSession, RecordingWebSocket]:
    websocket = RecordingWebSocket()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=FakeClient([]),  # type: ignore[arg-type]
        model_name="no-vad-asr",
        transcription_config=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            server_vad=False,
        ),
        strategy=FakeStrategy(),
        session_id="sess-no-vad",
    )
    return session, websocket


@pytest.mark.asyncio
async def test_model_without_server_vad_starts_in_manual_mode() -> None:
    session, websocket = no_vad_session()
    await session.send(
        session_module.TranscriptionSessionCreated(session=session.session_object())
    )

    assert websocket.events[-1]["session"]["turn_detection"] is None
    assert session.vad is None
    await session.dispatch(audio_event(make_pcm(0.5)))
    assert session.active_segment is not None
    await session.teardown()


@pytest.mark.asyncio
async def test_model_without_server_vad_rejects_turn_detection() -> None:
    session, websocket = no_vad_session()
    await session.dispatch(
        {
            "type": "session.update",
            "session": {"turn_detection": {"type": "server_vad"}},
        }
    )

    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "unsupported_turn_detection"
    assert session.vad is None
    await session.teardown()


def vad_session(
    monkeypatch: pytest.MonkeyPatch, *, max_segment_s: float = 30.0
) -> tuple[RealtimeTranscriptionSession, RecordingWebSocket]:
    use_fake_speech_model(monkeypatch)
    websocket = RecordingWebSocket()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=FakeClient([]),  # type: ignore[arg-type]
        model_name="qwen3-asr",
        transcription_config=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            server_vad=True,
            max_segment_s=max_segment_s,
        ),
        strategy=FakeStrategy(),
        session_id="sess-vad",
    )
    return session, websocket


class ContentStreamingVAD(StreamingVAD):
    def infer(self, frame: np.ndarray) -> float:
        return 1.0 if np.any(frame) else 0.0


@pytest.mark.asyncio
async def test_frame_vad_reports_the_same_boundaries_as_streaming_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vad_module, "load_silero_vad", lambda onnx: None)
    reference = ContentStreamingVAD(VADConfig())
    session, websocket = vad_session(monkeypatch)
    audio = (
        make_pcm(0.4, amplitude=0)
        + make_pcm(1.0)
        + make_pcm(0.8, amplitude=0)
        + make_pcm(0.6)
        + make_pcm(0.3, amplitude=0)  # a pause shorter than silence_duration_ms
        + make_pcm(0.5)
        + make_pcm(0.8, amplitude=0)
    )

    expected = []
    packet_bytes = 1000 * 2  # not a multiple of the 512-sample frame
    for start in range(0, len(audio), packet_bytes):
        packet = audio[start : start + packet_bytes]
        for emit in reference.process(packet):
            started = emit.event_type == VADEvent.SPEECH_STARTED
            expected.append((started, vad_module.offsets_to_ms(emit.sample_offset)))
        await session.dispatch(audio_event(packet))

    reported = [
        (
            event["type"].endswith("speech_started"),
            event.get("audio_start_ms", event.get("audio_end_ms")),
        )
        for event in websocket.events
        if event["type"].startswith("input_audio_buffer.speech_")
    ]
    assert [started for started, _ms in expected] == [True, False, True, False]
    assert reported == expected
    await session.teardown()


@pytest.mark.asyncio
async def test_hard_cut_inside_trailing_silence_leaves_no_empty_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket = vad_session(monkeypatch, max_segment_s=1.0)

    # Speech ends at 0.9 s, the hard cut lands at 1.0 s inside the silence,
    # and the offset fires later for the utterance the cut already closed.
    for packet in [make_pcm(0.9)] + [make_pcm(0.1, amplitude=0)] * 8:
        await session.dispatch(audio_event(packet))
    await session.dispatch({"type": "transcription.done"})

    stops = [
        event
        for event in websocket.events
        if event["type"] == "input_audio_buffer.speech_stopped"
    ]
    assert [event["segment_id"] for event in stops] == [0]
    committed = [
        event["segment_id"]
        for event in websocket.events
        if event["type"] == "input_audio_buffer.committed"
    ]
    finals = [
        event["segment_id"]
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert committed == finals == [0]
