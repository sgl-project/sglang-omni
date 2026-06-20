# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client import GenerateChunk
from sglang_omni.client.types import SpeechResult
from sglang_omni.serve import create_app
from sglang_omni.serve import speech_ws as speech_ws_module
from sglang_omni.serve.protocol import SpeechStreamSessionConfig
from sglang_omni.serve.speech_service import (
    MAX_SPEECH_INPUT_CHARS,
    SpeechRequestValidator,
)
from sglang_omni.serve.speech_ws import (
    MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION,
    MAX_TEXT_MESSAGE_BYTES,
    SpeechWebSocketSession,
)


class StreamingSpeechClient:
    def __init__(self, *, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.generated_prompts: list[str] = []
        self.speech_prompts: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        self.generated_prompts.append(request.prompt)
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=self.sample_rate,
            finish_reason="stop",
        )

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request_id, speed, allow_format_fallback
        self.speech_prompts.append(request.prompt)
        return SpeechResult(
            audio_bytes=b"RIFF",
            mime_type=f"audio/{response_format}",
            format=response_format,
            sample_rate=self.sample_rate,
        )

    async def abort(self, request_id: str) -> None:
        del request_id


class BlockingStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        self.started.set()
        await asyncio.Future()
        yield GenerateChunk(request_id=request_id or "speech-ws")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request, request_id, response_format, speed, allow_format_fallback
        self.started.set()
        await asyncio.Future()
        return SpeechResult(audio_bytes=b"", mime_type="audio/wav", format="wav")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class ReleasableSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted: list[str] = []

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request, request_id, response_format, speed, allow_format_fallback
        self.started.set()
        await self.release.wait()
        return SpeechResult(audio_bytes=b"RIFF", mime_type="audio/wav", format="wav")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class TwoChunkStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.0],
            sample_rate=24000,
            finish_reason="stop",
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class PausingStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        await asyncio.Future()
        yield GenerateChunk(request_id=request_id or "speech-ws")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class InvalidAudioStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-ws",
            modality="audio",
            audio_data=object(),
            sample_rate=24000,
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class CountingSpeechRequestValidator(SpeechRequestValidator):
    def __init__(self) -> None:
        super().__init__(default_model="tts")
        self.prepare_count = 0

    def prepare_generation_request(self, request: Any) -> Any:
        self.prepare_count += 1
        return super().prepare_generation_request(request)


class CompletedSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request, request_id, response_format, speed, allow_format_fallback
        return SpeechResult(audio_bytes=b"RIFF", mime_type="audio/wav", format="wav")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class RecordingWebSocket:
    def __init__(
        self,
        *,
        fail_bytes: bool = False,
        receive_messages: list[dict[str, Any]] | None = None,
        receive_after_bytes: int = 0,
    ) -> None:
        self.fail_bytes = fail_bytes
        self.receive_messages = list(receive_messages or [])
        self.receive_after_bytes = receive_after_bytes
        self.application_state = WebSocketState.CONNECTED
        self.client_state = WebSocketState.CONNECTED
        self.sent_text: list[dict[str, Any]] = []
        self.sent_bytes: list[bytes] = []

    async def send_text(self, payload: str) -> None:
        self.sent_text.append(json.loads(payload))

    async def send_bytes(self, payload: bytes) -> None:
        if self.fail_bytes:
            self.application_state = WebSocketState.DISCONNECTED
            self.client_state = WebSocketState.DISCONNECTED
            raise RuntimeError("client disconnected")
        self.sent_bytes.append(payload)

    async def receive(self) -> dict[str, Any]:
        while self.receive_after_bytes > len(self.sent_bytes):
            await asyncio.sleep(0)
        if self.receive_messages:
            message = self.receive_messages.pop(0)
            if message.get("type") == "websocket.disconnect":
                self.application_state = WebSocketState.DISCONNECTED
                self.client_state = WebSocketState.DISCONNECTED
            return message
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


def test_speech_websocket_streams_sentences_as_binary_frames() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "session": {
                    "response_format": "pcm",
                    "stream_audio": True,
                    "split_granularity": "sentence",
                },
            }
        )
        configured = websocket.receive_json()
        assert configured["type"] == "session.configured"

        websocket.send_json({"type": "input.text", "text": "Hello. Second"})
        first_start = websocket.receive_json()
        first_audio = websocket.receive_bytes()
        first_done = websocket.receive_json()

        assert first_start["type"] == "audio.start"
        assert first_start["sentence_text"] == "Hello."
        assert first_audio
        assert first_done["type"] == "audio.done"
        assert first_done["error"] is False

        websocket.send_json({"type": "input.done"})
        second_start = websocket.receive_json()
        second_audio = websocket.receive_bytes()
        second_done = websocket.receive_json()
        session_done = websocket.receive_json()

        assert second_start["sentence_text"] == "Second"
        assert second_audio
        assert second_done["error"] is False
        assert session_done["type"] == "session.done"
        assert session_done["total_sentences"] == 2

    assert client_impl.generated_prompts == ["Hello.", "Second"]


def test_speech_websocket_rejects_missing_initial_config() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "input.text", "text": "hello"})
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert event["param"] == "type"


def test_speech_websocket_rejects_binary_client_frames() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "pcm"})
        assert websocket.receive_json()["type"] == "session.configured"

        websocket.send_bytes(b"not-json-text")
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert "text frames" in event["message"]

        websocket.send_json({"type": "input.done"})
        assert websocket.receive_json()["type"] == "session.done"


def test_speech_websocket_supports_non_streaming_sentence_frames() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "response_format": "wav",
                "stream_audio": False,
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"

        websocket.send_json({"type": "input.text", "text": "Hello."})
        assert websocket.receive_json()["type"] == "audio.start"
        assert websocket.receive_bytes() == b"RIFF"
        assert websocket.receive_json()["type"] == "audio.done"
        websocket.send_json({"type": "input.done"})
        assert websocket.receive_json()["type"] == "session.done"

    assert client_impl.speech_prompts == ["Hello."]


def test_speech_websocket_stream_audio_defaults_to_non_streaming() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "wav"})
        configured = websocket.receive_json()
        assert configured["type"] == "session.configured"
        assert configured["stream_audio"] is False

        websocket.send_json({"type": "input.text", "text": "Hello."})
        assert websocket.receive_json()["type"] == "audio.start"
        assert websocket.receive_bytes() == b"RIFF"
        assert websocket.receive_json()["type"] == "audio.done"

    assert client_impl.speech_prompts == ["Hello."]


def test_speech_websocket_unknown_message_type_is_recoverable() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "pcm"})
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json({"type": "unexpected"})
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["param"] == "type"
        websocket.send_json({"type": "input.done"})
        assert websocket.receive_json()["type"] == "session.done"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("stream_audio", "true"),
        ("speed", "1.2"),
        ("max_new_tokens", "5"),
        ("token_count", "5"),
        ("duration_tokens", "5"),
    ],
)
def test_speech_websocket_rejects_stringified_config_types(
    field_name: str, value: str
) -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                field_name: value,
                "response_format": "pcm",
            }
        )
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert event["param"] == field_name


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("token_count", 0),
        ("token_count", -1),
        ("duration_tokens", 0),
        ("duration_tokens", -1),
    ],
)
def test_speech_websocket_rejects_non_positive_duration_fields(
    field_name: str, value: int
) -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                field_name: value,
                "response_format": "pcm",
            }
        )
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert event["param"] == field_name


def test_speech_websocket_streaming_accepts_speed() -> None:
    client = TestClient(create_app(StreamingSpeechClient(), model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
                "speed": 1.1,
            }
        )
        configured = websocket.receive_json()
        assert configured["type"] == "session.configured"


def test_speech_websocket_default_config_does_not_mark_generation_params_explicit() -> (
    None
):
    async def run() -> None:
        speech_service = SpeechRequestValidator(default_model="tts")
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=StreamingSpeechClient(),
            speech_service=speech_service,
        )

        session.config = await session._parse_config(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
            }
        )
        request = session._speech_request_from_config(sentence="Hello.", stream=True)
        gen_req = speech_service.build_generate_request(
            request,
            validate=False,
            reference_descriptors=session._config_reference_descriptors(),
            uploaded_voice=session._config_uploaded_voice(),
        )

        assert "explicit_generation_params" not in gen_req.metadata["tts_params"]

    asyncio.run(run())


def test_speech_websocket_preserves_explicit_generation_params() -> None:
    async def run() -> None:
        speech_service = SpeechRequestValidator(default_model="tts")
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=StreamingSpeechClient(),
            speech_service=speech_service,
        )

        session.config = await session._parse_config(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
                "temperature": 0.7,
                "top_k": 20,
            }
        )
        request = session._speech_request_from_config(sentence="Hello.", stream=True)
        gen_req = speech_service.build_generate_request(
            request,
            validate=False,
            reference_descriptors=session._config_reference_descriptors(),
            uploaded_voice=session._config_uploaded_voice(),
        )

        assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
            "temperature",
            "top_k",
        ]

    asyncio.run(run())


def test_speech_websocket_rejects_oversized_sentence_before_generation() -> None:
    client_impl = StreamingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "pcm"})
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json(
            {
                "type": "input.text",
                "text": "x" * (MAX_SPEECH_INPUT_CHARS + 1) + ".",
            }
        )
        error = websocket.receive_json()
        done = websocket.receive_json()

    assert error["type"] == "error"
    assert error["param"] == "input"
    assert done["type"] == "audio.done"
    assert done["error"] is True
    assert client_impl.generated_prompts == []


def test_speech_websocket_stream_start_uses_chunk_sample_rate() -> None:
    client = TestClient(
        create_app(StreamingSpeechClient(sample_rate=44100), model_name="tts")
    )

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json({"type": "input.text", "text": "Hello."})
        start = websocket.receive_json()

    assert start["type"] == "audio.start"
    assert start["sample_rate"] == 44100


def test_speech_websocket_non_streaming_start_uses_result_sample_rate() -> None:
    client = TestClient(
        create_app(StreamingSpeechClient(sample_rate=44100), model_name="tts")
    )

    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json({"type": "session.config", "response_format": "pcm"})
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json({"type": "input.text", "text": "Hello."})
        start = websocket.receive_json()

    assert start["type"] == "audio.start"
    assert start["sample_rate"] == 44100


def test_speech_websocket_cancellation_aborts_active_request() -> None:
    async def run() -> None:
        client_impl = BlockingStreamingSpeechClient()
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        task = asyncio.create_task(session._generate_sentence("Hello."))
        await client_impl.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_parent_cancellation_cleans_generation_tasks() -> None:
    async def run() -> None:
        client_impl = BlockingStreamingSpeechClient()
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)
        before_tasks = asyncio.all_tasks()

        task = asyncio.create_task(session._generate_sentence("Hello."))
        await client_impl.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        leaked_tasks = [
            pending_task
            for pending_task in asyncio.all_tasks()
            if pending_task not in before_tasks and not pending_task.done()
        ]
        assert leaked_tasks == []
        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_send_failure_aborts_active_stream() -> None:
    async def run() -> None:
        client_impl = TwoChunkStreamingSpeechClient()
        websocket = RecordingWebSocket(fail_bytes=True)
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_stream_exception_aborts_active_request() -> None:
    async def run() -> None:
        client_impl = InvalidAudioStreamingSpeechClient()
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        await session._generate_sentence("Hello.")

        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert websocket.sent_text[-2]["type"] == "error"
        assert websocket.sent_text[-1]["type"] == "audio.done"
        assert websocket.sent_text[-1]["error"] is True
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_client_disconnect_skips_error_and_done_sends() -> None:
    async def run() -> None:
        client_impl = InvalidAudioStreamingSpeechClient()
        websocket = RecordingWebSocket()
        websocket.client_state = WebSocketState.DISCONNECTED
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        await session._generate_sentence("Hello.")

        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert websocket.sent_text == []
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_completed_send_failure_does_not_abort() -> None:
    async def run() -> None:
        client_impl = CompletedSpeechClient()
        websocket = RecordingWebSocket(fail_bytes=True)
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=False)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert client_impl.aborted == []
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_peer_disconnect_aborts_blocked_speech() -> None:
    async def run() -> None:
        client_impl = BlockingSpeechClient()
        websocket = RecordingWebSocket(
            receive_messages=[{"type": "websocket.disconnect"}]
        )
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=False)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_peer_disconnect_aborts_blocked_stream() -> None:
    async def run() -> None:
        client_impl = BlockingStreamingSpeechClient()
        websocket = RecordingWebSocket(
            receive_messages=[{"type": "websocket.disconnect"}]
        )
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_peer_disconnect_aborts_between_stream_chunks() -> None:
    async def run() -> None:
        client_impl = PausingStreamingSpeechClient()
        websocket = RecordingWebSocket(
            receive_messages=[{"type": "websocket.disconnect"}],
            receive_after_bytes=1,
        )
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert websocket.sent_bytes
        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None

    asyncio.run(run())


def test_speech_websocket_reuses_prepared_session_references() -> None:
    async def run() -> None:
        client_impl = StreamingSpeechClient()
        speech_service = CountingSpeechRequestValidator()
        websocket = RecordingWebSocket()
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=speech_service,
        )

        session.config = await session._parse_config(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
            }
        )
        await session._generate_sentence("First.")
        await session._generate_sentence("Second.")

        assert speech_service.prepare_count == 1
        assert client_impl.generated_prompts == ["First.", "Second."]

    asyncio.run(run())


def test_speech_websocket_disconnect_watch_preserves_client_frames() -> None:
    async def run() -> None:
        client_impl = ReleasableSpeechClient()
        queued_message = {
            "type": "websocket.receive",
            "text": json.dumps({"type": "input.done"}),
        }
        websocket = RecordingWebSocket(receive_messages=[queued_message])
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=False)

        task = asyncio.create_task(session._generate_sentence("Hello."))
        await client_impl.started.wait()
        await asyncio.wait_for(_wait_for_buffered_receive(session), timeout=1.0)
        client_impl.release.set()
        await task

        raw = await session._receive_text_frame(
            timeout_s=0.01,
            max_bytes=MAX_TEXT_MESSAGE_BYTES,
            message_kind="text",
        )

        assert json.loads(raw)["type"] == "input.done"
        assert client_impl.aborted == []

    asyncio.run(run())


def test_speech_websocket_disconnect_watch_aborts_on_buffer_overflow() -> None:
    async def run() -> None:
        client_impl = PausingStreamingSpeechClient()
        websocket = RecordingWebSocket(
            receive_messages=[
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "input.text", "text": str(index)}),
                }
                for index in range(MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION + 1)
            ],
            receive_after_bytes=1,
        )
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert (
            len(session.buffered_receive_messages)
            == MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION
        )
        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None
        assert session.closed is True

    asyncio.run(run())


def test_speech_websocket_disconnect_watch_rejects_oversized_buffered_frame() -> None:
    async def run() -> None:
        client_impl = PausingStreamingSpeechClient()
        websocket = RecordingWebSocket(
            receive_messages=[
                {
                    "type": "websocket.receive",
                    "text": "x" * (MAX_TEXT_MESSAGE_BYTES + 1),
                }
            ],
            receive_after_bytes=1,
        )
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert len(session.buffered_receive_messages) == 0
        assert session.buffered_receive_message_bytes == 0
        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None
        assert session.closed is True

    asyncio.run(run())


def test_speech_websocket_disconnect_watch_aborts_on_buffered_byte_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        first_message = json.dumps({"type": "input.text", "text": "queued"})
        byte_cap = len(first_message) + 1
        monkeypatch.setattr(
            speech_ws_module,
            "MAX_BUFFERED_RECEIVE_BYTES_DURING_GENERATION",
            byte_cap,
        )
        client_impl = PausingStreamingSpeechClient()
        websocket = RecordingWebSocket(
            receive_messages=[
                {"type": "websocket.receive", "text": first_message},
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "input.text", "text": "overflow"}),
                },
            ],
            receive_after_bytes=1,
        )
        session = SpeechWebSocketSession(
            websocket,
            client=client_impl,
            speech_service=SpeechRequestValidator(default_model="tts"),
        )
        session.config = SpeechStreamSessionConfig(stream_audio=True)

        with pytest.raises(WebSocketDisconnect):
            await session._generate_sentence("Hello.")

        assert len(session.buffered_receive_messages) == 1
        assert session.buffered_receive_message_bytes == len(first_message)
        assert session.buffered_receive_message_bytes <= byte_cap
        assert client_impl.aborted == [f"{session.session_id}-0"]
        assert session.active_request_id is None
        assert session.closed is True

    asyncio.run(run())


async def _wait_for_buffered_receive(session: SpeechWebSocketSession) -> None:
    while not session.buffered_receive_messages:
        await asyncio.sleep(0)
