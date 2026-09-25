# SPDX-License-Identifier: Apache-2.0
"""Stateful WebSocket serving for text-to-speech streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Generator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Protocol, TypeVar

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, ClientError
from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    apply_speed,
    encode_pcm,
    select_audio_delta,
)
from sglang_omni.serve.protocol import CreateSpeechRequest, SpeechStreamSessionConfig
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    speech_generation_error,
    speech_websocket_error_payload,
)
from sglang_omni.serve.speech_limits import (
    MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES,
    MAX_SPEECH_WS_TEXT_MESSAGE_BYTES,
    SPEECH_WS_CONFIG_TIMEOUT_S,
)
from sglang_omni.serve.speech_service import (
    PreparedSpeechRequest,
    SpeechReferenceDescriptor,
    SpeechRequestValidator,
)
from sglang_omni.utils.json import JsonValue

if TYPE_CHECKING:
    from sglang_omni.serve.speech_voices import UploadedVoiceReference
else:
    pass

PayloadValue = TypeVar("PayloadValue")

logger = logging.getLogger(__name__)

CONFIG_TIMEOUT_S = SPEECH_WS_CONFIG_TIMEOUT_S
IDLE_TIMEOUT_S = 30.0
MAX_CONFIG_MESSAGE_BYTES = MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES
MAX_TEXT_MESSAGE_BYTES = MAX_SPEECH_WS_TEXT_MESSAGE_BYTES
MAX_BUFFERED_TEXT_CHARS = 256 * 1024
MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION = 16
MAX_BUFFERED_RECEIVE_BYTES_DURING_GENERATION = (
    MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION * MAX_TEXT_MESSAGE_BYTES
)
SENTENCE_BOUNDARIES = frozenset(".!?。！？")
CLAUSE_BOUNDARIES = frozenset(".!?。！？,，;；")
SUPPORTED_SPLIT_GRANULARITIES = frozenset({"sentence", "clause"})


def new_speech_ws_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class CancellableAwaitable(Protocol):
    def done(self) -> bool: ...

    def cancel(self) -> bool: ...

    def __await__(self) -> Generator[object, None, object]: ...


async def cancel_tasks(*tasks: CancellableAwaitable) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
        else:
            pass
    await asyncio.gather(*tasks, return_exceptions=True)


class SpeechWebSocketSession:
    """Own one `/v1/audio/speech/stream` WebSocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        speech_service: SpeechRequestValidator,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.speech_service = speech_service
        self.session_id = new_speech_ws_id("speech_ws")
        self.closed = False
        self.config: SpeechStreamSessionConfig | None = None
        self.buffer = ""
        self.sentence_index = 0
        self.committed_sentence_count = 0
        self.segment_index = 0
        self.active_request_id: str | None = None
        self.buffered_receive_messages: deque[MutableMapping[str, object]] = deque()
        self.buffered_receive_message_bytes = 0
        self.config_prepared_request: PreparedSpeechRequest | None = None

    async def run(self) -> None:
        try:
            configured = await self.receive_config()
            if not configured:
                return
            else:
                pass
            await self.message_loop()
        finally:
            await self.teardown()

    async def receive_config(self) -> bool:
        try:
            raw = await self.receive_text_frame(
                timeout_s=CONFIG_TIMEOUT_S,
                max_bytes=MAX_CONFIG_MESSAGE_BYTES,
                message_kind="session",
            )
            payload = self.parse_message(raw)
            if payload.get("type") != "session.config":
                await self.send_error(
                    bad_request(
                        "first WebSocket message must be session.config",
                        param="type",
                    )
                )
                return False
            else:
                pass
            self.config = await self.parse_config(payload)
            await self.send_json(
                {
                    "type": "session.configured",
                    "session_id": self.session_id,
                    "response_format": self.config.response_format,
                    "stream_audio": self.config.stream_audio,
                    "split_granularity": self.config.split_granularity,
                }
            )
            return True
        except asyncio.TimeoutError:
            await self.send_error(
                bad_request("session.config was not received before timeout")
            )
        except (SpeechAPIError, ValidationError) as exc:
            await self.send_error(speech_error_from_exception(exc))
        except (json.JSONDecodeError, ValueError) as exc:
            await self.send_error(bad_request(str(exc)))
        except WebSocketDisconnect:
            pass
        return False

    async def message_loop(self) -> None:
        while not self.closed:
            try:
                raw = await self.receive_text_frame(
                    timeout_s=IDLE_TIMEOUT_S,
                    max_bytes=MAX_TEXT_MESSAGE_BYTES,
                    message_kind="text",
                )
                payload = self.parse_message(raw)
            except asyncio.TimeoutError:
                await self.send_error(bad_request("speech WebSocket idle timeout"))
                return
            except json.JSONDecodeError as exc:
                await self.send_error(bad_request(str(exc)))
                continue
            except ValueError as exc:
                await self.send_error(bad_request(str(exc)))
                continue
            except WebSocketDisconnect:
                return

            message_type = payload.get("type")
            if message_type == "input.text":
                await self.handle_input_text(payload)
            elif message_type == "input.commit":
                await self.handle_input_commit()
            elif message_type == "input.done":
                await self.handle_input_done()
                return
            else:
                await self.send_error(
                    bad_request(
                        f"unsupported speech WebSocket message type: {message_type!r}",
                        param="type",
                    )
                )

    async def handle_input_text(self, payload: dict[str, PayloadValue]) -> None:
        text = payload.get("text")
        if not isinstance(text, str):
            await self.send_error(bad_request("input.text text must be a string"))
            return
        else:
            pass
        if not text:
            return
        else:
            pass
        if len(self.buffer) + len(text) > MAX_BUFFERED_TEXT_CHARS:
            self.buffer = ""
            await self.send_error(
                bad_request(
                    f"buffered speech text exceeds {MAX_BUFFERED_TEXT_CHARS} characters",
                    param="text",
                )
            )
            self.closed = True
            return
        else:
            pass
        self.buffer += text
        for sentence in self.pop_complete_segments():
            await self.generate_sentence(sentence)

    async def flush_buffer(self) -> None:
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            await self.generate_sentence(remaining)
        else:
            pass

    async def handle_input_commit(self) -> None:
        await self.flush_buffer()
        segment_sentences = self.sentence_index - self.committed_sentence_count
        self.committed_sentence_count = self.sentence_index
        await self.send_json(
            {
                "type": "input.committed",
                "session_id": self.session_id,
                "segment_index": self.segment_index,
                "segment_sentences": segment_sentences,
                "total_sentences": self.sentence_index,
            }
        )
        self.segment_index += 1

    async def handle_input_done(self) -> None:
        await self.flush_buffer()
        await self.send_json(
            {
                "type": "session.done",
                "session_id": self.session_id,
                "total_sentences": self.sentence_index,
            }
        )

    async def parse_config(
        self,
        payload: dict[str, PayloadValue],
    ) -> SpeechStreamSessionConfig:
        raw_config: PayloadValue | dict[str, PayloadValue] | None = payload.get(
            "session"
        )
        if raw_config is None:
            raw_config = {key: value for key, value in payload.items() if key != "type"}
        else:
            pass
        if not isinstance(raw_config, dict):
            raise bad_request(
                "session.config session must be an object",
                param="session",
            )
        else:
            pass
        self.speech_service.validate_raw_speech_fields(raw_config)
        validate_raw_session_fields(raw_config)
        config = SpeechStreamSessionConfig.model_validate(raw_config)
        if config.split_granularity not in SUPPORTED_SPLIT_GRANULARITIES:
            supported = ", ".join(sorted(SUPPORTED_SPLIT_GRANULARITIES))
            raise bad_request(
                f"split_granularity must be one of: {supported}",
                param="split_granularity",
            )
        else:
            pass
        if config.stream_audio and config.response_format.lower() != "pcm":
            raise bad_request(
                "stream_audio=true requires response_format='pcm'",
                param="response_format",
            )
        else:
            pass
        prepared = await asyncio.to_thread(
            self.speech_service.parse_generation_request,
            self.speech_payload_from_config(config, "probe"),
        )
        config_fields = set(SpeechStreamSessionConfig.model_fields)
        prepared_updates = {
            key: value
            for key, value in prepared.request.model_dump().items()
            if key in config_fields
        }
        self.config_prepared_request = prepared
        config = config.model_copy(update=prepared_updates)
        return config

    async def generate_sentence(self, sentence: str) -> None:
        assert self.config is not None
        sentence_index = self.sentence_index
        self.sentence_index += 1
        request_id = f"{self.session_id}-{sentence_index}"
        self.active_request_id = request_id
        total_bytes = 0
        failed = False
        try:
            if self.config.stream_audio:
                total_bytes = await self.run_generation_until_disconnect(
                    self.stream_sentence_audio(
                        sentence,
                        request_id=request_id,
                        sentence_index=sentence_index,
                    )
                )
            else:
                total_bytes = await self.run_generation_until_disconnect(
                    self.send_sentence_audio(
                        sentence,
                        request_id=request_id,
                        sentence_index=sentence_index,
                    )
                )
        except asyncio.CancelledError:
            failed = True
            await self.abort_request(request_id)
            raise
        except WebSocketDisconnect:
            failed = True
            await self.abort_request(request_id)
            raise
        except Exception as exc:
            failed = True
            await self.abort_request(request_id)
            error = speech_generation_error(exc)
            if error.status_code == 500:
                logger.exception("TTS WebSocket sentence failed: %s", request_id)
            else:
                logger.warning(
                    "Rejecting TTS WebSocket sentence %s: %s",
                    request_id,
                    error.message,
                )
            await self.send_error(error)
        finally:
            if self.active_request_id == request_id:
                self.active_request_id = None
            else:
                pass
            await self.send_json(
                {
                    "type": "audio.done",
                    "id": request_id,
                    "sentence_index": sentence_index,
                    "total_bytes": total_bytes,
                    "error": failed,
                }
            )

    async def stream_sentence_audio(
        self,
        sentence: str,
        *,
        request_id: str,
        sentence_index: int,
    ) -> int:
        assert self.config is not None
        request = self.speech_request_from_config(sentence=sentence, stream=True)
        gen_req = self.speech_service.build_generate_request(
            request,
            validate=False,
            reference_descriptors=self.config_reference_descriptors(),
            uploaded_voice=self.config_uploaded_voice(),
        )
        emitted_samples = 0
        total_bytes = 0
        chunk_count = 0
        started = False
        async for chunk in self.client.generate(gen_req, request_id=request_id):
            if chunk.audio_data is None:
                continue
            else:
                pass
            sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
            audio_data, emitted_samples = select_audio_delta(
                chunk.audio_data,
                emitted_samples=emitted_samples,
                is_terminal=chunk.finish_reason is not None,
            )
            if audio_data is None:
                continue
            else:
                pass
            if self.config.speed != 1.0:
                audio_data, sample_rate = apply_speed(
                    audio_data, self.config.speed, sample_rate
                )
            else:
                pass
            audio_bytes = encode_pcm(audio_data, sample_rate)
            if not audio_bytes:
                continue
            else:
                pass
            if not started:
                await self.send_audio_start(
                    request_id=request_id,
                    sentence_index=sentence_index,
                    sentence=sentence,
                    sample_rate=sample_rate,
                )
                started = True
            else:
                pass
            await self.send_audio_frame(audio_bytes, active_request_id=request_id)
            total_bytes += len(audio_bytes)
            chunk_count += 1
        if chunk_count == 0:
            raise ClientError("No audio output generated from the pipeline.")
        else:
            pass
        return total_bytes

    async def send_sentence_audio(
        self,
        sentence: str,
        *,
        request_id: str,
        sentence_index: int,
    ) -> int:
        assert self.config is not None
        request = self.speech_request_from_config(sentence=sentence, stream=False)
        gen_req = self.speech_service.build_generate_request(
            request,
            validate=False,
            reference_descriptors=self.config_reference_descriptors(),
            uploaded_voice=self.config_uploaded_voice(),
        )
        result = await self.client.speech(
            gen_req,
            request_id=request_id,
            response_format=request.response_format,
            speed=request.speed,
            allow_format_fallback=False,
        )
        if self.active_request_id == request_id:
            self.active_request_id = None
        else:
            pass
        await self.send_audio_start(
            request_id=request_id,
            sentence_index=sentence_index,
            sentence=sentence,
            sample_rate=result.sample_rate or DEFAULT_SAMPLE_RATE,
        )
        await self.send_audio_frame(result.audio_bytes)
        return len(result.audio_bytes)

    def speech_request_from_config(
        self,
        config: SpeechStreamSessionConfig | None = None,
        sentence: str = "",
        *,
        stream: bool | None = None,
    ) -> CreateSpeechRequest:
        request = CreateSpeechRequest.model_validate(
            self.speech_payload_from_config(config, sentence, stream=stream)
        )
        self.speech_service.validate_input_text(request.input)
        return request

    def speech_payload_from_config(
        self,
        config: SpeechStreamSessionConfig | None = None,
        sentence: str = "",
        *,
        stream: bool | None = None,
    ) -> dict[str, object]:
        config = config or self.config
        assert config is not None
        payload = config.model_dump(
            exclude={"stream_audio", "split_granularity"},
            exclude_none=True,
        )
        payload["input"] = sentence
        payload["stream"] = config.stream_audio if stream is None else stream
        return payload

    def config_reference_descriptors(self) -> list[SpeechReferenceDescriptor]:
        if self.config_prepared_request is None:
            return []
        else:
            pass
        return self.config_prepared_request.reference_descriptors

    def config_uploaded_voice(self) -> UploadedVoiceReference | None:
        if self.config_prepared_request is None:
            return None
        else:
            pass
        return self.config_prepared_request.uploaded_voice

    def pop_complete_segments(self) -> list[str]:
        assert self.config is not None
        boundaries = (
            CLAUSE_BOUNDARIES
            if self.config.split_granularity == "clause"
            else SENTENCE_BOUNDARIES
        )
        segments: list[str] = []
        start = 0
        for index, char in enumerate(self.buffer):
            if char in boundaries:
                segment = self.buffer[start : index + 1].strip()
                if segment:
                    segments.append(segment)
                else:
                    pass
                start = index + 1
            else:
                pass
        self.buffer = self.buffer[start:]
        return segments

    def parse_message(self, raw: str) -> dict[str, JsonValue]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("speech WebSocket messages must be JSON objects")
        else:
            pass
        return payload

    async def run_generation_until_disconnect(self, generation: Awaitable[int]) -> int:
        generation_task = asyncio.ensure_future(generation)
        disconnect_task = asyncio.create_task(self.watch_client_disconnect())
        try:
            done, _ = await asyncio.wait(
                {generation_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if generation_task in done:
                # control frames can arrive while generation owns the receive loop
                await asyncio.sleep(0)
                if disconnect_task.done():
                    disconnect_task.result()
                else:
                    pass
                await cancel_tasks(disconnect_task)
                return generation_task.result()
            else:
                pass

            await cancel_tasks(generation_task)
            disconnect_task.result()
            raise WebSocketDisconnect
        except asyncio.CancelledError:
            await cancel_tasks(generation_task, disconnect_task)
            raise
        except Exception:
            await cancel_tasks(generation_task, disconnect_task)
            raise

    async def watch_client_disconnect(self) -> None:
        while True:
            message = await self.websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            else:
                pass
            message_size = self.receive_message_size(message)
            if (
                len(self.buffered_receive_messages)
                >= MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION
                or message_size > MAX_TEXT_MESSAGE_BYTES
                or self.buffered_receive_message_bytes + message_size
                > MAX_BUFFERED_RECEIVE_BYTES_DURING_GENERATION
            ):
                self.closed = True
                raise WebSocketDisconnect
            else:
                pass
            self.buffered_receive_messages.append(message)
            self.buffered_receive_message_bytes += message_size

    async def receive_text_frame(
        self,
        *,
        timeout_s: float,
        max_bytes: int,
        message_kind: str,
    ) -> str:
        if self.buffered_receive_messages:
            message = self.buffered_receive_messages.popleft()
            self.buffered_receive_message_bytes = max(
                0,
                self.buffered_receive_message_bytes
                - self.receive_message_size(message),
            )
        else:
            message = await asyncio.wait_for(
                self.websocket.receive(), timeout=timeout_s
            )
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            raise WebSocketDisconnect
        else:
            pass
        if message_type != "websocket.receive":
            raise ValueError(
                f"unsupported speech WebSocket ASGI message: {message_type}"
            )
        else:
            pass

        raw = message.get("text")
        if raw is None:
            frame_bytes = message.get("bytes")
            if frame_bytes is not None and len(frame_bytes) > max_bytes:
                raise ValueError(
                    f"{message_kind} WebSocket message exceeds {max_bytes} bytes"
                )
            else:
                pass
            raise ValueError("speech WebSocket client messages must be text frames")
        else:
            pass
        self.validate_message_size(raw, max_bytes, message_kind)
        return raw

    @staticmethod
    def receive_message_size(message: Mapping[str, PayloadValue]) -> int:
        text = message.get("text")
        if isinstance(text, str):
            return len(text.encode("utf-8"))
        else:
            pass
        frame_bytes = message.get("bytes")
        if isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            return len(frame_bytes)
        else:
            pass
        return 0

    @staticmethod
    def validate_message_size(raw: str, max_bytes: int, message_kind: str) -> None:
        if len(raw.encode("utf-8")) > max_bytes:
            raise ValueError(
                f"{message_kind} WebSocket message exceeds {max_bytes} bytes"
            )
        else:
            pass

    async def send_json(self, payload: dict[str, PayloadValue]) -> None:
        if not self.can_send():
            return
        else:
            pass
        await self.websocket.send_text(json.dumps(payload))

    async def send_error(self, error: SpeechAPIError) -> None:
        await self.send_json(speech_websocket_error_payload(error))

    async def send_audio_start(
        self,
        *,
        request_id: str,
        sentence_index: int,
        sentence: str,
        sample_rate: int,
    ) -> None:
        assert self.config is not None
        await self.send_json(
            {
                "type": "audio.start",
                "id": request_id,
                "sentence_index": sentence_index,
                "sentence_text": sentence,
                "format": self.config.response_format,
                "sample_rate": sample_rate,
            }
        )

    async def send_audio_frame(
        self, audio_bytes: bytes, *, active_request_id: str | None = None
    ) -> None:
        try:
            await self.websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if active_request_id is not None:
                await self.abort_request(active_request_id)
            else:
                pass
            raise
        except Exception as exc:
            if active_request_id is not None:
                await self.abort_request(active_request_id)
            else:
                pass
            raise WebSocketDisconnect from exc

    async def abort_request(self, request_id: str) -> None:
        if self.active_request_id != request_id:
            return
        else:
            pass
        self.active_request_id = None
        await self.client.abort(request_id)

    async def abort_active_request(self) -> None:
        if self.active_request_id is not None:
            await self.abort_request(self.active_request_id)
        else:
            pass

    def can_send(self) -> bool:
        return (
            not self.closed
            and self.websocket.application_state == WebSocketState.CONNECTED
            and self.websocket.client_state == WebSocketState.CONNECTED
        )

    async def teardown(self) -> None:
        self.closed = True
        await self.abort_active_request()
        if (
            self.websocket.application_state == WebSocketState.CONNECTED
            and self.websocket.client_state == WebSocketState.CONNECTED
        ):
            await self.websocket.close()
        else:
            pass


def speech_error_from_exception(exc: Exception) -> SpeechAPIError:
    if isinstance(exc, SpeechAPIError):
        return exc
    else:
        pass
    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0] if exc.errors() else {}
        message = first_error.get("msg") or "invalid speech WebSocket config"
        location = ".".join(str(item) for item in first_error.get("loc", ()))
        return bad_request(f"{location}: {message}" if location else str(message))
    else:
        pass
    return bad_request(str(exc))


def validate_raw_session_fields(payload: dict[str, PayloadValue]) -> None:
    if "stream_audio" in payload and payload["stream_audio"] is not None:
        if not isinstance(payload["stream_audio"], bool):
            raise bad_request(
                "stream_audio must be a boolean",
                param="stream_audio",
            )
        else:
            pass
    else:
        pass
    if "split_granularity" in payload and payload["split_granularity"] is not None:
        if not isinstance(payload["split_granularity"], str):
            raise bad_request(
                "split_granularity must be a string",
                param="split_granularity",
            )
        else:
            pass
    else:
        pass
