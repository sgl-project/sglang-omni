# SPDX-License-Identifier: Apache-2.0
"""Stateful WebSocket serving for text-to-speech streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from collections.abc import Awaitable
from typing import Any

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
from sglang_omni.serve.speech_errors import SpeechAPIError, bad_request, internal_error
from sglang_omni.serve.speech_service import (
    MAX_REFERENCE_AUDIO_BYTES,
    PreparedSpeechRequest,
    SpeechRequestValidator,
)

logger = logging.getLogger(__name__)

CONFIG_TIMEOUT_S = 10.0
IDLE_TIMEOUT_S = 30.0
BASE64_ENCODED_REFERENCE_AUDIO_BYTES = ((MAX_REFERENCE_AUDIO_BYTES + 2) // 3) * 4
MAX_CONFIG_MESSAGE_BYTES = BASE64_ENCODED_REFERENCE_AUDIO_BYTES + 1024 * 1024
MAX_TEXT_MESSAGE_BYTES = 128 * 1024
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


async def _cancel_tasks(*tasks: asyncio.Task[Any]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
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
        self.active_request_id: str | None = None
        self.buffered_receive_messages: deque[dict[str, Any]] = deque()
        self.buffered_receive_message_bytes = 0
        self.config_prepared_request: PreparedSpeechRequest | None = None

    async def run(self) -> None:
        try:
            configured = await self._receive_config()
            if not configured:
                return
            await self._message_loop()
        finally:
            await self.teardown()

    async def _receive_config(self) -> bool:
        try:
            raw = await self._receive_text_frame(
                timeout_s=CONFIG_TIMEOUT_S,
                max_bytes=MAX_CONFIG_MESSAGE_BYTES,
                message_kind="session",
            )
            payload = self._parse_message(raw)
            if payload.get("type") != "session.config":
                await self._send_error(
                    bad_request(
                        "first WebSocket message must be session.config",
                        param="type",
                    )
                )
                return False
            self.config = await self._parse_config(payload)
            await self._send_json(
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
            await self._send_error(
                bad_request("session.config was not received before timeout")
            )
        except (SpeechAPIError, ValidationError) as exc:
            await self._send_error(_speech_error_from_exception(exc))
        except (json.JSONDecodeError, ValueError) as exc:
            await self._send_error(bad_request(str(exc)))
        except WebSocketDisconnect:
            pass
        return False

    async def _message_loop(self) -> None:
        while not self.closed:
            try:
                raw = await self._receive_text_frame(
                    timeout_s=IDLE_TIMEOUT_S,
                    max_bytes=MAX_TEXT_MESSAGE_BYTES,
                    message_kind="text",
                )
                payload = self._parse_message(raw)
            except asyncio.TimeoutError:
                await self._send_error(bad_request("speech WebSocket idle timeout"))
                return
            except json.JSONDecodeError as exc:
                await self._send_error(bad_request(str(exc)))
                continue
            except ValueError as exc:
                await self._send_error(bad_request(str(exc)))
                continue
            except WebSocketDisconnect:
                return

            message_type = payload.get("type")
            if message_type == "input.text":
                await self._handle_input_text(payload)
            elif message_type == "input.done":
                await self._handle_input_done()
                return
            else:
                await self._send_error(
                    bad_request(
                        f"unsupported speech WebSocket message type: {message_type!r}",
                        param="type",
                    )
                )

    async def _handle_input_text(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if not isinstance(text, str):
            await self._send_error(bad_request("input.text text must be a string"))
            return
        if not text:
            return
        if len(self.buffer) + len(text) > MAX_BUFFERED_TEXT_CHARS:
            self.buffer = ""
            await self._send_error(
                bad_request(
                    f"buffered speech text exceeds {MAX_BUFFERED_TEXT_CHARS} characters",
                    param="text",
                )
            )
            self.closed = True
            return
        self.buffer += text
        for sentence in self._pop_complete_segments():
            await self._generate_sentence(sentence)

    async def _handle_input_done(self) -> None:
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            await self._generate_sentence(remaining)
        await self._send_json(
            {
                "type": "session.done",
                "session_id": self.session_id,
                "total_sentences": self.sentence_index,
            }
        )

    async def _parse_config(
        self,
        payload: dict[str, Any],
    ) -> SpeechStreamSessionConfig:
        raw_config = payload.get("session")
        if raw_config is None:
            raw_config = {key: value for key, value in payload.items() if key != "type"}
        if not isinstance(raw_config, dict):
            raise bad_request(
                "session.config session must be an object",
                param="session",
            )
        self.speech_service.validate_raw_speech_fields(raw_config)
        _validate_raw_session_fields(raw_config)
        config = SpeechStreamSessionConfig.model_validate(raw_config)
        if config.split_granularity not in SUPPORTED_SPLIT_GRANULARITIES:
            supported = ", ".join(sorted(SUPPORTED_SPLIT_GRANULARITIES))
            raise bad_request(
                f"split_granularity must be one of: {supported}",
                param="split_granularity",
            )
        if config.stream_audio and config.response_format.lower() != "pcm":
            raise bad_request(
                "stream_audio=true requires response_format='pcm'",
                param="response_format",
            )
        prepared = await asyncio.to_thread(
            self.speech_service.parse_generation_request,
            self._speech_payload_from_config(config, "probe"),
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

    async def _generate_sentence(self, sentence: str) -> None:
        assert self.config is not None
        sentence_index = self.sentence_index
        self.sentence_index += 1
        request_id = f"{self.session_id}-{sentence_index}"
        self.active_request_id = request_id
        total_bytes = 0
        failed = False
        try:
            if self.config.stream_audio:
                total_bytes = await self._run_generation_until_disconnect(
                    self._stream_sentence_audio(
                        sentence,
                        request_id=request_id,
                        sentence_index=sentence_index,
                    )
                )
            else:
                total_bytes = await self._run_generation_until_disconnect(
                    self._send_sentence_audio(
                        sentence,
                        request_id=request_id,
                        sentence_index=sentence_index,
                    )
                )
        except asyncio.CancelledError:
            failed = True
            await self._abort_request(request_id)
            raise
        except WebSocketDisconnect:
            failed = True
            await self._abort_request(request_id)
            raise
        except Exception as exc:
            failed = True
            await self._abort_request(request_id)
            if isinstance(exc, SpeechAPIError):
                error = exc
            else:
                error = internal_error(str(exc))
                logger.exception("TTS WebSocket sentence failed: %s", request_id)
            await self._send_error(error)
        finally:
            if self.active_request_id == request_id:
                self.active_request_id = None
            await self._send_json(
                {
                    "type": "audio.done",
                    "id": request_id,
                    "sentence_index": sentence_index,
                    "total_bytes": total_bytes,
                    "error": failed,
                }
            )

    async def _stream_sentence_audio(
        self,
        sentence: str,
        *,
        request_id: str,
        sentence_index: int,
    ) -> int:
        assert self.config is not None
        request = self._speech_request_from_config(sentence=sentence, stream=True)
        gen_req = self.speech_service.build_generate_request(
            request,
            validate=False,
            reference_descriptors=self._config_reference_descriptors(),
            uploaded_voice=self._config_uploaded_voice(),
        )
        emitted_samples = 0
        total_bytes = 0
        chunk_count = 0
        started = False
        async for chunk in self.client.generate(gen_req, request_id=request_id):
            if chunk.audio_data is None:
                continue
            sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
            audio_data, emitted_samples = select_audio_delta(
                chunk.audio_data,
                emitted_samples=emitted_samples,
                is_terminal=chunk.finish_reason is not None,
            )
            if audio_data is None:
                continue
            if self.config.speed != 1.0:
                audio_data, sample_rate = apply_speed(
                    audio_data, self.config.speed, sample_rate
                )
            audio_bytes = encode_pcm(audio_data, sample_rate)
            if not audio_bytes:
                continue
            if not started:
                await self._send_audio_start(
                    request_id=request_id,
                    sentence_index=sentence_index,
                    sentence=sentence,
                    sample_rate=sample_rate,
                )
                started = True
            await self._send_audio_frame(audio_bytes, active_request_id=request_id)
            total_bytes += len(audio_bytes)
            chunk_count += 1
        if chunk_count == 0:
            raise ClientError("No audio output generated from the pipeline.")
        return total_bytes

    async def _send_sentence_audio(
        self,
        sentence: str,
        *,
        request_id: str,
        sentence_index: int,
    ) -> int:
        assert self.config is not None
        request = self._speech_request_from_config(sentence=sentence, stream=False)
        gen_req = self.speech_service.build_generate_request(
            request,
            validate=False,
            reference_descriptors=self._config_reference_descriptors(),
            uploaded_voice=self._config_uploaded_voice(),
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
        await self._send_audio_start(
            request_id=request_id,
            sentence_index=sentence_index,
            sentence=sentence,
            sample_rate=result.sample_rate or DEFAULT_SAMPLE_RATE,
        )
        await self._send_audio_frame(result.audio_bytes)
        return len(result.audio_bytes)

    def _speech_request_from_config(
        self,
        config: SpeechStreamSessionConfig | None = None,
        sentence: str = "",
        *,
        stream: bool | None = None,
    ) -> CreateSpeechRequest:
        request = CreateSpeechRequest.model_validate(
            self._speech_payload_from_config(config, sentence, stream=stream)
        )
        self.speech_service.validate_input_text(request.input)
        return request

    def _speech_payload_from_config(
        self,
        config: SpeechStreamSessionConfig | None = None,
        sentence: str = "",
        *,
        stream: bool | None = None,
    ) -> dict[str, Any]:
        config = config or self.config
        assert config is not None
        payload = config.model_dump(
            exclude={"stream_audio", "split_granularity"},
            exclude_none=True,
        )
        payload["input"] = sentence
        payload["stream"] = config.stream_audio if stream is None else stream
        return payload

    def _config_reference_descriptors(self) -> list[dict[str, Any]]:
        if self.config_prepared_request is None:
            return []
        return self.config_prepared_request.reference_descriptors

    def _config_uploaded_voice(self) -> Any:
        if self.config_prepared_request is None:
            return None
        return self.config_prepared_request.uploaded_voice

    def _pop_complete_segments(self) -> list[str]:
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
                start = index + 1
        self.buffer = self.buffer[start:]
        return segments

    def _parse_message(self, raw: str) -> dict[str, Any]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("speech WebSocket messages must be JSON objects")
        return payload

    async def _run_generation_until_disconnect(self, generation: Awaitable[int]) -> int:
        generation_task = asyncio.ensure_future(generation)
        disconnect_task = asyncio.create_task(self._watch_client_disconnect())
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
                await _cancel_tasks(disconnect_task)
                return generation_task.result()

            await _cancel_tasks(generation_task)
            disconnect_task.result()
            raise WebSocketDisconnect
        except asyncio.CancelledError:
            await _cancel_tasks(generation_task, disconnect_task)
            raise
        except Exception:
            await _cancel_tasks(generation_task, disconnect_task)
            raise

    async def _watch_client_disconnect(self) -> None:
        while True:
            message = await self.websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            message_size = self._receive_message_size(message)
            if (
                len(self.buffered_receive_messages)
                >= MAX_BUFFERED_RECEIVE_MESSAGES_DURING_GENERATION
                or message_size > MAX_TEXT_MESSAGE_BYTES
                or self.buffered_receive_message_bytes + message_size
                > MAX_BUFFERED_RECEIVE_BYTES_DURING_GENERATION
            ):
                self.closed = True
                raise WebSocketDisconnect
            self.buffered_receive_messages.append(message)
            self.buffered_receive_message_bytes += message_size

    async def _receive_text_frame(
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
                - self._receive_message_size(message),
            )
        else:
            message = await asyncio.wait_for(
                self.websocket.receive(), timeout=timeout_s
            )
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            raise WebSocketDisconnect
        if message_type != "websocket.receive":
            raise ValueError(
                f"unsupported speech WebSocket ASGI message: {message_type}"
            )

        raw = message.get("text")
        if raw is None:
            frame_bytes = message.get("bytes")
            if frame_bytes is not None and len(frame_bytes) > max_bytes:
                raise ValueError(
                    f"{message_kind} WebSocket message exceeds {max_bytes} bytes"
                )
            raise ValueError("speech WebSocket client messages must be text frames")
        self._validate_message_size(raw, max_bytes, message_kind)
        return raw

    @staticmethod
    def _receive_message_size(message: dict[str, Any]) -> int:
        text = message.get("text")
        if isinstance(text, str):
            return len(text.encode("utf-8"))
        frame_bytes = message.get("bytes")
        if isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            return len(frame_bytes)
        return 0

    @staticmethod
    def _validate_message_size(raw: str, max_bytes: int, message_kind: str) -> None:
        if len(raw.encode("utf-8")) > max_bytes:
            raise ValueError(
                f"{message_kind} WebSocket message exceeds {max_bytes} bytes"
            )

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._can_send():
            return
        await self.websocket.send_text(json.dumps(payload))

    async def _send_error(self, error: SpeechAPIError) -> None:
        payload: dict[str, Any] = {"type": "error", "message": error.message}
        if error.error_type is not None:
            payload["error_type"] = error.error_type
        if error.param is not None:
            payload["param"] = error.param
        if error.code is not None:
            payload["code"] = error.code
        await self._send_json(payload)

    async def _send_audio_start(
        self,
        *,
        request_id: str,
        sentence_index: int,
        sentence: str,
        sample_rate: int,
    ) -> None:
        assert self.config is not None
        await self._send_json(
            {
                "type": "audio.start",
                "id": request_id,
                "sentence_index": sentence_index,
                "sentence_text": sentence,
                "format": self.config.response_format,
                "sample_rate": sample_rate,
            }
        )

    async def _send_audio_frame(
        self, audio_bytes: bytes, *, active_request_id: str | None = None
    ) -> None:
        try:
            await self.websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if active_request_id is not None:
                await self._abort_request(active_request_id)
            raise
        except Exception as exc:
            if active_request_id is not None:
                await self._abort_request(active_request_id)
            raise WebSocketDisconnect from exc

    async def _abort_request(self, request_id: str) -> None:
        if self.active_request_id != request_id:
            return
        self.active_request_id = None
        await self.client.abort(request_id)

    async def _abort_active_request(self) -> None:
        if self.active_request_id is not None:
            await self._abort_request(self.active_request_id)

    def _can_send(self) -> bool:
        return (
            not self.closed
            and self.websocket.application_state == WebSocketState.CONNECTED
            and self.websocket.client_state == WebSocketState.CONNECTED
        )

    async def teardown(self) -> None:
        self.closed = True
        await self._abort_active_request()
        if (
            self.websocket.application_state == WebSocketState.CONNECTED
            and self.websocket.client_state == WebSocketState.CONNECTED
        ):
            await self.websocket.close()


def _speech_error_from_exception(exc: Exception) -> SpeechAPIError:
    if isinstance(exc, SpeechAPIError):
        return exc
    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0] if exc.errors() else {}
        message = first_error.get("msg") or "invalid speech WebSocket config"
        location = ".".join(str(item) for item in first_error.get("loc", ()))
        return bad_request(f"{location}: {message}" if location else str(message))
    return bad_request(str(exc))


def _validate_raw_session_fields(payload: dict[str, Any]) -> None:
    if "stream_audio" in payload and payload["stream_audio"] is not None:
        if not isinstance(payload["stream_audio"], bool):
            raise bad_request(
                "stream_audio must be a boolean",
                param="stream_audio",
            )
    if "split_granularity" in payload and payload["split_granularity"] is not None:
        if not isinstance(payload["split_granularity"], str):
            raise bad_request(
                "split_granularity must be a string",
                param="split_granularity",
            )
