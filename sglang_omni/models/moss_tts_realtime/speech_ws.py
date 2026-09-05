# SPDX-License-Identifier: Apache-2.0
"""Model-level implementation of the realtime speech WebSocket endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, ClientError, GenerateRequest, SamplingParams
from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE, encode_pcm, select_audio_delta
from sglang_omni.client.realtime import RealtimeHandle, open_realtime
from sglang_omni.models.moss_tts_realtime.config import MossTTSRealtimeResourceLimits
from sglang_omni.models.moss_tts_realtime.observability import (
    emit_realtime_event as _emit_event,
)
from sglang_omni.models.moss_tts_realtime.observability import realtime_events_active
from sglang_omni.models.moss_tts_realtime.protocol import (
    MossTTSRealtimeClientEvent,
    MossTTSRealtimeInputDone,
    MossTTSRealtimeInputEvent,
    MossTTSRealtimeInputText,
    MossTTSRealtimeInputTokens,
    MossTTSRealtimeSessionClose,
    MossTTSRealtimeSpeechSessionConfig,
    MossTTSRealtimeTurnCancel,
    MossTTSRealtimeTurnStart,
    moss_tts_realtime_event_fingerprint,
    parse_moss_tts_realtime_client_event,
    speech_websocket_session_config_payload,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeInputUpdate,
)
from sglang_omni.models.moss_tts_realtime.text_delta import (
    MossTTSRealtimeTextDeltaSnapshot,
    MossTTSRealtimeTextDeltaTokenizer,
    initialize_moss_tts_realtime_tokenizer_vocab_size,
    validate_moss_tts_realtime_text_token_ids,
)
from sglang_omni.proto import InputUpdateMessage
from sglang_omni.serve.speech_errors import SpeechAPIError, bad_request, internal_error
from sglang_omni.serve.speech_service import (
    MAX_REFERENCE_AUDIO_BYTES,
    PreparedSpeechRequest,
    SpeechRequestValidator,
)

logger = logging.getLogger(__name__)

CONFIG_TIMEOUT_S = 10.0
BASE64_ENCODED_REFERENCE_AUDIO_BYTES = ((MAX_REFERENCE_AUDIO_BYTES + 2) // 3) * 4
MAX_CONFIG_MESSAGE_BYTES = BASE64_ENCODED_REFERENCE_AUDIO_BYTES + 1024 * 1024
MOSS_TTS_REALTIME_INPUT_MODES = ("text", "tokens")


@dataclass(slots=True)
class _ActiveRealtimeTurn:
    turn_id: str
    request_id: str
    turn_index: int
    delta_tokenizer: MossTTSRealtimeTextDeltaTokenizer
    realtime_handle: RealtimeHandle | None = None
    generation_task: asyncio.Task[None] | None = None
    client_started: asyncio.Event = field(default_factory=asyncio.Event)
    input_mode: Literal["text", "tokens"] | None = None
    next_seq_no: int = 0
    input_done: bool = False
    received_token_count: int = 0
    accepted_fingerprints: dict[int, str] = field(default_factory=dict)
    cancel_requested: bool = False
    audio_started: bool = False
    audio_chunks: int = 0
    audio_bytes: int = 0


class MossTTSRealtimeSpeechWebSocketSession:
    """Own one ``/v1/audio/speech/realtime`` WebSocket session."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        speech_service: SpeechRequestValidator,
        session_id: str,
        tokenizer: Any,
        limits: MossTTSRealtimeResourceLimits | None = None,
        realtime_input_stage: str | None = None,
        idle_timeout_s: float = 30.0,
        max_message_bytes: int = 128 * 1024,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.speech_service = speech_service
        self.session_id = session_id
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must implement encode()")
        self.tokenizer = tokenizer
        initialize_moss_tts_realtime_tokenizer_vocab_size(tokenizer)
        self.limits = limits or MossTTSRealtimeResourceLimits()
        self.realtime_input_stage = realtime_input_stage or "tts_engine"
        self.idle_timeout_s = idle_timeout_s
        self.max_message_bytes = max_message_bytes
        self.config: MossTTSRealtimeSpeechSessionConfig | None = None
        self.config_prepared_request: PreparedSpeechRequest | None = None
        self.active_turn: _ActiveRealtimeTurn | None = None
        self.successful_turns = 0
        self.used_turn_ids: set[str] = set()
        self.closed = False
        self.disconnected = False
        self._send_lock = asyncio.Lock()
        self._teardown_complete = False
        self._backend_session_closed = False

    def _emit_turn_event(
        self,
        turn: _ActiveRealtimeTurn,
        event_name: str,
        **metadata: Any,
    ) -> None:
        _emit_event(
            request_id=turn.request_id,
            stage="coordinator",
            event_name=event_name,
            metadata={
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "turn_index": turn.turn_index,
                **metadata,
            },
        )

    @staticmethod
    def _audio_sample_count(audio_data: Any) -> int | None:
        shape = getattr(audio_data, "shape", None)
        if shape is not None:
            try:
                return int(shape[-1])
            except (IndexError, TypeError, ValueError):
                return None
        try:
            return len(audio_data)
        except TypeError:
            return None

    async def run(self) -> None:
        """Read configuration and drive one realtime speech session."""

        try:
            raw = await self._receive_text_frame(
                timeout_s=CONFIG_TIMEOUT_S,
                max_message_bytes=MAX_CONFIG_MESSAGE_BYTES,
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
                return
            await self._configure(payload)
            await self._message_loop()
        except asyncio.TimeoutError:
            await self._send_error(
                bad_request("session.config was not received before timeout")
            )
        except (json.JSONDecodeError, ValueError) as exc:
            await self._send_error(bad_request(str(exc)))
        except WebSocketDisconnect:
            self.disconnected = True
        finally:
            await self.teardown()

    async def _configure(self, payload: dict[str, Any]) -> None:
        try:
            raw_config = speech_websocket_session_config_payload(payload)
            config = MossTTSRealtimeSpeechSessionConfig.model_validate(raw_config)
            prepared = await asyncio.to_thread(
                self.speech_service.parse_generation_request,
                self._speech_probe_payload(config),
            )
            prepared_fields = prepared.request.model_dump()
            config_updates = {
                name: prepared_fields[name]
                for name in MossTTSRealtimeSpeechSessionConfig.model_fields
                if name in prepared_fields and prepared_fields[name] is not None
            }
            self.config = config.model_copy(update=config_updates)
            self.config_prepared_request = prepared
        except (SpeechAPIError, ValidationError) as exc:
            await self._send_error(_realtime_error_from_exception(exc))
            raise WebSocketDisconnect from exc
        except (TypeError, ValueError) as exc:
            await self._send_error(bad_request(str(exc)))
            raise WebSocketDisconnect from exc
        except Exception as exc:
            logger.exception("Failed to configure MOSS-TTS-Realtime WebSocket")
            await self._send_error(internal_error(str(exc)))
            raise WebSocketDisconnect from exc

        assert self.config is not None
        await self._send_json(
            {
                "type": "session.configured",
                "session_id": self.session_id,
                "response_format": self.config.response_format,
                "sample_rate": self.config.sample_rate,
                "stream_audio": self.config.stream_audio,
                "input_modes": list(MOSS_TTS_REALTIME_INPUT_MODES),
                "max_active_turns": 1,
            }
        )

    async def _message_loop(self) -> None:
        while not self.closed:
            try:
                raw = await self._receive_text_frame()
                payload = self._parse_message(raw)
                event = parse_moss_tts_realtime_client_event(payload)
                if event is None:
                    await self._send_error(
                        bad_request(
                            "unsupported MOSS-TTS-Realtime WebSocket message type: "
                            f"{payload.get('type')!r}",
                            param="type",
                        )
                    )
                    continue
                await self._dispatch(event)
            except asyncio.TimeoutError:
                await self._send_error(
                    bad_request("MOSS-TTS-Realtime WebSocket idle timeout")
                )
                return
            except ValidationError as exc:
                await self._send_error(_realtime_error_from_exception(exc))
            except json.JSONDecodeError as exc:
                await self._send_error(bad_request(str(exc)))
            except ValueError as exc:
                await self._send_error(bad_request(str(exc)))
            except WebSocketDisconnect:
                raise

    async def _dispatch(self, event: MossTTSRealtimeClientEvent) -> None:
        if isinstance(event, MossTTSRealtimeTurnStart):
            await self._handle_turn_start(event)
        elif isinstance(
            event,
            (
                MossTTSRealtimeInputText,
                MossTTSRealtimeInputTokens,
                MossTTSRealtimeInputDone,
            ),
        ):
            await self._handle_input(event)
        elif isinstance(event, MossTTSRealtimeTurnCancel):
            await self._handle_turn_cancel(event)
        elif isinstance(event, MossTTSRealtimeSessionClose):
            await self._handle_session_close()
        else:  # pragma: no cover - parser and dispatch table are kept together.
            raise AssertionError(f"unhandled realtime event {type(event).__name__}")

    async def _handle_turn_start(self, event: MossTTSRealtimeTurnStart) -> None:
        if self.active_turn is not None:
            await self._send_error(
                bad_request(
                    f"session already has active turn {self.active_turn.turn_id!r}",
                    param="turn_id",
                )
            )
            return
        if event.turn_id in self.used_turn_ids:
            await self._send_error(
                bad_request("turn_id must be unique within a session", param="turn_id")
            )
            return
        request_id = f"{self.session_id}:{event.turn_id}"
        turn = _ActiveRealtimeTurn(
            turn_id=event.turn_id,
            request_id=request_id,
            turn_index=self.successful_turns,
            delta_tokenizer=MossTTSRealtimeTextDeltaTokenizer(
                self.tokenizer,
                max_text_bytes=self.limits.max_pending_text_bytes,
                max_token_ids=self.limits.max_pending_text_tokens,
            ),
        )
        self.active_turn = turn
        self.used_turn_ids.add(event.turn_id)
        submitted = asyncio.get_running_loop().create_future()
        turn.generation_task = asyncio.create_task(
            self._run_turn_generation(turn, event, submitted),
            name=f"moss-tts-realtime:{request_id}",
        )
        try:
            await submitted
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.gather(turn.generation_task, return_exceptions=True)
            return
        if self.active_turn is not turn or turn.cancel_requested:
            return
        await self._send_json(
            {
                "type": "turn.started",
                "session_id": self.session_id,
                "turn_id": event.turn_id,
                "request_id": request_id,
                "next_seq_no": 0,
            }
        )
        turn.client_started.set()

    async def _handle_input(self, event: MossTTSRealtimeInputEvent) -> None:
        turn = await self._active_turn_for(event.turn_id)
        if turn is None:
            return

        observe_events = realtime_events_active()
        if observe_events:
            supplied_text_bytes = (
                len(event.text.encode("utf-8"))
                if isinstance(event, MossTTSRealtimeInputText)
                else 0
            )
            supplied_token_count = (
                len(event.token_ids)
                if isinstance(event, MossTTSRealtimeInputTokens)
                else 0
            )
            self._emit_turn_event(
                turn,
                "ws_input_received",
                seq_no=event.seq_no,
                input_type=event.type,
                input_done=isinstance(event, MossTTSRealtimeInputDone),
                supplied_text_bytes=supplied_text_bytes,
                supplied_token_count=supplied_token_count,
                stable_token_count=turn.received_token_count,
            )

        fingerprint = moss_tts_realtime_event_fingerprint(event)
        previous = turn.accepted_fingerprints.get(event.seq_no)
        if previous is not None:
            if previous != fingerprint:
                await self._send_error(
                    bad_request(
                        f"input seq_no {event.seq_no} was retried with different content",
                        param="seq_no",
                    )
                )
                return
            await self._send_input_ack(turn, event.seq_no)
            return
        if event.seq_no != turn.next_seq_no:
            await self._send_error(
                bad_request(
                    f"input sequence gap: expected {turn.next_seq_no}, got {event.seq_no}",
                    param="seq_no",
                )
            )
            return
        if turn.input_done:
            await self._send_error(
                bad_request("cannot append input after input.done", param="type")
            )
            return
        if len(turn.accepted_fingerprints) >= self.limits.max_input_updates:
            await self._send_error(
                bad_request(
                    "realtime input update limit exceeded",
                    param="seq_no",
                )
            )
            return

        snapshot: MossTTSRealtimeTextDeltaSnapshot | None = None
        new_mode = turn.input_mode
        input_done = isinstance(event, MossTTSRealtimeInputDone)
        try:
            if isinstance(event, MossTTSRealtimeInputText):
                if turn.input_mode == "tokens":
                    raise bad_request(
                        "one turn cannot mix input.text and input.tokens",
                        param="type",
                    )
                new_mode = "text"
                snapshot = turn.delta_tokenizer.snapshot()
                if observe_events:
                    self._emit_turn_event(
                        turn,
                        "text_tokenize_start",
                        seq_no=event.seq_no,
                        operation="push_delta",
                        stable_token_count=len(turn.delta_tokenizer.emitted_token_ids),
                    )
                delta = turn.delta_tokenizer.push_delta(event.text)
                if observe_events:
                    self._emit_turn_event(
                        turn,
                        "text_tokenize_end",
                        seq_no=event.seq_no,
                        operation="push_delta",
                        new_stable_token_count=len(delta.token_ids),
                        stable_token_count=len(turn.delta_tokenizer.emitted_token_ids),
                        tokenizer_token_count=len(turn.delta_tokenizer.token_ids),
                        tokenizer_text_bytes=turn.delta_tokenizer.total_text_bytes,
                    )
                token_ids = delta.token_ids
                byte_count = delta.byte_count
            elif isinstance(event, MossTTSRealtimeInputTokens):
                if turn.input_mode == "text":
                    raise bad_request(
                        "one turn cannot mix input.text and input.tokens",
                        param="type",
                    )
                new_mode = "tokens"
                token_ids = validate_moss_tts_realtime_text_token_ids(
                    event.token_ids,
                )
                if len(token_ids) > self.limits.max_pending_text_tokens:
                    raise bad_request(
                        "input.tokens exceeds the pending token limit",
                        param="token_ids",
                    )
                byte_count = 0
            else:
                if turn.input_mode == "text":
                    snapshot = turn.delta_tokenizer.snapshot()
                    if observe_events:
                        self._emit_turn_event(
                            turn,
                            "text_tokenize_start",
                            seq_no=event.seq_no,
                            operation="flush",
                            stable_token_count=len(
                                turn.delta_tokenizer.emitted_token_ids
                            ),
                        )
                    delta = turn.delta_tokenizer.flush()
                    if observe_events:
                        self._emit_turn_event(
                            turn,
                            "text_tokenize_end",
                            seq_no=event.seq_no,
                            operation="flush",
                            new_stable_token_count=len(delta.token_ids),
                            stable_token_count=len(
                                turn.delta_tokenizer.emitted_token_ids
                            ),
                            tokenizer_token_count=len(turn.delta_tokenizer.token_ids),
                            tokenizer_text_bytes=turn.delta_tokenizer.total_text_bytes,
                        )
                    token_ids = delta.token_ids
                    byte_count = delta.byte_count
                else:
                    token_ids = ()
                    byte_count = 0
                if turn.received_token_count + len(token_ids) == 0:
                    raise bad_request(
                        "input.done cannot close an empty realtime turn",
                        param="type",
                    )

            update = MossTTSRealtimeInputUpdate(
                seq_no=event.seq_no,
                token_ids=token_ids,
                byte_count=byte_count,
                input_done=input_done,
            )
            await self._submit_input_update(turn, update)
        except SpeechAPIError as exc:
            if snapshot is not None:
                turn.delta_tokenizer.restore(snapshot)
            await self._send_error(exc)
            return
        except (TypeError, ValueError, RuntimeError, ClientError) as exc:
            if snapshot is not None:
                turn.delta_tokenizer.restore(snapshot)
            await self._send_error(bad_request(str(exc)))
            return
        except Exception as exc:
            if snapshot is not None:
                turn.delta_tokenizer.restore(snapshot)
            logger.exception("MOSS-TTS-Realtime input admission failed")
            await self._send_error(internal_error(str(exc)))
            return

        turn.input_mode = new_mode
        turn.input_done = input_done
        turn.received_token_count += len(token_ids)
        turn.accepted_fingerprints[event.seq_no] = fingerprint
        turn.next_seq_no += 1
        await self._send_input_ack(turn, event.seq_no)

    async def _handle_turn_cancel(self, event: MossTTSRealtimeTurnCancel) -> None:
        turn = await self._active_turn_for(event.turn_id)
        if turn is None:
            return
        await self._terminate_turn(turn, reason="client_cancelled", send_events=True)

    async def _handle_session_close(self) -> None:
        if self.active_turn is not None:
            await self._terminate_turn(
                self.active_turn,
                reason="session_closed",
                send_events=True,
            )
        await self._close_backend_session()
        await self._send_json(
            {
                "type": "session.closed",
                "session_id": self.session_id,
            }
        )
        self.closed = True

    async def _submit_input_update(
        self,
        turn: _ActiveRealtimeTurn,
        update: MossTTSRealtimeInputUpdate,
    ) -> None:
        handle = turn.realtime_handle
        if handle is None:
            raise ClientError("realtime turn is not open")
        await handle.send_input(
            InputUpdateMessage(
                request_id=turn.request_id,
                session_id=self.session_id,
                turn_id=turn.turn_id,
                seq_no=update.seq_no,
                token_ids=update.token_ids,
                byte_count=update.byte_count,
                input_done=update.input_done,
            )
        )

    async def _run_turn_generation(
        self,
        turn: _ActiveRealtimeTurn,
        event: MossTTSRealtimeTurnStart,
        submitted: asyncio.Future[None],
    ) -> None:
        emitted_samples = 0
        finish_reason: str | None = None
        try:
            request = self._build_turn_request(turn, event)
            handle = await open_realtime(
                self.client,
                request,
                request_id=turn.request_id,
                session_id=self.session_id,
                turn_id=turn.turn_id,
                input_stage=self.realtime_input_stage,
            )
            turn.realtime_handle = handle
            if not submitted.done():
                submitted.set_result(None)
            async for chunk in handle:
                await turn.client_started.wait()
                if turn.cancel_requested:
                    return
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                if chunk.audio_data is None:
                    continue
                sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
                if sample_rate != self.config.sample_rate:
                    raise ClientError(
                        "MOSS-TTS-Realtime emitted unexpected sample rate "
                        f"{sample_rate}"
                    )
                audio_data, emitted_samples = select_audio_delta(
                    chunk.audio_data,
                    emitted_samples=emitted_samples,
                    is_terminal=chunk.finish_reason is not None,
                )
                if audio_data is None:
                    continue
                sample_count = None
                capture_first_pcm = turn.audio_chunks == 0 and realtime_events_active()
                if capture_first_pcm:
                    sample_count = self._audio_sample_count(audio_data)
                    capture_first_pcm = sample_count != 0
                pcm_metadata: dict[str, Any] | None = None
                if capture_first_pcm:
                    pcm_metadata = {
                        "seq_no": turn.next_seq_no - 1 if turn.next_seq_no else None,
                        "stable_token_count": turn.received_token_count,
                        "sample_rate": sample_rate,
                        "sample_count": sample_count,
                    }
                    self._emit_turn_event(
                        turn,
                        "pcm_encode_start",
                        **pcm_metadata,
                    )
                audio_bytes = encode_pcm(audio_data, sample_rate)
                if not audio_bytes:
                    continue
                if capture_first_pcm:
                    assert pcm_metadata is not None
                    pcm_metadata["pcm_bytes"] = len(audio_bytes)
                    self._emit_turn_event(
                        turn,
                        "pcm_host_ready",
                        **pcm_metadata,
                    )
                if not turn.audio_started:
                    await self._send_audio_start(turn)
                    turn.audio_started = True
                if capture_first_pcm:
                    self._emit_turn_event(
                        turn,
                        "pcm_send_begin",
                        audio_start_sent=turn.audio_started,
                        **pcm_metadata,
                    )
                await self._send_bytes(audio_bytes)
                if capture_first_pcm:
                    self._emit_turn_event(
                        turn,
                        "pcm_send_end",
                        audio_start_sent=turn.audio_started,
                        **pcm_metadata,
                    )
                turn.audio_chunks += 1
                turn.audio_bytes += len(audio_bytes)

            if turn.cancel_requested:
                return
            if turn.audio_chunks == 0:
                raise ClientError("No audio output generated from the pipeline.")
            await self._send_turn_terminal(
                turn,
                committed=True,
                reason=finish_reason or "stop",
                error=False,
            )
            if self.active_turn is turn:
                self.active_turn = None
                self.successful_turns += 1
        except asyncio.CancelledError:
            if not submitted.done():
                submitted.cancel()
            raise
        except WebSocketDisconnect as exc:
            if not submitted.done():
                submitted.set_exception(exc)
            self.disconnected = True
            turn.cancel_requested = True
            await self._abort_turn(turn)
        except Exception as exc:
            if not submitted.done():
                submitted.set_exception(exc)
            if turn.cancel_requested:
                return
            await self._abort_turn(turn)
            if isinstance(exc, SpeechAPIError):
                error = exc
            elif isinstance(exc, ClientError):
                error = bad_request(str(exc))
            else:
                logger.exception("MOSS-TTS-Realtime turn failed: %s", turn.request_id)
                error = internal_error(str(exc))
            await self._send_error(error)
            await self._send_turn_terminal(
                turn,
                committed=False,
                reason="error",
                error=True,
            )
            if self.active_turn is turn:
                self.active_turn = None

    def _build_turn_request(
        self,
        turn: _ActiveRealtimeTurn,
        event: MossTTSRealtimeTurnStart,
    ) -> GenerateRequest:
        assert self.config is not None
        assert self.config_prepared_request is not None
        prepared = self.config_prepared_request
        prompt: dict[str, Any] = {
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "turn_index": turn.turn_index,
            "voice": self.config.voice,
            "ref_audio": prepared.request.ref_audio,
            "ref_text": prepared.request.ref_text,
            "language": self.config.language,
            "instructions": self.config.instructions,
            "initial_token_ids": [],
            "input_done": False,
        }
        if prepared.reference_descriptors:
            prompt["references"] = prepared.reference_descriptors
        if event.user is not None:
            prompt["user"] = event.user.model_dump(exclude_none=True)

        return GenerateRequest(
            model=self.config.model or self.speech_service.default_model,
            prompt=prompt,
            sampling=SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                seed=self.config.seed,
                max_new_tokens=self.config.max_new_tokens,
            ),
            stage_params=self.config.stage_params,
            stream=True,
            output_modalities=["audio"],
            metadata={
                "task": "tts",
                "session_id": self.session_id,
            },
        )

    async def _active_turn_for(self, turn_id: str) -> _ActiveRealtimeTurn | None:
        turn = self.active_turn
        if turn is None:
            await self._send_error(
                bad_request("turn.start is required before this message", param="type")
            )
            return None
        if turn.turn_id != turn_id:
            await self._send_error(
                bad_request(
                    f"message targets turn {turn_id!r}, active turn is {turn.turn_id!r}",
                    param="turn_id",
                )
            )
            return None
        return turn

    async def _terminate_turn(
        self,
        turn: _ActiveRealtimeTurn,
        *,
        reason: str,
        send_events: bool,
    ) -> None:
        turn.cancel_requested = True
        await self._abort_turn(turn)
        task = turn.generation_task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if self.active_turn is turn:
            self.active_turn = None
        if send_events:
            await self._send_turn_terminal(
                turn,
                committed=False,
                reason=reason,
                error=False,
            )

    async def _send_input_ack(
        self,
        turn: _ActiveRealtimeTurn,
        accepted_seq_no: int,
    ) -> None:
        await self._send_json(
            {
                "type": "input.ack",
                "turn_id": turn.turn_id,
                "accepted_seq_no": accepted_seq_no,
                "next_seq_no": turn.next_seq_no,
            }
        )

    async def _send_audio_start(self, turn: _ActiveRealtimeTurn) -> None:
        await self._send_json(
            {
                "type": "audio.start",
                "id": turn.request_id,
                "turn_id": turn.turn_id,
                "format": "pcm",
                "sample_rate": self.config.sample_rate,
            }
        )

    async def _send_turn_terminal(
        self,
        turn: _ActiveRealtimeTurn,
        *,
        committed: bool,
        reason: str,
        error: bool,
    ) -> None:
        await self._send_json(
            {
                "type": "audio.done",
                "id": turn.request_id,
                "turn_id": turn.turn_id,
                "total_chunks": turn.audio_chunks,
                "total_bytes": turn.audio_bytes,
                "error": error,
            }
        )
        await self._send_json(
            {
                "type": "turn.done",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "request_id": turn.request_id,
                "committed": committed,
                "reason": reason,
            }
        )

    async def _abort_turn(self, turn: _ActiveRealtimeTurn) -> None:
        try:
            if turn.realtime_handle is not None:
                await turn.realtime_handle.aclose()
            else:
                await self.client.abort(turn.request_id)
        except Exception:
            logger.exception("Failed to abort realtime request %s", turn.request_id)

    async def _close_backend_session(self) -> None:
        if self._backend_session_closed:
            return
        self._backend_session_closed = True
        try:
            # Target both lifecycle owners: the engine releases the session KV
            # and the streaming vocoder releases the session-keyed codec slot.
            # Both admin handlers are idempotent for unknown sessions.
            result = await self.client.admin(
                "close_realtime_session",
                {"session_id": self.session_id},
                stages=[self.realtime_input_stage, "vocoder"],
                timeout_s=30.0,
            )
            if not bool(result.get("success", False)):
                raise ClientError(
                    str(
                        result.get("message") or "backend realtime session close failed"
                    )
                )
        except Exception:
            logger.exception(
                "Failed to close backend realtime session %s", self.session_id
            )

    def _speech_probe_payload(
        self,
        config: MossTTSRealtimeSpeechSessionConfig,
    ) -> dict[str, Any]:
        payload = config.model_dump(
            exclude={"sample_rate", "stream_audio", "repetition_window"},
            exclude_none=True,
        )
        payload["input"] = "probe"
        payload["stream"] = True
        return payload

    async def _receive_text_frame(
        self,
        *,
        timeout_s: float | None = None,
        max_message_bytes: int | None = None,
        message_kind: str = "MOSS-TTS-Realtime",
    ) -> str:
        resolved_timeout_s = self.idle_timeout_s if timeout_s is None else timeout_s
        resolved_max_bytes = (
            self.max_message_bytes if max_message_bytes is None else max_message_bytes
        )
        message = await asyncio.wait_for(
            self.websocket.receive(),
            timeout=resolved_timeout_s,
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
            if (
                isinstance(frame_bytes, (bytes, bytearray, memoryview))
                and len(frame_bytes) > resolved_max_bytes
            ):
                raise ValueError(
                    f"{message_kind} WebSocket message exceeds "
                    f"{resolved_max_bytes} bytes"
                )
            raise ValueError("speech WebSocket client messages must be text frames")
        if len(raw.encode("utf-8")) > resolved_max_bytes:
            raise ValueError(
                f"{message_kind} WebSocket message exceeds {resolved_max_bytes} bytes"
            )
        return raw

    @staticmethod
    def _parse_message(raw: str) -> dict[str, Any]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("speech WebSocket messages must be JSON objects")
        return payload

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._can_send():
            return
        async with self._send_lock:
            if self._can_send():
                await self.websocket.send_text(json.dumps(payload))

    async def _send_bytes(self, payload: bytes) -> None:
        if not self._can_send():
            raise WebSocketDisconnect
        try:
            async with self._send_lock:
                if not self._can_send():
                    raise WebSocketDisconnect
                await self.websocket.send_bytes(payload)
        except WebSocketDisconnect:
            raise
        except Exception as exc:
            raise WebSocketDisconnect from exc

    async def _send_error(self, error: SpeechAPIError) -> None:
        payload: dict[str, Any] = {"type": "error", "message": error.message}
        if error.error_type is not None:
            payload["error_type"] = error.error_type
        if error.param is not None:
            payload["param"] = error.param
        if error.code is not None:
            payload["code"] = error.code
        await self._send_json(payload)

    def _can_send(self) -> bool:
        return (
            not self.disconnected
            and self.websocket.application_state == WebSocketState.CONNECTED
            and self.websocket.client_state == WebSocketState.CONNECTED
        )

    async def teardown(self) -> None:
        if self._teardown_complete:
            return
        self._teardown_complete = True
        if self.active_turn is not None:
            await self._terminate_turn(
                self.active_turn,
                reason="client_disconnected" if self.disconnected else "session_ended",
                send_events=False,
            )
        await self._close_backend_session()
        self.closed = True


def create_moss_tts_realtime_speech_ws_handler(
    *,
    tokenizer: Any,
    limits: MossTTSRealtimeResourceLimits | None = None,
    realtime_input_stage: str | None = None,
) -> Callable[
    [WebSocket, Client, SpeechRequestValidator, str],
    Awaitable[None],
]:
    """Bind model-owned dependencies to the realtime speech endpoint."""

    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer must implement encode()")
    initialize_moss_tts_realtime_tokenizer_vocab_size(tokenizer)
    resolved_limits = limits or MossTTSRealtimeResourceLimits()
    resolved_input_stage = realtime_input_stage or "tts_engine"

    async def handle(
        websocket: WebSocket,
        client: Client,
        speech_service: SpeechRequestValidator,
        session_id: str,
    ) -> None:
        session = MossTTSRealtimeSpeechWebSocketSession(
            websocket,
            client=client,
            speech_service=speech_service,
            session_id=session_id,
            tokenizer=tokenizer,
            limits=resolved_limits,
            realtime_input_stage=resolved_input_stage,
        )
        await session.run()

    return handle


def _realtime_error_from_exception(exc: Exception) -> SpeechAPIError:
    if isinstance(exc, SpeechAPIError):
        return exc
    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0] if exc.errors() else {}
        message = first_error.get("msg") or "invalid realtime speech message"
        location = tuple(first_error.get("loc", ()))
        param = str(location[-1]) if location else None
        prefix = ".".join(str(item) for item in location)
        return bad_request(
            f"{prefix}: {message}" if prefix else str(message),
            param=param,
        )
    return bad_request(str(exc))


__all__ = [
    "MOSS_TTS_REALTIME_INPUT_MODES",
    "MossTTSRealtimeSpeechWebSocketSession",
]
