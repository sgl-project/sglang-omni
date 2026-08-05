from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, GenerateRequest, Message, SamplingParams
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferClear,
    ResponseCancel,
    SessionObject,
    SessionUpdate,
    make_event,
    parse_client_event,
)
from sglang_omni.serve.realtime.vad import (
    StreamingVAD,
    VADConfig,
    VADEvent,
    offsets_to_ms,
)

DEFAULT_INSTRUCTIONS = (
    "You are a helpful realtime voice assistant. Respond conversationally."
)

# Hardcoded — transcription must be verbatim regardless of session instructions.
_TRANSCRIPTION_PROMPT = (
    "You are a speech-to-text engine. Transcribe the user's spoken audio "
    "verbatim into the same language they spoke. Output ONLY the transcript "
    "— no descriptions, no refusals, no explanations."
)

HANDLERS: dict[type, str] = {
    SessionUpdate: "handle_session_update",
    InputAudioBufferAppend: "handle_audio_append",
    InputAudioBufferClear: "handle_audio_clear",
    ResponseCancel: "handle_response_cancel",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class ConversationItem:
    role: str  # "user" | "assistant"
    text: str


class RealtimeSession:
    """Owns one WebSocket and one OpenAI-Realtime audio-in session.

    Per turn (VAD ``speech_stopped`` → auto-commit):
      1. ``run_response`` consumes the audio + prior conversation, streams
         ``response.*`` events to the client. User sees their reply fast.
      2. ``run_transcription`` re-consumes the audio with a verbatim-transcribe
         prompt, streams ``conversation.item.input_audio_transcription.*`` for
         history/UI/log.
      3. Both transcript (user) and response (assistant) are appended to
         ``self.conversation`` so the next turn has full text context.
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        session_id: str | None = None,
        supports_audio_output: bool = False,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or new_id("sess")
        self.supports_audio_output = supports_audio_output

        self.session_object = SessionObject(
            id=self.session_id,
            model=model_name,
            modalities=["text"],
            instructions=DEFAULT_INSTRUCTIONS,
            input_audio_format="pcm16",
        )

        self.audio_buffer = RealtimeAudioBuffer(source_sr=16000, target_sr=16000)
        # (role, text) records — fed back as message history on the next turn.
        self.conversation: list[ConversationItem] = []
        self.closed = False

        self.active_request_id: str | None = None
        self.active_task: asyncio.Task | None = None
        self.active_response_task: asyncio.Task | None = None
        self.active_response_request_id: str | None = None
        self.active_response_has_audio = False
        self.response_cancel_reason: str | None = None
        self.cancelled_response_text = ""
        self.turn_cancel_requested = False
        self.cancel_cleanup_tasks: dict[asyncio.Task, asyncio.Task] = {}
        self.response_start_pending = False
        self.pending_response_cancel_reason: str | None = None
        # VAD may emit speech_stopped while engine is still busy on an
        # earlier utterance — serialize via FIFO.
        self.response_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.queue_drainer: asyncio.Task | None = None

        # VAD is created once with default config; session.update doesn't
        # touch it. Reconnect to change VAD params.
        self.vad = StreamingVAD(VADConfig())
        # Session-wall-clock sample offset of buffer byte 0; advances on
        # commit so speech timestamps stay correct after a buffer drop.
        self.buffer_origin_samples = 0
        self.utterance_start_byte: int | None = None
        # speech_started.item_id predicts the eventual committed id so
        # clients can align live VAD events to the transcript.
        self.utterance_item_id: str | None = None

    async def run(self) -> None:
        """Drive the WebSocket loop; ``websocket.disconnect`` arrives in-band."""
        await self.send(
            make_event(
                "session.created",
                session=self.session_object.model_dump(exclude_none=True),
            )
        )

        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message["text"]
            payload = json.loads(raw)
            assert isinstance(payload, dict), "Top-level payload must be a JSON object"
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event = parse_client_event(payload)
        assert event is not None, f"Unsupported event type: {payload.get('type')!r}"
        method_name = HANDLERS[type(event)]
        await getattr(self, method_name)(event)

    async def handle_session_update(self, event: SessionUpdate) -> None:
        # Validate a candidate first so a rejected update never lands in live state.
        update = event.session.model_dump(exclude_none=True, exclude_unset=True)
        candidate = SessionObject.model_validate(
            self.session_object.model_dump() | update
        )
        modalities = set(candidate.modalities)
        if modalities not in ({"text"}, {"text", "audio"}):
            await self.send_error(
                "invalid_request_error",
                "unsupported_modality",
                "modalities must be ['text'] or ['text', 'audio'].",
            )
            return
        audio_requested = "audio" in modalities
        if audio_requested and not self.supports_audio_output:
            await self.send_error(
                "invalid_request_error",
                "unsupported_modality",
                "Audio output is unavailable for this pipeline.",
            )
            return
        assert candidate.input_audio_format == "pcm16", "Only pcm16 is supported"
        if "output_audio_format" in update and candidate.output_audio_format != "pcm16":
            await self.send_error(
                "invalid_request_error",
                "unsupported_audio_format",
                "Only PCM16 output audio is supported.",
            )
            return
        self.session_object = candidate
        await self.send(
            make_event(
                "session.updated",
                session=self.session_object.model_dump(exclude_none=True),
            )
        )

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        decoded_len = self.audio_buffer.append_b64(event.audio)
        new_bytes = self.audio_buffer.tail(decoded_len)
        emits = await asyncio.to_thread(self.vad.process, new_bytes)
        for emit in emits:
            await self.handle_vad_emit(emit)

    async def handle_vad_emit(self, emit: Any) -> None:
        timestamp_ms = offsets_to_ms(self.buffer_origin_samples + emit.sample_offset)
        if emit.event_type == VADEvent.SPEECH_STARTED:
            # PCM16 mono: 2 bytes/sample.
            vad_byte = max(0, emit.sample_offset * 2)
            self.utterance_start_byte = min(vad_byte, self.audio_buffer.num_bytes)
            self.utterance_item_id = new_id("item")
            await self.send(
                make_event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=timestamp_ms,
                    item_id=self.utterance_item_id,
                )
            )
            turn_detection = self.session_object.turn_detection
            interrupt_response = (
                turn_detection is None or turn_detection.interrupt_response is not False
            )
            response_has_audio = self.active_response_has_audio or (
                self.response_start_pending
                and "audio" in self.session_object.modalities
            )
            if response_has_audio and interrupt_response:
                await self.cancel_active_response("turn_detected")
        elif emit.event_type == VADEvent.SPEECH_STOPPED:
            await self.send(
                make_event(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=timestamp_ms,
                    item_id=self.utterance_item_id or new_id("item"),
                )
            )
            await self.auto_commit_utterance(emit.sample_offset)

    def drop_buffer_and_reset_vad(self) -> None:
        self.buffer_origin_samples += self.audio_buffer.num_samples
        self.audio_buffer.clear()
        self.utterance_start_byte = None
        self.utterance_item_id = None
        self.vad.reset()

    async def auto_commit_utterance(self, end_sample_offset: int) -> None:
        if self.audio_buffer.is_empty():
            return
        start_byte = self.utterance_start_byte or 0
        end_byte = min(end_sample_offset * 2, self.audio_buffer.num_bytes)
        if end_byte <= start_byte:
            return
        payload = self.audio_buffer.to_sliced_wav_data_uri(
            start_byte=start_byte, end_byte=end_byte
        )
        item_id = self.utterance_item_id or new_id("item")
        self.drop_buffer_and_reset_vad()

        await self.send(make_event("input_audio_buffer.committed", item_id=item_id))
        await self.response_queue.put((item_id, payload))
        if self.queue_drainer is None or self.queue_drainer.done():
            self.queue_drainer = asyncio.create_task(self.drain_queue())

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        self.drop_buffer_and_reset_vad()
        await self.send(make_event("input_audio_buffer.cleared"))

    async def handle_response_cancel(self, event: ResponseCancel) -> None:
        await self.cancel_active_response("client_cancelled")

    async def cancel_active_response(self, reason: str) -> None:
        if self.response_start_pending:
            if self.pending_response_cancel_reason is None:
                self.pending_response_cancel_reason = reason
            return
        task = self.active_response_task
        request_id = self.active_response_request_id
        if task is None or task.done() or request_id is None:
            return
        cleanup_task = self.cancel_cleanup_tasks.get(task)
        if cleanup_task is not None and not cleanup_task.done():
            return
        self.response_cancel_reason = reason
        task.cancel()
        cleanup_task = asyncio.create_task(self._abort_and_drain(task, request_id))
        self.cancel_cleanup_tasks[task] = cleanup_task
        cleanup_task.add_done_callback(
            lambda _: self.cancel_cleanup_tasks.pop(task, None)
        )

    async def drain_queue(self) -> None:
        while not self.closed:
            item_id, payload = await self.response_queue.get()
            self.response_start_pending = True
            try:
                self.active_task = asyncio.create_task(self.run_turn(item_id, payload))
                await asyncio.gather(self.active_task, return_exceptions=True)
            finally:
                self.active_task = None
                self.response_start_pending = False
                self.pending_response_cancel_reason = None

    async def run_turn(self, item_id: str, audio_payload: str) -> None:
        """Pass 1: response (user-facing, streams fast).
        Pass 2: transcription (background, fills history).
        """
        self.turn_cancel_requested = False
        self.active_response_task = asyncio.create_task(
            self.run_response(audio_payload)
        )
        try:
            response_text = await self.active_response_task
        except asyncio.CancelledError:
            if self.response_cancel_reason is None or self.turn_cancel_requested:
                raise
            response_text = self.cancelled_response_text
        finally:
            self.active_response_task = None
            self.response_cancel_reason = None
            self.cancelled_response_text = ""
        transcript = await self.run_transcription(item_id, audio_payload)
        # Append in chronological order: user spoke first, assistant replied.
        if transcript:
            self.conversation.append(ConversationItem(role="user", text=transcript))
        if response_text:
            self.conversation.append(
                ConversationItem(role="assistant", text=response_text)
            )

    async def run_response(self, audio_payload: str) -> str:
        """Stream the assistant response and wait for every active terminal."""
        response_request = self.build_response_request(audio_payload)
        wants_audio = "audio" in (response_request.output_modalities or [])
        response_id = new_id("resp")
        resp_item_id = new_id("item")
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id
        self.active_response_request_id = request_id
        self.active_response_has_audio = wants_audio
        text_acc: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        saw_audio = False
        text_done = False
        audio_done = False
        response_done = False

        try:
            await self.send(
                make_event(
                    "response.created",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                )
            )

            self.response_start_pending = False
            if self.pending_response_cancel_reason is not None:
                reason = self.pending_response_cancel_reason
                self.pending_response_cancel_reason = None
                self.response_cancel_reason = reason
                await self.send(
                    make_event(
                        "response.text.done",
                        response_id=response_id,
                        item_id=resp_item_id,
                        output_index=0,
                        content_index=0,
                        text="",
                    )
                )
                await self._send_response_done(
                    response_id=response_id,
                    item_id=resp_item_id,
                    response_text="",
                    include_audio=False,
                    status="cancelled",
                    reason=reason,
                    usage=None,
                )
                response_done = True
                raise asyncio.CancelledError

            async for chunk in self.client.completion_stream(
                response_request,
                request_id=request_id,
                audio_format="pcm" if wants_audio else "wav",
            ):
                if chunk.text and (chunk.modality == "text" or not text_acc):
                    text_acc.append(chunk.text)
                    await self.send(
                        make_event(
                            "response.text.delta",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=0,
                            delta=chunk.text,
                        )
                    )

                if wants_audio and chunk.modality == "audio" and chunk.audio_b64:
                    saw_audio = True
                    await self.send(
                        make_event(
                            "response.audio.delta",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=1,
                            delta=chunk.audio_b64,
                        )
                    )

                if chunk.finish_reason is not None:
                    if chunk.modality == "text":
                        finish_reason = chunk.finish_reason
                    if chunk.usage is not None:
                        usage = dataclasses.asdict(chunk.usage)

                if (
                    chunk.modality == "text"
                    and chunk.finish_reason is not None
                    and not text_done
                ):
                    await self.send(
                        make_event(
                            "response.text.done",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=0,
                            text="".join(text_acc),
                        )
                    )
                    text_done = True
                elif (
                    wants_audio
                    and chunk.modality == "audio"
                    and chunk.finish_reason is not None
                    and saw_audio
                    and not audio_done
                ):
                    await self.send(
                        make_event(
                            "response.audio.done",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=1,
                        )
                    )
                    audio_done = True

            response_text = "".join(text_acc)
            if not text_done:
                await self.send(
                    make_event(
                        "response.text.done",
                        response_id=response_id,
                        item_id=resp_item_id,
                        output_index=0,
                        content_index=0,
                        text=response_text,
                    )
                )
                text_done = True
            if wants_audio and not saw_audio:
                await self.send_error(
                    "server_error",
                    "audio_output_missing",
                    "The configured pipeline completed without audio output.",
                )
                await self._send_response_done(
                    response_id=response_id,
                    item_id=resp_item_id,
                    response_text=response_text,
                    include_audio=False,
                    status="failed",
                    reason="audio_output_missing",
                    usage=usage,
                )
                response_done = True
                return ""

            if wants_audio and not audio_done:
                await self.send(
                    make_event(
                        "response.audio.done",
                        response_id=response_id,
                        item_id=resp_item_id,
                        output_index=0,
                        content_index=1,
                    )
                )
                audio_done = True

            await self._send_response_done(
                response_id=response_id,
                item_id=resp_item_id,
                response_text=response_text,
                include_audio=wants_audio,
                status="completed",
                reason=finish_reason,
                usage=usage,
            )
            response_done = True
            return response_text
        except asyncio.CancelledError:
            self.cancelled_response_text = "".join(text_acc)
            if not response_done:
                if not text_done:
                    await self.send(
                        make_event(
                            "response.text.done",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=0,
                            text="".join(text_acc),
                        )
                    )
                if wants_audio and saw_audio and not audio_done:
                    await self.send(
                        make_event(
                            "response.audio.done",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=1,
                        )
                    )
                await self._send_response_done(
                    response_id=response_id,
                    item_id=resp_item_id,
                    response_text="".join(text_acc),
                    include_audio=wants_audio and saw_audio,
                    status="cancelled",
                    reason=self.response_cancel_reason or "client_cancelled",
                    usage=usage,
                )
            raise
        except Exception as exc:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Realtime response generation failed",
                    "exception": exc,
                }
            )
            response_text = "".join(text_acc)
            if not response_done:
                if not text_done:
                    await self.send(
                        make_event(
                            "response.text.done",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=0,
                            text=response_text,
                        )
                    )
                if wants_audio and saw_audio and not audio_done:
                    await self.send(
                        make_event(
                            "response.audio.done",
                            response_id=response_id,
                            item_id=resp_item_id,
                            output_index=0,
                            content_index=1,
                        )
                    )
                await self.send_error(
                    "server_error",
                    "response_generation_failed",
                    "Realtime response generation failed.",
                )
                await self._send_response_done(
                    response_id=response_id,
                    item_id=resp_item_id,
                    response_text=response_text,
                    include_audio=wants_audio and saw_audio,
                    status="failed",
                    reason="error",
                    usage=usage,
                )
            return ""
        finally:
            if self.active_request_id == request_id:
                self.active_request_id = None
            if self.active_response_request_id == request_id:
                self.active_response_request_id = None
                self.active_response_has_audio = False

    async def _send_response_done(
        self,
        *,
        response_id: str,
        item_id: str,
        response_text: str,
        include_audio: bool,
        status: str,
        reason: str,
        usage: dict[str, Any] | None,
    ) -> None:
        content: list[dict[str, Any]] = [{"type": "text", "text": response_text}]
        if include_audio:
            content.append({"type": "audio", "transcript": response_text})
        await self.send(
            make_event(
                "response.done",
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": status,
                    "status_details": {"reason": reason},
                    "output": [
                        {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }
                    ],
                    "usage": usage,
                },
            )
        )

    async def run_transcription(self, item_id: str, audio_payload: str) -> str:
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.active_request_id = request_id
        try:
            text_acc: list[str] = []
            async for chunk in self.client.completion_stream(
                self.build_transcription_request(audio_payload),
                request_id=request_id,
            ):
                if chunk.modality == "text" and chunk.text:
                    text_acc.append(chunk.text)
                    await self.send(
                        make_event(
                            "conversation.item.input_audio_transcription.delta",
                            item_id=item_id,
                            content_index=0,
                            delta=chunk.text,
                        )
                    )
                if chunk.finish_reason is not None:
                    break

            transcript = "".join(text_acc)
            await self.send(
                make_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item_id,
                    content_index=0,
                    transcript=transcript,
                )
            )
            return transcript
        finally:
            self.active_request_id = None

    def _sampling(self) -> SamplingParams:
        max_tokens = self.session_object.max_response_output_tokens
        return SamplingParams(
            temperature=self.session_object.temperature,
            top_p=1.0,
            max_new_tokens=max_tokens if isinstance(max_tokens, int) else None,
        )

    def build_response_request(self, audio_payload: str) -> GenerateRequest:
        """Response pass: session instructions + conversation history + current audio.

        A trailing user message anchors the audio as *this turn's* user input.
        Without it Qwen3-Omni treats audio as background context and ignores it
        once any prior conversation exists, falling back to greeting on every
        turn.
        """
        messages: list[Message] = [
            Message(
                role="system",
                content=self.session_object.instructions or DEFAULT_INSTRUCTIONS,
            )
        ]
        for item in self.conversation:
            messages.append(Message(role=item.role, content=item.text))
        messages.append(
            Message(
                role="user",
                content="Listen to the spoken audio above and respond to it.",
            )
        )
        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self._sampling(),
            stream=True,
            output_modalities=list(self.session_object.modalities),
            metadata={"audios": [audio_payload]},
        )

    def build_transcription_request(self, audio_payload: str) -> GenerateRequest:
        """Transcription pass: hardcoded verbatim prompt + current audio only."""
        return GenerateRequest(
            model=self.model_name,
            messages=[
                Message(role="system", content=_TRANSCRIPTION_PROMPT),
                Message(role="user", content="Transcribe the spoken audio."),
            ],
            sampling=self._sampling(),
            stream=True,
            output_modalities=["text"],
            metadata={"audios": [audio_payload]},
        )

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        event.setdefault("event_id", new_id("evt"))
        await self.websocket.send_text(json.dumps(event))

    async def send_error(self, type_: str, code: str, message: str) -> None:
        await self.send(
            make_event(
                "error",
                error={"type": type_, "code": code, "message": message},
            )
        )

    async def _cancel_and_abort(
        self, task: asyncio.Task | None, request_id: str | None
    ) -> None:
        """Abort engine request, cancel task, absorb result.

        ``asyncio.gather(..., return_exceptions=True)`` is used instead of
        ``.exception()`` because the latter re-raises ``CancelledError`` on a
        cancelled task, turning a normal disconnect into a handler exception.
        """
        if task is None or task.done():
            return
        if task is self.active_task:
            self.turn_cancel_requested = True
        task.cancel()
        await self._abort_and_drain(task, request_id)

    async def _abort_and_drain(
        self, task: asyncio.Task, request_id: str | None
    ) -> None:
        try:
            if request_id is not None:
                await self.client.abort(request_id)
        except Exception as exc:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Realtime response abort failed",
                    "exception": exc,
                    "task": task,
                }
            )
        finally:
            await asyncio.gather(task, return_exceptions=True)

    async def teardown(self) -> None:
        self.closed = True
        if self.cancel_cleanup_tasks:
            await asyncio.gather(
                *list(self.cancel_cleanup_tasks.values()), return_exceptions=True
            )
        await self._cancel_and_abort(self.active_task, self.active_request_id)
        await self._cancel_and_abort(self.queue_drainer, None)
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()
