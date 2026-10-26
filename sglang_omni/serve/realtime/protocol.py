"""Strict direct-WebSocket projection of the shared realtime subset."""

import asyncio
import base64
import binascii
import json
import logging
import uuid

from pydantic import ValidationError
from starlette.status import (
    WS_1000_NORMAL_CLOSURE,
    WS_1008_POLICY_VIOLATION,
    WS_1011_INTERNAL_ERROR,
)
from starlette.websockets import WebSocket, WebSocketDisconnect

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.serve.realtime.constants import (
    CLOSE_REASON_CLIENT_CLOSED,
    CLOSE_REASON_DISCONNECT,
    CLOSE_REASON_IDLE_TIMEOUT,
    CLOSE_REASON_SESSION_UPDATE_TIMEOUT,
)
from sglang_omni.serve.realtime.control import ControlEvent, Failure
from sglang_omni.serve.realtime.projection import project_control, project_output
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import (
    CLIENT_EVENT,
    MAX_EVENT_ID_LENGTH,
    AudioAppendEvent,
    ConversationItemTruncateEvent,
    ImageAppendEvent,
    SessionUpdateEvent,
)
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import ProtocolError

logger = logging.getLogger(__name__)

MAX_CLOSE_REASON_BYTES = 123
POLICY_CLOSE_REASONS = frozenset(
    {
        CLOSE_REASON_SESSION_UPDATE_TIMEOUT,
        CLOSE_REASON_IDLE_TIMEOUT,
        ContextExhaustedError.CODE,
    }
)


def reject_nonfinite_number(constant: str) -> float:
    raise ValueError(f"nonfinite JSON number {constant}")


class SharedRealtimeSession:
    def __init__(self, websocket: WebSocket, runtime: SessionRuntime) -> None:
        self.websocket = websocket
        self.runtime = runtime
        self.session_id = runtime.session_id

    async def run(self) -> None:
        self.runtime.start()
        reader = asyncio.create_task(self.read())
        sender = asyncio.create_task(self.send())
        runtime_closed = asyncio.create_task(self.runtime.closed_event.wait())
        is_disconnected = False
        try:
            finished_tasks, _ = await asyncio.wait(
                (reader, sender, runtime_closed), return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished_tasks - {runtime_closed}:
                try:
                    has_client_left = task.result() is True
                except WebSocketDisconnect:
                    has_client_left = True
                is_disconnected = is_disconnected or has_client_left
        finally:
            try:
                await self.runtime.close(CLOSE_REASON_DISCONNECT)
                if not is_disconnected and not sender.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(sender),
                            self.runtime.limits.cleanup_timeout_s,
                        )
                    except (asyncio.TimeoutError, WebSocketDisconnect):
                        pass
                else:
                    pass
            finally:
                await cancel_local_tasks(
                    (reader, sender, runtime_closed),
                    self.runtime.limits.cleanup_timeout_s,
                )

    async def teardown(self) -> None:
        await self.runtime.close(CLOSE_REASON_DISCONNECT)

    async def send(self) -> None:
        async for envelope in self.runtime.outputs():
            if isinstance(envelope.event, ControlEvent):
                server_event = project_control(envelope.event)
            else:
                server_event = project_output(
                    envelope.event, output_modalities=envelope.output_modalities
                )
            server_event["event_id"] = "evt_" + uuid.uuid4().hex
            if envelope.unit is not None:
                unit = envelope.unit
                server_event.setdefault("sglang", {}).update(
                    {
                        "unit_id": unit.unit_id,
                        "chunk_seq": envelope.chunk_index,
                        "media_time": dict(
                            t_start_ms=self.runtime.capabilities.input_duration_ms(
                                unit.start_sample
                            ),
                            duration_ms=self.runtime.capabilities.input_duration_ms(
                                unit.real_samples
                            ),
                        ),
                    }
                )
            else:
                pass
            self.runtime.output_buffer.before_send(envelope)
            await self.websocket.send_text(json.dumps(server_event, allow_nan=False))
            if not isinstance(envelope.event, ControlEvent):
                self.runtime.mark_activity()
            else:
                pass
            self.runtime.output_buffer.sent(envelope)
        close_reason = self.runtime.close_reason
        assert close_reason is not None
        if close_reason in (CLOSE_REASON_CLIENT_CLOSED, CLOSE_REASON_DISCONNECT):
            close_code = WS_1000_NORMAL_CLOSURE
        elif close_reason in POLICY_CLOSE_REASONS:
            close_code = WS_1008_POLICY_VIOLATION
        else:
            close_code = WS_1011_INTERNAL_ERROR
        close_reason = close_reason.encode("utf-8")[:MAX_CLOSE_REASON_BYTES].decode(
            "utf-8", errors="ignore"
        )
        await self.websocket.close(close_code, close_reason)

    async def read(self) -> bool:
        """Returns whether the client disconnected."""
        while self.runtime.state != "CLOSED":
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return True
            else:
                pass
            event_id: str | None = None
            try:
                if message.get("bytes") is not None:
                    raise ProtocolError(
                        "invalid_request", "binary frames are unsupported"
                    )
                else:
                    pass
                try:
                    raw_event = json.loads(
                        message.get("text", ""),
                        parse_constant=reject_nonfinite_number,
                    )
                except (ValueError, TypeError) as exc:
                    raise ProtocolError("invalid_request", "invalid JSON") from exc
                client_event_id = (
                    raw_event.get("event_id") if isinstance(raw_event, dict) else None
                )
                if (
                    isinstance(client_event_id, str)
                    and 0 < len(client_event_id) <= MAX_EVENT_ID_LENGTH
                ):
                    event_id = client_event_id
                else:
                    pass
                await self.dispatch(raw_event)
            except ProtocolError as exc:
                try:
                    self.runtime.notify(
                        Failure(exc.code, str(exc), False, event_id, exc.param)
                    )
                except RuntimeError:
                    self.runtime.fail("outbound event budget exhausted")
                    return False
            except Exception as exc:
                logger.exception(f"Realtime session {self.session_id} dispatch failed")
                self.runtime.fail(str(exc), event_id=event_id)
                return False
        return False

    async def dispatch(self, raw_event: object) -> None:
        try:
            event = CLIENT_EVENT.validate_python(raw_event)
        except ValidationError as exc:
            error = exc.errors()[0]
            code = (
                "not_supported"
                if error["type"] == "union_tag_invalid"
                else "invalid_request"
            )
            raise ProtocolError(
                code, error["msg"], ".".join(str(part) for part in error["loc"])
            ) from exc
        if isinstance(event, SessionUpdateEvent):
            await self.runtime.update(event.session, event.event_id)
        elif isinstance(event, AudioAppendEvent):
            max_encoded_audio_chars = (self.runtime.limits.max_input_bytes + 2) // 3 * 4
            if len(event.audio) > max_encoded_audio_chars:
                raise ProtocolError(
                    "buffer_overflow", "encoded audio exceeds input budget"
                )
            else:
                pass
            try:
                pcm = base64.b64decode(event.audio, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError(
                    "invalid_request", "invalid base64 audio", "audio"
                ) from exc
            await self.runtime.append(
                pcm, event.sglang.seq, event.sglang.t_start_ms, event.event_id
            )
        elif isinstance(event, ImageAppendEvent):
            max_encoded_image_chars = (
                (self.runtime.capabilities.max_image_bytes + 2) // 3 * 4
            )
            if len(event.image) > max_encoded_image_chars:
                raise ProtocolError(
                    "buffer_overflow", "encoded image exceeds input budget"
                )
            else:
                pass
            try:
                image = base64.b64decode(event.image, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError(
                    "invalid_request", "invalid base64 image", "image"
                ) from exc
            await self.runtime.append_image(image, event.sglang.t_ms, event.event_id)
        elif event.type == "input_audio_buffer.clear":
            await self.runtime.clear(event.event_id)
        elif event.type == "response.cancel":
            await self.runtime.cancel()
        elif event.type == "input_audio_buffer.commit":
            await self.runtime.commit()
        elif isinstance(event, ConversationItemTruncateEvent):
            await self.runtime.truncate(event.item_id, event.audio_end_ms)
        elif event.type == "sglang.input_audio.end":
            await self.runtime.end(event.event_id)
        elif event.type == "session.close":
            await self.runtime.close(CLOSE_REASON_CLIENT_CLOSED, event.event_id)
        else:
            self.runtime.require_open()
            raise ProtocolError(
                "not_applicable", "manual turn commands are not granted"
            )
