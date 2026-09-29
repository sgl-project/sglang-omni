"""Strict direct-WebSocket projection of the shared realtime subset."""

import asyncio
import base64
import binascii
import json
import logging
import uuid

from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from sglang_omni.serve.realtime.control import ControlEvent, Failure
from sglang_omni.serve.realtime.projection import project_control, project_output
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import (
    CLIENT_EVENT,
    MAX_EVENT_ID_LENGTH,
    AudioAppendEvent,
    SessionUpdateEvent,
)
from sglang_omni.serve.realtime.types import ProtocolError

logger = logging.getLogger(__name__)


def reject_nonfinite_number(constant: str) -> float:
    raise ValueError(f"nonfinite JSON number {constant}")


class SharedRealtimeSession:
    def __init__(self, websocket: WebSocket, runtime: SessionRuntime) -> None:
        self.websocket = websocket
        self.runtime = runtime
        self.session_id = runtime.session_id

    async def run(self) -> None:
        self.runtime.notify_created()
        reader = asyncio.create_task(self.read())
        sender = asyncio.create_task(self.send())
        is_disconnected = False
        try:
            finished_tasks, _ = await asyncio.wait(
                (reader, sender), return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished_tasks:
                try:
                    has_client_left = task.result() is True
                except WebSocketDisconnect:
                    has_client_left = True
                is_disconnected = is_disconnected or has_client_left
        finally:
            await self.runtime.close("disconnect")
            if not is_disconnected and not sender.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(sender), self.runtime.limits.cleanup_timeout_s
                    )
                except (asyncio.TimeoutError, WebSocketDisconnect):
                    pass
            else:
                pass
            for task in (reader, sender):
                task.cancel()
            await asyncio.gather(reader, sender, return_exceptions=True)

    async def teardown(self) -> None:
        await self.runtime.close("disconnect")

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
            self.runtime.output_buffer.sent(envelope)
        await self.websocket.close()

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
        elif event.type == "input_audio_buffer.clear":
            await self.runtime.clear(event.event_id)
        elif event.type == "sglang.input_audio.end":
            await self.runtime.end(event.event_id)
        elif event.type == "session.close":
            await self.runtime.close("client_closed", event.event_id)
        else:
            self.runtime.require_open()
            raise ProtocolError(
                "not_applicable", "manual turn commands are not granted"
            )
