"""Strict direct-WebSocket projection of the shared realtime subset."""

import asyncio
import base64
import binascii
import json
import uuid

from starlette.websockets import WebSocketDisconnect

from sglang_omni.serve.realtime.control import ControlEvent, Failure
from sglang_omni.serve.realtime.projection import project_control, project_output
from sglang_omni.serve.realtime.runtime import ProtocolError


class SharedRealtimeSession:
    def __init__(self, websocket, runtime):
        self.websocket = websocket
        self.runtime = runtime
        self.session_id = runtime.session_id

    async def run(self):
        self.runtime.created()
        reader = asyncio.create_task(self._read())
        sender = asyncio.create_task(self._send())
        timer = asyncio.create_task(self.runtime.timeout())
        disconnected = False
        try:
            done, _ = await asyncio.wait(
                (reader, sender), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    result = task.result()
                    if task is reader and result == "disconnect":
                        disconnected = True
                except WebSocketDisconnect:
                    disconnected = True
        finally:
            await self.runtime.close("disconnect")
            if not disconnected and not sender.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(sender), self.runtime.limits.cleanup_timeout_s
                    )
                except (asyncio.TimeoutError, WebSocketDisconnect):
                    pass
            for task in (reader, sender, timer):
                task.cancel()
            await asyncio.gather(reader, sender, timer, return_exceptions=True)

    async def teardown(self):
        await self.runtime.close("disconnect")

    async def _send(self):
        async for envelope in self.runtime.outputs():
            if envelope.epoch != self.runtime.epoch and not envelope.control:
                continue
            if isinstance(envelope.event, ControlEvent):
                event = project_control(envelope.event)
            else:
                event = project_output(
                    envelope.event,
                    output_modalities=(
                        list(envelope.output_modalities)
                        if envelope.output_modalities is not None
                        else None
                    ),
                )
            event["event_id"] = "evt_" + uuid.uuid4().hex
            event.setdefault("sglang", {})["epoch"] = envelope.epoch
            if envelope.unit is not None:
                unit = envelope.unit
                event["sglang"].update(
                    unit_id=f"unit_{unit.seq}",
                    chunk_seq=envelope.chunk_seq,
                    media_time=dict(
                        t_start_ms=self.runtime.ms(unit.start_sample),
                        duration_ms=self.runtime.ms(unit.real_samples),
                    ),
                )
            self.runtime.before_send(envelope)
            await self.websocket.send_text(json.dumps(event, allow_nan=False))
            self.runtime.sent(envelope)
        await self.websocket.close()

    async def _read(self):
        while self.runtime.state != "CLOSED":
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return "disconnect"
            event_id = None
            try:
                if message.get("bytes") is not None:
                    raise ProtocolError(
                        "invalid_request", "binary frames are unsupported"
                    )
                try:
                    raw = json.loads(
                        message.get("text", ""),
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("nonfinite JSON number")
                        ),
                    )
                except (ValueError, TypeError) as exc:
                    raise ProtocolError("invalid_request", "invalid JSON") from exc
                if not isinstance(raw, dict):
                    raise ProtocolError(
                        "invalid_request", "event must be a JSON object"
                    )
                event_id = raw.get("event_id")
                if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
                    event_id = None
                    raise ProtocolError(
                        "invalid_request",
                        "event_id must be a nonempty string",
                        "event_id",
                    )
                await self.dispatch(raw)
            except ProtocolError as exc:
                try:
                    self.runtime.notify(
                        Failure(exc.code, str(exc), False, event_id, exc.param)
                    )
                except RuntimeError:
                    self.runtime.fail("outbound event budget exhausted")
                    return
            except Exception as exc:
                self.runtime.fail(str(exc), event_id=event_id)
                return

    async def dispatch(self, raw):
        typ, event_id = raw.get("type"), raw["event_id"]
        fields = {
            "session.update": {"session"},
            "input_audio_buffer.append": {"audio", "sglang"},
            "input_audio_buffer.clear": set(),
            "sglang.input_audio.end": set(),
            "response.cancel": set(),
            "session.close": set(),
            "input_audio_buffer.commit": set(),
            "response.create": set(),
            "sglang.playback.ack": {
                "response_id",
                "item_id",
                "content_index",
                "audio_end_ms",
                "sglang",
            },
        }
        if not isinstance(typ, str) or typ not in fields:
            raise ProtocolError("not_supported", "unsupported event type")
        if set(raw) - fields[typ] - {"type", "event_id"}:
            raise ProtocolError("invalid_request", "unsupported event fields")
        if typ == "session.update":
            await self.runtime.update(raw.get("session"), event_id)
        elif typ == "input_audio_buffer.append":
            extension = raw.get("sglang")
            if (
                not isinstance(extension, dict)
                or set(extension) - {"seq", "t_start_ms"}
                or "seq" not in extension
            ):
                raise ProtocolError(
                    "invalid_request", "append requires sglang.seq", "sglang.seq"
                )
            try:
                if not isinstance(raw.get("audio"), str):
                    raise ValueError("audio string required")
                if (
                    len(raw["audio"])
                    > (self.runtime.limits.max_input_bytes + 2) // 3 * 4
                ):
                    raise ProtocolError(
                        "buffer_overflow", "encoded audio exceeds input budget"
                    )
                pcm = base64.b64decode(raw["audio"], validate=True)
            except ProtocolError:
                raise
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError(
                    "invalid_request", "invalid base64 audio", "audio"
                ) from exc
            await self.runtime.append(
                pcm, extension["seq"], extension.get("t_start_ms"), event_id
            )
        elif typ == "input_audio_buffer.clear":
            await self.runtime.clear(event_id)
        elif typ == "sglang.input_audio.end":
            await self.runtime.end(event_id)
        elif typ == "response.cancel":
            await self.runtime.cancel(event_id)
        elif typ == "session.close":
            await self.runtime.close("client_closed", event_id)
        elif typ == "sglang.playback.ack":
            extension = raw.get("sglang")
            if not isinstance(extension, dict) or set(extension) != {"epoch"}:
                raise ProtocolError("invalid_request", "ACK requires sglang.epoch")
            if not all(
                isinstance(raw.get(k), str) and raw[k]
                for k in ("response_id", "item_id")
            ):
                raise ProtocolError("invalid_request", "ACK requires response/item IDs")
            await self.runtime.playback_ack(
                extension["epoch"],
                raw["response_id"],
                raw["item_id"],
                raw.get("content_index"),
                raw.get("audio_end_ms"),
            )
        else:
            self.runtime._open()
            raise ProtocolError(
                "not_applicable", "manual turn commands are not granted"
            )
