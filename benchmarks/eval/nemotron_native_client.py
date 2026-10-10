"""Measure one Nemotron native session without loading a model or dataset."""

import asyncio
import base64
import json
import math
import time
from typing import TypedDict

import msgspec
from pydantic import BaseModel
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException


class Event(BaseModel):
    type: str
    response_id: str | None = None
    delta: str | None = None
    text: str | None = None
    consumed_ms: float | None = None


class EventRecord(TypedDict):
    at_seconds: float
    raw_json: str


class PacketRecord(TypedDict):
    sequence: int
    at_seconds: float
    audio_end_seconds: float


class Measurement(msgspec.Struct):
    sample_id: str
    audio_seconds: float
    reference: str
    text: str = ""
    error: str | None = None
    first_text_seconds: float | None = None
    eos_to_final_seconds: float | None = None
    eos_to_drained_seconds: float | None = None
    wall_seconds: float | None = None
    events: list[EventRecord] = []
    packets: list[PacketRecord] = []


async def measure(
    url: str,
    pcm_bytes: bytes,
    sample_id: str,
    reference: str,
    *,
    packet_milliseconds: int = 20,
    paced: bool = True,
    timeout_seconds: float = 120,
) -> Measurement:
    if packet_milliseconds <= 0 or len(pcm_bytes) % 2 or timeout_seconds <= 0:
        raise ValueError(
            "Require positive packet_milliseconds/timeout and aligned PCM16"
        )
    else:
        result = Measurement(sample_id, len(pcm_bytes) / 32000, reference)
    encoded_packets = [
        base64.b64encode(pcm_bytes[offset : offset + packet_milliseconds * 32]).decode()
        for offset in range(0, len(pcm_bytes), packet_milliseconds * 32)
    ]
    origin_seconds = time.perf_counter()
    first_send_seconds: float | None = None
    eos_sent_seconds: float | None = None
    final_at_seconds: float | None = None
    drained_at_seconds: float | None = None
    deltas: list[str] = []
    response_ids: set[str] = set()
    packet_size_bytes = packet_milliseconds * 32

    try:
        async with (
            asyncio.timeout(timeout_seconds),
            connect(url, max_size=2**24) as websocket,
        ):

            async def receive() -> Event:
                raw_json = await websocket.recv()
                if isinstance(raw_json, bytes):
                    raw_json = raw_json.decode("utf-8")
                else:
                    pass
                result.events.append(
                    {
                        "at_seconds": time.perf_counter() - origin_seconds,
                        "raw_json": raw_json,
                    }
                )
                event = Event.model_validate_json(raw_json)
                if event.type == "error":
                    raise RuntimeError(f"Server error: {raw_json}")
                else:
                    return event

            if (await receive()).type != "session.created":
                raise ValueError("Expected session.created")
            else:
                await websocket.send(
                    json.dumps(
                        {
                            "event_id": "configure",
                            "type": "session.update",
                            "session": {"output_modalities": ["text"]},
                        }
                    )
                )
            if (await receive()).type != "session.updated":
                raise ValueError("Expected session.updated")
            else:
                first_send_seconds = time.perf_counter()

            async def send_audio() -> None:
                nonlocal eos_sent_seconds
                for sequence, offset in enumerate(
                    range(0, len(pcm_bytes), packet_size_bytes)
                ):
                    if paced:
                        await asyncio.sleep(
                            max(
                                0,
                                first_send_seconds
                                + offset / 32000
                                - time.perf_counter(),
                            )
                        )
                    else:
                        pass
                    packet = pcm_bytes[offset : offset + packet_size_bytes]
                    sent_at_seconds = time.perf_counter()
                    await websocket.send(
                        json.dumps(
                            {
                                "event_id": f"audio-{sequence}",
                                "type": "input_audio_buffer.append",
                                "audio": encoded_packets[sequence],
                                "sglang": {"seq": sequence},
                            }
                        )
                    )
                    result.packets.append(
                        {
                            "sequence": sequence,
                            "at_seconds": sent_at_seconds - origin_seconds,
                            "audio_end_seconds": (offset + len(packet)) / 32000,
                        }
                    )
                if paced:
                    await asyncio.sleep(
                        max(
                            0,
                            first_send_seconds
                            + result.audio_seconds
                            - time.perf_counter(),
                        )
                    )
                else:
                    pass
                eos_sent_seconds = time.perf_counter()
                await websocket.send(
                    json.dumps({"event_id": "end", "type": "sglang.input_audio.end"})
                )
                result.events.append(
                    {
                        "at_seconds": eos_sent_seconds - origin_seconds,
                        "raw_json": json.dumps({"sent": "sglang.input_audio.end"}),
                    }
                )

            async def receive_text() -> None:
                nonlocal final_at_seconds, drained_at_seconds
                while drained_at_seconds is None:
                    event = await receive()
                    now_seconds = time.perf_counter()
                    if event.type in (
                        "response.output_text.delta",
                        "response.output_text.done",
                    ):
                        response_id = event.response_id
                        if not isinstance(response_id, str):
                            raise ValueError("Text event lacks response_id")
                        else:
                            response_ids.add(response_id)
                    else:
                        pass
                    if event.type == "response.output_text.delta":
                        delta = event.delta
                        if final_at_seconds is not None or not isinstance(delta, str):
                            raise ValueError("Invalid delta or delta after final")
                        else:
                            deltas.append(delta)
                            if delta and result.first_text_seconds is None:
                                result.first_text_seconds = (
                                    now_seconds - first_send_seconds
                                )
                            else:
                                pass
                    elif event.type == "response.output_text.done":
                        text = event.text
                        if final_at_seconds is not None or not isinstance(text, str):
                            raise ValueError("Duplicate or malformed final text")
                        else:
                            final_at_seconds = now_seconds
                            result.text = text
                    elif event.type == "sglang.input_audio.drained":
                        drained_at_seconds = now_seconds
                        consumed = event.consumed_ms
                        if not isinstance(consumed, (float, int)) or not math.isclose(
                            consumed, result.audio_seconds * 1000, abs_tol=0.001
                        ):
                            raise ValueError(
                                "Drained audio duration does not match input"
                            )
                        else:
                            pass
                    else:
                        pass

            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_text())
            try:
                await asyncio.gather(sender, receiver)
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                    else:
                        pass
                await asyncio.gather(sender, receiver, return_exceptions=True)
            if (
                final_at_seconds is None
                or eos_sent_seconds is None
                or final_at_seconds < eos_sent_seconds
                or len(response_ids) != 1
                or "".join(deltas) != result.text
            ):
                raise ValueError(
                    "Final text, response identity, or EOS ordering violated"
                )
            else:
                result.eos_to_final_seconds = final_at_seconds - eos_sent_seconds
                result.eos_to_drained_seconds = drained_at_seconds - eos_sent_seconds
                result.wall_seconds = drained_at_seconds - first_send_seconds
            await websocket.send(
                json.dumps({"event_id": "close", "type": "session.close"})
            )
            while (await receive()).type != "session.closed":
                pass
    except (
        TimeoutError,
        OSError,
        ValueError,
        RuntimeError,
        WebSocketException,
    ) as error:
        result.error = f"{type(error).__name__}: {error}"
    return result
