# SPDX-License-Identifier: Apache-2.0
"""Record the native full-duplex session protocol over a real WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

import websockets

from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName

SAMPLE_RATE = 16000
PACKET_MS = 80
PACKET_BYTES = SAMPLE_RATE * PACKET_MS // 1000 * 2
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
POST_CLOSE_SECONDS = 0.2
FRAME_EXCERPT_CHARS = 200
# Note (wenyao): A closing session can still occupy the only admission slot.
ADMISSION_DENIED_STATUS = 503
ADMISSION_RETRIES = 3
ADMISSION_BACKOFF_S = 0.25
SEND_RECEIPTS_FILE = "input-send-receipts.json"
# Note (wenyao): Keepalive and compression would perturb packet timing.
TRANSPORT = {
    "max_message_bytes": MAX_MESSAGE_BYTES,
    "compression": None,
    "keepalive_ping": False,
    "post_close_s": POST_CLOSE_SECONDS,
    "admission_retries": ADMISSION_RETRIES,
    "admission_backoff_s": ADMISSION_BACKOFF_S,
    "input_end": "first append send start + unpadded input duration",
    "input_send_receipts": SEND_RECEIPTS_FILE,
}


async def run_session(
    url: str,
    pcm: bytes,
    *,
    scenario: str,
    trace_path: Path,
    timeout_s: float = 90.0,
    profile: ProfileName = DEFAULT_PROFILE,
) -> None:
    """Save observations and failures; classification belongs to offline replay."""
    if scenario != "continuous":
        raise ValueError(f"unsupported scenario: {scenario}")
    if not pcm or len(pcm) % 2:
        raise ValueError("Input must be nonempty PCM16")

    receipts: list[dict[str, Any]] = []
    with trace_path.open("x", encoding="utf-8", buffering=1) as trace_file:

        def record(direction: str, event: dict[str, Any]) -> float:
            now = time.perf_counter()
            trace_file.write(
                json.dumps(
                    {
                        "direction": direction,
                        "time_s": now,
                        "event": event,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            return now

        async def admit() -> websockets.ClientConnection:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return await websockets.connect(
                        url,
                        max_size=MAX_MESSAGE_BYTES,
                        compression=None,
                        ping_interval=None,
                    )
                except websockets.InvalidStatus as exc:
                    if exc.response.status_code != ADMISSION_DENIED_STATUS:
                        raise
                    elif attempt > ADMISSION_RETRIES:
                        raise RuntimeError(
                            f"admission denied {attempt} times with HTTP "
                            f"{ADMISSION_DENIED_STATUS}"
                        ) from exc
                    else:
                        record(
                            "admission",
                            {
                                "type": "connection_denied",
                                "http_status": exc.response.status_code,
                                "attempt": attempt,
                            },
                        )
                        await asyncio.sleep(ADMISSION_BACKOFF_S)

        async def exchange() -> None:
            async with await admit() as ws:
                seen: dict[str, asyncio.Event] = {}
                aborted = asyncio.Event()
                fatal = False

                async def send(event_type: str, **payload: Any) -> None:
                    event = {
                        "type": event_type,
                        "event_id": uuid.uuid4().hex,
                        **payload,
                    }
                    started = record("send", event)
                    await ws.send(json.dumps(event))
                    if event_type == "input_audio_buffer.append":
                        receipts.append(
                            {
                                "event_id": event["event_id"],
                                "seq": event["sglang"]["seq"],
                                "start_s": started,
                                "completed_s": time.perf_counter(),
                            }
                        )

                async def settle(event_type: str) -> bool:
                    receipt = seen.setdefault(event_type, asyncio.Event())
                    waiters = (
                        asyncio.create_task(receipt.wait()),
                        asyncio.create_task(aborted.wait()),
                    )
                    try:
                        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for waiter in waiters:
                            waiter.cancel()
                        await asyncio.gather(*waiters, return_exceptions=True)
                    return receipt.is_set()

                async def receive() -> None:
                    nonlocal fatal
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                            if not isinstance(event, dict):
                                raise ValueError("server event must be a JSON object")
                            event_type = event.get("type")
                            if not isinstance(event_type, str):
                                raise ValueError("server event type must be a string")
                            record("receive", event)
                        except ValueError as exc:
                            raise ValueError(
                                "malformed server frame "
                                f"{raw[:FRAME_EXCERPT_CHARS]!r}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        seen.setdefault(event_type, asyncio.Event()).set()
                        if event_type == "error":
                            # Note (wenyao): Close waits for adapter teardown.
                            extension = event.get("sglang")
                            fatal = fatal or bool(
                                isinstance(extension, dict) and extension.get("fatal")
                            )
                            aborted.set()
                    if not seen.get("session.closed", asyncio.Event()).is_set():
                        raise RuntimeError("Connection closed without session.closed")

                async def drive() -> None:
                    streamed = False
                    if await settle("session.created"):
                        await send(
                            "session.update",
                            session={
                                "output_modalities": list(
                                    PROFILES[profile].output_modalities
                                )
                            },
                        )
                    if await settle("session.updated"):
                        streamed = True
                        start_s = time.perf_counter()
                        for seq, offset in enumerate(range(0, len(pcm), PACKET_BYTES)):
                            await asyncio.sleep(
                                max(
                                    0.0,
                                    start_s
                                    + seq * PACKET_MS / 1000
                                    - time.perf_counter(),
                                )
                            )
                            if aborted.is_set():
                                streamed = False
                                break
                            await send(
                                "input_audio_buffer.append",
                                audio=base64.b64encode(
                                    pcm[offset : offset + PACKET_BYTES]
                                ).decode("ascii"),
                                sglang={
                                    "seq": seq,
                                    "t_start_ms": offset / 2 / SAMPLE_RATE * 1000,
                                },
                            )
                    if streamed:
                        await asyncio.sleep(
                            max(
                                0.0,
                                receipts[0]["start_s"]
                                + len(pcm) / (2 * SAMPLE_RATE)
                                - time.perf_counter(),
                            )
                        )
                        if not aborted.is_set():
                            await send("sglang.input_audio.end")
                            await settle("sglang.input_audio.drained")
                    closed = seen.setdefault("session.closed", asyncio.Event())
                    if not fatal and not closed.is_set():
                        await send("session.close")
                    await closed.wait()

                receiver = asyncio.create_task(receive())
                driver = asyncio.create_task(drive())
                try:
                    done, _ = await asyncio.wait(
                        {receiver, driver}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if receiver in done:
                        await receiver
                        # Note (wenyao): No receipts can arrive after receiver exit.
                        try:
                            await asyncio.wait_for(driver, POST_CLOSE_SECONDS)
                        except asyncio.TimeoutError:
                            record(
                                "error",
                                {"message": "driver stalled after the receive loop"},
                            )
                    else:
                        await driver
                        try:
                            await asyncio.wait_for(receiver, POST_CLOSE_SECONDS)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    receiver.cancel()
                    driver.cancel()
                    await asyncio.gather(receiver, driver, return_exceptions=True)

        try:
            await asyncio.wait_for(exchange(), timeout=timeout_s)
        except asyncio.TimeoutError:
            record("error", {"message": f"Session timeout after {timeout_s}s"})
        except (
            OSError,
            websockets.WebSocketException,
            ValueError,
            RuntimeError,
        ) as exc:
            record("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
                json.dumps({"appends": receipts}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
