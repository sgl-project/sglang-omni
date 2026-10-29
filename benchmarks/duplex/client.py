# SPDX-License-Identifier: Apache-2.0
"""Record native full-duplex, legacy or turn-based sessions over a real WebSocket."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import websockets
from pydantic import BaseModel, ConfigDict, JsonValue

from benchmarks.duplex.oracle_models import turn_detection_type
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName, Protocol

SAMPLE_RATE = 16000
PACKET_MS = 80
PACKET_BYTES = SAMPLE_RATE * PACKET_MS // 1000 * 2
# note (wenyao): Leave drain/close time before the server's 240 s deadline.
SESSION_LIMIT_S = 240.0
DRAIN_MARGIN_S = 10.0
MAX_TIMEOUT_S = SESSION_LIMIT_S - DRAIN_MARGIN_S
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
POST_CLOSE_SECONDS = 0.2
FRAME_EXCERPT_CHARS = 200
# note (wenyao): A closing session can still occupy the only admission slot.
ADMISSION_DENIED_STATUS = 503
ADMISSION_RETRIES = 3
ADMISSION_BACKOFF_S = 0.25
SEND_RECEIPTS_FILE = "input-send-receipts.json"
# Trace lines reach the disk in batches at most this far apart.
TRACE_FLUSH_INTERVAL_S = 0.5
# Sessions that end together are closed in one batch gathered for this long.
TRACE_CLOSE_BATCH_S = 0.05
# note (wenyao): Keepalive and compression would perturb packet timing.
TRANSPORT: dict[str, JsonValue] = {
    "max_message_bytes": MAX_MESSAGE_BYTES,
    "compression": None,
    "keepalive_ping": False,
    "post_close_s": POST_CLOSE_SECONDS,
    "admission_retries": ADMISSION_RETRIES,
    "admission_backoff_s": ADMISSION_BACKOFF_S,
    "input_end": "first append send start + unpadded input duration",
    "input_send_receipts": SEND_RECEIPTS_FILE,
}
# note (luojiaxuan): The legacy protocol has no end/drain receipts, so the session
# ends by a liveness probe a fixed tail after the input end instead.
LEGACY_TAIL_S = 1.0
LEGACY_TRANSPORT: dict[str, JsonValue] = {
    "input_receipts": "none: no accepted, ended or drained events",
    "session_end": "input end + tail, then a no-op session.update liveness probe; "
    "the socket closes after its session.updated without session.close",
}
# note (luojiaxuan): A turn-based server answers the utterance still open at the
# input end before it drains, so only the session deadline bounds that wait.
TURN_BASED_TRANSPORT: dict[str, JsonValue] = {
    "session_end": "sglang.input_audio.end at the input end, then session.close "
    "after sglang.input_audio.drained, which follows the server's answer to the "
    "last utterance; no deadline other than the session timeout bounds that wait",
}


class TurnDetection(BaseModel):
    """session.update turn_detection; the server validates every other field."""

    model_config = ConfigDict(extra="allow", strict=True)
    type: Literal["server_vad", "semantic_vad"]


def profile_protocol(
    profile: ProfileName, turn_detection: TurnDetection | None
) -> Protocol:
    """The protocol profile records; every protocol but native takes turn_detection."""
    protocol = PROFILES[profile].protocol
    if (protocol != "native") != (turn_detection is not None):
        raise ValueError(
            "turn_detection is required by legacy-protocol profiles and turn-based "
            "profiles and rejected by native ones"
        )
    else:
        return protocol


class InputAudioMetadata(TypedDict):
    seq: int
    t_start_ms: float


class SendReceipt(TypedDict):
    event_id: str
    seq: int
    start_s: float
    completed_s: float


@dataclass(frozen=True, kw_only=True)
class TraceLine:
    trace_file: io.TextIOWrapper
    line: str


@dataclass(frozen=True, kw_only=True)
class ReceiptsWrite:
    trace_file: io.TextIOWrapper
    receipts_path: Path
    receipts: list[SendReceipt]


@dataclass(frozen=True, kw_only=True)
class TraceClose:
    trace_file: io.TextIOWrapper
    loop: asyncio.AbstractEventLoop
    closed: asyncio.Future[None]


def settle_close(closed: asyncio.Future[None], failure: OSError | None) -> None:
    if closed.done():
        pass
    elif failure is None:
        closed.set_result(None)
    else:
        closed.set_exception(failure)


def reject_constant(constant: str) -> float:
    raise ValueError(f"Out of range float values are not JSON compliant: {constant}")


# A frame lands in the trace as received, so it must already be strict JSON.
SERVER_EVENT_DECODER = json.JSONDecoder(parse_constant=reject_constant)


# note (kangwenqiao): A write can wait hundreds of ms for disk writeback of its page,
# so it leaves the loop that paces every session. Every other thread that wakes takes
# the GIL from that loop, so one thread writes on a fixed period for all sessions.
class TraceWriter:
    """One thread writes the trace and receipts of every session of the process."""

    def __init__(self) -> None:
        self.pending: queue.SimpleQueue[TraceLine | ReceiptsWrite | TraceClose] = (
            queue.SimpleQueue()
        )
        self.close_requested: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        self.open_traces: int = 0
        self.thread: threading.Thread | None = None

    def open(self, trace_path: Path) -> io.TextIOWrapper:
        trace_file = trace_path.open("x", encoding="utf-8")
        with self.lock:
            self.open_traces += 1
            if self.thread is None:
                # A session that never closes has no lines left that it waits for.
                self.thread = threading.Thread(
                    target=self.run, name="trace-writer", daemon=True
                )
                self.thread.start()
            else:
                pass
        return trace_file

    def write_line(self, trace_file: io.TextIOWrapper, line: str) -> None:
        self.pending.put(TraceLine(trace_file=trace_file, line=line))

    def write_receipts(
        self,
        trace_file: io.TextIOWrapper,
        receipts_path: Path,
        receipts: list[SendReceipt],
    ) -> None:
        self.pending.put(
            ReceiptsWrite(
                trace_file=trace_file, receipts_path=receipts_path, receipts=receipts
            )
        )

    async def close(self, trace_file: io.TextIOWrapper) -> None:
        """Returns once the session's files are on disk; raises their write error."""
        loop = asyncio.get_running_loop()
        closed: asyncio.Future[None] = loop.create_future()
        self.pending.put(TraceClose(trace_file=trace_file, loop=loop, closed=closed))
        self.close_requested.set()
        await closed

    def run(self) -> None:
        failures: dict[io.TextIOWrapper, OSError] = {}
        while True:
            if self.close_requested.wait(TRACE_FLUSH_INTERVAL_S):
                time.sleep(TRACE_CLOSE_BATCH_S)
            else:
                pass
            self.close_requested.clear()
            batches: dict[io.TextIOWrapper, list[str]] = {}
            receipts_writes: list[ReceiptsWrite] = []
            closes: list[TraceClose] = []
            while not self.pending.empty():
                request = self.pending.get_nowait()
                if isinstance(request, TraceLine):
                    batches.setdefault(request.trace_file, []).append(request.line)
                elif isinstance(request, ReceiptsWrite):
                    receipts_writes.append(request)
                else:
                    closes.append(request)
            for trace_file, lines in batches.items():
                if trace_file not in failures:
                    try:
                        trace_file.write("".join(lines))
                    except OSError as exc:
                        failures[trace_file] = exc
                else:
                    pass
            for receipts_write in receipts_writes:
                try:
                    # note (kangwenqiao): Compact, because the indented dump is
                    # pure Python and holds the GIL from the pacing loop.
                    receipts_write.receipts_path.write_text(
                        json.dumps(
                            {"appends": receipts_write.receipts}, allow_nan=False
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                except OSError as exc:
                    failures.setdefault(receipts_write.trace_file, exc)
            for close_request in closes:
                try:
                    close_request.trace_file.close()
                except OSError as exc:
                    failures.setdefault(close_request.trace_file, exc)
                try:
                    close_request.loop.call_soon_threadsafe(
                        settle_close,
                        close_request.closed,
                        failures.pop(close_request.trace_file, None),
                    )
                except RuntimeError:
                    # A loop torn down mid-session has nobody left to wake.
                    pass
            with self.lock:
                self.open_traces -= len(closes)
                if self.open_traces == 0 and self.pending.empty():
                    self.thread = None
                    return
                else:
                    pass


TRACE_WRITER = TraceWriter()


@contextlib.asynccontextmanager
async def open_trace(trace_path: Path) -> AsyncIterator[io.TextIOWrapper]:
    """Trace file written by TRACE_WRITER; complete on exit, or raises its write error."""
    trace_file = TRACE_WRITER.open(trace_path)
    try:
        yield trace_file
    finally:
        await TRACE_WRITER.close(trace_file)


async def run_session(
    url: str,
    pcm: bytes,
    *,
    trace_path: Path,
    timeout_s: float = 90.0,
    profile: ProfileName = DEFAULT_PROFILE,
    turn_detection: TurnDetection | None = None,
    tail_s: float = LEGACY_TAIL_S,
) -> None:
    """Save observations and failures; classification belongs to offline replay.

    turn_detection configures a legacy-protocol or turn-based session; tail_s is the
    delay after the input end before a legacy-protocol session's liveness probe.
    """
    if not pcm or len(pcm) % 2:
        raise ValueError("Input must be nonempty PCM16")
    else:
        pass
    protocol = profile_protocol(profile, turn_detection)
    legacy = protocol == "legacy"
    requested_type = turn_detection.type if turn_detection is not None else None
    configure: dict[str, JsonValue]
    if turn_detection is None:
        configure = {"output_modalities": ["audio"]}
    elif legacy:
        configure = {
            "modalities": ["text", "audio"],
            "turn_detection": turn_detection.model_dump(mode="json"),
        }
    else:
        configure = {
            "output_modalities": ["audio"],
            "audio": {
                "input": {"turn_detection": turn_detection.model_dump(mode="json")}
            },
        }
    input_s = len(pcm) / (2 * SAMPLE_RATE)

    receipts: list[SendReceipt] = []
    async with open_trace(trace_path) as trace_file:

        def record(
            direction: Literal["send", "receive", "error", "admission"],
            event: dict[str, JsonValue],
        ) -> float:
            observed_time_s = time.perf_counter()
            TRACE_WRITER.write_line(
                trace_file,
                json.dumps(
                    {
                        "direction": direction,
                        "time_s": observed_time_s,
                        "event": event,
                    },
                    allow_nan=False,
                )
                + "\n",
            )
            return observed_time_s

        # note (edwingao28): Embedding the frame saves a second encode on the pacing
        # loop; a non-ASCII or multi-line frame is encoded again to keep ASCII lines.
        def record_frame(
            direction: Literal["send", "receive"],
            frame: str,
            event: dict[str, JsonValue],
            observed_time_s: float,
        ) -> None:
            if not frame.isascii() or "\n" in frame or "\r" in frame:
                line = json.dumps(
                    {
                        "direction": direction,
                        "time_s": observed_time_s,
                        "event": event,
                    },
                    allow_nan=False,
                )
            else:
                line = (
                    f'{{"direction": "{direction}", "time_s": {observed_time_s!r}, '
                    f'"event": {frame}}}'
                )
            TRACE_WRITER.write_line(trace_file, line + "\n")

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
            async with await admit() as websocket:
                seen: dict[str, asyncio.Event] = {}
                latest: dict[str, dict[str, JsonValue]] = {}
                aborted = asyncio.Event()
                fatal = False

                async def send(
                    event_type: Literal[
                        "session.update",
                        "input_audio_buffer.append",
                        "sglang.input_audio.end",
                        "session.close",
                    ],
                    *,
                    session: dict[str, JsonValue] | None = None,
                    audio: str | None = None,
                    sglang: InputAudioMetadata | None = None,
                ) -> None:
                    event_id = uuid.uuid4().hex
                    event: dict[str, JsonValue] = {
                        "type": event_type,
                        "event_id": event_id,
                    }
                    if session is not None:
                        event["session"] = session
                    else:
                        pass
                    if audio is not None:
                        event["audio"] = audio
                    else:
                        pass
                    if sglang is not None:
                        event["sglang"] = sglang
                    else:
                        pass
                    send_started_s = time.perf_counter()
                    frame = json.dumps(event, allow_nan=False)
                    record_frame("send", frame, event, send_started_s)
                    await websocket.send(frame)
                    if event_type == "input_audio_buffer.append":
                        assert sglang is not None
                        receipts.append(
                            {
                                "event_id": event_id,
                                "seq": sglang["seq"],
                                "start_s": send_started_s,
                                "completed_s": time.perf_counter(),
                            }
                        )
                    else:
                        pass

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
                    async for frame in websocket:
                        received_s = time.perf_counter()
                        try:
                            text = frame if isinstance(frame, str) else frame.decode()
                            event = SERVER_EVENT_DECODER.decode(text)
                            if not isinstance(event, dict):
                                raise ValueError("server event must be a JSON object")
                            else:
                                pass
                            event_type = event.get("type")
                            if not isinstance(event_type, str):
                                raise ValueError("server event type must be a string")
                            else:
                                pass
                            record_frame("receive", text, event, received_s)
                        except ValueError as exc:
                            raise ValueError(
                                "malformed server frame "
                                f"{frame[:FRAME_EXCERPT_CHARS]!r}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        latest[event_type] = event
                        seen.setdefault(event_type, asyncio.Event()).set()
                        if event_type == "error":
                            # note (wenyao): Close waits for adapter teardown.
                            extension = event.get("sglang")
                            fatal = fatal or bool(
                                isinstance(extension, dict) and extension.get("fatal")
                            )
                            aborted.set()
                        else:
                            pass
                    if not seen.get("session.closed", asyncio.Event()).is_set():
                        raise RuntimeError("Connection closed without session.closed")
                    else:
                        pass

                async def drive() -> None:
                    streamed = False
                    if await settle("session.created"):
                        await send("session.update", session=configure)
                    else:
                        pass
                    if await settle("session.updated"):
                        # note (luojiaxuan): The legacy facade substitutes server_vad
                        # when no Smart Turn model is loaded, so the echo is the only
                        # evidence of which detector ran.
                        echoed = (
                            turn_detection_type(
                                latest["session.updated"].get("session"), protocol
                            )
                            if turn_detection is not None
                            else None
                        )
                        if echoed != requested_type:
                            raise RuntimeError(
                                f"session.updated echoed turn_detection.type {echoed!r}, "
                                f"requested {requested_type!r}"
                            )
                        else:
                            pass
                        streamed = True
                        for sequence, byte_offset in enumerate(
                            range(0, len(pcm), PACKET_BYTES)
                        ):
                            # note (edwingao28): Replay measures pacing from the
                            # first append's send, so the deadlines start there too.
                            if receipts:
                                await asyncio.sleep(
                                    max(
                                        0.0,
                                        receipts[0]["start_s"]
                                        + sequence * PACKET_MS / 1000
                                        - time.perf_counter(),
                                    )
                                )
                            else:
                                pass
                            if aborted.is_set():
                                streamed = False
                                break
                            else:
                                pass
                            await send(
                                "input_audio_buffer.append",
                                audio=base64.b64encode(
                                    pcm[byte_offset : byte_offset + PACKET_BYTES]
                                ).decode("ascii"),
                                sglang={
                                    "seq": sequence,
                                    "t_start_ms": byte_offset / 2 / SAMPLE_RATE * 1000,
                                },
                            )
                    else:
                        pass
                    if streamed:
                        await asyncio.sleep(
                            max(
                                0.0,
                                receipts[0]["start_s"]
                                + input_s
                                + (tail_s if legacy else 0.0)
                                - time.perf_counter(),
                            )
                        )
                        if aborted.is_set():
                            pass
                        elif legacy:
                            # note (luojiaxuan): The facade acknowledges no other
                            # client event after the input, so a no-op update is the
                            # liveness proof at the end of the window.
                            seen["session.updated"].clear()
                            await send("session.update", session={})
                            await settle("session.updated")
                        else:
                            await send("sglang.input_audio.end")
                            await settle("sglang.input_audio.drained")
                    else:
                        pass
                    if legacy:
                        return
                    else:
                        pass
                    closed = seen.setdefault("session.closed", asyncio.Event())
                    if not fatal and not closed.is_set():
                        await send("session.close")
                    else:
                        pass
                    await closed.wait()

                receiver = asyncio.create_task(receive())
                driver = asyncio.create_task(drive())
                try:
                    done, _ = await asyncio.wait(
                        {receiver, driver}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if receiver in done:
                        await receiver
                        # note (wenyao): No receipts can arrive after receiver exit.
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
            TRACE_WRITER.write_receipts(
                trace_file, trace_path.with_name(SEND_RECEIPTS_FILE), receipts
            )
