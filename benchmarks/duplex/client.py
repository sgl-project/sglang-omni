# SPDX-License-Identifier: Apache-2.0
"""Record native full-duplex, legacy or turn-based sessions over a real WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

import websockets
from pydantic import BaseModel, ConfigDict, JsonValue

from benchmarks.duplex.oracle_models import turn_detection_type
from benchmarks.duplex.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    PlayoutCut,
    ProfileName,
    Protocol,
)

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
# note (luojiaxuan): An end-of-input rule two compared clients can share: tail_silence_s of
# zero PCM16 streamed after the input at the input's pace, the input end after that tail,
# then at most end_wait_s for the server to finish before the close. A reply counts as
# finished only if its response.done arrived before that input end.
END_OF_INPUT_RULE = (
    "tail_silence_s of zero PCM16 appended after the input, paced by media time; the "
    "input end (EOS) after the tail; then at most end_wait_s for the server to finish "
    "(sglang.input_audio.drained) before session.close; a reply is finished only if its "
    "response.done arrived before the input end"
)
PLAYBACK_REPORT = (
    "conversation.item.truncate on each response.done with status cancelled, with "
    "audio_end_ms the part of that reply a zero-buffer player had played at its "
    "receipt: each audio delta plays from the later of its arrival and the end of "
    "the previous delta, and the cut drops what is still unplayed"
)
OPENAI_REALTIME_PLAYBACK_REPORT = (
    "conversation.item.truncate at most once per item, with audio_end_ms the part of "
    "that item a zero-buffer player had played at the cut: at "
    "input_audio_buffer.speech_started, unless turn_detection.interrupt_response is "
    "false, for each reply with audio still unplayed, and at a response.done with "
    "status cancelled for that reply; each audio delta plays from the later of its "
    "arrival and the end of the previous delta, and a cut drops what is still unplayed"
)
PLAYBACK_REPORTS: dict[frozenset[PlayoutCut], str] = {
    PROFILES["turn-based"].playout_cuts: PLAYBACK_REPORT,
    PROFILES["openai-realtime"].playout_cuts: OPENAI_REALTIME_PLAYBACK_REPORT,
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


def session_cuts(
    cuts: frozenset[PlayoutCut], turn_detection: JsonValue
) -> frozenset[PlayoutCut]:
    """A profile's playout cuts in a session configured with this turn_detection
    object; speech_started cuts only when the server interrupts a reply on speech."""
    settings = turn_detection if isinstance(turn_detection, dict) else {}
    if settings.get("interrupt_response") is False:
        return cuts - {"input_audio_buffer.speech_started"}
    else:
        return cuts


def stops_playout(event: dict[str, JsonValue], cuts: frozenset[PlayoutCut]) -> bool:
    event_type = event.get("type")
    return event_type in cuts and (
        event_type != "response.done"
        or (event.get("response") or {}).get("status") == "cancelled"
    )


@dataclass
class ZeroBufferPlayer:
    """The player PLAYBACK_REPORTS describe: when its cursor frees, each audio
    item's reply and (start, seconds) deltas, and the items already truncated.

    One reply can carry several audio items (gpt-realtime-2.1 speaks a commentary
    item before its final answer), so played audio is counted per item."""

    output_rate: int
    cuts: frozenset[PlayoutCut]
    cursor_s: float = 0.0
    item_audio: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    item_replies: dict[str, str] = field(default_factory=dict)
    truncated_items: set[str] = field(default_factory=set)

    def played(
        self, event: dict[str, JsonValue], now_s: float
    ) -> list[dict[str, JsonValue]]:
        """Play an audio delta, or stop at a playout cut and return the
        conversation.item.truncate fields of each item it cuts: the items of the reply
        a cancelled response.done names, or at any other cut every item with audio
        unplayed."""
        event_type = event["type"]
        if event_type == "response.output_audio.delta":
            seconds = len(base64.b64decode(event["delta"])) / 2 / self.output_rate
            start_s = max(self.cursor_s, now_s)
            self.cursor_s = start_s + seconds
            self.item_audio.setdefault(event["item_id"], []).append(
                (start_s, seconds)
            )
            self.item_replies[event["item_id"]] = event["response_id"]
            return []
        elif stops_playout(event, self.cuts):
            self.cursor_s = min(self.cursor_s, now_s)
            cut_items = [
                item_id
                for item_id, deltas in self.item_audio.items()
                if (
                    self.item_replies[item_id] == event["response"]["id"]
                    if event_type == "response.done"
                    else any(start_s + seconds > now_s for start_s, seconds in deltas)
                )
            ]
            truncates: list[dict[str, JsonValue]] = []
            for item_id in cut_items:
                if item_id is None or item_id in self.truncated_items:
                    continue
                else:
                    pass
                self.truncated_items.add(item_id)
                played_s = sum(
                    min(seconds, max(0.0, now_s - start_s))
                    for start_s, seconds in self.item_audio[item_id]
                )
                truncates.append(
                    {
                        "item_id": item_id,
                        "content_index": 0,
                        "audio_end_ms": int(played_s * 1000),
                    }
                )
            return truncates
        else:
            return []


class InputAudioMetadata(TypedDict):
    seq: int
    t_start_ms: float


class SendReceipt(TypedDict):
    event_id: str
    seq: int
    start_s: float
    completed_s: float


async def run_session(
    url: str,
    pcm: bytes,
    *,
    trace_path: Path,
    timeout_s: float = 90.0,
    profile: ProfileName = DEFAULT_PROFILE,
    turn_detection: TurnDetection | None = None,
    tail_s: float = LEGACY_TAIL_S,
    report_playback: bool = False,
    tail_silence_s: float = 0.0,
    end_wait_s: float | None = None,
) -> None:
    """Save observations and failures; classification belongs to offline replay.

    turn_detection configures a legacy-protocol or turn-based session; tail_s is the
    delay after the input end before a legacy-protocol session's liveness probe;
    report_playback makes a turn-based session report each cut reply's played
    audio as PLAYBACK_REPORTS describes; tail_silence_s and end_wait_s set the
    end of input as END_OF_INPUT_RULE describes (end_wait_s None: only the session
    timeout bounds the drain).
    """
    if PROFILES[profile].wire != "sglang":
        # note (wenyao): openai_client builds on this module, so it loads on dispatch.
        from benchmarks.duplex.openai_client import run_openai_session

        return await run_openai_session(
            url,
            pcm,
            trace_path=trace_path,
            timeout_s=timeout_s,
            profile=profile,
            turn_detection=turn_detection,
            report_playback=report_playback,
            tail_silence_s=tail_silence_s,
            end_wait_s=end_wait_s,
        )
    else:
        pass
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
    tail_bytes = round(tail_silence_s * SAMPLE_RATE) * 2
    input_s = (len(pcm) + tail_bytes) / (2 * SAMPLE_RATE)
    # note (luojiaxuan): The tail starts its own packets, so the input's appends stay
    # exactly as without it.
    packet_offsets = [*range(0, len(pcm), PACKET_BYTES)] + [
        *range(len(pcm), len(pcm) + tail_bytes, PACKET_BYTES)
    ]
    stream = pcm + bytes(tail_bytes)
    output_rate = PROFILES[profile].output_sample_rate

    receipts: list[SendReceipt] = []
    with trace_path.open("x", encoding="utf-8", buffering=1) as trace_file:

        def record(
            direction: Literal["send", "receive", "error", "admission"],
            event: dict[str, JsonValue],
        ) -> float:
            observed_time_s = time.perf_counter()
            trace_file.write(
                json.dumps(
                    {
                        "direction": direction,
                        "time_s": observed_time_s,
                        "event": event,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            return observed_time_s

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
                        "conversation.item.truncate",
                        "session.close",
                    ],
                    *,
                    session: dict[str, JsonValue] | None = None,
                    audio: str | None = None,
                    sglang: InputAudioMetadata | None = None,
                    truncate: dict[str, JsonValue] | None = None,
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
                    if truncate is not None:
                        event.update(truncate)
                    else:
                        pass
                    send_started_s = record("send", event)
                    await websocket.send(json.dumps(event))
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

                player = ZeroBufferPlayer(
                    output_rate,
                    session_cuts(
                        PROFILES[profile].playout_cuts,
                        (
                            turn_detection.model_dump(mode="json")
                            if turn_detection is not None
                            else None
                        ),
                    ),
                )

                async def receive() -> None:
                    nonlocal fatal
                    async for frame in websocket:
                        try:
                            event = json.loads(frame)
                            if not isinstance(event, dict):
                                raise ValueError("server event must be a JSON object")
                            else:
                                pass
                            event_type = event.get("type")
                            if not isinstance(event_type, str):
                                raise ValueError("server event type must be a string")
                            else:
                                pass
                            received_s = record("receive", event)
                        except ValueError as exc:
                            raise ValueError(
                                "malformed server frame "
                                f"{frame[:FRAME_EXCERPT_CHARS]!r}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        latest[event_type] = event
                        seen.setdefault(event_type, asyncio.Event()).set()
                        for truncate in (
                            player.played(event, received_s) if report_playback else []
                        ):
                            await send("conversation.item.truncate", truncate=truncate)
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
                        start_s = time.perf_counter()
                        for sequence, byte_offset in enumerate(packet_offsets):
                            packet_end = (
                                len(pcm)
                                if byte_offset < len(pcm)
                                else len(stream)
                            )
                            await asyncio.sleep(
                                max(
                                    0.0,
                                    start_s
                                    + byte_offset / (2 * SAMPLE_RATE)
                                    - time.perf_counter(),
                                )
                            )
                            if aborted.is_set():
                                streamed = False
                                break
                            else:
                                pass
                            await send(
                                "input_audio_buffer.append",
                                audio=base64.b64encode(
                                    stream[
                                        byte_offset : min(
                                            byte_offset + PACKET_BYTES, packet_end
                                        )
                                    ]
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
                            if end_wait_s is None:
                                await settle("sglang.input_audio.drained")
                            else:
                                try:
                                    await asyncio.wait_for(
                                        settle("sglang.input_audio.drained"),
                                        end_wait_s,
                                    )
                                except asyncio.TimeoutError:
                                    raise RuntimeError(
                                        f"input not drained within {end_wait_s}s "
                                        "of the input end"
                                    ) from None
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
            trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
                json.dumps({"appends": receipts}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
