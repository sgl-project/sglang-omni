# SPDX-License-Identifier: Apache-2.0
"""One conversation with a cascade that carries a speech-avatar node (speech_video: jpeg_frame, or speech_motion: smplx_motion), over /v1/realtime or /v1/session.

Adapted from avtr_avatar_e2e.py. Streams a 16 kHz mono WAV as the user's
audio in real time, then silence for --tail-s, records when each reply audio
chunk and each plugin chunk arrives, and checks: the plugin media are JPEGs of
the port's size (video) or whole smplx_motion rows (motion); the media of a
reply cover its audio within one chunk (plus the stage's tail); their
t_start_ms join without a gap or overlap but where an interrupted turn's
frames were dropped or taken back; some
reply was heard. With --barge-in-s the question is spoken again that long
after the first reply audio arrives: the interrupted reply's media must stop
within two chunks of the interrupt reaching the client.

It reports:
- motion: the latency of each block, from the input packet that makes it
  ready leaving the client to the block arriving (a video chunk renders once
  its reply audio reached the node, which the client cannot see, as the
  server paces the audio it sends; the worker logs each render's time);
- the first plugin chunk's latency from the first input packet;
- under the playground's hold policy, how long each reply's audio waits for
  its first picture (frames play on t_start_ms from the first chunk's arrival
  plus --playout-delay-ms, at most reply_media_wait_ms), how late later media
  are for it, and under free how far the picture trails the voice.

    python -m benchmarks.eval.speech_avatar_e2e --url ws://127.0.0.1:9467 \\
        --protocol realtime --audio question.wav --output realtime.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import msgpack
import numpy as np
from PIL import Image

from benchmarks.eval.avtr_avatar_e2e import PACKET_MS, packets, read_wav, stats
from sglang_omni.models.speech_avatar.formats import (
    JPEG_FRAME,
    SMPLX_FEATURES_PER_FRAME,
    SMPLX_MOTION,
)
from sglang_omni.proto.streams import TURN_INTERRUPTED

FORMATS = (JPEG_FRAME, SMPLX_MOTION)
# The playground's default playout delay (reply_media_gate.js DEFAULT_PLAYOUT_DELAY_MS).
PLAYOUT_DELAY_MS = 100
# Frames of one rendered chunk arrive together; a new chunk starts when the next frame comes this much later.
CHUNK_ARRIVAL_GAP_S = 0.02


class Record:
    """Send times of the input packets, arrival times of reply audio, plugin chunks and interrupts, per reply."""

    def __init__(self, playout_delay_ms: float, tail_ms: float) -> None:
        self.began = time.perf_counter()
        self.sent: dict[int, float] = {}
        self.barge_in_s: float | None = None
        self.audio: dict[object, list[tuple[float, float]]] = defaultdict(list)
        # (arrival, reply, t_start_ms, duration_ms, payload)
        self.media: list[tuple[float, object, float, float, bytes]] = []
        self.cancelled: dict[object, float] = {}
        self.port: dict | None = None
        self.errors: list[str] = []
        self.playout_delay_ms = playout_delay_ms
        self.tail_ms = tail_ms

    @property
    def params(self) -> dict:
        return self.port["params"] if self.port else {}

    @property
    def is_video(self) -> bool:
        return bool(self.port) and self.port["format"] == JPEG_FRAME

    def add_audio(self, reply: object, seconds: float) -> None:
        self.audio[reply].append((time.perf_counter(), seconds))

    def add_media(
        self, reply: object, t_start_ms: float, duration_ms: float, payload: bytes
    ) -> None:
        self.media.append(
            (time.perf_counter(), reply, t_start_ms, duration_ms, payload)
        )

    def add_cancel(self, reply: object) -> None:
        self.cancelled.setdefault(reply, time.perf_counter())

    def video_chunks(self) -> list[list[tuple[float, object, float, float, bytes]]]:
        """The rendered (non-empty) frames grouped into the chunks they came in."""
        chunks: list[list] = []
        for frame in sorted(
            (m for m in self.media if m[4]), key=lambda m: (m[0], m[2])
        ):
            if chunks and frame[0] - chunks[-1][0][0] < CHUNK_ARRIVAL_GAP_S:
                chunks[-1].append(frame)
            else:
                chunks.append([frame])
        return chunks

    def chunk_latency(self) -> list[float]:
        """Motion only: from the input packet that makes a block ready leaving to the block arriving. A video chunk renders as soon as its reply audio reached the node, which the client cannot see (the server paces the audio it sends), so its render time is read from the worker's log instead."""
        latency = []
        if not self.is_video:
            for arrived, _, t_ms, duration_ms, _ in self.media:
                ready_packet = math.ceil((t_ms + duration_ms) / PACKET_MS - 1e-6) - 1
                if ready_packet in self.sent:
                    latency.append((arrived - self.sent[ready_packet]) * 1000)
                else:
                    pass
        else:
            pass
        return latency

    def summary(self) -> dict:
        media = self.media
        wait_ms = float(self.params.get("reply_media_wait_ms", 1000))
        chunk_ms = float(self.params.get("chunk_ms", 200))
        # The playground plays each chunk at its t_start_ms, counted from the first chunk's arrival plus the playout delay, or when it arrives if later.
        origin = (
            media[0][0] + self.playout_delay_ms / 1000 - media[0][2] / 1000
            if media
            else 0.0
        )
        shown = [max(m[0], origin + m[2] / 1000) for m in media]
        replies = {}
        held_ms, late_hold, behind_free = [], [], []
        replies_past_wait = 0
        for reply, audio in self.audio.items():
            own = [(m, shown[i]) for i, m in enumerate(media) if m[1] == reply]
            audio_s = sum(seconds for _, seconds in audio)
            media_s = sum(m[3] for m, _ in own) / 1000
            replies[str(reply)] = {
                "audio_s": round(audio_s, 3),
                "media_s": round(media_s, 3),
                "first_audio_s": round(audio[0][0] - self.began, 3),
                "first_media_s": round(own[0][0][0] - self.began, 3) if own else None,
            }
            if not own:
                continue
            else:
                first_audio = audio[0][0]
                start = min(own[0][1], first_audio + wait_ms / 1000)
                replies_past_wait += own[0][1] > start
            held_ms.append((start - first_audio) * 1000)
            for (arrived, _, t_ms, _, _), at in own:
                offset = (t_ms - own[0][0][2]) / 1000
                late_hold.append((arrived - (start + offset)) * 1000)
                behind_free.append((at - (first_audio + offset)) * 1000)
        ordered = sorted(media, key=lambda m: m[2])
        gaps = [
            (later[2] - earlier[2] - earlier[3], earlier[1])
            for earlier, later in zip(ordered, ordered[1:])
        ]
        interrupted = (
            list(self.audio)[0] if self.barge_in_s is not None and self.audio else None
        )
        heard = [r for r in replies.values() if r["audio_s"] > 0]
        barge_in = None
        if self.barge_in_s is not None and self.audio:
            first, *later = list(self.audio)
            cut = [m[0] for m in media if m[1] == first]
            cancel = self.cancelled.get(first)
            barge_in = {
                "interrupted_reply_last_media_after_barge_in_s": (
                    round(max(cut) - self.barge_in_s, 3) if cut else None
                ),
                "interrupt_reached_client_after_barge_in_s": (
                    round(cancel - self.barge_in_s, 3) if cancel else None
                ),
                "interrupted_reply_last_media_after_interrupt_s": (
                    round(max(cut) - cancel, 3) if cut and cancel else None
                ),
            }
            heard = [replies[str(r)] for r in later if replies[str(r)]["audio_s"] > 0]
        else:
            pass
        payloads = [m[4] for m in media if m[4]]
        if self.is_video:
            size = Image.open(io.BytesIO(payloads[0])).size if payloads else None
            media_ok = bool(payloads) and size == (
                self.params.get("width"),
                self.params.get("height"),
            )
        else:
            size = None
            media_ok = bool(payloads) and all(
                len(p) % (4 * SMPLX_FEATURES_PER_FRAME) == 0 for p in payloads
            )
        latency = self.chunk_latency()
        first_media = (
            round(media[0][0] - self.sent[0], 3) if media and 0 in self.sent else None
        )
        duration_s = (media[-1][0] - media[0][0]) if media else 0
        checks = {
            "reply_heard": bool(heard),
            "media_of_the_port_format": media_ok,
            "reply_media_cover_reply_audio": bool(heard)
            and all(
                r["media_s"] >= r["audio_s"] - 0.1
                and r["media_s"] - r["audio_s"] < (chunk_ms + self.tail_ms) / 1000 + 0.1
                for r in heard
            ),
            "stamps_contiguous_but_the_dropped_interrupted_turn": all(
                abs(gap) < 0.5 or reply == interrupted for gap, reply in gaps
            ),
            "no_errors": not self.errors,
        }
        if barge_in is not None:
            last = barge_in["interrupted_reply_last_media_after_interrupt_s"]
            checks["interrupted_reply_media_stop_within_two_chunks"] = (
                last is None or last < 2 * chunk_ms / 1000
            )
        else:
            pass
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "port": self.port,
            "chunks": len(media),
            "rendered_chunks": (
                len(self.video_chunks()) if self.is_video else len(payloads)
            ),
            "frame_size": size,
            "payload_bytes_mean": (
                float(np.mean([len(p) for p in payloads])) if payloads else None
            ),
            "media_kbit_s": (
                round(sum(len(p) for p in payloads) * 8 / 1000 / duration_s, 1)
                if duration_s
                else None
            ),
            "first_media_latency_s": first_media,
            "chunk_latency_ms": stats(latency),
            "replies": replies,
            "media_pairs": len(gaps),
            "playout_delay_ms": self.playout_delay_ms,
            "hold": {
                "reply_audio_held_ms": stats(held_ms),
                "replies_past_reply_media_wait": replies_past_wait,
                "media_late_for_audio_ms": stats(late_hold),
            },
            "free": {"picture_behind_voice_ms": stats(behind_free)},
            "errors": self.errors,
            "barge_in": barge_in,
            "raw": {
                "began": self.began,
                "sent": {str(k): v for k, v in self.sent.items()},
                "audio": {str(k): v for k, v in self.audio.items()},
                "media": [[m[0], str(m[1]), m[2], m[3], len(m[4])] for m in media],
                "cancelled": {str(k): v for k, v in self.cancelled.items()},
                "barge_in_s": self.barge_in_s,
            },
        }


async def send_paced(
    send, question: list[bytes], tail_s: float, record: Record, barge_in_after_s
) -> None:
    """The question, then silence for tail_s; with barge_in_after_s, the question again that long after the first reply audio arrived."""
    queue = question + [bytes(len(question[0]))] * int(tail_s * 1000 / PACKET_MS)
    began = time.perf_counter()
    index = 0
    while index < len(queue):
        await asyncio.sleep(
            max(began + (index + 1) * PACKET_MS / 1000 - time.perf_counter(), 0)
        )
        first_audio = min(
            (audio[0][0] for audio in record.audio.values()), default=None
        )
        if (
            barge_in_after_s is not None
            and record.barge_in_s is None
            and first_audio is not None
            and time.perf_counter() >= first_audio + barge_in_after_s
        ):
            record.barge_in_s = time.perf_counter()
            queue[index:index] = question
        else:
            pass
        await send(index, queue[index], index == len(queue) - 1)
        record.sent[index] = time.perf_counter()
        index += 1


async def realtime(url: str, stream, record: Record, settle_s: float) -> None:
    async with (
        aiohttp.ClientSession() as http,
        http.ws_connect(f"{url}/v1/realtime", max_msg_size=0) as ws,
    ):
        output_rate = None

        async def send(index: int, packet: bytes, last: bool) -> None:
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"evt_append_{index}",
                    "sglang": {"seq": index, "t_start_ms": index * PACKET_MS},
                    "audio": base64.b64encode(packet).decode(),
                }
            )

        async def receive() -> None:
            nonlocal output_rate
            async for message in ws:
                event = json.loads(message.data)
                kind = event["type"]
                if kind == "session.created":
                    await ws.send_json(
                        {
                            "type": "session.update",
                            "event_id": "evt_session_update",
                            "session": {"output_modalities": ["audio"]},
                        }
                    )
                elif kind == "session.updated":
                    output_rate = event["session"]["sglang"]["granted"][
                        "output_audio_format"
                    ]["rate"]
                    ports = [
                        port
                        for port in event["session"]["sglang"].get(
                            "plugin_output_ports", []
                        )
                        if port["format"] in FORMATS
                    ]
                    if len(ports) != 1 or "chunk_ms" not in ports[0]["params"]:
                        record.errors.append(f"session advertises {ports}")
                    else:
                        record.port = ports[0]
                    ready.set()
                elif kind == "response.output_audio.delta":
                    record.add_audio(
                        event.get("response_id"),
                        len(base64.b64decode(event["delta"])) / 2 / output_rate,
                    )
                elif kind == "response.done" and (
                    event["response"].get("status") == "cancelled"
                ):
                    record.add_cancel(event["response"].get("id"))
                elif kind == "sglang.output_chunk" and event["format"] in FORMATS:
                    record.add_media(
                        event.get("response_id"),
                        event["t_start_ms"],
                        event["duration_ms"],
                        base64.b64decode(event["payload_b64"]),
                    )
                elif kind == "error":
                    record.errors.append(json.dumps(event))
                else:
                    pass

        ready = asyncio.Event()
        reader = asyncio.create_task(receive())
        await asyncio.wait_for(ready.wait(), 60)
        await stream(send)
        await asyncio.sleep(settle_s)
        await ws.close()
        await reader


async def session(url: str, stream, record: Record, settle_s: float) -> None:
    async with aiohttp.ClientSession() as http:
        async with http.get(f"{url}/v1/realtime/capabilities") as answer:
            ports = [
                port
                for port in (await answer.json()).get("plugin_output_ports", [])
                if port["format"] in FORMATS
            ]
            record.port = ports[0] if len(ports) == 1 else None
        async with http.ws_connect(f"{url}/v1/session", max_msg_size=0) as ws:

            async def send(index: int, packet: bytes, last: bool) -> None:
                chunk = {
                    "modality": "audio",
                    "t_start_ms": float(index * PACKET_MS),
                    "duration_ms": float(PACKET_MS),
                    "seq": index,
                    "payload": packet,
                    "format": "pcm16",
                    "eos": last,
                    "turn": None,
                }
                await ws.send_bytes(
                    msgpack.packb({"op": "append", "chunk": chunk}, use_bin_type=True)
                )

            async def receive() -> None:
                async for message in ws:
                    frame = msgpack.unpackb(message.data, raw=False)
                    if frame["op"] == "output":
                        chunk = frame["chunk"]
                        if chunk["format"] == "pcm_f32le" and chunk["payload"]:
                            record.add_audio(chunk["turn"], chunk["duration_ms"] / 1000)
                        elif chunk["format"] in FORMATS:
                            record.add_media(
                                chunk["turn"],
                                chunk["t_start_ms"],
                                chunk["duration_ms"],
                                chunk["payload"],
                            )
                        elif (
                            isinstance(chunk["payload"], dict)
                            and chunk["payload"].get("type") == TURN_INTERRUPTED
                        ):
                            record.add_cancel(chunk["turn"])
                        else:
                            pass
                    elif frame["op"] in ("error", "rejected"):
                        record.errors.append(json.dumps(frame, default=str))
                    elif frame["op"] == "closed":
                        return
                    else:
                        pass

            await ws.send_bytes(
                msgpack.packb({"op": "open", "params": {}}, use_bin_type=True)
            )
            opened = msgpack.unpackb((await ws.receive()).data, raw=False)
            if opened["op"] != "opened":
                record.errors.append(json.dumps(opened, default=str))
                return
            else:
                pass
            reader = asyncio.create_task(receive())
            await stream(send)
            await asyncio.sleep(settle_s)
            await ws.send_bytes(msgpack.packb({"op": "close"}, use_bin_type=True))
            await asyncio.wait_for(reader, 120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--protocol", choices=("realtime", "session"), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--tail-s", type=float, default=15.0)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--playout-delay-ms", type=float, default=PLAYOUT_DELAY_MS)
    parser.add_argument(
        "--stage-tail-ms",
        type=float,
        default=0.0,
        help="the speech_video stage's tail_ms: its chunks run that long past a reply",
    )
    parser.add_argument(
        "--barge-in-s",
        type=float,
        help="speak the question again this long after the first reply audio arrives",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = Record(args.playout_delay_ms, args.stage_tail_ms)
    run = realtime if args.protocol == "realtime" else session
    question = packets(read_wav(args.audio))

    async def stream(send) -> None:
        await send_paced(send, question, args.tail_s, record, args.barge_in_s)

    asyncio.run(run(args.url, stream, record, args.settle_s))
    result = {"protocol": args.protocol, **record.summary()}
    args.output.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {key: result[key] for key in result if key not in ("replies", "raw")},
            indent=1,
        )
    )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
