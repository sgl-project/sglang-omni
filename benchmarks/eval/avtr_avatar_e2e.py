# SPDX-License-Identifier: Apache-2.0
"""One conversation with a cascade that carries the AVTR-1 avatar node, over /v1/realtime or /v1/session.

Streams a 16 kHz mono WAV as the user's audio in real time, then silence
for --tail-s, records when each reply audio chunk and each jpeg_frame
arrives, and checks that frames came as JPEGs of the port's size, one per
40 ms, that the frames carrying a reply cover its audio (within one 200 ms
chunk) and that some reply was heard. With --barge-in-s the question is
spoken again that long after the first reply audio arrives: the
interrupted reply's frames must stop within two chunks of the interrupt
reaching the client, and only later replies are checked for coverage. The
frames' t_start_ms must join without a gap, except where the interrupted
turn's frames were dropped: the client keeps its last picture over that
span. Exits 0 when the checks hold.

It reports the latency of each chunk (the input packet that makes a chunk
ready leaving the client to the chunk's last frame arriving), the first
frame's latency, and what a client sees under both A/V sync policies of the
playground, which plays frames on their t_start_ms from the first frame's
arrival: under hold a reply's audio waits for the reply's first frame, at
most the port's reply_media_wait_ms (how long the audio was held, how late
later frames are for it); under free the audio plays on arrival (how far the
picture trails the voice). Both come from the same recording: the policy is
the client's, the server does not see it.

    python -m benchmarks.eval.avtr_avatar_e2e --url ws://127.0.0.1:8765 \\
        --protocol realtime --audio question.wav --output realtime.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import aiohttp
import msgpack
import numpy as np
from PIL import Image

from sglang_omni.models.avtr.formats import JPEG_FRAME, REPLY_MEDIA_WAIT_MS, VIDEO
from sglang_omni.proto.streams import TURN_INTERRUPTED

PACKET_MS = 200
FRAME_MS = 40
CHUNK_FRAMES = 5
CHUNK_MS = CHUNK_FRAMES * FRAME_MS


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        return source.readframes(source.getnframes())


def packets(pcm: bytes) -> list[bytes]:
    size = 16000 * 2 * PACKET_MS // 1000
    pcm = pcm + bytes(-len(pcm) % size)
    return [pcm[start : start + size] for start in range(0, len(pcm), size)]


def stats(values: list[float]) -> dict | None:
    return (
        {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "n": len(values),
        }
        if values
        else None
    )


class Record:
    """Send times of the input packets, arrival times of reply audio, frames and interrupts, per reply."""

    def __init__(self) -> None:
        self.began = time.perf_counter()
        self.sent: dict[int, float] = {}
        self.barge_in_s: float | None = None
        self.audio: dict[object, list[tuple[float, float]]] = defaultdict(list)
        # (arrival, reply, t_start_ms, payload)
        self.frames: list[tuple[float, object, float, bytes]] = []
        self.cancelled: dict[object, float] = {}
        self.port: dict | None = None
        self.errors: list[str] = []
        self.reply_media_wait_ms = REPLY_MEDIA_WAIT_MS

    def add_audio(self, reply: object, seconds: float) -> None:
        self.audio[reply].append((time.perf_counter(), seconds))

    def add_frame(self, reply: object, t_start_ms: float, payload: bytes) -> None:
        self.frames.append((time.perf_counter(), reply, t_start_ms, payload))

    def add_cancel(self, reply: object) -> None:
        self.cancelled.setdefault(reply, time.perf_counter())

    def chunk_latency(self) -> tuple[list[float], int]:
        """Per fully rendered chunk: from the packet that makes it ready leaving to its last frame arriving; and the number of held chunks."""
        chunks: dict[int, list[tuple[float, bytes]]] = defaultdict(list)
        for arrived, _, t_ms, payload in self.frames:
            chunks[int(t_ms // CHUNK_MS)].append((arrived, payload))
        latency, held = [], 0
        for index, frames in chunks.items():
            ready_packet = index * CHUNK_MS // PACKET_MS - 1
            if not all(payload for _, payload in frames):
                held += 1
            elif len(frames) == CHUNK_FRAMES and ready_packet in self.sent:
                latency.append(
                    (max(a for a, _ in frames) - self.sent[ready_packet]) * 1000
                )
            else:
                pass
        return latency, held

    def summary(self) -> dict:
        # The playground's view plays each frame at its t_start_ms, counted from the first frame's arrival, or when it arrives if later.
        origin = self.frames[0][0] - self.frames[0][2] / 1000 if self.frames else 0.0
        shown = [
            max(arrived, origin + t_ms / 1000) for arrived, _, t_ms, _ in self.frames
        ]
        replies = {}
        held_ms, late_hold, behind_free = [], [], []
        replies_past_wait = 0
        for reply, audio in self.audio.items():
            frames = [
                (frame, shown[index])
                for index, frame in enumerate(self.frames)
                if frame[1] == reply
            ]
            audio_s = sum(seconds for _, seconds in audio)
            replies[str(reply)] = {
                "audio_s": round(audio_s, 3),
                "frames": len(frames),
                "first_audio_s": round(audio[0][0] - self.began, 3),
                "first_frame_s": (
                    round(frames[0][0][0] - self.began, 3) if frames else None
                ),
            }
            if not frames:
                continue
            else:
                first_audio = audio[0][0]
                start = min(frames[0][1], first_audio + self.reply_media_wait_ms / 1000)
                replies_past_wait += frames[0][1] > start
            held_ms.append((start - first_audio) * 1000)
            for (arrived, _, t_ms, _), at in frames:
                offset = (t_ms - frames[0][0][2]) / 1000
                late_hold.append((arrived - (start + offset)) * 1000)
                behind_free.append((at - (first_audio + offset)) * 1000)
        ordered = sorted(self.frames, key=lambda frame: frame[2])
        gaps = [
            (later[2] - earlier[2] - FRAME_MS, earlier[1])
            for earlier, later in zip(ordered, ordered[1:])
        ]
        interrupted = (
            list(self.audio)[0] if self.barge_in_s is not None and self.audio else None
        )
        heard = [r for r in replies.values() if r["audio_s"] > 0]
        barge_in = None
        if self.barge_in_s is not None:
            first, *later = list(self.audio)
            cut = [frame[0] for frame in self.frames if frame[1] == first]
            cancel = self.cancelled.get(first)
            barge_in = {
                "interrupted_reply_last_frame_after_barge_in_s": (
                    round(max(cut) - self.barge_in_s, 3) if cut else None
                ),
                "interrupt_reached_client_after_barge_in_s": (
                    round(cancel - self.barge_in_s, 3) if cancel else None
                ),
                "interrupted_reply_last_frame_after_interrupt_s": (
                    round(max(cut) - cancel, 3) if cut and cancel else None
                ),
                "later_replies_frames": [replies[str(r)]["frames"] for r in later],
            }
            # The interrupted reply's audio arrived ahead of its playback, so only later replies must be covered.
            heard = [replies[str(r)] for r in later if replies[str(r)]["audio_s"] > 0]
        else:
            pass
        pictures = [payload for _, _, _, payload in self.frames if payload]
        size = Image.open(io.BytesIO(pictures[0])).size if pictures else None
        expected = (
            (self.port["params"]["width"], self.port["params"]["height"])
            if self.port
            else size
        )
        latency, held_chunks = self.chunk_latency()
        first_frame = (
            round(self.frames[0][0] - self.sent[0], 3)
            if self.frames and 0 in self.sent
            else None
        )
        duration_s = (self.frames[-1][0] - self.frames[0][0]) if self.frames else 0
        checks = {
            "reply_heard": bool(heard),
            "frames_are_jpegs_of_the_port_size": bool(pictures) and size == expected,
            "reply_frames_cover_reply_audio": bool(heard)
            and all(
                abs(r["frames"] * FRAME_MS / 1000 - r["audio_s"]) < CHUNK_MS / 1000
                for r in heard
            ),
            "frame_stamps_contiguous_but_the_dropped_interrupted_turn": all(
                gap == 0 or (reply == interrupted and gap > 0 and gap % FRAME_MS == 0)
                for gap, reply in gaps
            ),
            "no_errors": not self.errors,
        }
        if barge_in is not None:
            last = barge_in["interrupted_reply_last_frame_after_interrupt_s"]
            checks["interrupted_reply_frames_stop_within_two_chunks"] = (
                last is not None and last < 2 * CHUNK_MS / 1000
            )
        else:
            pass
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "frames": len(self.frames),
            "held_frames": len(self.frames) - len(pictures),
            "held_chunks": held_chunks,
            "frame_size": size,
            "jpeg_bytes_mean": (
                float(np.mean([len(p) for p in pictures])) if pictures else None
            ),
            "video_kbit_s": (
                round(sum(len(p) for p in pictures) * 8 / 1000 / duration_s, 1)
                if duration_s
                else None
            ),
            "first_frame_latency_s": first_frame,
            "chunk_latency_ms": stats(latency),
            "replies": replies,
            "frame_pairs": len(gaps),
            "reply_media_wait_ms": self.reply_media_wait_ms,
            "hold": {
                "reply_audio_held_ms": stats(held_ms),
                "replies_past_reply_media_wait": replies_past_wait,
                "frames_late_for_audio_ms": stats(late_hold),
            },
            "free": {"picture_behind_voice_ms": stats(behind_free)},
            "errors": self.errors,
            "barge_in": barge_in,
        }


async def send_paced(
    send,
    question: list[bytes],
    tail_s: float,
    record: Record,
    barge_in_after_s: float | None,
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
                    ports = event["session"]["sglang"].get("plugin_output_ports", [])
                    video = [
                        port
                        for port in ports
                        if port["modality"] == VIDEO and port["format"] == JPEG_FRAME
                    ]
                    if len(video) != 1 or not {
                        "fps",
                        "timeline",
                        "follows_replies",
                        "width",
                        "height",
                    } <= set(video[0]["params"]):
                        record.errors.append(f"session advertises {ports}")
                    else:
                        record.port = video[0]
                        record.reply_media_wait_ms = video[0]["params"].get(
                            "reply_media_wait_ms", REPLY_MEDIA_WAIT_MS
                        )
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
                elif kind == "sglang.output_chunk" and event["format"] == JPEG_FRAME:
                    record.add_frame(
                        event.get("response_id"),
                        event["t_start_ms"],
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
    async with (
        aiohttp.ClientSession() as http,
        http.ws_connect(f"{url}/v1/session", max_msg_size=0) as ws,
    ):

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
                    elif chunk["format"] == JPEG_FRAME:
                        record.add_frame(
                            chunk["turn"], chunk["t_start_ms"], chunk["payload"]
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
    parser.add_argument(
        "--barge-in-s",
        type=float,
        help="speak the question again this long after the first reply audio arrives",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = Record()
    run = realtime if args.protocol == "realtime" else session
    question = packets(read_wav(args.audio))

    async def stream(send) -> None:
        await send_paced(send, question, args.tail_s, record, args.barge_in_s)

    asyncio.run(run(args.url, stream, record, args.settle_s))
    result = {"protocol": args.protocol, **record.summary()}
    args.output.write_text(json.dumps(result, indent=1))
    print(
        json.dumps({key: result[key] for key in result if key != "replies"}, indent=1)
    )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
