# SPDX-License-Identifier: Apache-2.0
"""One conversation with a cascade that carries the ELLSA 2 motion node, over /v1/realtime or /v1/session.

Streams a 16 kHz mono WAV as the user's audio in real time, then silence
for --tail-s, records when each reply audio chunk and each smplh_flame_motion
chunk arrives, and checks that motion arrived in 24-frame blocks, that the
blocks carrying a reply cover its audio (frames / 30 within one 0.8 s block
of the reply's duration) and that some reply was heard. With --barge-in-s
the question is spoken again that long after the first reply audio arrives:
the interrupted reply's motion must stop within two blocks and only later
replies are checked for coverage. The blocks' t_start_ms must join without a
gap, except where the interrupted turn's blocks were dropped: the client
keeps its last pose over that span. Exits 0 when the checks hold.

It also reports what a client sees under the playground's hold policy, where
motion plays on its t_start_ms from the first block's arrival and a reply's
audio waits for the reply's first motion frame, at most the port's
reply_media_wait_ms: the added reply start delay, how late each later block
of the reply arrives for its audio, and, for comparison, how far motion
trails audio played on arrival.

    python -m benchmarks.eval.ellsa2_motion_e2e --url ws://127.0.0.1:8765 \\
        --protocol realtime --audio question.wav --output realtime.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import aiohttp
import msgpack
import numpy as np

from sglang_omni.models.ellsa2.formats import (
    FEATURES_PER_FRAME,
    MOTION_FPS,
    REPLY_MEDIA_WAIT_MS,
    SMPLH_FLAME_MOTION,
    SMPLH_FLAME_MOTION_PORT,
)

PACKET_MS = 200
BLOCK_MS = 800
FRAMES_PER_BLOCK = 24


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        return source.readframes(source.getnframes())


def packets(pcm: bytes) -> list[bytes]:
    size = 16000 * 2 * PACKET_MS // 1000
    pcm = pcm + bytes(-len(pcm) % size)
    return [pcm[start : start + size] for start in range(0, len(pcm), size)]


class Record:
    """Arrival times of reply audio and motion, per reply."""

    def __init__(self) -> None:
        self.began = time.perf_counter()
        self.barge_in_s: float | None = None
        self.audio: dict[object, list[tuple[float, float]]] = defaultdict(list)
        self.motion: list[tuple[float, object, float, int]] = []
        self.errors: list[str] = []
        self.reply_media_wait_ms = REPLY_MEDIA_WAIT_MS

    def add_audio(self, reply: object, seconds: float) -> None:
        self.audio[reply].append((time.perf_counter(), seconds))

    def add_motion(self, reply: object, t_start_ms: float, payload: bytes) -> None:
        self.motion.append(
            (
                time.perf_counter(),
                reply,
                t_start_ms,
                len(payload) // (4 * FEATURES_PER_FRAME),
            )
        )

    def summary(self) -> dict:
        # The playground's motion view plays each block at its t_start_ms, counted from the first block's arrival, or when it arrives if later.
        origin = self.motion[0][0] - self.motion[0][2] / 1000 if self.motion else 0.0
        shown = [
            max(arrived, origin + t_ms / 1000) for arrived, _, t_ms, _ in self.motion
        ]
        replies = {}
        start_delay, late, behind = [], [], []
        replies_past_wait = 0
        for reply, audio in self.audio.items():
            blocks = [
                (block, shown[index])
                for index, block in enumerate(self.motion)
                if block[1] == reply
            ]
            audio_s = sum(seconds for _, seconds in audio)
            replies[str(reply)] = {
                "audio_s": audio_s,
                "motion_frames": sum(block[3] for block, _ in blocks),
                "motion_blocks": len(blocks),
            }
            if not blocks:
                continue
            else:
                first_audio = audio[0][0]
                start = min(blocks[0][1], first_audio + self.reply_media_wait_ms / 1000)
                replies_past_wait += blocks[0][1] > start
            start_delay.append((start - first_audio) * 1000)
            for (arrived, _, t_ms, _), _ in blocks:
                offset = (t_ms - blocks[0][0][2]) / 1000
                late.append((arrived - (start + offset)) * 1000)
                behind.append((arrived - (first_audio + offset)) * 1000)

        def stats(values: list[float]) -> dict | None:
            return (
                {
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "n": len(values),
                }
                if values
                else None
            )

        heard = [r for r in replies.values() if r["audio_s"] > 0]
        # Each pair of consecutive blocks by t_start_ms: the gap between them and the reply of the earlier one.
        ordered = sorted(self.motion, key=lambda block: block[2])
        gaps = [
            (later[2] - (earlier[2] + earlier[3] * 1000 / MOTION_FPS), earlier[1])
            for earlier, later in zip(ordered, ordered[1:])
        ]
        interrupted = (
            list(self.audio)[0] if self.barge_in_s is not None and self.audio else None
        )
        frozen_ms = [gap for gap, reply in gaps if gap != 0 and reply == interrupted]
        barge_in = None
        if self.barge_in_s is not None:
            first, *later = list(self.audio)
            cut = [m[0] for m in self.motion if m[1] == first]
            barge_in = {
                "interrupted_reply_last_motion_after_barge_in_s": (
                    round(max(cut) - self.barge_in_s, 3) if cut else None
                ),
                "later_replies_motion_blocks": [
                    replies[str(reply)]["motion_blocks"] for reply in later
                ],
            }
            # The interrupted reply's audio arrived ahead of its playback, so only later replies must be covered.
            heard = [
                replies[str(reply)]
                for reply in later
                if replies[str(reply)]["audio_s"] > 0
            ]
        else:
            pass
        checks = {
            "reply_heard": bool(heard),
            "motion_in_24_frame_blocks": bool(self.motion)
            and all(m[3] == FRAMES_PER_BLOCK for m in self.motion),
            "reply_motion_covers_reply_audio": bool(heard)
            and all(
                abs(r["motion_frames"] / MOTION_FPS - r["audio_s"]) < BLOCK_MS / 1000
                for r in heard
            ),
            "motion_stamps_contiguous_but_the_dropped_interrupted_turn": all(
                gap == 0 or (reply == interrupted and gap > 0 and gap % BLOCK_MS == 0)
                for gap, reply in gaps
            ),
            "no_errors": not self.errors,
        }
        if barge_in is not None:
            checks["interrupted_reply_motion_stops_within_two_blocks"] = (
                barge_in["interrupted_reply_last_motion_after_barge_in_s"] is not None
                and barge_in["interrupted_reply_last_motion_after_barge_in_s"]
                < 2 * BLOCK_MS / 1000 + 0.5
            )
        else:
            pass
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "motion_blocks": len(self.motion),
            "motion_frames": sum(m[3] for m in self.motion),
            "replies": replies,
            "motion_block_pairs": len(gaps),
            "frozen_after_interrupt_ms": frozen_ms,
            "reply_media_wait_ms": self.reply_media_wait_ms,
            "replies_past_reply_media_wait": replies_past_wait,
            "reply_start_delay_ms": stats(start_delay),
            "motion_late_for_audio_ms": stats(late),
            "motion_behind_audio_on_arrival_ms": stats(behind),
            "errors": self.errors,
            "barge_in": barge_in,
            "motion_timeline": [
                [round(arrived - self.began, 3), str(reply), t_ms]
                for arrived, reply, t_ms, _ in self.motion
            ],
            "reply_audio_first_arrival_s": {
                str(reply): round(audio[0][0] - self.began, 3)
                for reply, audio in self.audio.items()
            },
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
        index += 1


async def realtime(url: str, stream, record: Record, settle_s: float) -> None:
    async with (
        aiohttp.ClientSession() as http,
        http.ws_connect(f"{url}/v1/realtime") as ws,
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
                    if SMPLH_FLAME_MOTION_PORT.to_dict() not in ports:
                        record.errors.append(
                            f"session advertises {ports}, not {SMPLH_FLAME_MOTION_PORT.to_dict()}"
                        )
                    else:
                        record.reply_media_wait_ms = SMPLH_FLAME_MOTION_PORT.params[
                            "reply_media_wait_ms"
                        ]
                    ready.set()
                elif kind == "response.output_audio.delta":
                    record.add_audio(
                        event.get("response_id"),
                        len(base64.b64decode(event["delta"])) / 2 / output_rate,
                    )
                elif (
                    kind == "sglang.output_chunk"
                    and event["format"] == SMPLH_FLAME_MOTION
                ):
                    record.add_motion(
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
        http.ws_connect(f"{url}/v1/session") as ws,
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
                    elif chunk["format"] == SMPLH_FLAME_MOTION:
                        record.add_motion(
                            chunk["turn"], chunk["t_start_ms"], chunk["payload"]
                        )
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
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
