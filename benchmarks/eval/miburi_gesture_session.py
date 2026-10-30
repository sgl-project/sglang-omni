# SPDX-License-Identifier: Apache-2.0
"""Stream a recording into a MIBURI server over /v1/session and record its reply audio and motion.

Units of 80 ms are sent in real time. The report pairs every motion chunk with the reply chunk of the same frame and gives how far motion trails the reply on arrival; reply.wav and motion.npy (frames × 271) are written for a visual check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import msgpack
import numpy as np
import websockets

from sglang_omni.models.miburi.formats import FEATURES_PER_FRAME, SMPLX_MOTION
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.session_hooks import PERSONAPLEX_REPLY
from sglang_omni.proto.session import OutputChunkDict
from sglang_omni.utils.audio import load_audio

UNIT_MILLISECONDS = SAMPLES_PER_FRAME * 1000 / SAMPLE_RATE


async def receive_outputs(
    socket: websockets.ClientConnection,
    started: float,
    outputs: list[tuple[float, OutputChunkDict]],
) -> None:
    async for frame in socket:
        message = msgpack.unpackb(frame, raw=False)
        if message["op"] == "output":
            outputs.append((time.monotonic() - started, message["chunk"]))
        elif message["op"] == "error":
            raise RuntimeError(message["message"])
        elif message["op"] == "closed":
            return
        else:
            pass


async def send_units(
    socket: websockets.ClientConnection, audio: np.ndarray, started: float
) -> None:
    """The recording as 80 ms PCM16 units in real time, then an empty EOS unit."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    num_units = len(pcm) // SAMPLES_PER_FRAME
    for seq in range(num_units + 1):
        unit = (
            pcm[seq * SAMPLES_PER_FRAME : (seq + 1) * SAMPLES_PER_FRAME].tobytes()
            if seq < num_units
            else b""
        )
        chunk = {
            "modality": "audio",
            "t_start_ms": seq * UNIT_MILLISECONDS,
            "duration_ms": UNIT_MILLISECONDS if seq < num_units else 0.0,
            "seq": seq,
            "payload": unit,
            "format": "pcm16",
            "eos": seq == num_units,
            "turn": None,
        }
        await socket.send(
            msgpack.packb({"op": "append", "chunk": chunk}, use_bin_type=True)
        )
        await asyncio.sleep(
            max(
                0.0,
                started + (seq + 1) * UNIT_MILLISECONDS / 1000 - time.monotonic(),
            )
        )


async def run(
    url: str, audio: np.ndarray, params: dict[str, str], end_format: str
) -> list[tuple[float, OutputChunkDict]]:
    """Every output chunk with its arrival time in seconds from the first append."""
    outputs: list[tuple[float, OutputChunkDict]] = []
    async with websockets.connect(url, max_size=None) as socket:
        await socket.send(
            msgpack.packb({"op": "open", "params": params}, use_bin_type=True)
        )
        opened = msgpack.unpackb(await socket.recv(), raw=False)
        if opened.get("op") != "opened":
            raise RuntimeError(f"open failed: {opened}")
        else:
            pass
        started = time.monotonic()
        receiver = asyncio.create_task(receive_outputs(socket, started, outputs))
        await send_units(socket, audio, started)
        while not any(
            chunk["eos"] and chunk["format"] == end_format for _, chunk in outputs
        ):
            if receiver.done():
                receiver.result()
                raise RuntimeError(f"session ended before the {end_format} EOS")
            else:
                await asyncio.sleep(0.05)
        await socket.send(msgpack.packb({"op": "close"}, use_bin_type=True))
        await receiver
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="ws://localhost:8000/v1/session")
    parser.add_argument(
        "--audio", type=Path, required=True, help="Caller recording, any rate"
    )
    parser.add_argument(
        "--voice", help="Voice; pass an empty string for the moshiko backbone"
    )
    parser.add_argument(
        "--instructions",
        help="Role prompt; pass an empty string for the moshiko backbone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for report.json, reply.wav, motion.npy",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="The server is a PersonaPlex session without a gesture node; report the reply only",
    )
    return parser.parse_args()


def build_report(
    replies: list[tuple[float, OutputChunkDict]],
    motions: list[tuple[float, OutputChunkDict]],
) -> dict[str, object]:
    """Frame coverage, motion lag behind the reply frame of the same t_start_ms, and arrival gaps."""
    reply_arrival_by_ms: dict[float, float] = {}
    for arrival, chunk in replies:
        frames = round(chunk["duration_ms"] / UNIT_MILLISECONDS)
        for index in range(frames):
            reply_arrival_by_ms[chunk["t_start_ms"] + index * UNIT_MILLISECONDS] = (
                arrival
            )
    lags_ms = [
        1000 * (arrival - reply_arrival_by_ms[chunk["t_start_ms"]])
        for arrival, chunk in motions
        if chunk["t_start_ms"] in reply_arrival_by_ms
    ]
    gaps_ms = [
        1000 * (later[0] - earlier[0]) for earlier, later in zip(motions, motions[1:])
    ]
    reply_gaps_ms = [
        1000 * (later[0] - earlier[0]) for earlier, later in zip(replies, replies[1:])
    ]
    return {
        "reply_frames": len(reply_arrival_by_ms),
        "motion_chunks": len(motions),
        "motion_without_reply_frame": sum(
            chunk["t_start_ms"] not in reply_arrival_by_ms for _, chunk in motions
        ),
        "motion_lag_ms": (
            {
                name: float(np.percentile(lags_ms, q))
                for name, q in (("p50", 50), ("p95", 95), ("max", 100))
            }
            if lags_ms
            else None
        ),
        "motion_max_gap_ms": max(gaps_ms) if gaps_ms else None,
        "reply_max_gap_ms": max(reply_gaps_ms) if reply_gaps_ms else None,
        "first_reply_s": replies[0][0] if replies else None,
        "first_motion_s": motions[0][0] if motions else None,
    }


def write_outputs(
    out: Path,
    replies: list[tuple[float, OutputChunkDict]],
    motions: list[tuple[float, OutputChunkDict]],
    report: dict[str, object],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out / "reply.wav"), "wb") as reply_wav:
        reply_wav.setnchannels(1)
        reply_wav.setsampwidth(2)
        reply_wav.setframerate(SAMPLE_RATE)
        reply_wav.writeframes(b"".join(chunk["payload"]["pcm"] for _, chunk in replies))
    np.save(
        out / "motion.npy",
        (
            np.concatenate(
                [
                    np.frombuffer(chunk["payload"], "<f4").reshape(
                        -1, FEATURES_PER_FRAME
                    )
                    for _, chunk in motions
                ]
            )
            if motions
            else np.zeros((0, FEATURES_PER_FRAME), "<f4")
        ),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    params = {
        name: value
        for name, value in (("voice", args.voice), ("instructions", args.instructions))
        if value is not None
    }
    audio = load_audio(
        str(args.audio), source_name="caller", target_sample_rate=SAMPLE_RATE, mono=True
    )
    outputs = asyncio.run(
        run(
            args.url,
            np.asarray(audio, dtype=np.float32).reshape(-1),
            params,
            PERSONAPLEX_REPLY if args.no_motion else SMPLX_MOTION,
        )
    )
    replies = [
        (arrival, chunk)
        for arrival, chunk in outputs
        if chunk["format"] == PERSONAPLEX_REPLY and chunk["kind"] == "data"
    ]
    motions = [
        (arrival, chunk)
        for arrival, chunk in outputs
        if chunk["format"] == SMPLX_MOTION
        and chunk["kind"] == "data"
        and chunk["payload"]
    ]
    report = build_report(replies, motions)
    write_outputs(args.out, replies, motions, report)
    print(json.dumps(report, indent=2))
    if not args.no_motion and (
        report["motion_chunks"] != report["reply_frames"]
        or report["motion_without_reply_frame"]
    ):
        raise RuntimeError(
            f"motion does not cover the reply one chunk per frame: {report}"
        )
    else:
        pass


if __name__ == "__main__":
    main()
