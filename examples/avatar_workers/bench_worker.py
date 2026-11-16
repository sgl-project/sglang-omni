# SPDX-License-Identifier: Apache-2.0
"""Steady-state speed of a running chunk worker: one Session's chunks rendered back to back over HTTP from a speech file (and a user file for a listening worker), as the avatar_worker stage sends them.

python examples/avatar_workers/bench_worker.py --url http://127.0.0.1:8123 --speech agent.wav [--listen user.wav] [--chunks 60] [--out result.json] [--save-frames dir]

Reports per chunk the round trip (HTTP, model and JPEG encoding, what the
stage waits for) and the worker's own X-Render-Ms; the first chunk of the
Session (cache priming included) is reported apart and left out of the
steady-state percentiles, with the warm-up the worker did at load. The
real-time factor is the chunk's duration over the p50 round trip, and the
first picture's latency adds the chunk and the look-ahead the worker waits
for before it can start.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

RATE = 16000


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = (
            np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
            / 32768
        )
    pcm = pcm.reshape(-1, channels).mean(1)
    if rate != RATE:
        pcm = np.interp(
            np.arange(int(len(pcm) * RATE / rate)) * rate / RATE,
            np.arange(len(pcm)),
            pcm,
        ).astype(np.float32)
    return pcm


def pcm16(x: np.ndarray) -> bytes:
    return np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes()


def post(url: str, body: bytes):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/octet-stream"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.headers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--listen")
    parser.add_argument("--chunks", type=int, default=60)
    parser.add_argument("--out")
    parser.add_argument(
        "--save-frames",
        help="write the frames (JPEG or float32 rows) here, outside the timing",
    )
    args = parser.parse_args()
    info = json.loads(urllib.request.urlopen(args.url + "/info").read())
    chunk = round(info["chunk_frames"] * RATE / info["fps"])
    window = chunk + info["lookahead_samples"]
    speech = read_wav(args.speech)
    listen = read_wav(args.listen) if args.listen else np.zeros_like(speech)
    total = args.chunks * chunk + window
    speech = np.pad(speech, (0, max(0, total - len(speech))))[:total]
    listen = np.pad(listen, (0, max(0, total - len(listen))))[:total]
    state = b""
    rows = []
    saved = []
    for index in range(args.chunks):
        start = index * chunk
        body = pcm16(speech[start : start + window])
        if info.get("listens"):
            # The user's audio lags the chunk by the window, as the stage sends it.
            low = start - window
            past = listen[max(low, 0) : start]
            body += pcm16(
                np.concatenate([np.zeros(window - len(past), np.float32), past])
            )
        began = time.perf_counter()
        content, headers = post(
            args.url + "/render", struct.pack("<I", len(state)) + state + body
        )
        took = (time.perf_counter() - began) * 1000
        state_bytes = int(headers["X-State-Length-Bytes"])
        lengths = [int(n) for n in headers["X-Payload-Lengths"].split(",") if n]
        state = content[:state_bytes]
        rows.append(
            {
                "chunk": index,
                "round_trip_ms": round(took, 1),
                "render_ms": float(headers["X-Render-Ms"]),
                "frames": len(lengths),
                "bytes": sum(lengths),
            }
        )
        if args.save_frames:
            offset = state_bytes
            for length in lengths:
                saved.append(content[offset : offset + length])
                offset += length
    urllib.request.urlopen(
        urllib.request.Request(
            args.url + "/release",
            data=state,
            headers={"Content-Type": "application/octet-stream"},
        )
    )
    chunk_ms = 1000 * info["chunk_frames"] / info["fps"]
    steady = np.array([r["round_trip_ms"] for r in rows[1:]])
    model = np.array([r["render_ms"] for r in rows[1:]])
    result = {
        "info": info,
        "chunk_ms": chunk_ms,
        "lookahead_ms": 1000 * info["lookahead_samples"] / RATE,
        "chunks": args.chunks,
        "first_chunk_round_trip_ms": rows[0]["round_trip_ms"],
        "first_picture_latency_ms": round(
            chunk_ms
            + 1000 * info["lookahead_samples"] / RATE
            + rows[0]["round_trip_ms"],
            1,
        ),
        "steady_round_trip_ms": {
            q: round(float(np.percentile(steady, p)), 1)
            for q, p in (("p50", 50), ("p95", 95), ("max", 100))
        },
        "steady_model_ms": {
            q: round(float(np.percentile(model, p)), 1)
            for q, p in (("p50", 50), ("p95", 95))
        },
        "realtime_factor_p50": round(chunk_ms / float(np.percentile(steady, 50)), 3),
        "realtime_factor_p95": round(chunk_ms / float(np.percentile(steady, 95)), 3),
        "kb_per_frame": round(
            float(np.mean([r["bytes"] / max(r["frames"], 1) for r in rows])) / 1024, 1
        ),
        "rows": rows,
    }
    print(
        json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False)
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if args.save_frames:
        out = Path(args.save_frames)
        out.mkdir(parents=True, exist_ok=True)
        if info["media"] == "jpeg_frame":
            for index, data in enumerate(saved):
                (out / f"{index:05d}.jpg").write_bytes(data)
        else:
            np.save(
                out / "rows.npy",
                np.stack([np.frombuffer(d, "<f4") for d in saved if d]),
            )


if __name__ == "__main__":
    main()
