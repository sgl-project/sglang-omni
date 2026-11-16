# SPDX-License-Identifier: Apache-2.0
"""Drive a speech-avatar worker with a 16 kHz mono WAV chunk by chunk, back to back, and report its per-chunk times; standard library and numpy only.

    python bench_worker.py --url http://127.0.0.1:8031 --wav speech.wav [--save-dir frames/] [--chunks 30]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
import urllib.request
import wave

import numpy as np


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--chunks", type=int, default=0, help="0: the whole file")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    info = json.loads(get(f"{args.url}/info"))
    with wave.open(args.wav) as source:
        assert source.getframerate() == 16000 and source.getsampwidth() == 2
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        if source.getnchannels() > 1:
            pcm = pcm[:: source.getnchannels()]
    chunk = info["chunk_frames"] * 16000 // info["fps"]
    context, lookahead = info["context_samples"], info["lookahead_samples"]
    total = len(pcm) // chunk if not args.chunks else args.chunks
    padded = np.concatenate(
        [
            np.zeros(context, "<i2"),
            pcm,
            np.zeros(chunk * (total + 1) + lookahead, "<i2"),
        ]
    )
    state = b""
    wall, render, sizes = [], [], []
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    for index in range(total):
        window = padded[index * chunk : index * chunk + context + chunk + lookahead]
        body = struct.pack("<I", len(state)) + state + window.astype("<i2").tobytes()
        request = urllib.request.Request(f"{args.url}/render", data=body, method="POST")
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            data = response.read()
            headers = response.headers
        wall.append((time.perf_counter() - start) * 1000)
        render.append(float(headers["X-Render-Ms"]))
        state_length = int(headers["X-State-Length-Bytes"])
        state = data[:state_length]
        lengths = [int(n) for n in headers["X-Payload-Lengths"].split(",")]
        sizes.extend(lengths)
        if args.save_dir:
            offset = state_length
            for frame, length in enumerate(lengths):
                suffix = "jpg" if info["kind"] == "video" else "f32"
                name = (
                    f"{index * info['chunk_frames'] + frame:06d}.{suffix}"
                    if info["kind"] == "video"
                    else f"{index:06d}.{suffix}"
                )
                with open(os.path.join(args.save_dir, name), "wb") as out:
                    out.write(data[offset : offset + length])
                offset += length
    steady = wall[1:] if len(wall) > 1 else wall
    summary = {
        "info": info,
        "chunks": total,
        "chunk_ms": chunk / 16,
        "first_wall_ms": wall[0],
        "wall_ms_p50": float(np.percentile(steady, 50)),
        "wall_ms_p95": float(np.percentile(steady, 95)),
        "wall_ms_max": float(np.max(steady)),
        "render_ms_p50": float(np.percentile(render[1:] or render, 50)),
        "render_ms_p95": float(np.percentile(render[1:] or render, 95)),
        "payload_bytes_mean": float(np.mean(sizes)),
        "state_bytes": len(state),
    }
    print(json.dumps(summary, indent=1))
    if args.out_json:
        with open(args.out_json, "w") as out:
            json.dump({"summary": summary, "wall_ms": wall, "render_ms": render}, out)


if __name__ == "__main__":
    main()
