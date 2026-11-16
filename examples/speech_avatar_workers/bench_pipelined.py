# SPDX-License-Identifier: Apache-2.0
"""Drive a pipelined speech-avatar worker (liveavatar_worker.py) with a 16 kHz mono WAV as a live Session would, and report steady-state fps, per-chunk latency and first-frame latency; standard library and numpy only.

    python bench_pipelined.py --url http://127.0.0.1:9440 --wav speech.wav \
        [--pace realtime|burst] [--depth 5] [--sessions 3] [--skip-chunks 8] \
        [--save-dir out/] [--out-json result.json]

A chunk's speech is "available" once the stream reached its end plus the
worker's look-ahead: at the start plus that much audio time (--pace
realtime), or at once (--pace burst, the worker's top rate). It is sent as
soon as it is available and fewer than --depth chunks are in flight, with
the handle (key, index) the worker's state protocol names, so the pipeline
fills. Latency is from availability to the answer; first-frame latency is
the first chunk's of a Session. Steady state drops each Session's first
--skip-chunks chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import threading
import time
import urllib.request
import uuid
import wave

import numpy as np

HANDLE = struct.Struct("<16sI")
STATE_LENGTH = struct.Struct("<I")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as source:
        assert source.getframerate() == 16000 and source.getsampwidth() == 2, path
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        if source.getnchannels() > 1:
            pcm = pcm[:: source.getnchannels()]
        else:
            pass
    return pcm


def percentile(values: list, q: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q)) if values.size else float("nan")


def run_session(args, info: dict, pcm: np.ndarray, label: str) -> dict:
    chunk = info["chunk_frames"] * 16000 // info["fps"]
    context, lookahead = info["context_samples"], info["lookahead_samples"]
    chunks = len(pcm) // chunk if not args.chunks else min(args.chunks, len(pcm) // chunk)
    padded = np.concatenate([np.zeros(context, "<i2"), pcm, np.zeros(chunk + lookahead, "<i2")])
    key = uuid.uuid4().bytes
    slots = threading.Semaphore(args.depth)
    results: list = [None] * chunks
    threads = []
    start = time.monotonic() + 0.2

    def send(index: int, available: float) -> None:
        window = padded[index * chunk : index * chunk + context + chunk + lookahead]
        state = HANDLE.pack(key, index)
        body = STATE_LENGTH.pack(len(state)) + state + window.tobytes()
        sent = time.monotonic()
        request = urllib.request.Request(f"{args.url}/render", data=body, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as answer:
                payload = answer.read()
                headers = answer.headers
            done = time.monotonic()
            state_length = int(headers["X-State-Length-Bytes"])
            lengths = [int(n) for n in headers["X-Payload-Lengths"].split(",") if n]
            offset = state_length
            frames = []
            for length in lengths:
                frames.append(payload[offset : offset + length])
                offset += length
            results[index] = {
                "index": index,
                "available": available - start,
                "sent": sent - start,
                "done": done - start,
                "latency_ms": (done - available) * 1000,
                "frames": len(frames),
                "worker": json.loads(headers.get("X-Timings", "{}")),
                "jpegs": frames if args.save_dir else None,
            }
        except Exception as error:  # noqa: BLE001
            results[index] = {"index": index, "error": repr(error), "available": available - start}
        finally:
            slots.release()

    for index in range(chunks):
        if args.pace == "realtime":
            available = start + ((index + 1) * chunk + lookahead) / 16000
        else:
            available = start
        delay = available - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            pass
        slots.acquire()
        thread = threading.Thread(target=send, args=(index, available), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]
    steady = [r for r in good if r["index"] >= args.skip_chunks]
    frame_ms = 1000 / info["fps"]
    # Frames play from the first chunk's arrival on, one per frame_ms; a frame that arrives after its slot is late.
    first_done = good[0]["done"] if good else 0.0
    late = 0
    worst_slip_ms = 0.0
    for r in good:
        due = first_done + r["index"] * info["chunk_frames"] * frame_ms / 1000
        slip = (r["done"] - due) * 1000
        worst_slip_ms = max(worst_slip_ms, slip)
        late += slip > frame_ms
    span = steady[-1]["done"] - steady[0]["done"] if len(steady) > 1 else float("nan")
    summary = {
        "label": label,
        "pace": args.pace,
        "depth": args.depth,
        "chunks": chunks,
        "errors": len(errors),
        "first_frame_latency_ms": good[0]["latency_ms"] if good else None,
        "steady_chunks": len(steady),
        "steady_fps": (len(steady) - 1) * info["chunk_frames"] / span if len(steady) > 1 else None,
        "chunk_latency_ms": {
            "p50": percentile([r["latency_ms"] for r in steady], 50),
            "p95": percentile([r["latency_ms"] for r in steady], 95),
            "max": max([r["latency_ms"] for r in steady], default=float("nan")),
        },
        "answer_interval_ms": {
            "p50": percentile(np.diff([r["done"] for r in steady]) * 1000, 50) if len(steady) > 2 else None,
            "p95": percentile(np.diff([r["done"] for r in steady]) * 1000, 95) if len(steady) > 2 else None,
        },
        "worker_render_ms_p95": percentile([r["worker"].get("render_ms", float("nan")) for r in steady], 95),
        "worker_queue_ms_p95": percentile([r["worker"].get("queue_ms", float("nan")) for r in steady], 95),
        "worker_pipeline_ms_p50": percentile([r["worker"].get("pipeline_ms", float("nan")) for r in steady], 50),
        "worker_decode_ms_p50": percentile([r["worker"].get("decode_ms", float("nan")) for r in steady], 50),
        "playback_late_chunks": late,
        "playback_worst_slip_ms": worst_slip_ms,
        "real_time": None,
    }
    # Real time: played from the first chunk's arrival on, no chunk misses its slot by more than a frame, and the steady rate keeps up.
    if args.pace == "realtime":
        summary["real_time"] = bool(not errors and late == 0)
    else:
        summary["real_time"] = (summary["steady_fps"] or 0) >= info["fps"]
    if args.save_dir:
        directory = os.path.join(args.save_dir, label)
        os.makedirs(directory, exist_ok=True)
        number = 0
        for r in good:
            for jpeg in r["jpegs"]:
                with open(os.path.join(directory, f"{number:06d}.jpg"), "wb") as out:
                    out.write(jpeg)
                number += 1
    else:
        pass
    for r in good:
        r.pop("jpegs", None)
    summary["chunks_detail"] = results
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--pace", choices=["realtime", "burst"], default="realtime")
    parser.add_argument("--depth", type=int, default=0, help="Chunks in flight; 0: the worker's pipeline_depth.")
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--skip-chunks", type=int, default=8)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    info = json.loads(get(f"{args.url}/info"))
    args.depth = args.depth or info.get("pipeline_depth", 1)
    pcm = read_wav(args.wav)
    sessions = []
    for number in range(args.sessions):
        summary = run_session(args, info, pcm, f"{args.pace}-d{args.depth}-s{number}")
        brief = {k: v for k, v in summary.items() if k != "chunks_detail"}
        print(json.dumps(brief), flush=True)
        sessions.append(summary)
        time.sleep(1.0)
    if args.out_json:
        with open(args.out_json, "w") as out:
            json.dump({"info": info, "args": vars(args), "sessions": sessions}, out, indent=1)
    else:
        pass


if __name__ == "__main__":
    main()
