#!/usr/bin/env python3
"""Self-contained thinker + talker pipeline latency benchmark.

Starts the sglang-omni server, runs benchmarks, prints results, then shuts down.
Server output goes directly to stdout so you can see everything.

Metrics:
    Thinker TTFT:  request → first text token (preprocessing + encoding + prefill)
    Thinker TPS:   text token decode throughput
    RTF:           per_token_decode_time / 80ms  (Qwen3-Omni paper convention)
                   = (t_first_text → t_done) / audio_duration

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_pipeline.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only -N 5
    CUDA_VISIBLE_DEVICES=0,1 python bench_pipeline.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct -N 5
"""
from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import os
import signal
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field

import requests

TIMEOUT = 600
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT = 900


@dataclass
class Run:
    t_start: float = 0.0
    t_first_text: float | None = None
    t_last_text: float | None = None
    t_first_audio: float | None = None
    t_done: float = 0.0
    n_text: int = 0
    n_audio: int = 0
    audio_b64: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def thinker_ttft(self) -> float | None:
        if self.t_first_text is None:
            return None
        return (self.t_first_text - self.t_start) * 1000

    @property
    def first_packet_latency(self) -> float | None:
        """Time from request to first audio chunk (ms). Paper reports 234ms."""
        if self.t_first_audio is None:
            return None
        return (self.t_first_audio - self.t_start) * 1000

    @property
    def thinker_tps(self) -> float | None:
        if self.t_first_text is None or self.t_last_text is None or self.n_text < 2:
            return None
        dt = self.t_last_text - self.t_first_text
        return (self.n_text - 1) / dt if dt > 0 else None

    @property
    def decode_time(self) -> float | None:
        """Full decode time: thinker + talker + MTP + codec decoder."""
        if self.t_first_text is None:
            return None
        return self.t_done - self.t_first_text

    @property
    def audio_duration(self) -> float | None:
        return _wav_duration(self.audio_b64)

    @property
    def rtf(self) -> float | None:
        """RTF per Qwen3-Omni paper: per_token_decode_time / 80ms.

        Equivalent to decode_time / audio_duration since each codec token = 80ms.
        """
        dt = self.decode_time
        dur = self.audio_duration
        if dt is None or dur is None or dur <= 0 or dt <= 0:
            return None
        return dt / dur

    @property
    def e2e(self) -> float:
        return (self.t_done - self.t_start) * 1000


def _single_wav_duration(raw: bytes) -> float | None:
    if len(raw) < 44:
        return None
    try:
        buf = io.BytesIO(raw)
        if buf.read(4) != b"RIFF":
            return None
        buf.read(4)
        if buf.read(4) != b"WAVE":
            return None
        sr = bps = nch = 0
        while True:
            cid = buf.read(4)
            if len(cid) < 4:
                return None
            csz = struct.unpack("<I", buf.read(4))[0]
            if cid == b"fmt ":
                fmt = buf.read(csz)
                nch = struct.unpack("<H", fmt[2:4])[0]
                sr = struct.unpack("<I", fmt[4:8])[0]
                bps = struct.unpack("<H", fmt[14:16])[0]
            elif cid == b"data":
                if sr == 0 or bps == 0 or nch == 0:
                    return None
                return csz / (nch * (bps // 8) * sr)
            else:
                buf.read(csz)
    except Exception:
        return None


def _wav_duration(b64_chunks: list[str]) -> float | None:
    """Sum durations of individually-encoded WAV chunks."""
    if not b64_chunks:
        return None
    total = 0.0
    for chunk in b64_chunks:
        try:
            raw = base64.b64decode(chunk)
        except Exception:
            continue
        dur = _single_wav_duration(raw)
        if dur is not None:
            total += dur
    return total if total > 0 else None


def start_server(model_path: str, port: int, text_only: bool) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        model_path,
        "--port",
        str(port),
    ]
    if text_only:
        cmd.append("--text-only")

    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

    def _cleanup():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    atexit.register(_cleanup)
    return proc


def wait_healthy(url: str, proc: subprocess.Popen) -> bool:
    t0 = time.time()
    while time.time() - t0 < HEALTH_TIMEOUT:
        if proc.poll() is not None:
            print(f"Server exited with code {proc.returncode}")
            return False
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return False


def run_once(url: str, prompt: str, max_tokens: int, text_only: bool) -> Run:
    r = Run()
    modalities = ["text"] if text_only else ["text", "audio"]
    payload: dict = {
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "modalities": modalities,
    }
    if not text_only:
        payload["audio"] = {"format": "wav"}

    r.t_start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        r.error = str(e)
        r.t_done = time.perf_counter()
        return r

    if resp.status_code != 200:
        r.error = f"HTTP {resp.status_code}"
        r.t_done = time.perf_counter()
        return r

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode()
        if not decoded.startswith("data: "):
            continue
        s = decoded[6:]
        if s == "[DONE]":
            break
        try:
            chunk = json.loads(s)
        except json.JSONDecodeError:
            continue

        t = time.perf_counter()
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        if delta.get("content"):
            r.n_text += 1
            if r.t_first_text is None:
                r.t_first_text = t
            r.t_last_text = t

        audio = delta.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            r.n_audio += 1
            if r.t_first_audio is None:
                r.t_first_audio = t
            r.audio_b64.append(audio["data"])

    r.t_done = time.perf_counter()
    return r


def _fmt(v: float | None, prec: int = 1) -> str:
    if v is None:
        return "-"
    return f"{v:.{prec}f}"


def _cell(fn, runs: list[Run], prec: int = 1) -> str:
    vals = [fn(r) for r in runs if fn(r) is not None]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else None
    if s is not None and s > 0:
        return f"{m:.{prec}f} +/- {s:.{prec}f}"
    return f"{m:.{prec}f}"


def print_table(runs: list[Run], text_only: bool) -> None:
    w = 50
    print()
    print("=" * w)
    print(f"{'Metric':<25} {'Value':>20}")
    print("-" * w)
    print(f"{'Thinker TTFT (ms)':<25} {_cell(lambda r: r.thinker_ttft, runs):>20}")
    print(f"{'Thinker TPS (tok/s)':<25} {_cell(lambda r: r.thinker_tps, runs):>20}")
    if not text_only:
        print(
            f"{'First-packet (ms)':<25} {_cell(lambda r: r.first_packet_latency, runs):>20}"
        )
        print(f"{'RTF':<25} {_cell(lambda r: r.rtf, runs, 3):>20}")
        print(
            f"{'Audio duration (s)':<25} {_cell(lambda r: r.audio_duration, runs, 2):>20}"
        )
    print(f"{'E2E (ms)':<25} {_cell(lambda r: r.e2e, runs):>20}")
    print("=" * w)

    for i, r in enumerate(runs):
        parts = [f"run {i + 1}:"]
        parts.append(f"ttft={_fmt(r.thinker_ttft)}ms")
        parts.append(f"tps={_fmt(r.thinker_tps)}")
        if not text_only:
            parts.append(f"first_pkt={_fmt(r.first_packet_latency)}ms")
            parts.append(f"rtf={_fmt(r.rtf, 3)}")
            parts.append(f"audio={_fmt(r.audio_duration, 2)}s")
        parts.append(f"e2e={_fmt(r.e2e)}ms")
        print("  " + " | ".join(parts))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Self-contained pipeline latency benchmark"
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("-N", "--iterations", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Say hello and introduce yourself briefly.")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("-o", "--output", type=str, default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"

    proc = start_server(args.model_path, args.port, args.text_only)

    if not wait_healthy(url, proc):
        print("Server failed to start.")
        sys.exit(1)
    print("\n--- Server ready, starting benchmark ---\n")

    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}...", flush=True)
        run_once(url, args.prompt, args.max_tokens, args.text_only)

    runs: list[Run] = []
    for i in range(args.iterations):
        r = run_once(url, args.prompt, args.max_tokens, args.text_only)
        runs.append(r)
        if r.error:
            print(f"run {i + 1}/{args.iterations}: ERROR {r.error}")
        else:
            print(
                f"run {i + 1}/{args.iterations}: "
                f"ttft={_fmt(r.thinker_ttft)}ms tps={_fmt(r.thinker_tps)} "
                f"first_pkt={_fmt(r.first_packet_latency)}ms "
                f"rtf={_fmt(r.rtf, 3)} audio={_fmt(r.audio_duration, 2)}s "
                f"e2e={_fmt(r.e2e)}ms"
            )

    good = [r for r in runs if r.error is None]
    if good:
        print_table(good, args.text_only)
    else:
        print("All runs failed.")

    if args.output:
        export = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "args": vars(args),
            "runs": [
                {
                    "thinker_ttft_ms": r.thinker_ttft,
                    "thinker_tps": r.thinker_tps,
                    "first_packet_latency_ms": r.first_packet_latency,
                    "rtf": r.rtf,
                    "audio_duration_s": r.audio_duration,
                    "e2e_ms": r.e2e,
                    "n_text": r.n_text,
                    "n_audio": r.n_audio,
                    "error": r.error,
                }
                for r in runs
            ],
        }
        with open(args.output, "w") as f:
            json.dump(export, f, indent=2)
        print(f"Saved to {args.output}")

    print("Shutting down server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


if __name__ == "__main__":
    main()
