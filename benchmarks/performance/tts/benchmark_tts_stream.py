# SPDX-License-Identifier: Apache-2.0
"""Benchmark inter-chunk timing for S2-Pro TTS streaming.

Profiles the chunk cadence of the /v1/audio/speech SSE stream: time-to-first-
chunk (TTFC), per-chunk gap, audio duration, and realtime ratio.  Complements
benchmark_tts_speed.py which measures overall latency and throughput.

Usage:
    # Launch a server first:
    python -m sglang_omni.cli.cli serve \\
        --model-path fishaudio/s2-pro \\
        --config examples/configs/s2pro_tts.yaml \\
        --port 8000

    python -m benchmarks.performance.tts.benchmark_tts_stream \\
        --port 8000 \\
        --text "The quick brown fox jumps over the lazy dog."

    python -m benchmarks.performance.tts.benchmark_tts_stream \\
        --base-url http://localhost:8000 \\
        --text "Hello world"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave

import httpx

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is used to test text-to-speech systems."
)


def _wav_duration_seconds(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if sample_rate <= 0:
        return 0.0
    return frame_count / sample_rate


def _parse_stream_event(line: str) -> tuple[bytes | None, bool]:
    """Parse one SSE data line. Returns (audio_bytes_or_None, is_done)."""
    if not line.startswith("data: "):
        return None, False
    data = line[len("data: ") :].strip()
    if data == "[DONE]":
        return None, True
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None, False
    audio = payload.get("audio")
    if not isinstance(audio, dict):
        return None, False
    raw = audio.get("data")
    if not isinstance(raw, str):
        return None, False
    import base64

    return base64.b64decode(raw), False


def stream_and_profile(
    api_base: str, text: str, model: str, *, verbose: bool = True
) -> None:
    url = f"{api_base.rstrip('/')}/v1/audio/speech"
    payload = {"input": text, "model": model, "stream": True}

    if verbose:
        print(f"\n{'='*60}")
        print("  S2-Pro Streaming Chunk Timing Benchmark")
        print(f"{'='*60}")
        print(f"  Endpoint: {url}")
        print(
            f"  Text ({len(text)} chars): {text[:80]}{'...' if len(text) > 80 else ''}"
        )
        print(f"{'='*60}\n")
        print(
            f"{'Chunk':>6}  {'Wall Time':>10}  {'Gap (s)':>9}  {'Audio (s)':>9}  {'RT Ratio':>9}"
        )
        print("-" * 60)

    request_start = time.perf_counter()
    chunks: list[dict] = []
    prev_wall = None
    chunk_idx = 0
    total_audio_s = 0.0

    try:
        with httpx.stream("POST", url, json=payload, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                now = time.perf_counter()
                if not line:
                    continue
                audio_bytes, is_done = _parse_stream_event(line)
                if is_done:
                    break
                if audio_bytes is None:
                    continue

                gap_s = (
                    (now - prev_wall)
                    if prev_wall is not None
                    else (now - request_start)
                )
                audio_s = _wav_duration_seconds(audio_bytes)
                rt_ratio = audio_s / gap_s if gap_s > 0 else float("inf")
                total_audio_s += audio_s

                entry = {
                    "index": chunk_idx,
                    "wall_time": now - request_start,
                    "gap_s": gap_s,
                    "audio_s": audio_s,
                    "rt_ratio": rt_ratio,
                }
                chunks.append(entry)

                if verbose:
                    marker = " <-- TTFC" if chunk_idx == 0 else ""
                    print(
                        f"{chunk_idx:>6}  "
                        f"{entry['wall_time']:>10.3f}  "
                        f"{gap_s:>9.3f}  "
                        f"{audio_s:>9.3f}  "
                        f"{rt_ratio:>9.2f}x"
                        f"{marker}"
                    )

                chunk_idx += 1
                prev_wall = now

    except httpx.HTTPStatusError as exc:
        print(
            f"\nERROR: HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total_wall = time.perf_counter() - request_start

    if not chunks:
        print("No audio chunks received.", file=sys.stderr)
        sys.exit(1)

    gaps = [c["gap_s"] for c in chunks[1:]] if len(chunks) > 1 else []
    ttfc = chunks[0]["wall_time"]

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  Chunks received:       {len(chunks)}")
    print(f"  Time to first chunk:   {ttfc:.3f}s")
    print(f"  Total wall time:       {total_wall:.3f}s")
    print(f"  Total audio:           {total_audio_s:.3f}s")
    print(f"  Overall RT ratio:      {total_audio_s / total_wall:.2f}x")
    if gaps:
        print(
            f"  Inter-chunk gaps (s):  min={min(gaps):.3f}  max={max(gaps):.3f}  mean={sum(gaps)/len(gaps):.3f}"
        )
        slow_gaps = [(i + 1, round(g, 3)) for i, g in enumerate(gaps) if g > 1.0]
        if slow_gaps:
            print(f"  Slow gaps (>1s):       {slow_gaps}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark S2-Pro TTS streaming chunk cadence."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="s2pro")
    parser.add_argument(
        "--text", type=str, default=DEFAULT_TEXT, help="Text to synthesize."
    )
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    stream_and_profile(base_url, args.text, args.model)


if __name__ == "__main__":
    main()
