#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Profile inter-chunk timing for S2-Pro TTS streaming.

Usage:
    python playground/tts/profile_stream.py [--api-base http://localhost:8000] [--text "..."]

Reports:
  - wall-clock timestamp for each SSE chunk received
  - inter-chunk gap (seconds)
  - audio duration per chunk (seconds)
  - realtime ratio per chunk (audio_duration / gap)
  - time-to-first-chunk (TTFC)
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from playground.tts.audio_stream import SpeechStreamEvent, parse_speech_stream_data, wav_duration_seconds

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is used to test text-to-speech systems."
)


def stream_and_profile(api_base: str, text: str, *, verbose: bool = True) -> None:
    api_base = api_base.rstrip("/")
    url = f"{api_base}/v1/audio/speech"

    payload = {
        "input": text,
        "model": "s2pro",
        "stream": True,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  S2-Pro Streaming Chunk Timing Profile")
        print(f"{'='*60}")
        print(f"  Endpoint: {url}")
        print(f"  Text ({len(text)} chars): {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"{'='*60}\n")
        print(f"{'Chunk':>6}  {'Wall Time':>10}  {'Gap (s)':>9}  {'Audio (s)':>9}  {'RT Ratio':>9}  {'Tokens':>7}")
        print("-" * 65)

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
                if not line or not line.startswith("data: "):
                    continue
                event = parse_speech_stream_data(line[len("data: "):])
                if event is None:
                    continue
                if event.is_done:
                    break

                if event.audio_bytes:
                    gap_s = (now - prev_wall) if prev_wall is not None else (now - request_start)
                    audio_s = wav_duration_seconds(event.audio_bytes)
                    rt_ratio = audio_s / gap_s if gap_s > 0 else float("inf")
                    total_audio_s += audio_s

                    entry = {
                        "index": chunk_idx,
                        "wall_time": now - request_start,
                        "gap_s": gap_s,
                        "audio_s": audio_s,
                        "rt_ratio": rt_ratio,
                        "finish_reason": event.finish_reason,
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
        print(f"\nERROR: HTTP {exc.response.status_code}: {exc.response.text[:200]}", file=sys.stderr)
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

    print(f"\n{'='*65}")
    print("  Summary")
    print(f"{'='*65}")
    print(f"  Chunks received:         {len(chunks)}")
    print(f"  Time to first chunk:     {ttfc:.3f}s")
    print(f"  Total wall time:         {total_wall:.3f}s")
    print(f"  Total audio generated:   {total_audio_s:.3f}s")
    print(f"  Overall RT ratio:        {total_audio_s / total_wall:.2f}x")
    if gaps:
        print(f"  Inter-chunk gaps (s):")
        print(f"    min={min(gaps):.3f}  max={max(gaps):.3f}  mean={sum(gaps)/len(gaps):.3f}")
        slow_gaps = [(i+1, g) for i, g in enumerate(gaps) if g > 1.0]
        if slow_gaps:
            print(f"  Slow gaps (>1s): {slow_gaps}")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile S2-Pro TTS streaming chunk timing")
    parser.add_argument("--api-base", default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    args = parser.parse_args()

    stream_and_profile(args.api_base, args.text)


if __name__ == "__main__":
    main()
