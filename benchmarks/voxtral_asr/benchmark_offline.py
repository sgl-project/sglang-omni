#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline ASR throughput benchmark for Voxtral realtime in sglang-omni.

Assumes an sglang-omni server is already running with the Voxtral ASR pipeline
(see examples/configs/voxtral_asr.yaml). The server exposes
``POST /v1/audio/transcriptions`` as a multipart file upload.

Example:
    python benchmarks/voxtral_asr/benchmark_offline.py \
        --audio /path/to/speech.wav --num-requests 64 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path


async def send_one(session, url: str, audio_bytes: bytes, language: str) -> float:
    import aiohttp

    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", "voxtral")
    if language:
        form.add_field("language", language)

    start = time.perf_counter()
    async with session.post(url, data=form) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"request failed: {resp.status} {body[:200]}")
    return time.perf_counter() - start


async def benchmark(
    url: str,
    *,
    audio_path: Path,
    language: str,
    num_requests: int,
    concurrency: int,
) -> None:
    import aiohttp

    audio_bytes = audio_path.read_bytes()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_send(session) -> float:
        async with semaphore:
            return await send_one(session, url, audio_bytes, language)

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(bounded_send(session)) for _ in range(num_requests)
        ]
        latencies = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - wall_start

    latencies = sorted(latencies)
    print(f"Audio: {audio_path} ({audio_path.stat().st_size / 1024:.0f} KiB)")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"Wall time: {wall_s:.2f}s")
    print(f"Mean latency: {sum(latencies) / len(latencies):.3f}s")
    print(f"P50: {latencies[len(latencies) // 2]:.3f}s")
    print(f"P99: {latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]:.3f}s")
    print(f"Throughput: {num_requests / wall_s:.2f} req/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Voxtral ASR throughput")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/audio/transcriptions"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to a real speech WAV (16 kHz mono recommended).",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    asyncio.run(
        benchmark(
            args.url,
            audio_path=args.audio,
            language=args.language,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
