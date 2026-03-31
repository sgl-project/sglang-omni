# SPDX-License-Identifier: Apache-2.0
"""TTS Speed benchmark for Omni models (Qwen3-Omni).

Measures latency, RTF, and throughput via /v1/chat/completions with
modalities: ["text", "audio"].  Supports voice cloning (default) and
plain TTS modes.

Usage:
    # Voice cloning (default)
    python benchmarks/eval/benchmark_omni_tts_speed.py \
        --model qwen3-omni --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 10

    # Plain TTS (no voice cloning)
    python benchmarks/eval/benchmark_omni_tts_speed.py \
        --model qwen3-omni --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 10 --no-ref-audio

    # Concurrency test
    python benchmarks/eval/benchmark_omni_tts_speed.py \
        --model qwen3-omni --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 10 --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts_speed import (
    make_omni_tts_send_fn,
    print_speed_summary,
    save_speed_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def benchmark(args: argparse.Namespace) -> None:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    if not os.path.isfile(args.testset):
        logger.error("Testset not found: %s", args.testset)
        return

    samples = load_seedtts_samples(args.testset, args.max_samples)
    logger.info("Prepared %d requests", len(samples))

    save_audio_dir = None
    if args.save_audio and args.output_dir:
        save_audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    send_fn = make_omni_tts_send_fn(
        args.model,
        api_url,
        lang=args.lang,
        voice_clone=not args.no_ref_audio,
        speaker=args.speaker,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_audio_dir=save_audio_dir,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    print_speed_summary(metrics, args.model)

    if args.output_dir:
        config = {
            "model": args.model,
            "base_url": base_url,
            "testset": args.testset,
            "no_ref_audio": args.no_ref_audio,
            "lang": args.lang,
            "speaker": args.speaker,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
        }
        save_speed_results(outputs, metrics, config, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TTS speed for Omni models (Qwen3-Omni)."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Language for prompt construction.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Speaker voice for TTS.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="seed-tts-eval/en/meta.lst",
        help="Path to a meta.lst file (seed-tts-eval format).",
    )
    parser.add_argument(
        "--no-ref-audio",
        action="store_true",
        help="Skip ref audio/text (TTS without voice cloning).",
    )
    parser.add_argument("--output-dir", type=str, default="results/omni_tts_speed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument(
        "--save-audio", action="store_true", help="Save generated WAV files."
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
