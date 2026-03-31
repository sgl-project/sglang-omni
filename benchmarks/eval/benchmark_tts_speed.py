# SPDX-License-Identifier: Apache-2.0
"""TTS Speed benchmark.

Measures latency, RTF, throughput, and token throughput for TTS models via the
/v1/audio/speech API.  Supports voice cloning (default) and plain TTS modes,
both streaming and non-streaming.

Usage:
    # Voice cloning, non-streaming
    python benchmarks/eval/benchmark_tts_speed.py \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 10

    # Voice cloning, streaming
    python benchmarks/eval/benchmark_tts_speed.py \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 10 --stream

    # Plain TTS, non-streaming
    python benchmarks/eval/benchmark_tts_speed.py \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 10 --no-ref-audio
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
    make_tts_send_fn,
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
    api_url = f"{base_url}/v1/audio/speech"

    if not os.path.isfile(args.testset):
        logger.error("Testset not found: %s", args.testset)
        return

    # Note (Chenyang): We use the seed-tts-eval dataset by default.
    # TODO (Chenyang): Make datasets configurable.
    samples = load_seedtts_samples(args.testset, args.max_samples)
    logger.info("Prepared %d requests", len(samples))

    save_audio_dir = None
    if args.save_audio and args.output_dir:
        save_audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    gen_kwargs: dict = {}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    send_fn = make_tts_send_fn(
        args.model,
        api_url,
        stream=args.stream,
        no_ref_audio=args.no_ref_audio,
        save_audio_dir=save_audio_dir,
        **gen_kwargs,
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
            "stream": args.stream,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
        }
        save_speed_results(outputs, metrics, config, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark online serving for TTS models."
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
        default="fishaudio/s2-pro",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="seed-tts-eval/en/meta.lst",
        help=(
            "Path to a meta.lst file (one sample per line).  "
            "Accepts any dataset in seed-tts-eval format."
        ),
    )
    parser.add_argument(
        "--no-ref-audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_speed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Send requests with stream=true (SSE audio chunks).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
