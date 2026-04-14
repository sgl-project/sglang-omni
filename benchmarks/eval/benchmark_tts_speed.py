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

    # Voice cloning, non-streaming, high concurrency
    python benchmarks/eval/benchmark_tts_speed.py \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst --max-samples 50 \
        --concurrency 20

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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts_speed import (
    build_speed_results,
    make_tts_send_fn,
    print_speed_summary,
    save_generated_audio_metadata,
    save_speed_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TtsSpeedBenchmarkConfig:
    model: str
    testset: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    no_ref_audio: bool = False
    output_dir: str | None = None
    max_samples: int | None = None
    max_new_tokens: int | None = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    warmup: int = 1
    concurrency: int = 1
    request_rate: float = float("inf")
    save_audio: bool = False
    stream: bool = False
    disable_tqdm: bool = False


def _build_base_url(config: TtsSpeedBenchmarkConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


def _build_generation_kwargs(config: TtsSpeedBenchmarkConfig) -> dict:
    generation_kwargs: dict = {}
    if config.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = config.max_new_tokens
    if config.temperature is not None:
        generation_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        generation_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        generation_kwargs["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = config.repetition_penalty
    return generation_kwargs


def _build_results_config(
    config: TtsSpeedBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    return {
        "model": config.model,
        "base_url": base_url,
        "testset": config.testset,
        "no_ref_audio": config.no_ref_audio,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "warmup": config.warmup,
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
    }


async def run_tts_speed_benchmark(config: TtsSpeedBenchmarkConfig) -> dict:
    if not os.path.isfile(config.testset):
        raise FileNotFoundError(f"Testset not found: {config.testset}")

    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"

    # Note (Chenyang): We use the seed-tts-eval dataset by default.
    # TODO (Chenyang): Make datasets configurable.
    samples = load_seedtts_samples(config.testset, config.max_samples)
    logger.info("Prepared %d requests", len(samples))

    save_audio_dir = None
    if config.save_audio and config.output_dir:
        save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
        os.makedirs(save_audio_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    send_fn = make_tts_send_fn(
        config.model,
        api_url,
        stream=config.stream,
        no_ref_audio=config.no_ref_audio,
        save_audio_dir=save_audio_dir,
        **generation_kwargs,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    if config.output_dir:
        save_speed_results(outputs, metrics, results_config, config.output_dir)
        if config.save_audio:
            save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def _config_from_args(args: argparse.Namespace) -> TtsSpeedBenchmarkConfig:
    return TtsSpeedBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        testset=args.testset,
        no_ref_audio=args.no_ref_audio,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        save_audio=args.save_audio,
        stream=args.stream,
        disable_tqdm=args.disable_tqdm,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_tts_speed_benchmark(config)
    print_speed_summary(
        results["summary"],
        config.model,
        concurrency=config.concurrency,
    )
    return results


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
        "--concurrency",
        dest="concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests.",
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
