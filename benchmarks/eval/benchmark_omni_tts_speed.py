# SPDX-License-Identifier: Apache-2.0
"""TTS Speed benchmark for Omni models (Qwen3-Omni).

Measures latency, RTF, throughput, and token throughput via /v1/chat/completions
with modalities: ["text", "audio"].

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
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import get_wav_duration, wait_for_service
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts_speed import (
    build_speed_results,
    print_speed_summary,
    save_speed_results,
)
from benchmarks.tasks.voice_clone import VoiceCloneOmni

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_PREVIEW_LENGTH = 60


@dataclass
class OmniTtsSpeedBenchmarkConfig:
    model: str
    testset: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    lang: str = "en"
    speaker: str = "Ethan"
    no_ref_audio: bool = False
    output_dir: str | None = None
    max_samples: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    warmup: int = 1
    max_concurrency: int = 1
    request_rate: float = float("inf")
    save_audio: bool = False
    disable_tqdm: bool = False


def _build_base_url(config: OmniTtsSpeedBenchmarkConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


def _build_results_config(
    config: OmniTtsSpeedBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    return {
        "model": config.model,
        "base_url": base_url,
        "testset": config.testset,
        "no_ref_audio": config.no_ref_audio,
        "lang": config.lang,
        "speaker": config.speaker,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "warmup": config.warmup,
        "max_concurrency": config.max_concurrency,
        "request_rate": config.request_rate,
    }


def make_omni_tts_send_fn(
    model_name: str,
    api_url: str,
    *,
    lang: str = "en",
    voice_clone: bool = True,
    speaker: str = "Ethan",
    max_tokens: int = 256,
    temperature: float = 0.7,
    save_audio_dir: str | None = None,
) -> SendFn:
    """Return a *send_fn* for Omni models via VoiceCloneOmni."""
    task = VoiceCloneOmni()

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        start_time = time.perf_counter()
        try:
            wav_bytes, _, usage = await task.generate_speech(
                session,
                api_url,
                model_name,
                sample,
                lang,
                speaker=speaker,
                max_tokens=max_tokens,
                temperature=temperature,
                voice_clone=voice_clone,
            )
            result.audio_duration_s = get_wav_duration(wav_bytes)
            elapsed = time.perf_counter() - start_time
            if result.audio_duration_s > 0:
                result.is_success = True
                result.rtf = elapsed / result.audio_duration_s
            else:
                result.error = f"Invalid audio ({len(wav_bytes)} bytes)"

            if usage:
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

            # Note (chenyang): engine_time_s should be the time taken by
            # the engine. Current omni chat completions has no X-Engine-Time
            # header, so we use request elapsed time as engine_time_s proxy.
            # This shall largely affect the results at high concurrency,
            # since the wait time is included in the request elapsed time.

            result.engine_time_s = elapsed
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s

            if save_audio_dir:
                path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
                with open(path, "wb") as f:
                    f.write(wav_bytes)
                result.wav_path = path
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


async def run_omni_tts_speed_benchmark(
    config: OmniTtsSpeedBenchmarkConfig,
) -> dict:
    if not os.path.isfile(config.testset):
        raise FileNotFoundError(f"Testset not found: {config.testset}")

    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_seedtts_samples(config.testset, config.max_samples)
    logger.info("Prepared %d requests", len(samples))

    save_audio_dir = None
    if config.save_audio and config.output_dir:
        save_audio_dir = os.path.join(config.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    send_fn = make_omni_tts_send_fn(
        config.model,
        api_url,
        lang=config.lang,
        voice_clone=not config.no_ref_audio,
        speaker=config.speaker,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        save_audio_dir=save_audio_dir,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
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
    return benchmark_results


def _config_from_args(args: argparse.Namespace) -> OmniTtsSpeedBenchmarkConfig:
    return OmniTtsSpeedBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        lang=args.lang,
        speaker=args.speaker,
        testset=args.testset,
        no_ref_audio=args.no_ref_audio,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        save_audio=args.save_audio,
        disable_tqdm=args.disable_tqdm,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_omni_tts_speed_benchmark(config)
    print_speed_summary(results["summary"], config.model)
    return results


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
