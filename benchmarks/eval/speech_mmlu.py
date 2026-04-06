# SPDX-License-Identifier: Apache-2.0
"""Speech MMLU benchmark: evaluate audio comprehension accuracy.

Sends spoken MMLU questions to a running sglang-omni server and evaluates
multiple-choice accuracy. Supports text-only and text+audio output modes.

Dataset: XiaomiMiMo/SpeechMMLU (8,549 samples, 34 subjects)

Usage::

    # Audio-in -> Text-out (accuracy only)
    python benchmarks/eval/speech_mmlu.py \\
        --model qwen3-omni --port 8000 \\
        --modalities text --max-samples 100

    # Audio-in -> Text+Audio-out (accuracy + audio metrics)
    python benchmarks/eval/speech_mmlu.py \\
        --model qwen3-omni --port 8000 \\
        --modalities text+audio --max-samples 100 --save-audio

    # Filter by subject
    python benchmarks/eval/speech_mmlu.py \\
        --model qwen3-omni --port 8000 \\
        --subjects anatomy,virology --max-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.speech_mmlu import load_speech_mmlu_samples
from benchmarks.metrics.accuracy import compute_accuracy_metrics
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.speech_mmlu import (
    build_speech_mmlu_results,
    make_speech_mmlu_send_fn,
    print_speech_mmlu_summary,
    save_speech_mmlu_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SpeechMmluBenchmarkConfig:
    model: str = "qwen3-omni"
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    modalities: list[str] = field(default_factory=lambda: ["text"])
    cache_dir: str = "benchmarks/cache/speech_mmlu"
    output_dir: str = "results/speech_mmlu"
    max_samples: int | None = None
    subjects: list[str] | None = None
    prompt: str | None = None
    max_tokens: int = 32
    temperature: float = 0.0
    warmup: int = 1
    max_concurrency: int = 1
    request_rate: float = float("inf")
    save_audio: bool = False
    disable_tqdm: bool = False
    seed: int | None = None


def _build_base_url(config: SpeechMmluBenchmarkConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_speech_mmlu_benchmark(config: SpeechMmluBenchmarkConfig) -> dict:
    """Run the Speech MMLU benchmark end-to-end."""
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    # Load dataset
    samples = load_speech_mmlu_samples(
        cache_dir=config.cache_dir,
        max_samples=config.max_samples,
        subjects=config.subjects,
        seed=config.seed,
    )
    if not samples:
        raise ValueError("No samples loaded. Check --subjects filter or dataset.")
    logger.info("Loaded %d samples for evaluation", len(samples))

    # Prepare audio save dir
    save_audio_dir = None
    if config.save_audio and config.output_dir:
        save_audio_dir = os.path.join(config.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    # Build send function
    send_fn_kwargs = {
        "modalities": config.modalities,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "save_audio_dir": save_audio_dir,
    }
    if config.prompt is not None:
        send_fn_kwargs["prompt"] = config.prompt

    send_fn = make_speech_mmlu_send_fn(config.model, api_url, **send_fn_kwargs)

    # Run benchmark
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    # Post-process
    speech_results = build_speech_mmlu_results(
        request_results, samples, config.modalities
    )
    accuracy_metrics = compute_accuracy_metrics(
        [
            {
                "subject": r.subject,
                "correct_answer": r.correct_answer,
                "predicted_answer": r.predicted_answer,
                "is_correct": r.is_correct,
                "is_parseable": r.is_parseable,
            }
            for r in speech_results
        ]
    )

    # Speed metrics (useful for both modes, but especially text+audio)
    speed_metrics = compute_speed_metrics(
        request_results, wall_clock_s=runner.wall_clock_s
    )

    # Print summary
    print_speech_mmlu_summary(
        accuracy_metrics,
        config.model,
        speed_metrics=speed_metrics if "audio" in config.modalities else None,
    )

    # Save results
    if config.output_dir:
        results_config = {
            "model": config.model,
            "base_url": base_url,
            "modalities": config.modalities,
            "max_samples": config.max_samples,
            "subjects": config.subjects,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "warmup": config.warmup,
            "max_concurrency": config.max_concurrency,
            "seed": config.seed,
        }
        save_speech_mmlu_results(
            speech_results,
            accuracy_metrics,
            results_config,
            config.output_dir,
            speed_metrics=speed_metrics if "audio" in config.modalities else None,
        )

    return {
        "accuracy": accuracy_metrics,
        "speed": speed_metrics,
    }


def _parse_modalities(value: str) -> list[str]:
    """Parse modalities from CLI arg: 'text' or 'text+audio'."""
    if value == "text":
        return ["text"]
    elif value == "text+audio":
        return ["text", "audio"]
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid modalities: {value}. Use 'text' or 'text+audio'."
        )


def _config_from_args(args: argparse.Namespace) -> SpeechMmluBenchmarkConfig:
    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]

    return SpeechMmluBenchmarkConfig(
        model=args.model,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        modalities=_parse_modalities(args.modalities),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        subjects=subjects,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        save_audio=args.save_audio,
        disable_tqdm=args.disable_tqdm,
        seed=args.seed,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    return await run_speech_mmlu_benchmark(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speech MMLU benchmark: evaluate audio comprehension accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    # Model
    parser.add_argument(
        "--model", type=str, default="qwen3-omni", help="Model name for API requests."
    )

    # Evaluation mode
    parser.add_argument(
        "--modalities",
        type=str,
        choices=["text", "text+audio"],
        default="text",
        help="Output modalities: 'text' (accuracy only) or 'text+audio' (accuracy + audio).",
    )

    # Dataset
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="benchmarks/cache/speech_mmlu",
        help="Directory for cached dataset files.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples.")
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subjects to evaluate (e.g. anatomy,virology).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for subsampling."
    )

    # Generation
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for the model. Default: standard MCQ instruction.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Runner
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))

    # Output
    parser.add_argument("--output-dir", type=str, default="results/speech_mmlu")
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio files (text+audio mode only).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")

    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
