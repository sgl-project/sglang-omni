# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark for sglang-omni models.

Evaluates VLM accuracy and performance on the MMMU validation set via
``/v1/chat/completions`` with image input and text-only output.

Usage:
    python benchmarks/eval/mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 20

    # With concurrency
    python benchmarks/eval/mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.mmmu import (
    compute_mmmu_metrics,
    make_mmmu_send_fn,
    print_mmmu_accuracy_summary,
    save_mmmu_results,
)
from benchmarks.tasks.tts_speed import print_speed_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MMMUEvalConfig:
    model: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 1
    request_rate: float = float("inf")
    disable_tqdm: bool = False


def _build_base_url(config: MMMUEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_mmmu_eval(config: MMMUEvalConfig) -> dict:
    """Run full MMMU evaluation and return results dict.

    Returns a dict with ``summary``, ``speed``, ``config``, and
    ``per_sample`` keys.
    """
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_mmmu_samples(config.max_samples)
    logger.info("Prepared %d MMMU samples", len(samples))

    send_fn = make_mmmu_send_fn(
        config.model,
        api_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    summary, per_sample = compute_mmmu_metrics(samples, request_results)
    speed_metrics = compute_speed_metrics(
        request_results, wall_clock_s=runner.wall_clock_s
    )

    config_dict = {
        "model": config.model,
        "base_url": base_url,
        "max_samples": config.max_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "max_concurrency": config.max_concurrency,
        "warmup": config.warmup,
    }

    results = {
        "summary": summary,
        "speed": speed_metrics,
        "config": config_dict,
        "per_sample": per_sample,
    }

    if config.output_dir:
        save_mmmu_results(results, config.output_dir)

    return results


def _config_from_args(args: argparse.Namespace) -> MMMUEvalConfig:
    return MMMUEvalConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_mmmu_eval(config)
    print_mmmu_accuracy_summary(results["summary"], config.model)
    print_speed_summary(
        results["speed"], config.model, config.max_concurrency, title="MMMU Speed"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMMU benchmark for VLM models served by sglang-omni."
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
    parser.add_argument("--output-dir", type=str, default="results/mmmu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
