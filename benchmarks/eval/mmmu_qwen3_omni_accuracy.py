# SPDX-License-Identifier: Apache-2.0
"""MMMU accuracy benchmark for Qwen3-Omni.

Evaluates VLM accuracy on the MMMU validation set via ``/v1/chat/completions``
with image input and text-only output.

Usage:
    python benchmarks/eval/mmmu_qwen3_omni_accuracy.py \
        --model qwen3-omni --port 8000 --max-samples 20

    # With concurrency
    python benchmarks/eval/mmmu_qwen3_omni_accuracy.py \
        --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmmu import (
    MMMUSample,
    eval_open,
    load_mmmu_samples,
    parse_multi_choice_response,
    parse_open_response,
)

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
    warmup: int = 0
    request_rate: float = float("inf")
    disable_tqdm: bool = False


def _build_base_url(config: MMMUEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


def make_mmmu_send_fn(
    model_name: str,
    api_url: str,
    *,
    max_tokens: int = 30,
    temperature: float = 0.0,
) -> SendFn:
    """Return a *send_fn* that sends an MMMUSample to /v1/chat/completions.

    Uses the sglang-omni request format with a top-level ``images`` field
    and ``modalities: ["text"]`` (text-only output, no audio generation).
    """

    async def send_fn(
        session: aiohttp.ClientSession, sample: MMMUSample
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.prompt[:60],
        )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": sample.prompt}],
            "images": sample.image_uris,
            "modalities": ["text"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                body = await response.json()

            content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            result.text = content or ""
            result.is_success = True

            usage = body.get("usage", {})
            if usage:
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def compute_mmmu_metrics(
    samples: list[MMMUSample],
    results: list[RequestResult],
    eval_time_s: float,
) -> tuple[dict, list[dict]]:
    """Parse answers, compute accuracy, and build per-sample detail list.

    Returns ``(summary_dict, per_sample_list)``.
    """
    correct = 0
    failed = 0
    per_sample: list[dict] = []

    for sample, result in zip(samples, results):
        if not result.is_success:
            per_sample.append(
                {
                    "sample_id": sample.sample_id,
                    "subject": sample.subject,
                    "expected": sample.answer,
                    "predicted": "",
                    "raw_response": result.error,
                    "is_correct": False,
                    "latency_s": round(result.latency_s, 4),
                    "is_success": False,
                    "error": result.error,
                }
            )
            failed += 1
            continue

        gold = sample.answer
        if (
            sample.question_type == "multiple-choice"
            and sample.all_choices
            and sample.index2ans
        ):
            predicted = parse_multi_choice_response(
                result.text,
                sample.all_choices,
                sample.index2ans,
            )
            is_correct = gold is not None and predicted == gold
        else:
            parsed_list = parse_open_response(result.text)
            is_correct = gold is not None and eval_open(gold, parsed_list)
            predicted = ", ".join(map(str, parsed_list))

        if is_correct:
            correct += 1

        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "subject": sample.subject,
                "question_type": sample.question_type,
                "expected": sample.answer,
                "predicted": predicted,
                "raw_response": result.text,
                "is_correct": is_correct,
                "latency_s": round(result.latency_s, 4),
                "is_success": True,
                "error": "",
            }
        )

    total = len(samples)
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "eval_time_s": round(eval_time_s, 3),
        "failed": failed,
    }
    return summary, per_sample


def print_mmmu_summary(metrics: dict, model_name: str) -> None:
    """Print formatted MMMU accuracy summary to stdout."""
    lw = 28
    print(f"\n{'=' * 50}")
    print(f"  MMMU Accuracy — {model_name}")
    print(f"{'=' * 50}")
    print(f"  {'Total samples:':<{lw}} {metrics['total_samples']}")
    print(f"  {'Correct:':<{lw}} {metrics['correct']}")
    print(
        f"  {'Accuracy:':<{lw}} {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.1f}%)"
    )
    print(f"  {'Failed requests:':<{lw}} {metrics['failed']}")
    print(f"  {'Eval time (s):':<{lw}} {metrics['eval_time_s']:.1f}")
    print(f"{'=' * 50}\n")


def save_mmmu_results(
    per_sample: list[dict],
    metrics: dict,
    config_dict: dict,
    output_dir: str,
) -> None:
    """Save results to ``mmmu_results.json`` in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "mmmu_results.json")
    payload = {
        "summary": metrics,
        "config": config_dict,
        "per_sample": per_sample,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", path)


async def run_mmmu_eval(config: MMMUEvalConfig) -> dict:
    """Run full MMMU evaluation and return results dict.

    Returns a dict with ``summary``, ``config``, and ``per_sample`` keys.
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
    results = await runner.run(samples, send_fn)

    summary, per_sample = compute_mmmu_metrics(
        samples, results, eval_time_s=runner.wall_clock_s
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

    if config.output_dir:
        save_mmmu_results(per_sample, summary, config_dict, config.output_dir)

    return {
        "summary": summary,
        "config": config_dict,
        "per_sample": per_sample,
    }


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
    print_mmmu_summary(results["summary"], config.model)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMMU accuracy benchmark for VLM models served by sglang-omni."
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
    parser.add_argument("--warmup", type=int, default=0)
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
