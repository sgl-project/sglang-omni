# SPDX-License-Identifier: Apache-2.0
"""MMAU audio-understanding benchmark for SGLang Omni models."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmau import MmauSample, load_mmau_samples
from benchmarks.metrics.mmsu import compute_mmsu_metrics, print_mmsu_summary
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.audio_understanding import (
    build_mmsu_results,
    make_mmsu_send_fn,
    save_mmsu_results,
)

DEFAULT_PROMPT = (
    "Listen to the audio and answer the multiple-choice question. "
    "Reply with only the option letter."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run(
    args: argparse.Namespace,
    *,
    samples: list[MmauSample] | None = None,
) -> dict:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"
    if samples is None:
        samples = load_mmau_samples(
            max_samples=args.max_samples,
            categories=args.categories.split(",") if args.categories else None,
            tasks=args.tasks.split(",") if args.tasks else None,
            seed=args.seed,
            repo_id=args.repo_id,
            split=args.split,
        )

    send_fn_kwargs = dict(
        modalities=["text"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.prompt:
        send_fn_kwargs["prompt"] = args.prompt
    send_fn = make_mmsu_send_fn(args.model, api_url, **send_fn_kwargs)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
            timeout_s=args.timeout_s,
        )
    )
    request_results = await runner.run(samples, send_fn)

    results = build_mmsu_results(request_results, samples, ["text"])
    metrics = compute_mmsu_metrics(results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    output = {
        "accuracy": metrics,
        "speed": speed,
        "per_sample": [asdict(result) for result in results],
    }

    if args.output_dir:
        save_mmsu_results(
            results,
            metrics,
            {
                "model": args.model,
                "base_url": base_url,
                "repo_id": args.repo_id,
                "split": args.split,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            args.output_dir,
            benchmark_name="mmau",
            speed_metrics=speed,
        )

    return output


def main() -> None:
    p = argparse.ArgumentParser(description="MMAU benchmark.")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", type=str, default="qwen3-omni")
    p.add_argument("--output-dir", type=str, default="results/mmau")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--tasks", type=str, default=None)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--disable-tqdm", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--repo-id", type=str, default="lmms-lab/mmau")
    p.add_argument("--split", type=str, default="test_mini")

    args = p.parse_args()
    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    output = asyncio.run(run(args))
    print_mmsu_summary(
        output["accuracy"],
        args.model,
        benchmark_name="MMAU",
        speed_metrics=output["speed"],
    )


if __name__ == "__main__":
    main()
