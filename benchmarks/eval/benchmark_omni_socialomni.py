# SPDX-License-Identifier: Apache-2.0
"""SocialOmni benchmark entrypoint for sglang-omni.

Usage:
    python -m benchmarks.dataset.prepare --dataset socialomni
    python -m benchmarks.eval.benchmark_omni_socialomni \
        --model qwen3-omni --port 8000 --level level1

    python -m benchmarks.eval.benchmark_omni_socialomni \
        --model qwen3-omni --port 8000 --level level2 --judges 1 \
        --judge-base-url http://localhost:8001 --judge-model qwen3-omni-judge

    python -m benchmarks.eval.benchmark_omni_socialomni \
        --model qwen3-omni --port 8000 --level level2 --judges 3 \
        --judge-base-url http://localhost:8001 --judge-model judge-a \
        --judge-base-url http://localhost:8002 --judge-model judge-b \
        --judge-base-url https://openai-compatible.example/v1 --judge-model judge-c
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.socialomni import (
    DEFAULT_SOCIALOMNI_DIRS,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
)
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.socialomni import (
    build_base_url,
    build_chat_api_url,
    build_socialomni_level2_metrics,
    build_socialomni_level2_summary,
    compute_socialomni_level1_results,
    make_socialomni_level1_send_fn,
    preflight_chat_completion_endpoint,
    print_socialomni_level1_summary,
    print_socialomni_level2_summary,
    run_socialomni_level2_benchmark,
    validate_judge_specs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@dataclass(frozen=True)
class SocialOmniEvalConfig:
    model: str
    level: str
    dataset_name: str = "socialomni"
    dataset_dir: str | None = None
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    output_dir: str = "results/socialomni"
    max_samples: int | None = None
    max_tokens: int = 64
    temperature: float = 0.0
    warmup: int = 0
    max_concurrency: int = 1
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    timeout_s: int = 300
    judges: int = 1
    judge_base_urls: tuple[str, ...] = ()
    judge_models: tuple[str, ...] = ()


def _resolve_dataset_root(config: SocialOmniEvalConfig) -> Path:
    if config.dataset_dir:
        return Path(config.dataset_dir)
    return Path(DEFAULT_SOCIALOMNI_DIRS[config.dataset_name])


def _result_filename(dataset_name: str, level: str) -> str:
    return f"{dataset_name.replace('-', '_')}_{level}_results.json"


async def run_socialomni_level1_eval(config: SocialOmniEvalConfig) -> dict[str, Any]:
    """Run the SocialOmni level1 benchmark."""
    dataset_root = _resolve_dataset_root(config)
    samples = load_socialomni_level1_samples(dataset_root, config.max_samples)
    base_url = build_base_url(
        base_url=config.base_url, host=config.host, port=config.port
    )
    api_url = build_chat_api_url(base_url)

    timeout = aiohttp.ClientTimeout(total=config.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await preflight_chat_completion_endpoint(
            session,
            api_url=api_url,
            model_name=config.model,
            endpoint_name="main endpoint",
        )

    send_fn = make_socialomni_level1_send_fn(
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
            timeout_s=config.timeout_s,
        )
    )
    request_results = await runner.run(samples, send_fn)
    summary, per_sample = compute_socialomni_level1_results(samples, request_results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)

    results = {
        "benchmark": "socialomni",
        "level": "level1",
        "config": {
            "model": config.model,
            "base_url": base_url,
            "dataset_name": config.dataset_name,
            "dataset_root": str(dataset_root),
            "max_samples": config.max_samples,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "max_concurrency": config.max_concurrency,
            "warmup": config.warmup,
        },
        "summary": summary,
        "speed": speed,
        "per_sample": per_sample,
    }
    save_json_results(
        results, config.output_dir, _result_filename(config.dataset_name, config.level)
    )
    print_socialomni_level1_summary(summary)
    return results


async def run_socialomni_level2_eval(config: SocialOmniEvalConfig) -> dict[str, Any]:
    """Run the SocialOmni level2 benchmark."""
    dataset_root = _resolve_dataset_root(config)
    samples = load_socialomni_level2_samples(dataset_root, config.max_samples)
    base_url = build_base_url(
        base_url=config.base_url, host=config.host, port=config.port
    )
    api_url = build_chat_api_url(base_url)
    judge_specs = validate_judge_specs(
        config.judges,
        list(config.judge_base_urls),
        list(config.judge_models),
    )

    timeout = aiohttp.ClientTimeout(total=config.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await preflight_chat_completion_endpoint(
            session,
            api_url=api_url,
            model_name=config.model,
            endpoint_name="main endpoint",
        )
        for judge in judge_specs:
            await preflight_chat_completion_endpoint(
                session,
                api_url=judge.api_url,
                model_name=judge.model,
                endpoint_name=f"judge endpoint ({judge.model})",
            )

    per_sample, primary_requests, judge_requests, wall_clock_s = (
        await run_socialomni_level2_benchmark(
            samples,
            api_url=api_url,
            model_name=config.model,
            judge_specs=judge_specs,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_concurrency=config.max_concurrency,
            timeout_s=config.timeout_s,
        )
    )
    summary = build_socialomni_level2_summary(per_sample)
    metrics = build_socialomni_level2_metrics(
        per_sample,
        primary_requests,
        judge_requests,
        wall_clock_s=wall_clock_s,
    )

    results = {
        "benchmark": "socialomni",
        "level": "level2",
        "config": {
            "model": config.model,
            "base_url": base_url,
            "dataset_name": config.dataset_name,
            "dataset_root": str(dataset_root),
            "max_samples": config.max_samples,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "max_concurrency": config.max_concurrency,
            "judges": config.judges,
            "judge_specs": [
                {"base_url": judge.base_url, "model": judge.model}
                for judge in judge_specs
            ],
        },
        "summary": summary,
        "per_sample": per_sample,
        **metrics,
    }
    save_json_results(
        results, config.output_dir, _result_filename(config.dataset_name, config.level)
    )
    print_socialomni_level2_summary(summary)
    return results


async def run_socialomni_eval(config: SocialOmniEvalConfig) -> dict[str, Any]:
    """Run the requested SocialOmni benchmark level."""
    if config.level == "level1":
        return await run_socialomni_level1_eval(config)
    return await run_socialomni_level2_eval(config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SocialOmni benchmark for sglang-omni."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--level", choices=["level1", "level2"], required=True)
    parser.add_argument(
        "--dataset-name",
        choices=["socialomni", "socialomni-mini"],
        default="socialomni",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override the prepared dataset root (defaults to ./socialomni or ./socialomni-mini).",
    )
    parser.add_argument("--output-dir", type=str, default="results/socialomni")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--judges", choices=[1, 3], type=int, default=1)
    parser.add_argument("--judge-base-url", action="append", default=[])
    parser.add_argument("--judge-model", action="append", default=[])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = SocialOmniEvalConfig(
        model=args.model,
        level=args.level,
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        timeout_s=args.timeout_s,
        judges=args.judges,
        judge_base_urls=tuple(args.judge_base_url),
        judge_models=tuple(args.judge_model),
    )

    wait_for_service(
        build_base_url(base_url=config.base_url, host=config.host, port=config.port)
    )
    asyncio.run(run_socialomni_eval(config))


if __name__ == "__main__":
    main()
