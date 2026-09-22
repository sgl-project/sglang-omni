# SPDX-License-Identifier: Apache-2.0
"""Colocated SeedTTS benchmark for per-server latency under shared-pod runs.

Usage:

    # Start multiple Qwen3-Omni servers on the same host first, each on its
    # own port / GPU placement. Then point this harness at all of them:
    python -m benchmarks.eval.benchmark_omni_seedtts_colocated \
        --base-url http://localhost:8000 \
        --base-url http://localhost:8001 \
        --meta seedtts_testset/en/meta.lst \
        --model qwen3-omni \
        --max-samples 50 \
        --max-concurrency 16 \
        --rounds 2

The harness runs one SeedTTS generation-speed benchmark against each server in
parallel and reports both per-run and aggregate per-server metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)


@dataclass
class ColocatedSeedttsConfig:
    base_urls: list[str]
    model: str
    meta: str
    output_dir: str
    lang: str = "en"
    speaker: str = "Ethan"
    voice_clone: bool = False
    max_samples: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    warmup: int = 1
    max_concurrency: int = 1
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    rounds: int = 1
    server_timeout: int = 1200


def _validate_config(config: ColocatedSeedttsConfig) -> None:
    if not os.path.isfile(config.meta):
        raise FileNotFoundError(f"Meta file not found: {config.meta}")
    if not config.base_urls:
        raise ValueError("At least one --base-url is required")
    if len(set(config.base_urls)) != len(config.base_urls):
        raise ValueError("Duplicate --base-url values are not allowed")
    if config.rounds < 1:
        raise ValueError("--rounds must be >= 1")


def _make_server_config(
    config: ColocatedSeedttsConfig,
    *,
    base_url: str,
    server_index: int,
    round_index: int,
) -> OmniSeedttsBenchmarkConfig:
    output_dir = os.path.join(
        config.output_dir,
        f"n{len(config.base_urls)}",
        f"round_{round_index + 1}",
        f"server_{server_index + 1}",
    )
    return OmniSeedttsBenchmarkConfig(
        model=config.model,
        meta=config.meta,
        base_url=base_url,
        lang=config.lang,
        speaker=config.speaker,
        voice_clone=config.voice_clone,
        output_dir=output_dir,
        max_samples=config.max_samples,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        warmup=config.warmup,
        max_concurrency=config.max_concurrency,
        request_rate=config.request_rate,
        disable_tqdm=config.disable_tqdm,
    )


def _mean_metric(summaries: list[dict[str, Any]], key: str) -> float | None:
    values = [summary.get(key) for summary in summaries if summary.get(key) is not None]
    if not values:
        return None
    return round(float(mean(values)), 4)


def _summarize_runs(
    summaries: list[dict[str, Any]],
    *,
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        **extra_fields,
        "completed_requests_total": sum(
            int(summary.get("completed_requests", 0)) for summary in summaries
        ),
        "failed_requests_total": sum(
            int(summary.get("failed_requests", 0)) for summary in summaries
        ),
        "throughput_qps_mean": _mean_metric(summaries, "throughput_qps"),
        "latency_mean_s_mean": _mean_metric(summaries, "latency_mean_s"),
        "latency_p95_s_mean": _mean_metric(summaries, "latency_p95_s"),
        "tok_per_s_agg_mean": _mean_metric(summaries, "tok_per_s_agg"),
        "rtf_mean_mean": _mean_metric(summaries, "rtf_mean"),
    }


def _build_per_server_summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (int(run["server_index"]), str(run["base_url"]))
        grouped.setdefault(key, []).append(run["summary"])

    per_server: list[dict[str, Any]] = []
    for (server_index, base_url), summaries in sorted(grouped.items()):
        per_server.append(
            _summarize_runs(
                summaries,
                extra_fields={
                    "server_index": server_index,
                    "base_url": base_url,
                    "rounds": len(summaries),
                },
            )
        )
    return per_server


def _build_aggregate_summary(
    runs: list[dict[str, Any]],
    *,
    per_server: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [run["summary"] for run in runs]
    return _summarize_runs(
        summaries,
        extra_fields={
            "num_servers": len(per_server),
            "total_runs": len(runs),
        },
    )


def _print_per_server_summaries(per_server: list[dict[str, Any]]) -> None:
    for server in per_server:
        print(
            "  "
            f"server_{server['server_index']} "
            f"lat_mean={server.get('latency_mean_s_mean', 'N/A')}s "
            f"p95={server.get('latency_p95_s_mean', 'N/A')}s "
            f"qps={server.get('throughput_qps_mean', 'N/A')}"
        )


def _print_group_summary(
    aggregate: dict[str, Any],
    *,
    per_server: list[dict[str, Any]],
) -> None:
    print("\n============================================================")
    print("              Colocated SeedTTS Benchmark Result            ")
    print("============================================================")
    print(f"  Servers:                      {aggregate['num_servers']}")
    print(f"  Runs:                         {aggregate['total_runs']}")
    print(
        f"  Completed requests (total):   {aggregate['completed_requests_total']}"
    )
    print(f"  Failed requests (total):      {aggregate['failed_requests_total']}")
    print("------------------------------------------------------------")
    print(
        f"  Throughput qps (mean):        {aggregate.get('throughput_qps_mean', 'N/A')}"
    )
    print(
        f"  Latency mean (s, mean):       {aggregate.get('latency_mean_s_mean', 'N/A')}"
    )
    print(
        f"  Latency p95 (s, mean):        {aggregate.get('latency_p95_s_mean', 'N/A')}"
    )
    print(
        f"  Tok/s agg (mean):             {aggregate.get('tok_per_s_agg_mean', 'N/A')}"
    )
    print(
        f"  RTF mean (mean):              {aggregate.get('rtf_mean_mean', 'N/A')}"
    )
    print("------------------------------------------------------------")
    _print_per_server_summaries(per_server)
    print("============================================================")


async def run_colocated_seedtts_benchmark(
    config: ColocatedSeedttsConfig,
) -> dict[str, Any]:
    _validate_config(config)

    for base_url in config.base_urls:
        wait_for_service(base_url, timeout=config.server_timeout)

    runs: list[dict[str, Any]] = []

    for round_index in range(config.rounds):
        tasks = []
        for server_index, base_url in enumerate(config.base_urls, start=1):
            server_config = _make_server_config(
                config,
                base_url=base_url,
                server_index=server_index,
                round_index=round_index,
            )
            tasks.append(run_omni_seedtts_benchmark(server_config))

        round_results = await asyncio.gather(*tasks)
        for server_index, (base_url, result) in enumerate(
            zip(config.base_urls, round_results, strict=True),
            start=1,
        ):
            runs.append(
                {
                    "round": round_index + 1,
                    "server_index": server_index,
                    "base_url": base_url,
                    "summary": result["summary"],
                    "config": result["config"],
                }
            )

    per_server = _build_per_server_summaries(runs)
    aggregate = _build_aggregate_summary(runs, per_server=per_server)
    _print_group_summary(aggregate, per_server=per_server)

    return {
        "config": asdict(config),
        "aggregate": aggregate,
        "per_server": per_server,
        "runs": runs,
    }


def _config_from_args(args: argparse.Namespace) -> ColocatedSeedttsConfig:
    voice_clone = args.voice_clone and not args.no_ref_audio
    return ColocatedSeedttsConfig(
        base_urls=args.base_url,
        model=args.model,
        meta=args.meta,
        output_dir=args.output_dir,
        lang=args.lang,
        speaker=args.speaker,
        voice_clone=voice_clone,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        rounds=args.rounds,
        server_timeout=args.server_timeout,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SeedTTS generation-speed benchmarks against multiple colocated "
            "Qwen3-Omni servers in parallel and report per-server latency drift."
        )
    )
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        help=(
            "Server base URL. Repeat once per colocated server, e.g. "
            "--base-url http://localhost:8000 "
            "--base-url http://localhost:8001."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="seedtts_testset/en/meta.lst",
        help="Path to a meta.lst file (seed-tts-eval format).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/omni_seedtts_colocated",
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
    voice_clone_group = parser.add_mutually_exclusive_group()
    voice_clone_group.add_argument(
        "--voice-clone",
        dest="voice_clone",
        action="store_true",
        help="Pass ref_audio via 'audios' field for voice cloning.",
    )
    voice_clone_group.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Disable voice cloning.",
    )
    parser.set_defaults(voice_clone=False, no_ref_audio=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=1200)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _config_from_args(args)
    results = asyncio.run(run_colocated_seedtts_benchmark(config))
    save_json_results(results, config.output_dir, "colocated_eval_results.json")


if __name__ == "__main__":
    main()
