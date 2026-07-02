# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR concurrency-scaling benchmark on SeedTTS EN (issue #646).

Sweeps ASR transcription fan-out (concurrency) against a *running* Qwen3-ASR
SGLang Omni router and reports, for each concurrency level, the metrics tracked
in issue #646: corpus/per-sample WER, wall-clock, throughput, latency
percentiles, RTF, and per-worker routing balance. This produces the repeatable
concurrency-scaling data the issue's acceptance criteria ask for, and lets us
decide the right ASR fan-out for SeedTTS EN transcription / WER workloads.

This script transcribes the SeedTTS *reference* clips directly (no TTS
generation step), so it isolates ASR behavior from TTS.

This file is the executable CLI wrapper: the shared transcription/scoring
layer (run_asr_transcription and build_asr_eval_results) lives in
benchmarks.tasks.asr and is imported here and by the Qwen3-ASR correctness
gate (tests/test_model/test_qwen3_asr_ci.py), so the gate is just this
benchmark run plus thresholds.

Usage:

    # Download the test set once:
    python -m benchmarks.dataset.prepare --dataset seedtts

    # Launch Qwen3-ASR (DP=2 to match TTS CI):
    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-ASR-1.7B \
        --dp-size 2 \
        --port 8000

    # Sweep the issue's matrix (3 repeats each) over the full SeedTTS EN set:
    python -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
        --port 8000 \
        --concurrencies 1,2,4,8,16,32,64 \
        --repeats 3

    # Quick local smoke on a 20-sample subset:
    python -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
        --port 8000 --max-samples 20 --concurrencies 2,32 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics

import requests

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    build_asr_eval_results,
    run_asr_transcription,
)

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32,64"


def _fetch_worker_snapshot(host: str, port: int) -> dict | None:
    """Best-effort read of the router /workers snapshot (None if unavailable)."""
    try:
        response = requests.get(
            f"http://{host}:{port}/workers",
            timeout=10,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _worker_delta(before: dict | None, after: dict | None) -> dict:
    """Routed/successful/failed deltas and per-worker routed balance."""
    if not before or not after:
        return {}

    def _by_id(snapshot: dict, key: str) -> dict[str, int]:
        return {
            str(w.get("display_id")): int(w.get(key, 0))
            for w in snapshot.get("workers", [])
        }

    out: dict[str, object] = {}
    for key in ("routed_requests", "successful_requests", "failed_requests"):
        before_by_id = _by_id(before, key)
        after_by_id = _by_id(after, key)
        deltas = {
            wid: after_by_id.get(wid, 0) - before_by_id.get(wid, 0)
            for wid in after_by_id
        }
        out[f"total_{key}"] = sum(deltas.values())
        if key == "routed_requests":
            out["per_worker_routed"] = deltas
    return out


async def _run_repeat(args, samples, concurrency: int, repeat: int) -> dict:
    before = _fetch_worker_snapshot(args.host, args.port)
    outputs, wall_clock_s = await run_asr_transcription(
        samples,
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        lang=args.lang,
        concurrency=concurrency,
    )
    after = _fetch_worker_snapshot(args.host, args.port)

    results = build_asr_eval_results(
        samples,
        outputs,
        wall_clock_s,
        args.lang,
        model_path=args.model_path,
        concurrency=concurrency,
    )
    summary = results["summary"]
    speed = results["speed"]
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "evaluated": summary["evaluated"],
        "total": summary["total_samples"],
        "skipped": summary["skipped"],
        "corpus_wer": summary["corpus_wer"],
        "per_sample_wer_max": summary["wer_per_sample_max"],
        "wall_clock_s": wall_clock_s,
        "throughput_samples_per_s": speed["throughput_samples_per_s"],
        "latency_mean_s": speed["latency_mean_s"],
        "latency_p95_s": speed["latency_p95_s"],
        "latency_p99_s": speed["latency_p99_s"],
        "rtf_mean": speed["rtf_mean"],
        "rtf_p95": speed["rtf_p95"],
        "worker": _worker_delta(before, after),
    }


def _aggregate(repeats: list[dict]) -> dict:
    """Mean/best/worst across repeats for the headline metrics."""

    def _stat(key: str) -> dict:
        values = [r[key] for r in repeats]
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "concurrency": repeats[0]["concurrency"],
        "repeats": len(repeats),
        "evaluated": repeats[0]["evaluated"],
        "total": repeats[0]["total"],
        "skipped": repeats[0]["skipped"],
        "corpus_wer": _stat("corpus_wer"),
        "per_sample_wer_max": _stat("per_sample_wer_max"),
        "wall_clock_s": _stat("wall_clock_s"),
        "throughput_samples_per_s": _stat("throughput_samples_per_s"),
        "latency_mean_s": _stat("latency_mean_s"),
        "latency_p95_s": _stat("latency_p95_s"),
        "latency_p99_s": _stat("latency_p99_s"),
        "rtf_mean": _stat("rtf_mean"),
        "rtf_p95": _stat("rtf_p95"),
        "per_repeat": repeats,
    }


def _print_table(aggregates: list[dict]) -> None:
    header = (
        "| conc | reps | wall(s) mean | thrpt mean | thrpt best | "
        "lat mean(s) | lat p95(s) | rtf mean | rtf p95 | corpus WER | max WER |"
    )
    sep = "|---:" * 11 + "|"
    print("\n" + header)
    print(sep)
    for agg in aggregates:
        print(
            f"| {agg['concurrency']} | {agg['repeats']} "
            f"| {agg['wall_clock_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['max']:.3f} "
            f"| {agg['latency_mean_s']['mean']:.3f} "
            f"| {agg['latency_p95_s']['mean']:.3f} "
            f"| {agg['rtf_mean']['mean']:.4f} "
            f"| {agg['rtf_p95']['mean']:.4f} "
            f"| {agg['corpus_wer']['max']:.4f} "
            f"| {agg['per_sample_wer_max']['max']:.4f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of the running Qwen3-ASR SGLang Omni router.",
    )
    parser.add_argument(
        "--meta",
        default=DATASETS["seedtts"],
        help="SeedTTS source (HF repo id or local meta.lst).",
    )
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0 = full SeedTTS set; 1088 for EN).",
    )
    parser.add_argument(
        "--concurrencies",
        default=DEFAULT_CONCURRENCIES,
        help="Comma-separated ASR concurrency levels to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--model-path",
        default=QWEN3_ASR_MODEL_PATH,
        help="ASR model id served by the router.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded warmup pass before timing each concurrency.",
    )
    parser.add_argument(
        "--output",
        default="qwen3_asr_concurrency_results.json",
        help="Where to write the full JSON results.",
    )
    return parser.parse_args()


async def _sweep(args, samples, concurrencies: list[int]) -> list[dict]:
    aggregates: list[dict] = []
    for concurrency in concurrencies:
        if args.warmup:
            print(f"[conc={concurrency}] warmup pass ...")
            await run_asr_transcription(
                samples,
                host=args.host,
                port=args.port,
                model_path=args.model_path,
                lang=args.lang,
                concurrency=concurrency,
            )
        repeats: list[dict] = []
        for repeat in range(1, args.repeats + 1):
            result = await _run_repeat(args, samples, concurrency, repeat)
            repeats.append(result)
            print(
                f"[conc={concurrency} rep={repeat}] "
                f"wall={result['wall_clock_s']:.3f}s "
                f"thrpt={result['throughput_samples_per_s']:.3f}/s "
                f"lat_mean={result['latency_mean_s']:.3f}s "
                f"lat_p95={result['latency_p95_s']:.3f}s "
                f"rtf_mean={result['rtf_mean']:.4f} "
                f"corpus_wer={result['corpus_wer']:.4f} "
                f"skipped={result['skipped']}"
            )
            if result["worker"].get("per_worker_routed"):
                print(f"    routed per worker: {result['worker']['per_worker_routed']}")
        aggregates.append(_aggregate(repeats))
    return aggregates


def main() -> None:
    args = parse_args()
    concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    max_samples = args.max_samples if args.max_samples > 0 else None

    samples = load_seedtts_samples(args.meta, max_samples=max_samples, split=args.lang)
    print(
        f"Loaded {len(samples)} SeedTTS {args.lang} samples; "
        f"sweeping concurrency={concurrencies} x {args.repeats} repeats "
        f"against {args.host}:{args.port} ({args.model_path})"
    )

    aggregates = asyncio.run(_sweep(args, samples, concurrencies))
    _print_table(aggregates)

    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "meta": args.meta,
            "lang": args.lang,
            "model_path": args.model_path,
            "num_samples": len(samples),
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "results": aggregates,
    }
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
