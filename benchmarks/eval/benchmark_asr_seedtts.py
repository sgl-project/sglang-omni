# SPDX-License-Identifier: Apache-2.0
# Author:
# chenyang zhao: https://github.com/zhaochenyang20
# PoTaTo-Mika: https://github.com/PoTaTo-Mika
"""ASR concurrency benchmark on SeedTTS reference audio (issue #646).

This script transcribes SeedTTS reference clips directly through a running ASR
router and reports WER, throughput, latency, RTF, and worker routing balance.
It supports Qwen3-ASR, Fun-ASR-Nano, and Whisper through ``--model-path``.

Usage:

    # Download the test set once:
    python -m benchmarks.dataset.prepare --dataset seedtts

    # Launch a conservative single-RTX-4090 profile
    sgl-omni serve \
        --config examples/configs/fun_asr_rtx4090.yaml \
        --port 8000

    # Sweep the full SeedTTS EN set (3 measured repeats after a warmup):
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 \
        --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
        --concurrencies 1,2,4,8,16,32 \
        --repeats 3 --warmup \
        --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
        --dtype bfloat16 \
        --attention-backend flashinfer \
        --mm-attention-backend triton_attn \
        --cuda-graph --no-torch-compile \
        --max-running-requests 16 \
        --mem-fraction-static 0.65

    # Quick local smoke on a 20-sample subset:
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 --max-samples 20 --concurrencies 2,32 --repeats 3

    # Run the same sweep against Whisper Large v3:
    sgl-omni serve \
        --config examples/configs/whisper_asr_rtx4090.yaml --port 8000
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 --model-path openai/whisper-large-v3 \
        --concurrencies 1,2,4,8,16,32 --repeats 3 --warmup

Earlier reference results on the full SeedTTS EN set (1088 clips, bf16, single RTX
4080 SUPER 32 GB, DP=1, three repeats plus one discarded warmup per level):

* At concurrency 32, Qwen3-ASR-1.7B reached 55.07 samples/s with 0.577 s mean
  latency, 0.1247 mean RTF, and 0.0130 corpus WER.
* At concurrency 32, Fun-ASR-Nano reached 40.66 samples/s with 0.784 s mean
  latency, 0.1696 mean RTF, and 0.0171 corpus WER.
* At concurrency 1, Fun-ASR-Nano had roughly half the mean latency and RTF of
  Qwen3-ASR (0.081 s vs. 0.165 s; 0.0175 vs. 0.0359).

Both models saturated near concurrency 32. Fun-ASR completed every request at
all measured levels; Qwen3-ASR skipped 72 requests at concurrency 64 on the
single GPU. Audio duration was 4.69 s mean, 4.53 s median, and 8.81 s maximum.

Reference results on a single H100 80 GB (bf16, DP=1, three repeats plus one
discarded warmup per level):

* Fun-ASR-Nano on the full SeedTTS EN set (1088 clips): 26.44 samples/s with
  0.038 s mean latency at concurrency 1, saturating near 127.5 samples/s at
  concurrency 16 through 32, with 0.0171 corpus WER at every level through 32.
  The higher concurrency 64 figure counts completed samples only, after
  request shedding.
* Fun-ASR-Nano on the full SeedTTS ZH set (2020 clips): 167.4 samples/s at
  concurrency 32 with 0.190 s mean latency and 0.0135 corpus WER at every
  level through 32.
* Qwen3-ASR-1.7B on the same GPU and EN set: 97.9 samples/s at concurrency 32
  with 0.324 s mean latency and 0.0122 corpus WER. Fun-ASR-Nano was about 30
  percent faster at saturation, and at concurrency 1 kept 0.038 s mean latency
  versus 0.099 s for Qwen3-ASR, consistent with the 4080S comparison above.

On the H100 both languages shed roughly 2 to 5 percent of requests at
concurrency 64 with HTTP 500, because a single worker admits at most 16
pending request builds. The full tables live in docs/cookbook/fun_asr.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics

import requests

from benchmarks.dataset.prepare import DATASETS, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
    build_asr_eval_results,
    run_asr_transcription,
)

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32,64"
MODEL_REVISIONS = {
    QWEN3_ASR_MODEL_PATH: "7278e1e70fe206f11671096ffdd38061171dd6e5",
    FUN_ASR_MODEL_PATH: "854d88f94205cd17d2afdb24332130d86fbe654a",
    OMNI_WHISPER_MODEL_PATH: "06f233fe06e710322aca913c1bc4249a0d71fce1",
}
EXPECTED_SAMPLES = {"en": 1088, "zh": 2020}


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


async def run_asr_seedtts_once(
    samples: list[SampleInput],
    host: str,
    port: int,
    concurrency: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    lang: str = "en",
    warmup: int = 0,
    disable_tqdm: bool = True,
) -> dict:
    """Run one SeedTTS ASR benchmark pass and return WER/speed/worker metrics."""
    before = _fetch_worker_snapshot(host, port)
    outputs, wall_clock_s = await run_asr_transcription(
        samples,
        host=host,
        port=port,
        model_path=model_path,
        lang=lang,
        concurrency=concurrency,
        warmup=warmup,
        disable_tqdm=disable_tqdm,
    )
    after = _fetch_worker_snapshot(host, port)

    benchmark_result = build_asr_eval_results(
        samples,
        outputs,
        wall_clock_s,
        lang,
        model_path=model_path,
        concurrency=concurrency,
    )
    benchmark_result["wall_clock_s"] = wall_clock_s
    benchmark_result["worker"] = _worker_delta(before, after)
    return benchmark_result


async def _run_repeat(args, samples, concurrency: int, repeat: int) -> dict:
    monitor = (
        None
        if args.disable_resource_monitor
        else ResourceMonitor(
            gpu_index=args.gpu_index,
            interval_s=args.monitor_interval_s,
        ).start()
    )
    try:
        benchmark_result = await run_asr_seedtts_once(
            samples,
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            lang=args.lang,
            concurrency=concurrency,
        )
    finally:
        resources = (
            monitor.stop()
            if monitor is not None
            else {
                "available": False,
                "error": "resource monitoring disabled",
            }
        )
    summary = benchmark_result["summary"]
    speed = benchmark_result["speed"]
    wall_clock_s = benchmark_result["wall_clock_s"]
    audio_processed_s = float(speed.get("asr_audio_processed_s") or 0.0)
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "evaluated": summary["evaluated"],
        "total": summary["total_samples"],
        "skipped": summary["skipped"],
        "errors": summary["skipped"],
        "corpus_wer": summary["corpus_wer"],
        "per_sample_wer_max": summary["wer_per_sample_max"],
        "wall_clock_s": wall_clock_s,
        "throughput_samples_per_s": speed["throughput_samples_per_s"],
        "audio_seconds_per_s": (
            audio_processed_s / wall_clock_s if wall_clock_s > 0 else 0.0
        ),
        "latency_mean_s": speed["latency_mean_s"],
        "latency_p95_s": speed["latency_p95_s"],
        "latency_p99_s": speed["latency_p99_s"],
        "rtf_mean": speed["rtf_mean"],
        "rtf_p95": speed["rtf_p95"],
        "worker": benchmark_result["worker"],
        "resources": resources,
        "per_sample": benchmark_result["per_sample"],
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

    def _resource_metric(*path: str) -> dict | None:
        values: list[float] = []
        for repeat in repeats:
            current = repeat.get("resources")
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if isinstance(current, (int, float)):
                values.append(float(current))
        if not values:
            return None
        return {
            "per_repeat": values,
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
        "errors": _stat("errors"),
        "corpus_wer": _stat("corpus_wer"),
        "per_sample_wer_max": _stat("per_sample_wer_max"),
        "wall_clock_s": _stat("wall_clock_s"),
        "throughput_samples_per_s": _stat("throughput_samples_per_s"),
        "audio_seconds_per_s": _stat("audio_seconds_per_s"),
        "latency_mean_s": _stat("latency_mean_s"),
        "latency_p95_s": _stat("latency_p95_s"),
        "latency_p99_s": _stat("latency_p99_s"),
        "rtf_mean": _stat("rtf_mean"),
        "rtf_p95": _stat("rtf_p95"),
        "resources": {
            "gpu_memory_used_peak_mib": _resource_metric("gpu_memory_used_mib", "max"),
            "gpu_memory_used_steady_mib": _resource_metric(
                "gpu_memory_used_mib", "steady_mean"
            ),
            "gpu_process_memory_peak_mib": _resource_metric(
                "gpu_process_memory_mib", "max"
            ),
            "power_peak_w": _resource_metric("power_w", "max"),
            "system_cpu_peak_percent": _resource_metric("system_cpu_percent", "max"),
            "gpu_process_cpu_peak_percent": _resource_metric(
                "gpu_process_cpu_percent", "max"
            ),
            "monitor_errors": [
                repeat["resources"].get("error")
                for repeat in repeats
                if repeat.get("resources", {}).get("error")
            ],
        },
        "per_repeat": repeats,
    }


def _print_table(aggregates: list[dict]) -> None:
    header = (
        "| conc | reps | wall(s) mean | thrpt mean | thrpt best | "
        "audio s/s | lat mean(s) | lat p95(s) | rtf mean | rtf p95 | "
        "corpus WER | max WER |"
    )
    sep = "|---:" * 12 + "|"
    print("\n" + header)
    print(sep)
    for agg in aggregates:
        print(
            f"| {agg['concurrency']} | {agg['repeats']} "
            f"| {agg['wall_clock_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['max']:.3f} "
            f"| {agg['audio_seconds_per_s']['mean']:.3f} "
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
        help="Port of the running ASR SGLang Omni router.",
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
        help=(
            "ASR model id served by the router. Defaults to "
            f"{QWEN3_ASR_MODEL_PATH}; use "
            f"{FUN_ASR_MODEL_PATH} for Fun-ASR-Nano or "
            f"{OMNI_WHISPER_MODEL_PATH} for Whisper."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Resolved model revision used by the running server. Known ASR "
            "model paths use pinned defaults."
        ),
    )
    parser.add_argument(
        "--dataset-revision",
        default=SEEDTTS_DATASET_REVISION,
        help="Pinned HuggingFace SeedTTS dataset revision.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Served dtype recorded as provenance (for example, bfloat16).",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Selected language-model attention backend recorded as provenance.",
    )
    parser.add_argument(
        "--mm-attention-backend",
        default=None,
        help="Selected multimodal attention backend recorded as provenance.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the server uses CUDA Graphs, recorded as provenance.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the server uses torch.compile, recorded as provenance.",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="Server admission limit recorded as provenance.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="Server static-memory fraction recorded as provenance.",
    )
    parser.add_argument(
        "--launch-command",
        default=os.environ.get("SGLANG_OMNI_BENCHMARK_LAUNCH_COMMAND"),
        help="Exact server launch command stored in the result JSON.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Logical local GPU index sampled for memory, utilization, and power.",
    )
    parser.add_argument(
        "--monitor-interval-s",
        type=float,
        default=0.2,
        help="Resource monitor sampling interval.",
    )
    parser.add_argument(
        "--disable-resource-monitor",
        action="store_true",
        help="Disable local GPU/CPU resource sampling.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded warmup pass before timing each concurrency.",
    )
    parser.add_argument(
        "--output",
        default="asr_seedtts_results.json",
        help="Where to write the full JSON results.",
    )
    return parser.parse_args()


async def _sweep(args, samples, concurrencies: list[int]) -> list[dict]:
    aggregates: list[dict] = []
    for concurrency in concurrencies:
        if args.warmup:
            print(f"[conc={concurrency}] warmup pass ...", flush=True)
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
                f"skipped={result['skipped']}",
                flush=True,
            )
            if result["worker"].get("per_worker_routed"):
                print(
                    f"    routed per worker: {result['worker']['per_worker_routed']}",
                    flush=True,
                )
        aggregates.append(_aggregate(repeats))
    return aggregates


def main() -> None:
    args = parse_args()
    concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    if not concurrencies or any(concurrency < 1 for concurrency in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    max_samples = args.max_samples if args.max_samples > 0 else None
    model_revision = args.model_revision or MODEL_REVISIONS.get(args.model_path)

    samples = load_seedtts_samples(
        args.meta,
        max_samples=max_samples,
        split=args.lang,
        revision=args.dataset_revision,
    )
    if max_samples is None and len(samples) != EXPECTED_SAMPLES[args.lang]:
        raise RuntimeError(
            f"Expected full SeedTTS {args.lang} scope of "
            f"{EXPECTED_SAMPLES[args.lang]} samples, got {len(samples)}"
        )
    print(
        f"Loaded {len(samples)} SeedTTS {args.lang} samples; "
        f"sweeping concurrency={concurrencies} x {args.repeats} repeats "
        f"against {args.host}:{args.port} ({args.model_path})"
    )

    aggregates = asyncio.run(_sweep(args, samples, concurrencies))
    _print_table(aggregates)

    server_config = {
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "mm_attention_backend": args.mm_attention_backend,
        "cuda_graph": args.cuda_graph,
        "torch_compile": args.torch_compile,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
    }
    payload = {
        "schema_version": 2,
        "provenance": collect_benchmark_provenance(
            model_id=args.model_path,
            model_revision=model_revision,
            dataset_id=args.meta,
            dataset_revision=args.dataset_revision,
            launch_command=args.launch_command,
            server_config=server_config,
        ),
        "config": {
            "host": args.host,
            "port": args.port,
            "meta": args.meta,
            "lang": args.lang,
            "model_path": args.model_path,
            "model_revision": model_revision,
            "dataset_revision": args.dataset_revision,
            "num_samples": len(samples),
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "server": server_config,
            "resource_monitor": {
                "enabled": not args.disable_resource_monitor,
                "gpu_index": args.gpu_index,
                "interval_s": args.monitor_interval_s,
            },
        },
        "results": aggregates,
    }
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
