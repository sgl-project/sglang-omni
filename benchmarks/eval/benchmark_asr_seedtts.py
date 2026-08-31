"""ASR concurrency benchmark on SeedTTS reference audio (issue #646).

This script transcribes SeedTTS reference clips directly through a running ASR
router and reports WER, request throughput, RTFx, RTF, latency, and worker
routing balance.

Author:

    Chenyang Zhao https://github.com/zhaochenyang20
    PoTaTo-Mika https://github.com/PoTaTo-Mika

Usage:

    1. Download the test set once:
    python -m benchmarks.dataset.prepare --dataset seedtts

    2. Pin and launch Qwen3-ASR:
    MODEL_PATH=$(hf download Qwen/Qwen3-ASR-1.7B \
        --revision 7278e1e70fe206f11671096ffdd38061171dd6e5)
    sgl-omni serve \
        --model-path "${MODEL_PATH}" \
        --model-name Qwen/Qwen3-ASR-1.7B \
        --port 8000

    # Sweep the full SeedTTS EN set (3 repeats each):
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 \
        --concurrencies 1,2,4,8,16,32,64 \
        --repeats 3 --warmup \
        --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
        --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5

    # Quick local smoke on a 20-sample subset:
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 --max-samples 20 --concurrencies 2,32 --repeats 3

    # Run the same sweep against Fun-ASR-Nano:
    python -m sglang_omni.cli serve \
        --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
        --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup

    # Run Fun-ASR-Nano with SSE streaming and TTFT metrics:
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
        --concurrencies 32 --repeats 3 --warmup --stream

Reference results on the full SeedTTS EN set (1088 clips, bf16, single RTX
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
import hashlib
import json
import os
import statistics
import time

import requests

from benchmarks.dataset.prepare import DATASETS, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.asr_profiling import (
    UtilizationSampler,
    collect_environment_fingerprint,
    collect_server_identity,
    run_profiled_pass,
)
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
    build_asr_eval_results,
    run_asr_transcription,
)

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32,64"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _parse_concurrencies(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError(
            "concurrencies must be a non-empty comma-separated list"
        )
    return [_positive_int(token) for token in tokens]


def _evaluation_input_sha256(
    samples: list[SampleInput], *, namespace: str = "seedtts"
) -> str:
    """Fingerprint the exact evaluation input: ids, texts, and audio bytes."""
    digest = hashlib.sha256(f"{namespace}-evaluation-input-v1\0".encode())
    for sample in samples:
        for value in (sample.sample_id, sample.ref_text, sample.target_text):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        with open(sample.ref_audio, "rb") as audio:
            size = os.fstat(audio.fileno()).st_size
            digest.update(size.to_bytes(8, "big"))
            while chunk := audio.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


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
    stream: bool = False,
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
        stream=stream,
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


def _save_raw_per_sample(
    raw_dir: str, concurrency: int, label: str, per_sample: list[dict]
) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"conc{concurrency}_{label}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for record in per_sample:
            handle.write(json.dumps(record) + "\n")


async def _run_repeat(args, samples, concurrency: int, repeat: int) -> dict:
    monitor = (
        None
        if args.disable_resource_monitor
        else ResourceMonitor(
            gpu_index=args.gpu_index,
            interval_s=args.monitor_interval_s,
            gpu_process_pids=args.gpu_process_pids,
        ).start()
    )
    sampler = None
    try:
        if args.sample_util:
            gpu_ids = [int(g) for g in args.util_gpu_ids.split(",") if g.strip()]
            sampler = UtilizationSampler(gpu_ids=gpu_ids, interval_s=args.util_interval)
            sampler.start()
        benchmark_result = await run_asr_seedtts_once(
            samples,
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            lang=args.lang,
            concurrency=concurrency,
            stream=args.stream,
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
        util_summary = sampler.stop().to_dict() if sampler is not None else None
    summary = benchmark_result["summary"]
    speed = benchmark_result["speed"]
    has_evaluated_samples = summary["evaluated"] > 0
    has_rtf = speed["rtf_p95"] is not None
    result = {
        "concurrency": concurrency,
        "repeat": repeat,
        "evaluated": summary["evaluated"],
        "total": summary["total_samples"],
        "skipped": summary["skipped"],
        "corpus_wer": summary["corpus_wer"] if has_evaluated_samples else None,
        "per_sample_wer_max": (
            summary["wer_per_sample_max"] if has_evaluated_samples else None
        ),
        "wall_clock_s": benchmark_result["wall_clock_s"],
        "throughput_samples_per_s": speed["throughput_samples_per_s"],
        "rtfx": speed["rtfx"],
        "latency_mean_s": (speed["latency_mean_s"] if has_evaluated_samples else None),
        "latency_median_s": (
            speed["latency_median_s"] if has_evaluated_samples else None
        ),
        "latency_p95_s": (speed["latency_p95_s"] if has_evaluated_samples else None),
        "latency_p99_s": (speed["latency_p99_s"] if has_evaluated_samples else None),
        "rtf_mean": speed["rtf_mean"] if has_rtf else None,
        "rtf_p95": speed["rtf_p95"],
        "worker": benchmark_result["worker"],
        "resources": resources,
    }
    if util_summary is not None:
        result["utilization"] = util_summary
    if args.save_raw_dir:
        _save_raw_per_sample(
            args.save_raw_dir,
            concurrency,
            f"rep{repeat}",
            benchmark_result["per_sample"],
        )
    for key in (
        "text_ttft_mean_s",
        "text_ttft_median_s",
        "text_ttft_p95_s",
        "text_ttft_p99_s",
        "inter_chunk_mean_s",
        "inter_chunk_p95_s",
        "inter_chunk_p99_s",
    ):
        if key in speed:
            result[key] = speed[key]
    return result


def _aggregate(repeats: list[dict]) -> dict:
    """Aggregate repeat metrics without hiding failed or partial work."""
    if not repeats:
        raise ValueError("At least one repeat is required")

    def _stat(key: str) -> dict:
        values = [r[key] for r in repeats if r.get(key) is not None]
        if not values:
            return {"mean": None, "min": None, "max": None, "n": 0}
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
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

    total_requests = sum(r["total"] for r in repeats)
    evaluated = sum(r["evaluated"] for r in repeats)
    skipped = sum(r["skipped"] for r in repeats)

    aggregate = {
        "concurrency": repeats[0]["concurrency"],
        "repeats": len(repeats),
        "evaluated": evaluated,
        "total": total_requests,
        "skipped": skipped,
        "corpus_wer": _stat("corpus_wer"),
        "per_sample_wer_max": _stat("per_sample_wer_max"),
        "wall_clock_s": _stat("wall_clock_s"),
        "throughput_samples_per_s": _stat("throughput_samples_per_s"),
        "rtfx": _stat("rtfx"),
        "latency_mean_s": _stat("latency_mean_s"),
        "latency_median_s": _stat("latency_median_s"),
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
    for key in (
        "text_ttft_mean_s",
        "text_ttft_median_s",
        "text_ttft_p95_s",
        "text_ttft_p99_s",
        "inter_chunk_mean_s",
        "inter_chunk_p95_s",
        "inter_chunk_p99_s",
    ):
        if any(key in repeat for repeat in repeats):
            aggregate[key] = _stat(key)
    return aggregate


def _format_metric(value: float | None, digits: int, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _print_table(aggregates: list[dict]) -> None:
    include_ttft = any("text_ttft_mean_s" in agg for agg in aggregates)
    header = (
        "| conc | valid reps (lat/RTF/WER) | evaluated/requests | "
        "wall(s) mean | req/s mean | req/s best | RTFx mean | lat mean(s) | "
        "lat p50(s) | lat p95(s) | RTF mean | RTF p95 | "
    )
    if include_ttft:
        header += "TTFT mean(s) | TTFT p95(s) | ITL mean(s) | "
    header += "corpus WER | max WER |"
    sep = "|---:" * (17 if include_ttft else 14) + "|"
    print("\n" + header)
    print(sep)
    for agg in aggregates:
        row = (
            f"| {agg['concurrency']} "
            f"| {agg['latency_mean_s']['n']}/{agg['repeats']} / "
            f"{agg['rtf_mean']['n']}/{agg['repeats']} / "
            f"{agg['corpus_wer']['n']}/{agg['repeats']} "
            f"| {agg['evaluated']}/{agg['total']} "
            f"| {_format_metric(agg['wall_clock_s']['mean'], 3)} "
            f"| {_format_metric(agg['throughput_samples_per_s']['mean'], 3)} "
            f"| {_format_metric(agg['throughput_samples_per_s']['max'], 3)} "
            f"| {_format_metric(agg['rtfx']['mean'], 3)} "
            f"| {_format_metric(agg['latency_mean_s']['mean'], 3)} "
            f"| {_format_metric(agg['latency_median_s']['mean'], 3)} "
            f"| {_format_metric(agg['latency_p95_s']['mean'], 3)} "
            f"| {_format_metric(agg['rtf_mean']['mean'], 4)} "
            f"| {_format_metric(agg['rtf_p95']['mean'], 4)} "
        )
        if include_ttft:
            row += (
                f"| {_format_metric(agg.get('text_ttft_mean_s', {}).get('mean'), 4)} "
                f"| {_format_metric(agg.get('text_ttft_p95_s', {}).get('mean'), 4)} "
                f"| {_format_metric(agg.get('inter_chunk_mean_s', {}).get('mean'), 4)} "
            )
        row += (
            f"| {_format_metric(agg['corpus_wer']['max'], 4)} "
            f"| {_format_metric(agg['per_sample_wer_max']['max'], 4)} |"
        )
        print(row)


def add_common_args(
    parser: argparse.ArgumentParser, *, default_output: str
) -> argparse.ArgumentParser:
    """Add the router, sweep, provenance, monitoring, and output options."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of the running ASR SGLang Omni router.",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=_parse_concurrencies(DEFAULT_CONCURRENCIES),
        help="Comma-separated positive ASR concurrency levels to sweep.",
    )
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--model-path",
        default=QWEN3_ASR_MODEL_PATH,
        help=(
            "ASR model id served by the router. Defaults to "
            f"{QWEN3_ASR_MODEL_PATH}; use "
            f"{FUN_ASR_MODEL_PATH} for Fun-ASR-Nano."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Declared model revision loaded by the running server.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Exact dataset revision; canonical SeedTTS uses a pinned default.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Served dtype recorded as provenance (for example, bfloat16).",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Selected model quantization recorded as declared provenance.",
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
        "--gpu-process-pid",
        dest="gpu_process_pids",
        type=_positive_int,
        action="append",
        help=(
            "NVML/host PID to include in process memory and CPU metrics; repeat "
            "for multiple GPU processes. Without this option, process-specific "
            "metrics are unavailable instead of including every GPU workload."
        ),
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
        "--stream",
        action="store_true",
        help=(
            "Use SSE streaming transcription (stream=true). Fills text_ttft_* "
            "and inter_chunk_* speed metrics while preserving final-text WER."
        ),
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Where to write the full JSON results.",
    )
    parser.add_argument(
        "--save-raw-dir",
        default="",
        help=(
            "Directory for raw per-request JSONL (one file per concurrency "
            "and repeat). Empty disables raw preservation."
        ),
    )
    parser.add_argument(
        "--profile-events",
        action="store_true",
        help=(
            "After the measured repeats of each concurrency, run one extra "
            "profiled pass with request-level event recording and attach the "
            "stage/hop breakdown. Requires running on the server host."
        ),
    )
    parser.add_argument(
        "--profile-urls",
        default="",
        help=(
            "Comma-separated serve base URLs exposing /start_request_profile "
            "(the serve port for DP=1, each worker base URL behind a router). "
            "Defaults to http://<host>:<port>."
        ),
    )
    parser.add_argument(
        "--profile-event-dir",
        default="/tmp/asr_bench_profile",
        help="Server-side base directory for profiler event JSONL.",
    )
    parser.add_argument(
        "--sample-util",
        action="store_true",
        help=(
            "Sample host CPU and GPU utilization during each measured repeat "
            "and attach mean/max summaries to the results."
        ),
    )
    parser.add_argument(
        "--util-gpu-ids",
        default="",
        help="Comma-separated GPU indices to sample (empty = all visible).",
    )
    parser.add_argument(
        "--util-interval",
        type=float,
        default=1.0,
        help="Utilization sampling interval in seconds.",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help=(
            "Record an environment fingerprint (git state, package versions, "
            "dependency freeze hash, GPUs, cached model revision) in the "
            "output JSON."
        ),
    )
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill defaults that depend on other parsed arguments."""
    if not args.profile_urls:
        args.profile_urls = f"http://{args.host}:{args.port}"
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    add_common_args(parser, default_output="asr_seedtts_results.json")
    return finalize_args(parser.parse_args())


async def _run_profiled_pass(args, samples, concurrency: int) -> dict | None:
    """One extra profiled pass, excluded from measured aggregates.

    Request-event profiling adds server-side JSONL writes, so the measured
    repeats stay clean and one dedicated pass captures the stage breakdown.
    The event dir is a server-side path; building the report assumes the
    benchmark runs on the same host as the server.
    """
    run_id = f"asrbench-c{concurrency}-{int(time.time())}"
    event_dir = os.path.join(args.profile_event_dir, run_id)
    profile_urls = [u.strip() for u in args.profile_urls.split(",") if u.strip()]
    return await run_profiled_pass(
        run_id=run_id,
        event_dir=event_dir,
        profile_urls=profile_urls,
        run_pass=lambda: _run_repeat(args, samples, concurrency, repeat=0),
        log_prefix=f"[conc={concurrency}]",
    )


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
                stream=args.stream,
            )
        repeats: list[dict] = []
        for repeat in range(1, args.repeats + 1):
            result = await _run_repeat(args, samples, concurrency, repeat)
            repeats.append(result)
            line = (
                f"[conc={concurrency} rep={repeat}] "
                f"wall={result['wall_clock_s']:.3f}s "
                f"thrpt={result['throughput_samples_per_s']:.3f}/s "
                f"rtfx={result['rtfx']:.3f} "
                f"lat_mean={_format_metric(result['latency_mean_s'], 3, 's')} "
                f"lat_p95={_format_metric(result['latency_p95_s'], 3, 's')} "
                f"rtf_mean={_format_metric(result['rtf_mean'], 4)} "
            )
            if "text_ttft_mean_s" in result:
                line += (
                    f"ttft_mean="
                    f"{_format_metric(result['text_ttft_mean_s'], 4, 's')} "
                )
            line += (
                f"corpus_wer={_format_metric(result['corpus_wer'], 4)} "
                f"evaluated={result['evaluated']}/{result['total']}"
            )
            print(line)
            if result["worker"].get("per_worker_routed"):
                print(f"    routed per worker: {result['worker']['per_worker_routed']}")
        aggregate = _aggregate(repeats)
        if args.profile_events:
            print(f"[conc={concurrency}] profiled pass ...")
            aggregate["profile"] = await _run_profiled_pass(args, samples, concurrency)
        aggregates.append(aggregate)
    return aggregates


def main() -> None:
    args = parse_args()
    concurrencies = args.concurrencies
    max_samples = args.max_samples if args.max_samples > 0 else None
    model_revision = args.model_revision
    is_local_source = os.path.isfile(args.meta) or args.meta.endswith(".lst")
    if is_local_source:
        dataset_revision = None
    elif args.dataset_revision is not None:
        dataset_revision = args.dataset_revision
    elif args.meta == DATASETS["seedtts"]:
        dataset_revision = SEEDTTS_DATASET_REVISION
    else:
        dataset_revision = None

    samples = load_seedtts_samples(
        args.meta,
        max_samples=max_samples,
        split=args.lang,
        revision=dataset_revision,
    )
    if not samples:
        raise RuntimeError(f"No SeedTTS samples loaded from {args.meta!r}")
    evaluation_input_sha256 = _evaluation_input_sha256(samples)
    print(
        f"Loaded {len(samples)} SeedTTS {args.lang} samples; "
        f"sweeping concurrency={concurrencies} x {args.repeats} repeats "
        f"against {args.host}:{args.port} ({args.model_path})"
    )

    aggregates = asyncio.run(_sweep(args, samples, concurrencies))
    _print_table(aggregates)

    server_config = {
        "dtype": args.dtype,
        "quantization": args.quantization,
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
            dataset_revision=dataset_revision,
            launch_command=args.launch_command,
            server_config=server_config,
            evaluation_input_sha256=evaluation_input_sha256,
        ),
        "config": {
            "host": args.host,
            "port": args.port,
            "meta": args.meta,
            "lang": args.lang,
            "model_path": args.model_path,
            "declared_model_revision": model_revision,
            "dataset_revision": dataset_revision,
            "num_samples": len(samples),
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "stream": args.stream,
            "declared_server": server_config,
            "resource_monitor": {
                "enabled": not args.disable_resource_monitor,
                "gpu_index": args.gpu_index,
                "interval_s": args.monitor_interval_s,
                "gpu_process_pids": args.gpu_process_pids or [],
            },
            "profile_events": args.profile_events,
            "sample_util": args.sample_util,
        },
        "results": aggregates,
    }
    if args.fingerprint:
        # The client fingerprint equals the server's only when both share one
        # host and checkout; the server block records what the server reports.
        payload["environment_fingerprint"] = {
            "client": collect_environment_fingerprint(args.model_path),
            "server": collect_server_identity(f"http://{args.host}:{args.port}"),
        }
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
