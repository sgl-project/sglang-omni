# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR concurrency-scaling benchmark on SeedTTS EN (issue #646).

Sweeps ASR transcription fan-out (concurrency) against a *running* Qwen3-ASR
SGLang Omni router and reports, for each concurrency level, the metrics tracked
in issue #646: corpus/per-sample WER, wall-clock, throughput, latency
percentiles, RTF, and per-worker routing balance. This produces the repeatable
concurrency-scaling data the issue's acceptance criteria ask for, and lets us
decide the right ASR fan-out for the small correctness gate vs. the full
SeedTTS EN transcription / WER workloads.

This script transcribes the SeedTTS *reference* clips directly (no TTS
generation step), so it isolates ASR behavior from TTS.

``run_asr_transcription`` + ``build_asr_eval_results`` are the shared
transcription/scoring path; the Qwen3-ASR correctness gate
(``tests/test_model/test_qwen3_asr_ci.py``) imports them so the gate is just
this benchmark run plus thresholds. Both reuse the benchmark framework
abstractions (``BenchmarkRunner``, ``benchmarks.metrics``).

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

    # Quick check on the 20-sample correctness subset:
    python -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
        --port 8000 --max-samples 20 --concurrencies 2,32 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

import aiohttp
import requests
from jiwer import process_words

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.wer import calculate_asr_speed_metrics, calculate_wer_metrics
from benchmarks.tasks.tts import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MAX_NEW_TOKENS,
    QWEN3_ASR_MODEL_PATH,
    QWEN3_ASR_REQUEST_TIMEOUT_S,
    SampleOutput,
    normalize_text,
)

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32,64"


def make_asr_send_fn(
    model_name: str,
    api_url: str,
    *,
    lang: str = "en",
    max_new_tokens: int = QWEN3_ASR_MAX_NEW_TOKENS,
) -> SendFn:
    """Return a *send_fn(session, sample) -> RequestResult* that transcribes one
    SeedTTS reference clip via the Omni ``/v1/audio/transcriptions`` endpoint.

    Note: do NOT send temperature=0 — Qwen3-ASR degenerates under pure greedy
    (the server bumps it to 0.01). ``language`` selects the forced prefix.
    """

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        try:
            with open(sample.ref_audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        except OSError as exc:
            result.error = str(exc)
            return result
        result.audio_duration_s = get_wav_duration(audio_bytes)

        form = aiohttp.FormData()
        form.add_field("model", model_name)
        form.add_field("language", "en" if lang == "en" else lang)
        form.add_field("response_format", "json")
        form.add_field("max_new_tokens", str(max_new_tokens))
        form.add_field(
            "file",
            audio_bytes,
            filename=os.path.basename(sample.ref_audio),
            content_type="audio/wav",
        )

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, data=form) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                else:
                    payload = await response.json()
                    result.text = str(payload.get("text", ""))
                    result.is_success = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        if result.is_success and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send_fn


async def run_asr_transcription(
    samples: list[SampleInput],
    *,
    host: str = "127.0.0.1",
    port: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    lang: str = "en",
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    warmup: int = 0,
    request_timeout_s: int = QWEN3_ASR_REQUEST_TIMEOUT_S,
    disable_tqdm: bool = True,
) -> tuple[list[RequestResult], float]:
    """Transcribe ``samples`` against a running ASR router at one concurrency.

    Returns ``(outputs, wall_clock_s)`` via the shared ``BenchmarkRunner``.
    """
    api_url = f"http://{host}:{port}/v1/audio/transcriptions"
    send_fn = make_asr_send_fn(model_path, api_url, lang=lang)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=request_timeout_s,
        )
    )
    outputs = await runner.run(samples, send_fn)
    return outputs, runner.wall_clock_s


def build_asr_eval_results(
    samples: list[SampleInput],
    outputs: list[RequestResult],
    wall_clock_s: float,
    lang: str,
    *,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """Score transcriptions and assemble WER + speed metrics.

    Returns ``{"summary": wer, "speed": speed, "per_sample": [...]}`` with the
    exact ``summary.*`` / ``speed.*`` keys the Qwen3-ASR gate writes and the
    tune-ci-thresholds config reads. WER/speed reuse ``benchmarks.metrics``.
    """
    result_by_id = {result.request_id: result for result in outputs}
    sample_outputs: list[SampleOutput] = []
    per_sample: list[dict] = []
    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.ref_text,
        )
        if result is None or not result.is_success:
            output.error = (result.error if result else "") or "No transcription"
        else:
            output.latency_s = result.latency_s
            output.asr_latency_s = result.latency_s
            output.audio_duration_s = result.audio_duration_s
            output.whisper_text = result.text
            output.ref_norm = normalize_text(sample.ref_text, lang)
            output.hyp_norm = normalize_text(result.text, lang)
            if output.ref_norm:
                measures = process_words(output.ref_norm, output.hyp_norm)
                output.wer = measures.wer
                output.substitutions = measures.substitutions
                output.deletions = measures.deletions
                output.insertions = measures.insertions
                output.hits = measures.hits
                output.is_success = True
            else:
                output.error = "Empty reference after normalization"
        sample_outputs.append(output)
        per_sample.append(
            {
                "id": output.sample_id,
                "is_success": output.is_success,
                "wer": output.wer if output.is_success else None,
                "ref_text": output.target_text,
                "hyp_text": output.whisper_text,
                "ref_norm": output.ref_norm,
                "hyp_norm": output.hyp_norm,
                "audio_duration_s": output.audio_duration_s,
                "latency_s": output.latency_s,
                "error": output.error,
            }
        )

    wer_summary = calculate_wer_metrics(sample_outputs, lang)
    # note (Yue Yin): gate + tune-ci-thresholds read summary.corpus_wer
    wer_summary["corpus_wer"] = wer_summary["wer_corpus"]

    asr_speed = calculate_asr_speed_metrics(sample_outputs, wall_time_s=wall_clock_s)
    # note (Yue Yin): compute_speed_metrics supplies rtf_p95 (the asr metrics omit it)
    perf = compute_speed_metrics(outputs, wall_clock_s=wall_clock_s)
    speed = {
        **asr_speed,
        "asr_model": model_path,
        "asr_concurrency": concurrency,
        "asr_rtf_p95": perf.get("rtf_p95"),
        # note (Yue Yin): plain calibration keys read by tune-ci-thresholds + gate
        "throughput_samples_per_s": asr_speed["asr_throughput_samples_per_s"],
        "latency_mean_s": asr_speed["asr_latency_mean_s"],
        "latency_median_s": asr_speed["asr_latency_median_s"],
        "latency_p95_s": asr_speed["asr_latency_p95_s"],
        "latency_p99_s": asr_speed["asr_latency_p99_s"],
        "rtf_mean": asr_speed["asr_rtf_mean"],
        "rtf_median": asr_speed["asr_rtf_median"],
        "rtf_p95": perf.get("rtf_p95"),
    }
    return {"summary": wer_summary, "speed": speed, "per_sample": per_sample}


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
