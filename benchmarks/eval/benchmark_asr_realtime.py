# SPDX-License-Identifier: Apache-2.0
"""Realtime (WebSocket) ASR benchmark on SeedTTS reference audio.

Streams each clip through /v1/realtime?intent=transcription at wall-clock
pace and reports client-observed streaming latencies, protocol invariant
violations, and WER of the completed transcript. Optionally transcribes the
same clips over HTTP /v1/audio/transcriptions for a WER baseline.

The measurement lives in :mod:`benchmarks.realtime_asr`; this module is the
CLI plus :func:`run_asr_realtime_once` for CI reuse.

Author:

    Jeffro https://github.com/0xjeffro


Usage:

    1. Download the test set once:
    python -m benchmarks.dataset.prepare --dataset seedtts

    2. Launch Qwen3-ASR with the realtime endpoint:
    sgl-omni serve \\
        --model-path Qwen/Qwen3-ASR-1.7B \\
        --model-name Qwen/Qwen3-ASR-1.7B \\
        --enable-realtime --port 8000

    3. Stream 50 EN clips, one session at a time, with an HTTP WER baseline:
    python -m benchmarks.eval.benchmark_asr_realtime \\
        --port 8000 --max-samples 50 --http-baseline

    # Sweep session concurrency (one result per level, no cross-level report):
    python -m benchmarks.eval.benchmark_asr_realtime \\
        --port 8000 --max-samples 50 --concurrencies 1,4,8

    # Manual commit instead of server VAD:
    python -m benchmarks.eval.benchmark_asr_realtime \\
        --port 8000 --max-samples 50 --mode manual
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

import torch

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import (
    SEEDTTS_DATASET_REVISION,
    SampleInput,
    load_seedtts_samples,
)
from benchmarks.eval.benchmark_asr_seedtts import _parse_concurrencies, _positive_int
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.realtime_asr.client import (
    DEFAULT_TURN_DETECTION,
    SessionTrace,
    realtime_url,
    run_session,
)
from benchmarks.realtime_asr.metrics import (
    check_invariants,
    latency_metrics,
    paired_corpus_wer,
    summarize,
    wer_metrics,
)
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    _load_wav_mono_16k,
    apply_wer,
    build_asr_eval_results,
    run_asr_transcription,
)

MODES = ("vad", "manual")
DEFAULT_TRAILING_SILENCE_MS = 1000
"""Longer than the server VAD default ``silence_duration_ms`` (500), so the
last turn closes on VAD rather than on ``transcription.done``."""
LATENCY_KEYS = (
    "first_partial_latency_s",
    "partial_interval_s",
    "final_latency_s",
    "done_to_completed_s",
)


def load_pcm16(wav_path: str) -> bytes:
    """Mono 16 kHz PCM16 bytes for any WAV the SeedTTS loader hands out."""
    audio = _load_wav_mono_16k(wav_path)
    return (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()


def _session_kwargs(mode: str) -> dict[str, Any]:
    if mode == "vad":
        return {"turn_detection": DEFAULT_TURN_DETECTION, "manual_commit": False}
    if mode == "manual":
        return {"turn_detection": None, "manual_commit": True}
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _per_sample_record(
    sample: SampleInput, trace: SessionTrace, output: SampleOutput
) -> dict[str, Any]:
    record = {
        "sample_id": sample.sample_id,
        "error": trace.error,
        "violations": check_invariants(trace),
        "text": trace.completed_text,
        "ref_text": sample.ref_text,
        "wer": output.wer if output.is_success else None,
        **latency_metrics(trace),
    }
    return record


def _summary(
    per_sample: list[dict[str, Any]],
    outputs: list[SampleOutput],
    *,
    lang: str,
    wall_clock_s: float,
) -> dict[str, Any]:
    latencies: dict[str, list[float]] = {key: [] for key in LATENCY_KEYS}
    for record in per_sample:
        for key in LATENCY_KEYS:
            value = record[key]
            if isinstance(value, list):
                latencies[key].extend(value)
            elif value is not None:
                latencies[key].append(value)
    wer = calculate_wer_metrics(outputs, lang)
    clean = [record for record in per_sample if not record["violations"]]
    return {
        "total": len(per_sample),
        "evaluated": wer["evaluated"],
        "sessions_with_violations": len(per_sample) - len(clean),
        "sessions_with_client_error": sum(1 for r in per_sample if r["error"]),
        "corpus_wer": wer["wer_corpus"],
        "per_sample_wer_max": wer["wer_per_sample_max"],
        "segments_total": sum(r["segment_count"] for r in per_sample),
        "partials_total": sum(r["partial_count"] for r in per_sample),
        "audio_total_s": sum(r["audio_duration_s"] for r in per_sample),
        "wall_clock_s": wall_clock_s,
        **{key: summarize(values) for key, values in latencies.items()},
    }


async def _http_baseline(
    samples: list[SampleInput],
    *,
    host: str,
    port: int,
    concurrency: int,
    model_path: str,
    lang: str,
) -> tuple[dict[str, Any], list[SampleOutput]]:
    outputs, wall_clock_s = await run_asr_transcription(
        samples,
        host=host,
        port=port,
        model_path=model_path,
        lang=lang,
        concurrency=concurrency,
    )
    results = build_asr_eval_results(
        samples,
        outputs,
        wall_clock_s,
        lang,
        model_path=model_path,
        concurrency=concurrency,
    )
    http_outputs: list[SampleOutput] = []
    for row in results["per_sample"]:
        output = SampleOutput(sample_id=row["id"], target_text=row["ref_text"])
        if row["is_success"]:
            output = apply_wer(output, row["hyp_text"], lang)
        http_outputs.append(output)
    baseline = {
        "corpus_wer": results["summary"]["corpus_wer"],
        "per_sample_wer_max": results["summary"].get("wer_per_sample_max"),
        "evaluated": results["summary"]["evaluated"],
        "wall_clock_s": wall_clock_s,
        "per_sample": {row["id"]: row.get("wer") for row in results["per_sample"]},
    }
    return baseline, http_outputs


async def run_asr_realtime_once(
    samples: list[SampleInput],
    *,
    host: str,
    port: int,
    concurrency: int = 1,
    packet_ms: int = 200,
    mode: str = "vad",
    lang: str = "en",
    paced: bool = True,
    trailing_silence_ms: int = DEFAULT_TRAILING_SILENCE_MS,
    timeout_s: float = 120.0,
    with_http_baseline: bool = False,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    pcm_cache: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Stream every sample once, ``concurrency`` sessions at a time.

    Returns ``{"config", "summary", "per_sample", "http_baseline"?}``. The
    ``summary`` carries mean/p50/p95 for each latency in :data:`LATENCY_KEYS`,
    corpus WER, and how many sessions violated a protocol invariant. Nothing is
    asserted here; CI decides what to gate on.
    """
    session_kwargs = _session_kwargs(mode)
    url = realtime_url(host, port)
    pcm_cache = {} if pcm_cache is None else pcm_cache
    for sample in samples:
        if sample.ref_audio not in pcm_cache:
            pcm_cache[sample.ref_audio] = load_pcm16(sample.ref_audio)

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(sample: SampleInput) -> SessionTrace:
        async with semaphore:
            return await run_session(
                url,
                pcm_cache[sample.ref_audio],
                packet_ms=packet_ms,
                paced=paced,
                language=lang,
                trailing_silence_ms=trailing_silence_ms,
                timeout_s=timeout_s,
                **session_kwargs,
            )

    start = time.perf_counter()
    traces = await asyncio.gather(*(_one(sample) for sample in samples))
    wall_clock_s = time.perf_counter() - start

    outputs: list[SampleOutput] = []
    per_sample: list[dict[str, Any]] = []
    for sample, trace in zip(samples, traces):
        output = wer_metrics(trace, sample.ref_text, lang=lang)
        output.sample_id = sample.sample_id
        outputs.append(output)
        per_sample.append(_per_sample_record(sample, trace, output))

    decode_intervals = {t.session.get("decode_interval_ms") for t in traces}
    result: dict[str, Any] = {
        "config": {
            "mode": mode,
            "concurrency": concurrency,
            "packet_ms": packet_ms,
            "paced": paced,
            "trailing_silence_ms": trailing_silence_ms,
            "lang": lang,
            "model_path": model_path,
            "decode_interval_ms": (
                decode_intervals.pop() if len(decode_intervals) == 1 else None
            ),
        },
        "summary": _summary(per_sample, outputs, lang=lang, wall_clock_s=wall_clock_s),
        "per_sample": per_sample,
    }
    if with_http_baseline:
        baseline, http_outputs = await _http_baseline(
            samples,
            host=host,
            port=port,
            concurrency=concurrency,
            model_path=model_path,
            lang=lang,
        )
        result["http_baseline"] = baseline
        result["summary"]["http_corpus_wer"] = baseline["corpus_wer"]
        result["summary"]["http_evaluated"] = baseline["evaluated"]
        result["summary"].update(paired_corpus_wer(outputs, http_outputs, lang=lang))
        for record in per_sample:
            record["http_wer"] = baseline["per_sample"].get(record["sample_id"])
    return result


# --- CLI ---------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_stat(stat: dict[str, Any] | None) -> str:
    if stat is None:
        return "n/a"
    return f"{stat['mean']:.3f} / {stat['p50']:.3f} / {stat['p95']:.3f} (n={stat['n']})"


def _print_table(results: list[dict[str, Any]]) -> None:
    print()
    print(
        "| conc | mode | interval ms | sessions | violations | corpus WER | HTTP WER "
        "| ΔWER vs HTTP (common n) "
        "| first partial mean/p50/p95 s | partial gap mean/p50/p95 s "
        "| final mean/p50/p95 s | done→completed mean/p50/p95 s | wall s |"
    )
    print("|" + "---|" * 13)
    for result in results:
        config, summary = result["config"], result["summary"]
        print(
            f"| {config['concurrency']} | {config['mode']} "
            f"| {config['decode_interval_ms']} | {summary['total']} "
            f"| {summary['sessions_with_violations']} "
            f"| {_fmt(summary['corpus_wer'], 4)} "
            f"| {_fmt(summary.get('http_corpus_wer'), 4)} "
            f"| {_fmt(summary.get('corpus_wer_delta_vs_http'), 4)} "
            f"({summary.get('common_evaluated', 'n/a')}) "
            f"| {_fmt_stat(summary['first_partial_latency_s'])} "
            f"| {_fmt_stat(summary['partial_interval_s'])} "
            f"| {_fmt_stat(summary['final_latency_s'])} "
            f"| {_fmt_stat(summary['done_to_completed_s'])} "
            f"| {summary['wall_clock_s']:.1f} |"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", default=QWEN3_ASR_MODEL_PATH)
    parser.add_argument("--meta", default=DATASETS["seedtts"])
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--max-samples", type=int, default=50, help="0 streams the whole split."
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=[1],
        help="Comma-separated simultaneous session counts; one result each.",
    )
    parser.add_argument("--mode", choices=MODES, default="vad")
    parser.add_argument("--packet-ms", type=_positive_int, default=200)
    parser.add_argument(
        "--no-pace",
        action="store_true",
        help="Send packets as fast as possible (throughput only).",
    )
    parser.add_argument(
        "--trailing-silence-ms", type=int, default=DEFAULT_TRAILING_SILENCE_MS
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--http-baseline",
        action="store_true",
        help="Also transcribe over /v1/audio/transcriptions for a WER baseline.",
    )
    parser.add_argument("--output", default="asr_realtime_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None
    is_local_source = os.path.isfile(args.meta) or args.meta.endswith(".lst")
    if is_local_source:
        revision = None
    elif args.dataset_revision is not None:
        revision = args.dataset_revision
    elif args.meta == DATASETS["seedtts"]:
        revision = SEEDTTS_DATASET_REVISION
    else:
        revision = None
    samples = load_seedtts_samples(
        args.meta, max_samples=max_samples, split=args.lang, revision=revision
    )
    if not samples:
        raise RuntimeError(f"No SeedTTS samples loaded from {args.meta!r}")
    print(
        f"Loaded {len(samples)} SeedTTS {args.lang} samples; streaming "
        f"mode={args.mode} packet={args.packet_ms}ms paced={not args.no_pace} "
        f"concurrencies={args.concurrencies} against {args.host}:{args.port}"
    )

    async def _sweep() -> list[dict[str, Any]]:
        pcm_cache: dict[str, bytes] = {}
        results = []
        for concurrency in args.concurrencies:
            result = await run_asr_realtime_once(
                samples,
                host=args.host,
                port=args.port,
                concurrency=concurrency,
                packet_ms=args.packet_ms,
                mode=args.mode,
                lang=args.lang,
                paced=not args.no_pace,
                trailing_silence_ms=args.trailing_silence_ms,
                timeout_s=args.timeout_s,
                with_http_baseline=args.http_baseline,
                model_path=args.model_path,
                pcm_cache=pcm_cache,
            )
            summary = result["summary"]
            print(
                f"[conc={concurrency}] wall={summary['wall_clock_s']:.1f}s "
                f"violations={summary['sessions_with_violations']}/{summary['total']} "
                f"corpus_wer={_fmt(summary['corpus_wer'], 4)} "
                f"first_partial_p95={_fmt((summary['first_partial_latency_s'] or {}).get('p95'))}s"
            )
            results.append(result)
        return results

    results = asyncio.run(_sweep())
    _print_table(results)
    payload = {
        "schema_version": 1,
        "benchmark": "asr_realtime",
        "dataset": {
            "meta": args.meta,
            "revision": revision,
            "lang": args.lang,
            "num_samples": len(samples),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
