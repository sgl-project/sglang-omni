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

    # Nemotron native streaming sessions:
    python -m benchmarks.eval.benchmark_asr_realtime \\
        --port 8000 --max-samples 50 --protocol native \\
        --model-path nvidia/nemotron-3.5-asr-streaming-0.6b
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import torch
from pydantic import JsonValue

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
from benchmarks.realtime_asr.replay import TraceArtifact, save_trace, source_fingerprint
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    _load_wav_mono_16k,
    apply_wer,
    build_asr_eval_results,
    run_asr_transcription,
)
from sglang_omni.utils.logging import configure_dependency_loggers

MODES = ("vad", "manual")
DEFAULT_TRAILING_SILENCE_MS = 1000
"""Longer than the server VAD default ``silence_duration_ms`` (500), so the
last turn closes on VAD rather than on ``transcription.done``."""
LATENCY_KEYS = (
    "first_partial_latency_s",
    "partial_interval_s",
    "final_latency_s",
    "done_to_completed_s",
    "first_text_s",
)


def load_pcm16(wav_path: str, *, preserve_pcm: bool = False) -> bytes:
    """Mono 16 kHz PCM16 bytes for any WAV the SeedTTS loader hands out."""
    if preserve_pcm:
        try:
            with wave.open(wav_path, "rb") as audio_file:
                if (
                    audio_file.getnchannels(),
                    audio_file.getsampwidth(),
                    audio_file.getframerate(),
                ) == (1, 2, 16000):
                    return audio_file.readframes(audio_file.getnframes())
                else:
                    pass
        except wave.Error:
            pass
    else:
        pass
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
        "wer_error": output.error or None,
        **latency_metrics(trace),
    }
    return record


def _summary(
    per_sample: list[dict[str, Any]],
    outputs: list[SampleOutput],
    *,
    lang: str,
    wall_clock_s: float,
    protocol: Literal["transcription", "native"] = "transcription",
) -> dict[str, Any]:
    clean = [record for record in per_sample if not record["violations"]]
    latencies: dict[str, list[float]] = {key: [] for key in LATENCY_KEYS}
    for record in clean if protocol == "native" else per_sample:
        for key in LATENCY_KEYS:
            value = record[key]
            if isinstance(value, list):
                latencies[key].extend(value)
            elif value is not None:
                latencies[key].append(value)
    wer = calculate_wer_metrics(outputs, lang)
    return {
        "success": len(clean),
        "requests_per_second": len(clean) / wall_clock_s,
        "audio_seconds_per_second": sum(record["audio_duration_s"] for record in clean)
        / wall_clock_s,
        "request_wall_s": summarize(
            [record["wall_s"] for record in clean if record["wall_s"] is not None]
        ),
        "request_rtf": summarize(
            [
                record["wall_s"] / record["audio_duration_s"]
                for record in clean
                if record["wall_s"] is not None and record["audio_duration_s"] > 0
            ]
        ),
        "total": len(per_sample),
        "evaluated": wer["evaluated"],
        "sessions_with_violations": len(per_sample) - len(clean),
        "sessions_with_client_error": sum(1 for r in per_sample if r["error"]),
        "corpus_wer": (
            None if protocol == "native" and not wer["evaluated"] else wer["wer_corpus"]
        ),
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
    request_language: str,
) -> tuple[dict[str, Any], list[SampleOutput]]:
    outputs, wall_clock_s = await run_asr_transcription(
        samples,
        host=host,
        port=port,
        model_path=model_path,
        lang=request_language,
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
    protocol: Literal["transcription", "native"] = "transcription",
    mode: Literal["vad", "manual"] | None = None,
    lang: str = "en",
    paced: bool = True,
    trailing_silence_ms: int = DEFAULT_TRAILING_SILENCE_MS,
    timeout_s: float = 120.0,
    with_http_baseline: bool = False,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    pcm_cache: dict[str, bytes] | None = None,
    trace_dir: str | None = None,
    repeat: int = 1,
    warmup: bool = False,
) -> dict[str, Any]:
    """Stream every sample once, ``concurrency`` sessions at a time.

    Returns ``{"config", "summary", "per_sample", "http_baseline"?}``. The
    ``summary`` carries mean/p50/p95 for each latency in :data:`LATENCY_KEYS`,
    corpus WER, and how many sessions violated a protocol invariant. Nothing is
    asserted here; CI decides what to gate on.
    """
    if protocol == "native":
        if mode is not None:
            raise ValueError("mode is unavailable with the native protocol")
        else:
            pass
        session_kwargs = {"turn_detection": None, "manual_commit": False}
        trailing_silence_ms = 0
        url = f"{realtime_url(host, port, protocol='native')}?{urlencode({'model': model_path})}"
    else:
        mode = mode or "vad"
        session_kwargs = _session_kwargs(mode)
        url = realtime_url(host, port)
    pcm_cache = {} if pcm_cache is None else pcm_cache
    for sample in samples:
        if sample.ref_audio not in pcm_cache:
            pcm_cache[sample.ref_audio] = load_pcm16(
                sample.ref_audio, preserve_pcm=protocol == "native"
            )

    config = {
        "repeat": repeat,
        "warmup": warmup,
        "protocol": protocol,
        "mode": mode,
        "concurrency": concurrency,
        "packet_ms": packet_ms,
        "paced": paced,
        "trailing_silence_ms": trailing_silence_ms,
        "timeout_s": timeout_s,
        "lang": lang,
        "model_path": model_path,
    }
    if trace_dir is None:
        run_dir = None
        source = {}
        input_hashes = {}
    else:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"c{concurrency}-", dir=trace_dir))
        source = source_fingerprint()
        input_hashes = {
            sample.ref_audio: hashlib.sha256(pcm_cache[sample.ref_audio]).hexdigest()
            for sample in samples
        }
        for audio_path, digest in input_hashes.items():
            (run_dir / f"input-{digest}.pcm").write_bytes(pcm_cache[audio_path])
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "config": config,
                    "source": source,
                    "samples": [
                        {
                            "sample_id": sample.sample_id,
                            "trace_file": f"{index:06d}.json",
                            "input_pcm_sha256": input_hashes[sample.ref_audio],
                        }
                        for index, sample in enumerate(samples)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

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
                protocol=protocol,
                **session_kwargs,
            )

    start = time.perf_counter()
    traces = await asyncio.gather(*(_one(sample) for sample in samples))
    wall_clock_s = time.perf_counter() - start

    if run_dir is not None:
        for index, (sample, trace) in enumerate(zip(samples, traces)):
            save_trace(
                run_dir / f"{index:06d}.json",
                TraceArtifact(
                    schema_version=1,
                    sample_id=sample.sample_id,
                    config=config,
                    source=source,
                    input_pcm_sha256=input_hashes[sample.ref_audio],
                    trace=trace,
                ),
            )
    else:
        pass

    outputs: list[SampleOutput] = []
    per_sample: list[dict[str, Any]] = []
    for index, (sample, trace) in enumerate(zip(samples, traces)):
        output = wer_metrics(trace, sample.ref_text, lang=lang)
        output.sample_id = sample.sample_id
        outputs.append(output)
        record = _per_sample_record(sample, trace, output)
        if run_dir is not None:
            record["trace_file"] = str((run_dir / f"{index:06d}.json").resolve())
        else:
            pass
        per_sample.append(record)

    decode_intervals = {t.session.get("decode_interval_ms") for t in traces}
    native_units = {
        (t.session.get("sglang", {}).get("granted") or {}).get("native_unit_ms")
        for t in traces
    }
    result: dict[str, Any] = {
        "config": {
            **config,
            "decode_interval_ms": (
                decode_intervals.pop() if len(decode_intervals) == 1 else None
            ),
            "native_unit_ms": native_units.pop() if len(native_units) == 1 else None,
        },
        "summary": _summary(
            per_sample, outputs, lang=lang, wall_clock_s=wall_clock_s, protocol=protocol
        ),
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
            request_language="auto" if protocol == "native" else lang,
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
            f"| {config['decode_interval_ms'] if config['protocol'] == 'transcription' else config['native_unit_ms']} | {summary['total']} "
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
    parser.add_argument(
        "--protocol", choices=("transcription", "native"), default="transcription"
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="Turn closing for the transcription protocol (default: vad).",
    )
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--warmup", action="store_true")
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
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Save inputs and raw session observations here (default: OUTPUT.traces).",
    )
    args = parser.parse_args()
    if args.protocol == "native" and args.mode is not None:
        parser.error("--mode is only available with --protocol transcription")
    else:
        return args


async def run_sweep(
    arguments: argparse.Namespace, samples: list[SampleInput]
) -> list[dict[str, JsonValue]]:
    pcm_cache: dict[str, bytes] = {}
    results: list[dict[str, JsonValue]] = []
    for concurrency in arguments.concurrencies:
        for repeat in range(0 if arguments.warmup else 1, arguments.repeats + 1):
            result = await run_asr_realtime_once(
                samples,
                host=arguments.host,
                port=arguments.port,
                concurrency=concurrency,
                packet_ms=arguments.packet_ms,
                protocol=arguments.protocol,
                mode=arguments.mode,
                lang=arguments.lang,
                paced=not arguments.no_pace,
                trailing_silence_ms=arguments.trailing_silence_ms,
                timeout_s=arguments.timeout_s,
                with_http_baseline=arguments.http_baseline,
                model_path=arguments.model_path,
                pcm_cache=pcm_cache,
                trace_dir=arguments.trace_dir or f"{arguments.output}.traces",
                repeat=repeat,
                warmup=repeat == 0,
            )
            if repeat == 0:
                if result["summary"]["sessions_with_violations"]:
                    raise RuntimeError("Warmup failed; inspect saved traces")
                else:
                    pass
            else:
                summary = result["summary"]
                print(
                    f"[conc={concurrency}] wall={summary['wall_clock_s']:.1f}s "
                    f"violations={summary['sessions_with_violations']}/{summary['total']} "
                    f"corpus_wer={_fmt(summary['corpus_wer'], 4)} "
                    f"first_partial_p95={_fmt((summary['first_partial_latency_s'] or {}).get('p95'))}s"
                )
                results.append(result)
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    configure_dependency_loggers()
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
        f"protocol={args.protocol} mode={args.mode or 'default'} "
        f"packet={args.packet_ms}ms paced={not args.no_pace} "
        f"concurrencies={args.concurrencies} against {args.host}:{args.port}"
    )

    results = asyncio.run(run_sweep(args, samples))
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
    if args.protocol == "native" and any(
        result["summary"]["sessions_with_violations"] for result in results
    ):
        raise RuntimeError("Native requests failed; inspect saved results and traces")
    else:
        pass


if __name__ == "__main__":
    main()
