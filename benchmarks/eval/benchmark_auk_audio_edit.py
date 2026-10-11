# SPDX-License-Identifier: Apache-2.0
"""Run AuK speech editing on the Ming Freeform Audio Edit benchmark.

Usage:

    # Start an AuK server first:
    python -m sglang_omni.cli serve --model-path tencent/AuK --port 8000

    # Generate and score the published time-stretch subset:
    python -m benchmarks.eval.benchmark_auk_audio_edit \
        --task time_stretch --language en \
        --output-dir results/auk_ming_time_stretch

The runner supports every published edit task. Time-stretch and volume runs
also report the benchmark's official signal-level error metrics. Generated WAV
files and per-sample metadata are retained for WER, speaker-similarity, and
task-specific evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import (
    get_wav_duration,
    save_json_results,
    wait_for_service,
)
from benchmarks.dataset.ming_freeform_audio_edit import (
    SUPPORTED_TASKS,
    SpeechEditSample,
    load_ming_freeform_samples,
)
from benchmarks.dataset.prepare import (
    MING_FREEFORM_AUDIO_EDIT_DATASET_ID,
    MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION,
)
from benchmarks.metrics.performance import compute_speed_metrics, print_speed_summary
from benchmarks.metrics.speech_edit import (
    aggregate_signal_edit_scores,
    score_signal_edit,
)

logger = logging.getLogger(__name__)
_RESULTS_FILENAME = "ming_freeform_audio_edit_results.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _build_base_url(args: argparse.Namespace) -> str:
    return (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")


def _build_auk_edit_request(
    sample: SpeechEditSample,
    *,
    model: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": sample.instruction,
        "metadata": {"tts_params": {"ref_audio": sample.source_audio_url}},
        "output_modalities": ["audio"],
        "return_logprob": False,
        "sampling_params": {"seed": seed},
    }


def _make_auk_edit_send_fn(
    *,
    api_url: str,
    model: str,
    seed: int,
    generated_dir: Path,
) -> SendFn:
    generated_dir.mkdir(parents=True, exist_ok=True)

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: SpeechEditSample,
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.instruction[:80],
        )
        payload = _build_auk_edit_request(sample, model=model, seed=seed)
        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                    return result
                body = await response.json()

            audio = body.get("audio")
            if not isinstance(audio, dict):
                raise ValueError("Response does not contain an audio object")
            if audio.get("format") not in {None, "wav"}:
                raise ValueError(f"Expected WAV audio, got {audio.get('format')!r}")
            encoded = audio.get("data")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("Response does not contain base64 audio data")
            wav_bytes = base64.b64decode(encoded, validate=True)
            duration = get_wav_duration(wav_bytes)
            if duration <= 0:
                raise ValueError("Response contains an empty or invalid WAV")

            wav_path = (generated_dir / f"{sample.sample_id}.wav").resolve()
            wav_path.write_bytes(wav_bytes)
            meta_info = body.get("meta_info") or {}
            result.prompt_tokens = int(meta_info.get("prompt_tokens", 0) or 0)
            result.completion_tokens = int(meta_info.get("completion_tokens", 0) or 0)
            result.audio_duration_s = duration
            result.wav_path = str(wav_path)
            result.is_success = True
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            binascii.Error,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
            result.engine_time_s = result.latency_s
            if result.is_success:
                result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send_fn


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_ming_freeform_samples(
        args.task,
        language=args.language,
        semantic_subset=args.semantic_subset,
        max_samples=args.max_samples,
        revision=args.dataset_revision,
    )
    if not samples:
        raise ValueError("No benchmark samples were loaded")

    output_dir = Path(args.output_dir).resolve()
    send_fn = _make_auk_edit_send_fn(
        api_url=f"{_build_base_url(args)}/generate",
        model=args.model,
        seed=args.seed,
        generated_dir=output_dir / "generated",
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            warmup=args.warmup,
            timeout_s=args.timeout_s,
        )
    )
    outputs = await runner.run(samples, send_fn)
    speed = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)

    sample_by_id = {sample.sample_id: sample for sample in samples}
    signal_scores: list[dict[str, float]] = []
    per_sample: list[dict[str, Any]] = []
    for output in outputs:
        sample = sample_by_id[output.request_id]
        row = {
            **asdict(output),
            "task": sample.task,
            "language": sample.language,
            "instruction": sample.instruction,
            "original_text": sample.original_text,
            "edited_text": sample.edited_text,
            "source_audio": sample.source_audio,
            "source_audio_repo_path": sample.source_audio_repo_path,
            "scale": sample.scale,
            "signal_metrics": None,
            "metric_error": None,
        }
        if output.is_success and args.task in {"time_stretch", "volume"}:
            try:
                if sample.scale is None:
                    raise ValueError("Benchmark sample does not define an edit scale")
                score = score_signal_edit(
                    sample.source_audio,
                    output.wav_path,
                    task=args.task,
                    scale=sample.scale,
                )
                row["signal_metrics"] = score
                signal_scores.append(score)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                row["metric_error"] = str(exc)
        per_sample.append(row)

    signal_summary = aggregate_signal_edit_scores(signal_scores)
    result = {
        "config": {
            "model": args.model,
            "base_url": _build_base_url(args),
            "dataset": MING_FREEFORM_AUDIO_EDIT_DATASET_ID,
            "dataset_revision": args.dataset_revision,
            "task": args.task,
            "language": args.language,
            "semantic_subset": args.semantic_subset,
            "max_samples": args.max_samples,
            "max_concurrency": args.max_concurrency,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "speed": speed,
        "signal_metrics": signal_summary,
        "per_sample": per_sample,
    }
    save_json_results(result, str(output_dir), _RESULTS_FILENAME)
    print_speed_summary(
        speed,
        args.model,
        concurrency=args.max_concurrency,
        title="AuK Audio Edit Benchmark",
        dataset=f"Ming Freeform {args.task} {args.language}",
    )
    if signal_scores:
        logger.info("Signal metrics: %s", signal_summary)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="time_stretch")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--semantic-subset", choices=("basic", "full"), default="basic")
    parser.add_argument("--model", default="tencent/AuK")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url")
    parser.add_argument("--output-dir", default="results/auk_ming_audio_edit")
    parser.add_argument("--max-samples", type=_positive_int)
    parser.add_argument("--max-concurrency", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-s", type=_positive_int, default=600)
    parser.add_argument(
        "--dataset-revision",
        default=MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION,
    )
    parser.add_argument("--skip-health-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    base_url = _build_base_url(args)
    if not args.skip_health_check:
        wait_for_service(base_url, timeout=args.timeout_s)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
