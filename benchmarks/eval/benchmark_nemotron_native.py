"""Measure native ASR sessions with the project datasets and WER scorer."""

import argparse
import asyncio
import json
import math
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlencode

import msgspec

from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.nemotron_native_client import Measurement, measure
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import apply_wer


class Percentiles(TypedDict):
    n: int
    p50: float | None
    p95: float | None


def percentile(values: list[float], fraction: float) -> float | None:
    if values:
        return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]
    else:
        return None


def summarize_measurements(
    results: list[Measurement],
    outputs: list[SampleOutput],
    score_language: str,
    concurrency: int,
    repeat: int,
    elapsed_seconds: float,
) -> dict[str, int | float | None | Percentiles]:
    valid = [result for result in results if result.error is None]
    evaluated = sum(output.is_success for output in outputs)
    metrics = calculate_wer_metrics(outputs, score_language)
    summary: dict[str, int | float | None | Percentiles] = {
        "concurrency": concurrency,
        "repeat": repeat,
        "total": len(results),
        "success": len(valid),
        "wer_evaluated": evaluated,
        "wer_skipped": len(valid) - evaluated,
        "wer_percent": metrics["wer_corpus"] * 100 if evaluated else None,
        "wall_seconds": elapsed_seconds,
        "audio_seconds_per_second": sum(result.audio_seconds for result in valid)
        / elapsed_seconds,
        "requests_per_second": len(valid) / elapsed_seconds,
    }
    latencies: dict[str, list[float]] = {
        "request_wall_seconds": [
            result.wall_seconds for result in valid if result.wall_seconds is not None
        ],
        "request_rtf": [
            result.wall_seconds / result.audio_seconds
            for result in valid
            if result.wall_seconds is not None and result.audio_seconds > 0
        ],
        "first_text_seconds": [
            result.first_text_seconds
            for result in valid
            if result.first_text_seconds is not None
        ],
        "eos_to_final_seconds": [
            result.eos_to_final_seconds
            for result in valid
            if result.eos_to_final_seconds is not None
        ],
        "eos_to_drained_seconds": [
            result.eos_to_drained_seconds
            for result in valid
            if result.eos_to_drained_seconds is not None
        ],
    }
    for name, values in latencies.items():
        summary[name] = {
            "n": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }
    return summary


async def run(arguments: argparse.Namespace) -> None:
    samples = load_seedtts_samples(
        arguments.meta,
        max_samples=arguments.max_samples,
        split=arguments.score_language,
        revision=arguments.dataset_revision,
    )
    if not samples:
        raise ValueError("No audio samples")
    else:
        audio: dict[str, bytes] = {}
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError(
            "Sample identifiers must be unique; repeated audio paths are allowed"
        )
    else:
        pass
    for sample in samples:
        with wave.open(sample.ref_audio, "rb") as source:
            if (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            ) != (1, 2, 16000):
                raise ValueError(
                    f"Prepare mono 16 kHz PCM16 WAV before timing: {sample.ref_audio}"
                )
            else:
                audio[sample.sample_id] = source.readframes(source.getnframes())
    apply_wer(SampleOutput(target_text="hello"), "hello", arguments.score_language)
    arguments.output.mkdir(parents=True, exist_ok=False)
    (arguments.output / "config.json").write_text(
        json.dumps(
            {
                **vars(arguments),
                "output": str(arguments.output),
                "sample_ids": [sample.sample_id for sample in samples],
                "audio_total_seconds": sum(
                    len(value) / 32000 for value in audio.values()
                ),
            },
            indent=2,
        )
    )
    url = f"{arguments.url}?{urlencode({'model': arguments.model})}"
    summaries: list[dict[str, int | float | None | Percentiles]] = []
    failed_requests = 0
    for concurrency in arguments.concurrencies:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(index: int) -> Measurement:
            sample = samples[index]
            async with semaphore:
                return await measure(
                    url,
                    audio[sample.sample_id],
                    sample.sample_id,
                    sample.ref_text,
                    packet_milliseconds=arguments.packet_milliseconds,
                    paced=not arguments.burst,
                    timeout_seconds=arguments.timeout_seconds,
                )

        for repeat in range(0 if arguments.warmup else 1, arguments.repeats + 1):
            started = time.perf_counter()
            results = await asyncio.gather(
                *(one(index) for index in range(len(samples)))
            )
            elapsed_seconds = time.perf_counter() - started
            outputs: list[SampleOutput] = []
            path = arguments.output / f"c{concurrency}-r{repeat}.jsonl"
            with path.open("w") as handle:
                for result in results:
                    output = SampleOutput(
                        sample_id=result.sample_id,
                        target_text=result.reference,
                        audio_duration_s=result.audio_seconds,
                    )
                    if result.error is None:
                        output = apply_wer(
                            output, result.text, arguments.score_language
                        )
                    else:
                        output.error = result.error
                    outputs.append(output)
                    handle.write(
                        json.dumps(
                            {**msgspec.to_builtins(result), "score": asdict(output)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if repeat == 0:
                if any(result.error is not None for result in results):
                    raise RuntimeError(f"Warmup failed; see {path}")
                else:
                    continue
            else:
                pass
            summary = summarize_measurements(
                results,
                outputs,
                arguments.score_language,
                concurrency,
                repeat,
                elapsed_seconds,
            )
            failed_requests += sum(result.error is not None for result in results)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            (arguments.output / "summary.json").write_text(
                json.dumps(summaries, indent=2)
            )

    if failed_requests:
        raise RuntimeError(
            f"{failed_requests} measured requests failed; inspect saved results"
        )
    else:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    parser.add_argument("--score-language", choices=["en", "zh"], default="en")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--packet-milliseconds", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        min(
            arguments.max_samples,
            arguments.repeats,
            arguments.packet_milliseconds,
            arguments.timeout_seconds,
            *arguments.concurrencies,
        )
        <= 0
    ):
        parser.error(
            "Sample counts, repetitions, concurrency, packet size and timeout must be positive"
        )
    else:
        asyncio.run(run(arguments))


if __name__ == "__main__":
    main()
else:
    pass
