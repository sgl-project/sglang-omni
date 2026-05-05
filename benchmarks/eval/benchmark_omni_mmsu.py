# SPDX-License-Identifier: Apache-2.0
"""MMSU (audio understanding) benchmark for sglang-omni models.

Evaluates Qwen3-Omni accuracy and performance on the MMSU eval set via
/v1/chat/completions with optional audio input modality.

Usage:

    # Launch the server:
    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --version v1 --port 8000

    # If only text is needed:

    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --version v1 --text-only --port 8000

    # Prepare the dataset:
    python -m benchmarks.dataset.prepare --dataset mmsu

    # Text-only
    python benchmarks/eval/benchmark_omni_mmsu.py \
        --model qwen3-omni --port 8000 --max-samples 50

    # Text + audio
    python benchmarks/eval/benchmark_omni_mmsu.py \
        --model qwen3-omni --port 8000 --max-samples 50 \
        --modalities text+audio

    # Filter by task names or categories
    python benchmarks/eval/benchmark_omni_mmsu.py \
        --model qwen3-omni --port 8000 \
        --task-names casual_reasoning,continuation_writing


H200 Full-Set Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).

Benchmark: MMSU     |  Dataset: MMSU full (5000 samples)
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-05-04

Accuracy (accuracy)

| Model      | Config                | overall_accuracy | parseable_samples | unparseable_samples | Source                        |
| ---------- | --------------------- | ---------------- | ----------------- | ------------------- | ----------------------------- |
| Qwen3-Omni | modalities=text       | 70.96%           | 5000/5000         | 0                   | PR #393 [H200, V1-pipeline, full-set, c=8] |
| Qwen3-Omni | modalities=text+audio | 71.06%           | 5000/5000         | 0                   | PR #393 [H200, V1-pipeline, full-set, c=8] |
| Qwen3-Omni | modalities=text       | 71.10%           | 4999/5000         | 1                   | PR #351 [H100, full-set, c=8] |
| Qwen3-Omni | modalities=text+audio | 71.14%           | 5000/5000         | 0                   | PR #351 [H100, full-set, c=8] |

Per-task accuracy (accuracy.per_task; top-level task names only — full sub/sub-sub trees stay in JSON output)

| Model      | Config                | per_task breakdown (highlights)                                                                                                                                                                | Source                   |
| ---------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Qwen3-Omni | modalities=text       | strong: casual_reasoning / continuation_writing / long_speech_summarization / polysemy_reasoning = 100%; weak: dialogue_turn_counting 10.10%, pitch_comparison 22.22%, speed_comparison 22.94% | PR #316 [H200, full-set] |
| Qwen3-Omni | modalities=text+audio | strong: casual_reasoning / continuation_writing / long_speech_summarization / polysemy_reasoning = 100%; weak: dialogue_turn_counting 12.12%, speed_comparison 21.10%, pitch_comparison 24.07% | PR #316 [H200, full-set] |
| Qwen3-Omni | modalities=text       | strong: casual_reasoning / continuation_writing / long_speech_summarization / polysemy_reasoning = 100%; weak: dialogue_turn_counting 12.12%, speed_comparison 22.94%, pitch_comparison 23.15%   | PR #351 [H100, full-set] |
| Qwen3-Omni | modalities=text+audio | strong: casual_reasoning / continuation_writing / long_speech_summarization / polysemy_reasoning = 100%; weak: dialogue_turn_counting 13.13%, speed_comparison 22.02%, pitch_comparison 24.07%   | PR #351 [H100, full-set] |

Speed (speed)

| Model      | Config                | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                          |
| ---------- | --------------------- | -------------- | ------------- | -------------- | -------------- | ------------- | ----------------------------------------------- |
| Qwen3-Omni | modalities=text       | 0.226          | 0.335         | 35.33          | 9.1            | 9.1           | PR #393 [H200, V1-pipeline, full-set, c=8]                   |
| Qwen3-Omni | modalities=text+audio | 0.243          | 0.340         | 32.89          | 8.5            | 8.5           | PR #393 [H200, V1-pipeline, full-set, c=8, text-only server] |
| Qwen3-Omni | modalities=text       | 0.512          | 0.864         | 15.598         | 4.5            | 4.0           | PR #351 [H100, full-set, c=8]                   |
| Qwen3-Omni | modalities=text+audio | 0.515          | 0.884         | 15.521         | 4.4            | 4.0           | PR #351 [H100, full-set, c=8] (text-only server) |

Note (Xuesong): text + audio numbers above were measured against a text-only
Qwen3-Omni server (talker disabled) because a full-pipeline run is blocked on
Issue #276 (talker is c=1 only at ~2 min/sample). Numbers therefore reflect
text-only behavior and are near-identical to the `modalities=text` row;
re-run with talker enabled once #276 lands to get true full-pipeline reference.

Local v1 Pipeline Result (this workspace, 2026-05-01)

This local run used the stage-6 talker prompt on the full 2000-sample
`zhaochenyang20/mmsu-ci-2000` backing set, not the 5000-sample `ddwang2000/MMSU`
full set summarized above, so it is not directly apples-to-apples with the
reference rows.

Accuracy (summary)

| Model      | Config                         | overall_accuracy | parseable_samples | unparseable_samples | Source                                                                       |
| ---------- | ------------------------------ | ---------------- | ----------------- | ------------------- | ---------------------------------------------------------------------------- |
| Qwen3-Omni | stage6 talker, mmsu-ci-2000    | 70.05%           | 1996/2000         | 4                   | local v1 run [H200, 2000-sample stage-6 backing set, speech pipeline, c=1] |

Speed (speed)

| Model      | Config                         | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                                       |
| ---------- | ------------------------------ | -------------- | ------------- | -------------- | -------------- | ------------- | ---------------------------------------------------------------------------- |
| Qwen3-Omni | stage6 talker, mmsu-ci-2000    | 22.049         | 31.935        | 0.362          | 2.7            | 2.7           | local v1 run [H200, 2000-sample stage-6 backing set, speech pipeline, c=1] |

Additional local notes: `audio_returned=2000/2000`, `rtf_mean=1.2704`.
"""


from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmsu import MmsuSample, load_mmsu_samples
from benchmarks.metrics.mmsu import compute_mmsu_metrics, print_mmsu_summary
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.audio_understanding import (
    build_mmsu_results,
    make_mmsu_send_fn,
    save_mmsu_results,
)
from benchmarks.tasks.tts import compute_text_audio_consistency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run(
    args: argparse.Namespace,
    *,
    samples: list[MmsuSample] | None = None,
) -> dict:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"
    modalities = ["text", "audio"] if args.modalities == "text+audio" else ["text"]

    if samples is None:
        samples = load_mmsu_samples(
            max_samples=args.max_samples,
            task_names=args.task_names.split(",") if args.task_names else None,
            categories=args.categories.split(",") if args.categories else None,
            seed=args.seed,
            repo_id=args.repo_id,
        )

    save_audio_dir = None
    if args.save_audio and args.output_dir:
        save_audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    send_fn_kwargs = dict(
        modalities=modalities,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_audio_dir=save_audio_dir,
    )
    if args.prompt:
        send_fn_kwargs["prompt"] = args.prompt
    send_fn = make_mmsu_send_fn(args.model, api_url, **send_fn_kwargs)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
            timeout_s=args.timeout_s,
        )
    )
    request_results = await runner.run(samples, send_fn)

    results = build_mmsu_results(request_results, samples, modalities)
    metrics = compute_mmsu_metrics(results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    audio_mode = "audio" in modalities
    if audio_mode:
        speed["audio_returned"] = sum(
            1 for r in request_results if r.audio_duration_s > 0
        )
        speed["audio_expected"] = len(request_results)

    output: dict = {"accuracy": metrics, "speed": speed}
    wer_results = None
    if audio_mode:
        wer_results = compute_text_audio_consistency(
            request_results,
            args.lang,
            args.asr_device,
        )
        output["wer"] = wer_results

    if args.output_dir:
        save_mmsu_results(
            results,
            metrics,
            {
                "model": args.model,
                "base_url": base_url,
                "modalities": modalities,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            args.output_dir,
            speed_metrics=speed,
            wer_metrics=wer_results,
        )

    return output


def main() -> None:
    p = argparse.ArgumentParser(description="MMSU benchmark.")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", type=str, default="qwen3-omni")
    p.add_argument("--modalities", choices=["text", "text+audio"], default="text")
    p.add_argument("--output-dir", type=str, default="results/mmsu")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--task-names", type=str, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--save-audio", action="store_true")
    p.add_argument("--disable-tqdm", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo (e.g. 'zhaochenyang20/mmsu-ci-2000'). "
        "Defaults to loading the full ddwang2000/MMSU (train split).",
    )
    p.add_argument(
        "--lang", type=str, default="en", help="Language for ASR WER evaluation"
    )
    p.add_argument(
        "--asr-device", type=str, default="cuda:0", help="Device for ASR model"
    )

    args = p.parse_args()
    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    output = asyncio.run(run(args))
    print_mmsu_summary(output["accuracy"], args.model, speed_metrics=output["speed"])
    if "wer" in output:
        print_wer_summary(output["wer"]["summary"], args.model)


if __name__ == "__main__":
    main()
