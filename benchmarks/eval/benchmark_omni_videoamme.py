# SPDX-License-Identifier: Apache-2.0
"""Video-AMME benchmark for Qwen3-Omni video + audio input.

Video-AMME is derived from the Video-MME CI subset. The video is paired with a
spoken audio question; the text prompt contains only routing and answer-format
instructions.

Usage:
    python -m benchmarks.dataset.prepare --dataset videoamme-ci-50

    python examples/run_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --model-name qwen3-omni \
        --port 30000 \
        --thinker-max-seq-len 32768 \
        --mem-fraction-static 0.78

    python -m benchmarks.eval.benchmark_omni_videoamme \
        --model qwen3-omni --port 30000 \
        --repo-id zhaochenyang20/Video_AMME_ci \
        --max-samples 50 --max-concurrency 8 \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

H200 Reference Results

Benchmark: Video-AMME | Dataset: zhaochenyang20/Video_AMME_ci test split (50 questions)
Hardware:  1 x H200
Last verified: 2026-04-26

Accuracy

| Model      | Config                | accuracy | correct | failed | mc_fallback | Source                                                              |
| ---------- | --------------------- | -------- | ------- | ------ | ----------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 66.00%   | 33/50   | 0      | 0           | PR #356 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-10, c=8 | 50.00%   | 5/10    | 0      | 0           | PR #356 [H200, c=8, max_tokens=256] |

Speed

| Model      | Config                | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | tok_per_s_mean | tok_per_s_agg | gen_tokens_mean | gen_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                                              |
| ---------- | --------------------- | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | -------------- | ------------- | --------------- | ---------------- | ------------------ | ------------------- | -------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 54.545         | 48.362           | 95.787        | 108.239       | 0.9            | 0.8           | 46              | 2278             | 14336              | 716818              | 0.145          | PR #356 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-10, c=8 | 10        | 0      | 94.525         | 100.172          | 136.172       | 137.596       | 0.6            | 0.6           | 53              | 526              | 14434              | 144340              | 0.072          | PR #356 [H200, c=8, max_tokens=256] |

Talker WER

| Model      | Config                    | evaluated | skipped | wer_corpus | wer_per_sample_mean | wer_per_sample_p95 | wer_per_sample_max | n_above_50_pct_wer | rtf_mean | audio_duration_mean_s | Source                                                              |
| ---------- | ------------------------- | --------- | ------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ | -------- | --------------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-talker, ci-10, c=8 | 10        | 0       | 0.70%      | 0.90%               | 3.75%              | 4.00%              | 0                  | 6.3183   | 15.598                | PR #356 [H200, c=8, max_tokens=256] |
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.videomme import (
    DEFAULT_VIDEOAMME_REPO_ID as _VIDEOAMME_DEFAULT_REPO,
)
from benchmarks.dataset.videomme import VideoAMMESample, load_videoamme_samples
from benchmarks.eval.benchmark_omni_videomme import (
    VideoEvalConfig,
    add_video_eval_args,
    run_video_eval,
    video_eval_config_from_args,
)
from benchmarks.tasks.tts import print_speed_summary, print_wer_summary
from benchmarks.tasks.video_understanding import (
    VIDEOAMME_REQUEST_TEXT,
    print_videomme_accuracy_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run_videoamme_eval(
    config: VideoEvalConfig,
    *,
    samples: list[VideoAMMESample] | None = None,
) -> dict:
    return await run_video_eval(
        config,
        samples=samples,
        load_samples=load_videoamme_samples,
        task_label="Video-AMME",
        output_filename="videoamme_results.json",
        audio_output_dir_default="results/videoamme_audio",
        enable_audio_input=True,
        fixed_prompt=VIDEOAMME_REQUEST_TEXT,
    )


def _config_from_args(args: argparse.Namespace) -> VideoEvalConfig:
    return video_eval_config_from_args(args)


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_videoamme_eval(config)
    print_videomme_accuracy_summary(
        results["summary"],
        config.model,
        title="Video-AMME Accuracy",
    )
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="Video-AMME Speed",
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-AMME benchmark for video + audio question models."
    )
    add_video_eval_args(
        parser,
        repo_help=(
            "HuggingFace dataset repo for Video-AMME. "
            f"Defaults to {_VIDEOAMME_DEFAULT_REPO}."
        ),
    )
    args = parser.parse_args()

    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
