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
| Qwen3-Omni | thinker-only, ci-50, c=8 | 66.00%   | 33/50   | 0      | 0           | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 64.00%   | 32/50   | 0      | 0           | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 66.00%   | 33/50   | 0      | 0           | PR #411 [H100, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 68.00%   | 34/50   | 0      | 0           | local [H100, c=8, max_tokens=256] |

Speed

| Model      | Config                | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | output_tok_per_req_s | output_tokens_mean | output_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                                              |
| ---------- | --------------------- | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | ------------------------------ | --------------- | ---------------- | ------------------ | ------------------- | -------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 44.530         | 46.846           | 52.694        | 53.180        | 0.9                            | 40.0            | 2025             | 21684.0            | 1084218             | 0.167          | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50        | 0      | 40.423         | 40.063           | 63.088        | 81.046        | 1.0                            | 41.0            | 2050             | 21684.0            | 1084218             | 0.193          | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 38.408         | 40.320           | 44.883        | 45.764        | 1.1                            | 44.0            | 2181             | 21684.0            | 1084218             | 0.194          | PR #411 [H100, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50        | 0      | 33.618         | 33.365           | 46.380        | 58.321        | 1.3                            | 42.0            | 2123             | 21684.0            | 1084218             | 0.230          | local [H100, c=8, max_tokens=256] |


Talker WER

| Model      | Config                    | evaluated | skipped | wer_corpus | wer_per_sample_mean | wer_per_sample_p95 | wer_per_sample_max | n_above_50_pct_wer | rtf_mean | audio_duration_mean_s | Source                                                              |
| ---------- | ------------------------- | --------- | ------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ | -------- | --------------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50/50     | 0       | 2.58%      | 18.87%              | 155.00%            | 200.00%            | 6                  | 6.3183   | 11.750                | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50/50     | 0       | 2.29%      | 11.94%              | 100.00%            | 200.00%            | 4                  | 6.0412   | 12.356                | local [H100, c=8, max_tokens=256] |

Local v1 Pipeline Result (this workspace, 2026-05-01)

Accuracy

| Model      | Config                   | accuracy | correct | failed | mc_fallback | Source                                               |
| ---------- | ------------------------ | -------- | ------- | ------ | ----------- | ---------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 68.00%   | 34/50   | 0      | 0           | local v1 sweep [H200, ci-50, c=8, max_tokens=256]   |

Speed

| Model      | Config                   | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | output_tok_per_req_s | output_tokens_mean | output_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                               |
| ---------- | ------------------------ | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | ------------------------------ | --------------- | ---------------- | ------------------ | ------------------- | -------------- | ---------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 133.245        | 137.354          | 155.201       | 159.106       | 0.3                            | 43              | 2172             | 21684              | 1084218             | 0.058          | local v1 sweep [H200, ci-50, c=8, max_tokens=256]   |
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
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.video_understanding import VIDEOAMME_REQUEST_TEXT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run_videoamme_eval(
    config: VideoEvalConfig,
    *,
    samples: list[VideoAMMESample] | None = None,
    compute_wer: bool = True,
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
        compute_wer=compute_wer,
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
