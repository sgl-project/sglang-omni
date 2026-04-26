# SPDX-License-Identifier: Apache-2.0
"""Video-AMME benchmark for Qwen3-Omni video + audio input.

Video-AMME is derived from the Video-MME CI subset. The video is paired with a
spoken audio question; the text prompt contains only routing and answer-format
instructions.

Usage:
    python -m benchmarks.dataset.prepare --dataset video-amme-ci-50

    python examples/run_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --model-name qwen3-omni \
        --port 30000 \
        --thinker-max-seq-len 32768 \
        --mem-fraction-static 0.78

    python -m benchmarks.eval.benchmark_omni_video_amme \
        --model qwen3-omni --port 30000 \
        --repo-id Ratish21/Video_AMME_ci \
        --max-samples 50 --max-concurrency 8 \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

H200 Reference Results

Benchmark: Video-AMME | Dataset: Ratish21/Video_AMME_ci test split (50 questions)
Hardware:  1 x H200
Last verified: 2026-04-26

Accuracy

| Model      | Config                     | accuracy | correct | failed | mc_fallback | Source                                                              |
| ---------- | -------------------------- | -------- | ------- | ------ | ----------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, full-set, c=8 | 66.00%   | 33/50   | 0      | 0           | benchmark-video-amme 8fb2dcd65ed8a896a50c5ab8ab2cf10c08c787ff [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, c=8        | 50.00%   | 5/10    | 0      | 0           | benchmark-video-amme 8fb2dcd65ed8a896a50c5ab8ab2cf10c08c787ff [H200, c=8, max_tokens=256] |

Speed

| Model      | Config                     | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | tok_per_s_mean | tok_per_s_agg | gen_tokens_mean | gen_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                                              |
| ---------- | -------------------------- | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | -------------- | ------------- | --------------- | ---------------- | ------------------ | ------------------- | -------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, full-set, c=8 | 50        | 0      | 54.545         | 48.362           | 95.787        | 108.239       | 0.9            | 0.8           | 46              | 2278             | 14336              | 716818              | 0.145          | benchmark-video-amme 8fb2dcd65ed8a896a50c5ab8ab2cf10c08c787ff [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, c=8        | 10        | 0      | 94.525         | 100.172          | 136.172       | 137.596       | 0.6            | 0.6           | 53              | 526              | 14434              | 144340              | 0.072          | benchmark-video-amme 8fb2dcd65ed8a896a50c5ab8ab2cf10c08c787ff [H200, c=8, max_tokens=256] |

Talker WER

| Model      | Config              | evaluated | skipped | wer_corpus | wer_per_sample_mean | wer_per_sample_p95 | wer_per_sample_max | n_above_50_pct_wer | rtf_mean | audio_duration_mean_s | Source                                                              |
| ---------- | ------------------- | --------- | ------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ | -------- | --------------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-talker, c=8 | 10        | 0       | 0.70%      | 0.90%               | 3.75%              | 4.00%              | 0                  | 6.3183   | 15.598                | benchmark-video-amme 8fb2dcd65ed8a896a50c5ab8ab2cf10c08c787ff [H200, c=8, max_tokens=256] |
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.video_amme import DEFAULT_REPO_ID as _VIDEO_AMME_DEFAULT_REPO
from benchmarks.dataset.video_amme import VideoAMMESample, load_video_amme_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import (
    compute_text_audio_consistency,
    print_speed_summary,
    print_wer_summary,
)
from benchmarks.tasks.video_understanding import (
    compute_videomme_metrics,
    make_video_amme_send_fn,
    print_videomme_accuracy_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VideoAMMEEvalConfig:
    model: str
    split: str = "test"
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 0
    request_rate: float = float("inf")
    timeout_s: int = 300
    disable_tqdm: bool = False
    repo_id: str | None = None
    enable_audio: bool = False
    asr_device: str = "cuda:0"
    lang: str = "en"


def _build_base_url(config: VideoAMMEEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_video_amme_eval(
    config: VideoAMMEEvalConfig,
    *,
    samples: list[VideoAMMESample] | None = None,
) -> dict:
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    if samples is None:
        samples = load_video_amme_samples(
            repo_id=config.repo_id,
            split=config.split,
            max_samples=config.max_samples,
        )
    logger.info("Prepared %d Video-AMME samples", len(samples))
    audio_dir = None
    if config.enable_audio:
        output_root = Path(config.output_dir or "results/videoamme_audio")
        audio_dir = str(output_root / "audio")

    send_fn = make_video_amme_send_fn(
        config.model,
        api_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        video_fps=config.video_fps,
        video_max_frames=config.video_max_frames,
        video_min_pixels=config.video_min_pixels,
        video_max_pixels=config.video_max_pixels,
        video_total_pixels=config.video_total_pixels,
        enable_audio=config.enable_audio,
        audio_dir=audio_dir,
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
            timeout_s=config.timeout_s,
        )
    )
    request_results = await runner.run(samples, send_fn)

    summary, per_sample = compute_videomme_metrics(samples, request_results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    results = {
        "summary": summary,
        "speed": speed,
        "config": {
            "model": config.model,
            "base_url": base_url,
            "repo_id": config.repo_id,
            "split": config.split,
            "max_samples": config.max_samples,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "video_fps": config.video_fps,
            "video_max_frames": config.video_max_frames,
            "video_min_pixels": config.video_min_pixels,
            "video_max_pixels": config.video_max_pixels,
            "video_total_pixels": config.video_total_pixels,
            "max_concurrency": config.max_concurrency,
            "warmup": config.warmup,
            "enable_audio": config.enable_audio,
            "asr_device": config.asr_device,
            "lang": config.lang,
        },
        "per_sample": per_sample,
    }
    if config.enable_audio:
        results["wer"] = compute_text_audio_consistency(
            request_results,
            config.lang,
            config.asr_device,
        )

    if config.output_dir:
        save_json_results(results, config.output_dir, "videoamme_results.json")

    return results


def _config_from_args(args: argparse.Namespace) -> VideoAMMEEvalConfig:
    return VideoAMMEEvalConfig(
        model=args.model,
        repo_id=args.repo_id,
        split=args.split,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_total_pixels=args.video_total_pixels,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        timeout_s=args.timeout_s,
        enable_audio=args.enable_audio,
        asr_device=args.asr_device,
        lang=args.lang,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_video_amme_eval(config)
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
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3-omni")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset repo for Video-AMME. "
            f"Defaults to {_VIDEO_AMME_DEFAULT_REPO}."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--video-total-pixels", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Request text+audio output and compute text-audio WER.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cuda:0",
        help="Device for ASR model when --enable-audio is used.",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language for ASR transcription when --enable-audio is used.",
    )
    args = parser.parse_args()

    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
