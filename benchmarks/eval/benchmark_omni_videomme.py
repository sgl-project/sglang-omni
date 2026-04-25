# SPDX-License-Identifier: Apache-2.0
"""Video-MME benchmark for sglang-omni models.

Evaluates video understanding accuracy and performance on the Video-MME
test set via /v1/chat/completions with video input. Each sample is a
multiple-choice question (A-D) grounded in a YouTube video clip, covering
short, medium, and long durations across six domains.

Note (Qiujiang, Chenyang):

The full test split contains long videos whose prompts approach the 32k-token
thinker context. We set --thinker-max-seq-len 32768 to accommodate the longest
ones, and --encoder-mem-reserve 0.40 to hold back ~56 GB of GPU memory for the
co-located video encoder at peak activation.

TODO (Qiujiang, Chenyang):

We are facing extremely fragmented memory allocation when processing long videos.

In CI of test_qwen3_omni_videomme_ci.py, we use --encoder-mem-reserve 0.20
on the 50-sample videomme-ci-50 subset. The smaller reserve is sufficient
there because the per-server request budget never crosses the threshold where
encoder-activation fragmentation starts dropping requests. At 100 samples on
the test-split prefix,0.40 is the smallest reserve that empirically completes
100 sequential requests without dropped responses; going above 0.40 leaves
 SGLang with too little KV pool to boot.


Detailed usage of the serving args can be found in https://github.com/sgl-project/sglang-omni/pull/339

Usage:

    1. Download the dataset

    python -m benchmarks.dataset.prepare --dataset videomme

    2. Launch the thinker-only server

    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000 \
        --thinker-max-seq-len 32768 \
        --text-only \
        --encoder-mem-reserve 0.40

    3. Run the benchmark (--max-samples matches the reference table below)

    python benchmarks/eval/benchmark_omni_videomme.py \
        --model qwen3-omni --port 8000 \
        --max-concurrency 4 --max-tokens 256 --max-samples 100

H200 Reference Results

Benchmark: Video-MME | Dataset: lmms-lab/Video-MME test split (2520 questions full; first N samples used here)
Hardware:  1 x H200
Last verified: 2026-04-24

Accuracy (summary)

| Model      | Config                          | accuracy | correct | failed | mc_fallback | Source                                                              |
| ---------- | ------------------------------- | -------- | ------- | ------ | ----------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, encoder-reserve=0.40 | 77.00% | 77/100  | 0      | 1           | PR #327 [H200, first-100 prefix, c=4, max_tokens=256] |

Speed (speed)

| Model      | Config                             | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                |
| ---------- | ---------------------------------- | -------------- | ------------- | -------------- | -------------- | ------------- | ----------------------------------------------------- |
| Qwen3-Omni | thinker-only, encoder-reserve=0.40 | 42.53          | 67.62         | 0.094          | 2.70           | 2.60          | PR #327 [H200, first-100 prefix, c=4, max_tokens=256] |
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
from benchmarks.dataset.videomme import DEFAULT_REPO_ID as _VIDEOMME_DEFAULT_REPO
from benchmarks.dataset.videomme import VideoMMESample, load_videomme_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import (
    compute_text_audio_consistency,
    print_speed_summary,
    print_wer_summary,
)
from benchmarks.tasks.video_understanding import (
    compute_videomme_metrics,
    make_videomme_send_fn,
    print_videomme_accuracy_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VideoMMEEvalConfig:
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
    disable_tqdm: bool = False
    repo_id: str | None = None
    enable_audio: bool = False
    asr_device: str = "cuda:0"
    lang: str = "en"


def _build_base_url(config: VideoMMEEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_videomme_eval(
    config: VideoMMEEvalConfig,
    *,
    samples: list[VideoMMESample] | None = None,
) -> dict:
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    if samples is None:
        samples = load_videomme_samples(
            repo_id=config.repo_id,
            split=config.split,
            max_samples=config.max_samples,
        )
    logger.info("Prepared %d Video-MME samples", len(samples))
    audio_dir = None
    if config.enable_audio:
        output_root = Path(config.output_dir or "results/videomme_audio")
        audio_dir = str(output_root / "audio")

    send_fn = make_videomme_send_fn(
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
        save_json_results(results, config.output_dir, "videomme_results.json")

    return results


def _config_from_args(args: argparse.Namespace) -> VideoMMEEvalConfig:
    return VideoMMEEvalConfig(
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
        enable_audio=args.enable_audio,
        asr_device=args.asr_device,
        lang=args.lang,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_videomme_eval(config)
    print_videomme_accuracy_summary(results["summary"], config.model)
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="Video-MME Speed",
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-MME benchmark for video understanding models."
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
            "HuggingFace dataset repo for Video-MME. "
            f"Defaults to {_VIDEOMME_DEFAULT_REPO}."
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
