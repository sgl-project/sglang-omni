# SPDX-License-Identifier: Apache-2.0
"""Daily-Omni audio-visual reasoning benchmark for SGLang Omni models.

Daily-Omni contains 1,197 four-choice questions over 684 daily-life videos.
The official media archive stores video and audio as separate files, so the
default ``all`` input mode sends both paths to ``/v1/chat/completions``.

Usage:
    python -m benchmarks.dataset.prepare --dataset dailyomni

    python -m benchmarks.eval.benchmark_omni_dailyomni \
        --model your-model-name --port 8000 \
        --max-concurrency 4 --video-fps 2 \
        --video-max-frames 128 --video-max-pixels 401408
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.dailyomni import (
    DEFAULT_REPO_ID,
    SUPPORTED_INPUT_MODES,
    load_dailyomni_samples,
)
from benchmarks.dataset.videomme import VideoAMMESample
from benchmarks.eval.benchmark_omni_videomme import (
    VideoEvalConfig,
    add_video_eval_args,
    run_video_eval,
    video_eval_config_from_args,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from benchmarks.metrics.wer import print_wer_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@dataclass
class DailyOmniEvalConfig(VideoEvalConfig):
    split: str = "train"
    repo_id: str | None = DEFAULT_REPO_ID
    media_root: str | None = None
    input_mode: str = "all"


async def run_dailyomni_eval(
    config: DailyOmniEvalConfig,
    *,
    samples: list[VideoAMMESample] | None = None,
    compute_wer: bool = True,
) -> dict:
    """Run Daily-Omni and return accuracy, speed, config, and item records."""
    if config.input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Unsupported Daily-Omni input mode {config.input_mode!r}; "
            f"choose one of {SUPPORTED_INPUT_MODES}"
        )

    load_samples = partial(
        load_dailyomni_samples,
        media_root=config.media_root,
        input_mode=config.input_mode,
    )
    return await run_video_eval(
        config,
        samples=samples,
        load_samples=load_samples,
        task_label="Daily-Omni",
        output_filename="dailyomni_results.json",
        audio_output_dir_default="results/dailyomni_audio",
        enable_video_input=config.input_mode in {"all", "visual"},
        enable_audio_input=config.input_mode in {"all", "audio"},
        compute_wer=compute_wer,
        extra_config={
            "media_root": config.media_root,
            "input_mode": config.input_mode,
        },
    )


def _config_from_args(args: argparse.Namespace) -> DailyOmniEvalConfig:
    base_config = video_eval_config_from_args(args)
    return DailyOmniEvalConfig(
        **vars(base_config),
        media_root=args.media_root,
        input_mode=args.input_mode,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_dailyomni_eval(config)
    print_videomme_accuracy_summary(
        results["summary"],
        config.model,
        title="Daily-Omni Accuracy",
        dataset=f"{config.repo_id} ({config.input_mode})",
    )
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="Daily-Omni Speed",
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily-Omni benchmark for audio-visual reasoning models."
    )
    add_video_eval_args(
        parser,
        repo_help=(
            "Hugging Face dataset repo or local dataset directory for Daily-Omni. "
            f"Defaults to {DEFAULT_REPO_ID}."
        ),
    )
    parser.set_defaults(
        repo_id=DEFAULT_REPO_ID,
        split="train",
        output_dir="results/dailyomni",
    )
    parser.add_argument(
        "--media-root",
        type=str,
        default=None,
        help=(
            "Existing extracted Daily-Omni dataset directory or Videos directory. "
            "By default Videos.tar is downloaded and extracted into the cache."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=SUPPORTED_INPUT_MODES,
        default="all",
        help="Evaluate with video+audio (all), video only, or audio only.",
    )
    args = parser.parse_args()

    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
