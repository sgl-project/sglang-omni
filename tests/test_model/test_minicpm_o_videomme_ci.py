# SPDX-License-Identifier: Apache-2.0
"""Video-MME accuracy CI for MiniCPM-o (Video -> Text).

Usage:
    pytest tests/test_model/test_minicpm_o_videomme_ci.py -s -x

MiniCPM-o 4.5 defaults to an 8k thinker context. Sixteen frames at 1 FPS
keeps the 50-sample CI set within that context while exercising the video
path. The benchmark result is also printed so CI runs provide speed data.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.test_model.conftest import MINICPMO_MODEL_NAME
from tests.utils import MetricCheckCollector, ServerHandle

MAX_SAMPLES = 50
CONCURRENCY = 1
VIDEO_FPS = 1
VIDEO_MAX_FRAMES = 16
VIDEO_MAX_PIXELS = 401408

# Calibrated on MiniCPM-o 4.5, H100, 50 samples, 16 frames: 30/50 correct.
# Keep four samples of slack for ordinary model/runtime drift.
MINICPMO_VIDEOMME_MIN_ACCURACY = 0.50


@pytest.mark.benchmark
def test_minicpm_o_videomme_accuracy(
    minicpm_o_text_server: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run the 50-sample Video-MME CI set and validate serving correctness."""
    config = VideoEvalConfig(
        model=MINICPMO_MODEL_NAME,
        port=minicpm_o_text_server.port,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=VIDEO_FPS,
        video_max_frames=VIDEO_MAX_FRAMES,
        video_max_pixels=VIDEO_MAX_PIXELS,
        disable_tqdm=False,
        timeout_s=500,
    )
    results = asyncio.run(
        run_video_eval(
            config,
            task_label="MiniCPM-o Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default="results/minicpm_o_videomme_audio",
        )
    )

    summary = results["summary"]
    dataset_label = format_benchmark_dataset_label(
        dataset="videomme-ci-50",
        repo_id=config.repo_id,
    )
    print_videomme_accuracy_summary(
        summary,
        config.model,
        dataset=dataset_label,
    )
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="MiniCPM-o Video-MME Speed",
        dataset=dataset_label,
    )

    checks = MetricCheckCollector("MiniCPM-o Video-MME accuracy")
    checks.check(
        summary.get("total_samples") == MAX_SAMPLES,
        f"Expected {MAX_SAMPLES} samples, got {summary.get('total_samples')}",
    )
    checks.check(
        summary.get("failed", 0) == 0,
        f"Expected 0 failed samples, got {summary.get('failed')}",
    )
    accuracy = summary.get("accuracy")
    checks.check(
        accuracy is not None and accuracy >= MINICPMO_VIDEOMME_MIN_ACCURACY,
        f"Video-MME accuracy {accuracy!r} < "
        f"threshold {MINICPMO_VIDEOMME_MIN_ACCURACY}",
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
