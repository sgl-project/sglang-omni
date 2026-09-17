# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Super Reasoner accuracy CI (Text/Image/Video -> Text).

Runs the understanding benchmarks that back the paste-ready accuracy table,
scored Omni-vs-ground-truth (no HF reference runner):

  * MMMU        -- mmmu-ci-50            (image reasoner)
  * MMMU strict -- same 50, MC parse fallbacks counted as incorrect
  * VideoMME    -- videomme-ci-50        (video reasoner)

The Super Reasoner is a single native SRT deployment. This test serves it at
TP=2 on a 2-GPU node via examples/configs/cosmos3_super_reasoner.yaml (the
config pins the checkpoint revision, so managed_omni_server serves that pin).

Opt in on the GPU node, matching tests/integration/cosmos3/test_super_gpu.py:

    COSMOS3_SUPER_RUN_GPU=1 pytest tests/test_model/test_cosmos3_super_reasoner_ci.py -s -x

The test asserts PROVISIONAL accuracy floors as a regression guard and prints a
2107-style markdown table assembled from its own results.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.utils import MetricCheckCollector

pytestmark = pytest.mark.skipif(
    os.environ.get("COSMOS3_SUPER_RUN_GPU") != "1",
    reason="Set COSMOS3_SUPER_RUN_GPU=1 on the GPU node to opt in",
)

REASONER_CONFIG = "examples/configs/cosmos3_super_reasoner.yaml"
MODEL_REPO = "nvidia/Cosmos3-Super"
# Omni serves the model under the pipeline config's ``name:`` field, not the
# checkpoint id. Keep this in sync with cosmos3_super_reasoner.yaml.
SERVED_MODEL_NAME = "cosmos3-super-reasoner"

CONCURRENCY = 8
MAX_SAMPLES = 50

# Accuracy floors calibrated from a measured 2-GPU (TP=2) run of this harness
# against the pinned Super checkpoint (Omni vs. ground truth):
#   MMMU 30-32/50 (60-64%), strict 60-64%, VideoMME 28/50 (56%).
# Floors sit ~10-14 pts below the measured values to absorb concurrency-driven
# run-to-run variance (temperature=0 is not fully deterministic at concurrency 8).
MMMU_MIN_ACCURACY = 0.48
MMMU_STRICT_MIN_ACCURACY = 0.48
VIDEOMME_MIN_ACCURACY = 0.44


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def cosmos3_super_reasoner_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> "int":
    """Serve the Super Reasoner (TP=2) and yield its port."""
    port = _free_port()
    log_file = tmp_path_factory.mktemp("cosmos3_super_reasoner") / "server.log"
    with managed_omni_server(
        model_path=MODEL_REPO,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        server_config=REASONER_CONFIG,
    ):
        yield port


def _fraction(correct: int, total: int) -> str:
    pct = (correct / total * 100) if total else 0.0
    return f"{correct}/{total} ({pct:.0f}%)"


def _strict_mmmu(per_sample: list[dict]) -> tuple[int, int]:
    """Correct/total when MC parse fallbacks count as incorrect."""
    total = len(per_sample)
    correct = sum(
        1
        for record in per_sample
        if record["is_correct"] and not record["is_mc_fallback"]
    )
    return correct, total


def _print_accuracy_table(rows: list[tuple[str, str, str]]) -> None:
    """Print a paste-ready #2107-style markdown table from the run's results."""
    print("\n### Cosmos3-Super Reasoner accuracy (Omni vs. ground truth)\n")
    print("| Evaluation | Sample source | Omni |")
    print("| --- | --- | ---: |")
    for evaluation, source, omni in rows:
        print(f"| {evaluation} | {source} | {omni} |")
    print()


@pytest.mark.benchmark
def test_reasoner_accuracy(
    cosmos3_super_reasoner_server: int,
    tmp_path: Path,
) -> None:
    """Run MMMU + VideoMME and report Omni-vs-ground-truth accuracy."""
    port = cosmos3_super_reasoner_server
    checks = MetricCheckCollector("Cosmos3-Super Reasoner accuracy")
    table: list[tuple[str, str, str]] = []

    # --- MMMU (image) + strict variant ---
    mmmu = asyncio.run(
        run_mmmu_eval(
            MMMUEvalConfig(
                model=SERVED_MODEL_NAME,
                port=port,
                max_concurrency=CONCURRENCY,
                output_dir=str(tmp_path / "mmmu"),
                repo_id=DATASETS["mmmu-ci-50"],
                warmup=2,
            )
        )
    )
    mmmu_summary = mmmu["summary"]
    print_mmmu_accuracy_summary(
        mmmu_summary,
        SERVED_MODEL_NAME,
        dataset=format_benchmark_dataset_label(
            dataset="mmmu-ci-50", repo_id=DATASETS["mmmu-ci-50"]
        ),
    )
    mmmu_total = mmmu_summary.get("total_samples", 0)
    strict_correct, strict_total = _strict_mmmu(mmmu["per_sample"])
    strict_acc = strict_correct / strict_total if strict_total else 0.0
    table.append(("MMMU", "50 samples from CI", _fraction(mmmu_summary["correct"], mmmu_total)))
    table.append(
        (
            "MMMU, strict scoring",
            "Same 50 CI examples; random-fallback answers count as incorrect",
            _fraction(strict_correct, strict_total),
        )
    )

    # --- VideoMME (video) ---
    videomme = asyncio.run(
        run_video_eval(
            VideoEvalConfig(
                model=SERVED_MODEL_NAME,
                port=port,
                max_samples=MAX_SAMPLES,
                max_concurrency=CONCURRENCY,
                output_dir=str(tmp_path / "videomme"),
                repo_id=DATASETS["videomme-ci-50"],
                # The Cosmos3 reasoner rejects top-level video_* processing
                # options ("Use native chat content for media and its processing
                # options", reasoner.py); let native preprocessing use defaults.
                timeout_s=500,
            ),
            task_label="Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default=str(tmp_path / "videomme_audio"),
        )
    )
    videomme_summary = videomme["summary"]
    print_videomme_accuracy_summary(
        videomme_summary,
        SERVED_MODEL_NAME,
        dataset=format_benchmark_dataset_label(
            dataset="videomme-ci-50", repo_id=DATASETS["videomme-ci-50"]
        ),
    )
    videomme_total = videomme_summary.get("total_samples", 0)
    table.append(
        ("VideoMME", "50 samples from CI", _fraction(videomme_summary["correct"], videomme_total))
    )

    _print_accuracy_table(table)

    # --- assertions (failures + provisional floors) ---
    for name, summary in (
        ("MMMU", mmmu_summary),
        ("VideoMME", videomme_summary),
    ):
        failed = summary.get("failed", 0)
        total = summary.get("total_samples", 0)
        checks.check(
            failed == 0,
            f"{name} had {failed}/{total} failed requests (timeouts or empty "
            f"responses); any failure fails the test",
        )

    for name, accuracy, floor in (
        ("MMMU", mmmu_summary.get("accuracy"), MMMU_MIN_ACCURACY),
        ("MMMU strict", strict_acc, MMMU_STRICT_MIN_ACCURACY),
        ("VideoMME", videomme_summary.get("accuracy"), VIDEOMME_MIN_ACCURACY),
    ):
        if accuracy is None:
            checks.fail(f"{name} accuracy missing from summary")
        else:
            checks.check(
                accuracy >= floor,
                f"{name} accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
                f"provisional floor {floor} ({floor * 100:.0f}%)",
            )

    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
