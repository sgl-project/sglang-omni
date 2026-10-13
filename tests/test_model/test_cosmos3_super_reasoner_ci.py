# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Super Reasoner accuracy CI (Text/Image/Video -> Text).

Runs the understanding benchmarks that back the paste-ready accuracy table,
scored Omni-vs-ground-truth (no HF reference runner):

  * MMMU        -- mmmu-ci-50            (image reasoner)
  * MMMU strict -- same 50, MC parse fallbacks counted as incorrect
  * VideoMME    -- videomme-ci-50        (video reasoner)

The Super Reasoner uses the pinned Reasoner YAML and the allocation selected by
COSMOS3_SUPER_GPU_IDS (default 0,1). SDK and HTTP run sequentially against the
same request builders and scorers; select one with COSMOS3_SUPER_TRANSPORTS.

Opt in on the GPU node, matching tests/integration/cosmos3/test_super_gpu.py:

    COSMOS3_SUPER_RUN_GPU=1 pytest tests/test_model/test_cosmos3_super_reasoner_ci.py -s -x

The test asserts PROVISIONAL accuracy floors as a regression guard and prints a
2107-style markdown table assembled from its own results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.integration.cosmos3.super_harness import deployment, transports
from tests.integration.cosmos3.test_super_gpu import build_config
from tests.test_model.test_cosmos3_super_generator_ci import gpu_devices, resolve_model
from tests.utils import MetricCheckCollector

pytestmark = pytest.mark.skipif(
    os.environ.get("COSMOS3_SUPER_RUN_GPU") != "1",
    reason="Set COSMOS3_SUPER_RUN_GPU=1 on the GPU node to opt in",
)

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


def fraction(correct: int, total: int) -> str:
    pct = (correct / total * 100) if total else 0.0
    return f"{correct}/{total} ({pct:.0f}%)"


def strict_mmmu(per_sample: list[dict]) -> tuple[int, int]:
    """Correct/total when MC parse fallbacks count as incorrect."""
    total = len(per_sample)
    correct = sum(
        1
        for record in per_sample
        if record["is_correct"] and not record["is_mc_fallback"]
    )
    return correct, total


def print_accuracy_table(rows: list[tuple[str, str, str]]) -> None:
    """Print a paste-ready #2107-style markdown table from the run's results."""
    print("\n### Cosmos3-Super Reasoner accuracy (Omni vs. ground truth)\n")
    print("| Evaluation | Sample source | Omni |")
    print("| --- | --- | ---: |")
    for evaluation, source, omni in rows:
        print(f"| {evaluation} | {source} | {omni} |")
    print()


@pytest.mark.parametrize("transport", transports())
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_reasoner_accuracy(tmp_path: Path, transport: str) -> None:
    config = build_config(
        "reasoner", str(resolve_model("reasoner")), gpu_devices(), tmp_path
    )
    async with deployment(config, transport, tmp_path / transport) as client:
        await reasoner_accuracy(client, tmp_path / transport)


async def reasoner_accuracy(client, tmp_path):
    checks = MetricCheckCollector("Cosmos3-Super Reasoner accuracy")
    table: list[tuple[str, str, str]] = []

    # --- MMMU (image) + strict variant ---
    mmmu = await run_mmmu_eval(
        MMMUEvalConfig(
            model=SERVED_MODEL_NAME,
            base_url=f"{client.transport}://cosmos3-super",
            max_concurrency=CONCURRENCY,
            output_dir=str(tmp_path / "mmmu"),
            repo_id=DATASETS["mmmu-ci-50"],
            warmup=2,
        ),
        completion=client.completion,
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
    strict_correct, strict_total = strict_mmmu(mmmu["per_sample"])
    strict_acc = strict_correct / strict_total if strict_total else 0.0
    table.append(
        ("MMMU", "50 samples from CI", fraction(mmmu_summary["correct"], mmmu_total))
    )
    table.append(
        (
            "MMMU, strict scoring",
            "Same 50 CI examples; random-fallback answers count as incorrect",
            fraction(strict_correct, strict_total),
        )
    )

    # --- VideoMME (video) ---
    videomme = await run_video_eval(
        VideoEvalConfig(
            model=SERVED_MODEL_NAME,
            base_url=f"{client.transport}://cosmos3-super",
            max_samples=MAX_SAMPLES,
            max_concurrency=CONCURRENCY,
            output_dir=str(tmp_path / "videomme"),
            repo_id=DATASETS["videomme-ci-50"],
            # The Cosmos3 reasoner rejects top-level video_* processing
            # options ("Use native chat content for media and its processing
            # options", native_chat_scheduler.py); let native preprocessing use defaults.
            timeout_s=500,
        ),
        task_label="Video-MME",
        output_filename="videomme_results.json",
        audio_output_dir_default=str(tmp_path / "videomme_audio"),
        completion=client.completion,
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
        (
            "VideoMME",
            "50 samples from CI",
            fraction(videomme_summary["correct"], videomme_total),
        )
    )

    print_accuracy_table(table)

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
