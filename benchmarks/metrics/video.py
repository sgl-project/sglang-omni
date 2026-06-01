# SPDX-License-Identifier: Apache-2.0
"""Video understanding accuracy metrics and presentation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from benchmarks.metrics._format import (
    ACCURACY_LABEL_WIDTH,
    ACCURACY_LINE_WIDTH,
    print_accuracy_breakdown,
)

if TYPE_CHECKING:
    from benchmarks.tasks.video_understanding import VideoMMERecord


def _finalize_breakdown(
    buckets: dict[str, dict[str, int]]
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "total": value["total"],
            "correct": value["correct"],
            "accuracy": (
                round(value["correct"] / value["total"], 4)
                if value["total"] > 0
                else 0.0
            ),
        }
        for key, value in sorted(buckets.items())
    }


def compute_videomme_metrics(
    per_sample: list["VideoMMERecord"],
) -> dict[str, Any]:
    correct = 0
    failed = 0
    per_duration: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    per_domain: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    per_task_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )

    for record in per_sample:
        per_duration[record["duration"]]["total"] += 1
        per_domain[record["domain"]]["total"] += 1
        per_task_type[record["task_type"]]["total"] += 1

        if not record["is_success"]:
            failed += 1
            continue

        if record["is_correct"]:
            correct += 1
            per_duration[record["duration"]]["correct"] += 1
            per_domain[record["domain"]]["correct"] += 1
            per_task_type[record["task_type"]]["correct"] += 1

    total = len(per_sample)
    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "failed": failed,
        "mc_fallback": sum(record["is_mc_fallback"] for record in per_sample),
        "per_duration": _finalize_breakdown(per_duration),
        "per_domain": _finalize_breakdown(per_domain),
        "per_task_type": _finalize_breakdown(per_task_type),
    }
    return summary


def print_videomme_accuracy_summary(
    metrics: dict[str, Any],
    model_name: str,
    *,
    title: str = "Video-MME Accuracy",
) -> None:
    lw = ACCURACY_LABEL_WIDTH
    print(f"\n{'=' * ACCURACY_LINE_WIDTH}")
    print(f"  {title} — {model_name}")
    print(f"{'=' * ACCURACY_LINE_WIDTH}")
    print(f"  {'Total samples:':<{lw}} {metrics['total_samples']}")
    print(f"  {'Correct:':<{lw}} {metrics['correct']}")
    print(
        f"  {'Accuracy:':<{lw}} {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.1f}%)"
    )
    print(f"  {'Failed requests:':<{lw}} {metrics['failed']}")
    print(f"  {'MC parse fallback:':<{lw}} {metrics['mc_fallback']}")
    print_accuracy_breakdown("By duration", metrics.get("per_duration", {}))
    print_accuracy_breakdown("By domain", metrics.get("per_domain", {}))
    print_accuracy_breakdown("By task type", metrics.get("per_task_type", {}))
    print(f"{'=' * ACCURACY_LINE_WIDTH}\n")
