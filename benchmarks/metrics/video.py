# SPDX-License-Identifier: Apache-2.0
"""Video understanding accuracy metrics and presentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from benchmarks.benchmarker.utils import print_accuracy_breakdown

SUMMARY_LABEL_WIDTH = 28
SUMMARY_LINE_WIDTH = 50


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
    per_sample: list[dict[str, Any]],
    mc_fallback: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "mc_fallback": mc_fallback,
        "per_duration": _finalize_breakdown(per_duration),
        "per_domain": _finalize_breakdown(per_domain),
        "per_task_type": _finalize_breakdown(per_task_type),
    }
    return summary, per_sample


def print_videomme_accuracy_summary(
    metrics: dict[str, Any],
    model_name: str,
    *,
    title: str = "Video-MME Accuracy",
) -> None:
    lw = SUMMARY_LABEL_WIDTH
    print(f"\n{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  {title} — {model_name}")
    print(f"{'=' * SUMMARY_LINE_WIDTH}")
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
    print(f"{'=' * SUMMARY_LINE_WIDTH}\n")
