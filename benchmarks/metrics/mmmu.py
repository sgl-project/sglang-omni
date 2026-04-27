# SPDX-License-Identifier: Apache-2.0
"""MMMU accuracy metrics and presentation."""

from __future__ import annotations

SUMMARY_LABEL_WIDTH = 28
SUMMARY_LINE_WIDTH = 50


def compute_mmmu_metrics(
    per_sample: list[dict],
    mc_fallback: int,
) -> tuple[dict, list[dict]]:
    """Aggregate already-decided MMMU per-sample result records.

    Returns ``(summary_dict, per_sample_list)``.
    """
    correct = sum(1 for record in per_sample if record["is_correct"])
    failed = sum(1 for record in per_sample if not record["is_success"])
    total = len(per_sample)
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "failed": failed,
        "mc_fallback": mc_fallback,
    }
    return summary, per_sample


def print_mmmu_accuracy_summary(metrics: dict, model_name: str) -> None:
    """Print formatted MMMU accuracy summary to stdout."""
    lw = SUMMARY_LABEL_WIDTH
    print(f"\n{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  MMMU Accuracy — {model_name}")
    print(f"{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  {'Total samples:':<{lw}} {metrics['total_samples']}")
    print(f"  {'Correct:':<{lw}} {metrics['correct']}")
    print(
        f"  {'Accuracy:':<{lw}} {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.1f}%)"
    )
    print(f"  {'Failed requests:':<{lw}} {metrics['failed']}")
    print(f"  {'MC parse fallback:':<{lw}} {metrics['mc_fallback']}")
    print(f"{'=' * SUMMARY_LINE_WIDTH}\n")
