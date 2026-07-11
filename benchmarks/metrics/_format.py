# SPDX-License-Identifier: Apache-2.0
"""Shared metric presentation formatting."""

from __future__ import annotations

from typing import Any

ACCURACY_LABEL_WIDTH = 28
ACCURACY_LINE_WIDTH = 50
SPEED_LABEL_WIDTH = 30
SPEED_LINE_WIDTH = 60
BREAKDOWN_KEY_WIDTH = 14
MMSU_CATEGORY_WIDTH = 18


def print_speed_metric_line(lw: int, label: str, metrics: dict, key: str) -> None:
    value = metrics.get(key)
    if value is None:
        return
    print(f"  {label:<{lw}} {value}")


def format_benchmark_dataset_label(
    *,
    dataset: str | None = None,
    repo_id: str | None = None,
    split: str | None = None,
) -> str | None:
    """Build a single dataset label for benchmark/CI summary headers."""
    if dataset and repo_id:
        detail = repo_id if not split else f"{repo_id}, {split}"
        return f"{dataset} ({detail})"
    if repo_id:
        return repo_id if not split else f"{repo_id} ({split})"
    return dataset


def print_benchmark_dataset_line(label_width: int, dataset: str | None) -> None:
    if dataset:
        print(f"  {'Dataset:':<{label_width}} {dataset}")


def print_accuracy_breakdown(
    title: str,
    breakdown: dict[str, dict[str, Any]],
    *,
    key_width: int = BREAKDOWN_KEY_WIDTH,
) -> None:
    """Print a correct/total (accuracy%) table for a per-bucket breakdown."""
    if not breakdown:
        return
    print(f"  {title}:")
    for key, stat in breakdown.items():
        print(
            f"    {key:<{key_width}} {stat['correct']}/{stat['total']} "
            f"({stat['accuracy'] * 100:.1f}%)"
        )
