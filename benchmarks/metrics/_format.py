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
