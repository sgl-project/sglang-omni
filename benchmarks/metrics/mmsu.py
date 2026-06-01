# SPDX-License-Identifier: Apache-2.0
"""MMSU accuracy metrics and presentation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from benchmarks.metrics._format import MMSU_CATEGORY_WIDTH, SPEED_LINE_WIDTH

if TYPE_CHECKING:
    from benchmarks.tasks.audio_understanding import MmsuResult


def _build_group_metrics(
    results: list["MmsuResult"],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "parseable": 0}
    )
    for result in results:
        value = getattr(result, key)
        grouped[value]["total"] += 1
        if result.is_parseable:
            grouped[value]["parseable"] += 1
        if result.is_correct:
            grouped[value]["correct"] += 1

    metrics: dict[str, dict[str, Any]] = {}
    for name, counts in sorted(grouped.items()):
        metrics[name] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "parseable": counts["parseable"],
            "accuracy": round(counts["correct"] / counts["total"], 4),
        }
    return metrics


def compute_mmsu_metrics(results: list["MmsuResult"]) -> dict[str, Any]:
    total = len(results)
    parseable = sum(1 for result in results if result.is_parseable)
    successful = sum(1 for result in results if result.is_success)
    correct = sum(1 for result in results if result.is_correct)

    return {
        "total_samples": total,
        "parseable_samples": parseable,
        "unparseable_samples": total - parseable,
        "successful_samples": successful,
        "failed_samples": total - successful,
        "correct": correct,
        "incorrect": total - correct,
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "per_task": _build_group_metrics(results, "task_name"),
        "per_category": _build_group_metrics(results, "category"),
        "per_sub_category": _build_group_metrics(results, "sub_category"),
        "per_sub_sub_category": _build_group_metrics(results, "sub_sub_category"),
        "per_linguistics_sub_discipline": _build_group_metrics(
            results,
            "linguistics_sub_discipline",
        ),
    }


def print_mmsu_summary(
    metrics: dict[str, Any],
    model_name: str,
    *,
    speed_metrics: dict[str, Any] | None = None,
) -> None:
    print("\n" + "=" * SPEED_LINE_WIDTH)
    print(f"  MMSU Results - {model_name}")
    print("=" * SPEED_LINE_WIDTH)
    print(f"  Total samples:    {metrics['total_samples']}")
    print(
        f"  Successful:       {metrics.get('successful_samples', metrics['total_samples'])}"
    )
    print(f"  Parseable:        {metrics['parseable_samples']}")
    print(f"  Correct:          {metrics['correct']}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print("-" * SPEED_LINE_WIDTH)
    print(f"  {'Category':<{MMSU_CATEGORY_WIDTH}} {'Acc':>8} {'N':>6}")
    print("-" * SPEED_LINE_WIDTH)
    for name, info in metrics["per_category"].items():
        print(
            f"  {name:<{MMSU_CATEGORY_WIDTH}} "
            f"{info['accuracy']:>8.2%} {info['total']:>6}"
        )
    if speed_metrics:
        print("-" * SPEED_LINE_WIDTH)
        print(f"  Latency mean:     {speed_metrics.get('latency_mean_s', 0):.3f}s")
        print(f"  Latency p95:      {speed_metrics.get('latency_p95_s', 0):.3f}s")
        if speed_metrics.get("audio_duration_mean_s", 0) > 0:
            print(
                f"  Audio mean:       {speed_metrics.get('audio_duration_mean_s', 0):.3f}s"
            )
        if speed_metrics.get("rtf_mean") is not None:
            print(f"  RTF mean:         {speed_metrics.get('rtf_mean', 0):.4f}")
        print(f"  Throughput:       {speed_metrics.get('throughput_qps', 0):.2f} req/s")
        print(f"  Tok/s agg:        {speed_metrics.get('tok_per_s_agg', 0):.2f}")
        audio_returned = speed_metrics.get("audio_returned")
        audio_expected = speed_metrics.get("audio_expected")
        if audio_expected:
            print(f"  Audio returned:   {audio_returned}/{audio_expected}")
    print("=" * SPEED_LINE_WIDTH)
