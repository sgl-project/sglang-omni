# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-repeat FDB v1.5 summaries into one mean and deviation table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

CATEGORIES = (
    "all",
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
)
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
COLUMNS = (
    ("pairs", "Pairs", "count"),
    ("timed_overlap_sessions", "Timed overlap sessions", "count"),
    ("stop_mean_s", "Stop latency (s)", "seconds"),
    ("response_mean_s", "Response latency (s)", "seconds"),
    *((label, label, "percent") for label in C_LABELS),
    ("judged_pairs", "Judged pairs", "count"),
)
SEMANTIC_AXES = ("interaction_handling", "relevance", "grounding", "joint")
SEMANTIC_COLUMNS = tuple(
    (f"{axis}_{measure}", f"{axis} {measure}", "percent")
    for axis in SEMANTIC_AXES
    for measure in ("quality", "coverage")
)
JudgeName = Literal["qwen", "gpt"]
RepeatMetrics = dict[str, dict[str, float | None]]


def read_json(path: Path) -> dict[str, JsonValue]:
    return json.loads(path.read_text(encoding="utf-8"))


def repeat_metrics(repeat_dir: Path, engine: str, judge: JudgeName) -> RepeatMetrics:
    timing_groups = read_json(repeat_dir / "scores" / "summary.json")["engines"][engine]
    if judge == "qwen":
        behavior_groups = read_json(repeat_dir / "judge-qwen" / "summary.json")[
            "engines"
        ][engine]
    else:
        behavior_groups = {
            category: group["behavior"] for category, group in timing_groups.items()
        }
    metrics = {}
    for category in CATEGORIES:
        if category not in timing_groups:
            continue
        else:
            pass
        timing = timing_groups[category]["timing_official_overlap"]
        intervals = timing["official_all_intervals"]
        behavior = behavior_groups[category]
        metrics[category] = {
            "pairs": timing["population"],
            "timed_overlap_sessions": timing["status"].get("ok", 0),
            "stop_mean_s": intervals["stop"]["mean_s"],
            "response_mean_s": intervals["response"]["mean_s"],
            "judged_pairs": behavior["valid_n"],
            **{
                label: behavior["valid_label_proportions"][label]["proportion"]
                for label in C_LABELS
            },
        }
    return metrics


def semantic_metrics(repeat_dir: Path) -> RepeatMetrics:
    summary_path = repeat_dir / "semantic-qwen" / "summary.json"
    if not summary_path.is_file():
        return {}
    else:
        pass
    metrics = {}
    for category, axes in read_json(summary_path)["categories"].items():
        metrics[category] = {}
        for axis in SEMANTIC_AXES:
            for measure in ("quality", "coverage"):
                percent = axes[axis][f"{measure}_percent"]
                metrics[category][f"{axis}_{measure}"] = (
                    None if percent is None else percent / 100
                )
    return metrics


def render_table(
    per_repeat: list[RepeatMetrics], columns: tuple[tuple[str, str, str], ...]
) -> list[str]:
    lines = [
        "| Category | " + " | ".join(title for _, title, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for category in CATEGORIES:
        rows = [metrics[category] for metrics in per_repeat if category in metrics]
        if not rows:
            continue
        else:
            pass
        cells = [
            format_cell([row[key] for row in rows], unit) for key, _, unit in columns
        ]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return lines


def format_cell(values: list[float | None], unit: str) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "n/a"
    elif unit == "count":
        low, high = int(min(present)), int(max(present))
        return str(low) if low == high else f"{low}-{high}"
    else:
        pass
    scale = 100.0 if unit == "percent" else 1.0
    digits = 1 if unit == "percent" else 3
    suffix = "%" if unit == "percent" else ""
    mean = statistics.fmean(present) * scale
    if len(present) == 1:
        return f"{mean:.{digits}f}{suffix}"
    else:
        deviation = statistics.stdev(present) * scale
        return f"{mean:.{digits}f} ± {deviation:.{digits}f}{suffix}"


def render(run_root: Path, engine: str, judge: JudgeName) -> str:
    repeat_dirs = sorted(
        path
        for path in run_root.glob("repeat-*")
        if (path / "scores" / "summary.json").is_file()
    )
    if not repeat_dirs:
        raise SystemExit(f"no repeat-*/scores/summary.json under {run_root}")
    else:
        pass
    per_repeat = [repeat_metrics(path, engine, judge) for path in repeat_dirs]
    lines = [
        f"# FDB v1.5 results: {run_root.name}",
        "",
        f"Repeats: {', '.join(path.name for path in repeat_dirs)}. "
        f"Judge: {judge}. Cells are mean ± sample standard deviation across repeats.",
        "Latencies are pooled whole-file overlap intervals; "
        "label shares are over valid judge labels.",
        "",
        *render_table(per_repeat, COLUMNS),
    ]
    per_repeat_semantic = [semantic_metrics(path) for path in repeat_dirs]
    if any(per_repeat_semantic):
        lines += [
            "",
            "## Semantic A/F/U judge",
            "",
            "quality = A/(A+F); coverage = (A+F)/N. "
            "joint fails if any axis fails and accepts only if all three accept.",
            "",
            *render_table(per_repeat_semantic, SEMANTIC_COLUMNS),
        ]
    else:
        pass
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--judge", choices=("qwen", "gpt"), required=True)
    args = parser.parse_args()
    results = render(args.run_root, args.engine, args.judge)
    (args.run_root / "RESULTS.md").write_text(results, encoding="utf-8")
    print(results)
    print(f"Wrote {args.run_root / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
else:
    pass
