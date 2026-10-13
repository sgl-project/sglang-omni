# SPDX-License-Identifier: Apache-2.0
"""Aggregate the per-repeat v1.5 and v1.0 summaries of a run into mean and
deviation tables, one section per dataset the run selected."""

from __future__ import annotations

import statistics
from pathlib import Path

from benchmarks.duplex.fdb.common import (
    ENGINE_LABEL,
    V10_DIR,
    JudgeName,
    Settings,
    log,
    read_json,
)
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS

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
# note (luojiaxuan): the [Result] labels each pinned v1.0 evaluator prints, by task.
V10_TASK_RESULTS = {
    "pause_handling": {"takeover": "Average take turn"},
    "smooth_turn_taking": {
        "takeover": "Average take turn",
        "latency_s": "Average latency",
    },
    "user_interruption": {
        "takeover": "Average take turn",
        "latency_s": "Average latency",
        "relevance": "Average rating",
    },
    "backchannel": {
        "takeover": "TOR mean",
        "frequency": "Frequency mean",
        "jsd": "JSD mean",
    },
}
V10_COLUMNS = (
    ("selected", "Selected", "count"),
    ("eligible", "Eligible", "count"),
    ("evaluated", "Evaluated", "count"),
    ("takeover", "Takeover rate", "percent"),
    ("latency_s", "Latency (s)", "seconds"),
    ("relevance", "Relevance (0-5)", "rating"),
    ("judge_retries", "Judge retries", "count"),
    ("frequency", "Backchannel frequency", "rate"),
    ("jsd", "Backchannel JSD", "divergence"),
)
RepeatMetrics = dict[str, dict[str, float | None]]


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


def v10_metrics(repeat_dir: Path) -> RepeatMetrics:
    """One row per selected v1.0 subset from the export manifest; summary.json
    overlays the evaluated count and metrics, so a subset whose sessions all
    failed, or that is not evaluated, keeps its row with n/a cells."""
    tree = repeat_dir / V10_DIR / "reference"
    if not (tree / "manifest.json").is_file():
        return {}
    else:
        pass
    summary_path = tree / "summary.json"
    evaluated = read_json(summary_path)["subsets"] if summary_path.is_file() else {}
    metrics = {}
    for subset, counts in read_json(tree / "manifest.json")["counts"].items():
        if counts["selected"] == 0:
            continue
        else:
            pass
        row = {key: None for key, _, _ in V10_COLUMNS}
        row["selected"] = counts["selected"]
        row["eligible"] = counts["eligible"]
        # note (luojiaxuan): with no eligible sample there is nothing to evaluate.
        row["evaluated"] = 0 if counts["eligible"] == 0 else None
        if subset in evaluated:
            entry = evaluated[subset]
            row["evaluated"] = entry["evaluated"]
            for key, label in V10_TASK_RESULTS[entry["task"]].items():
                row[key] = entry["result"][label]
            if entry["task"] == "user_interruption":
                row["judge_retries"] = entry["judge"]["retries"]
            else:
                pass
        else:
            pass
        metrics[subset] = row
    return metrics


def v10_unevaluated(repeat_dir: Path) -> list[str]:
    """The eligible v1.0 subsets summary.json lacks, as pending or failed. The
    evaluator writes logs/evaluate-<subset>.log for every subset it runs, passing
    or failing, so a subset with that log and no result failed, and one without
    it has not been reached."""
    tree = repeat_dir / V10_DIR / "reference"
    if not (tree / "manifest.json").is_file():
        return []
    else:
        pass
    summary_path = tree / "summary.json"
    evaluated = read_json(summary_path)["subsets"] if summary_path.is_file() else {}
    notes = []
    for subset, counts in read_json(tree / "manifest.json")["counts"].items():
        if counts["eligible"] == 0 or subset in evaluated:
            continue
        elif (tree / "logs" / f"evaluate-{subset}.log").is_file():
            notes.append(
                f"- {repeat_dir.name} `{subset}`: evaluation failed; see "
                f"`{V10_DIR}/reference/logs/evaluate-{subset}.log`"
            )
        else:
            notes.append(
                f"- {repeat_dir.name} `{subset}`: not evaluated yet; run `judge`"
            )
    return notes


def render_table(
    per_repeat: list[RepeatMetrics],
    columns: tuple[tuple[str, str, str], ...],
    categories: tuple[str, ...] = CATEGORIES,
) -> list[str]:
    lines = [
        "| Category | " + " | ".join(title for _, title, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for category in categories:
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


def coverage_note(
    label: str, repeat_dirs: list[Path], per_repeat: list[RepeatMetrics]
) -> str:
    """Which repeats a table draws on, when some repeats lack that dataset."""
    carrying = [path.name for path, metrics in zip(repeat_dirs, per_repeat) if metrics]
    if len(carrying) == len(repeat_dirs):
        return ""
    else:
        return (
            f" {label} cells come from {', '.join(carrying)} only; the other "
            f"repeats have no {label} results."
        )


def render(run_root: Path, engine: str, judge: JudgeName) -> str:
    repeat_dirs = sorted(
        path
        for path in run_root.glob("repeat-*")
        if (path / "scores" / "summary.json").is_file()
        or (path / V10_DIR / "reference" / "manifest.json").is_file()
    )
    if not repeat_dirs:
        raise SystemExit(f"no finished repeat-* under {run_root}")
    else:
        pass
    lines = [
        f"# FDB results: {run_root.name}",
        "",
        f"Repeats: {', '.join(path.name for path in repeat_dirs)}. "
        f"Judge: {judge}. Cells are mean ± sample standard deviation across repeats.",
    ]
    per_repeat = [
        (
            repeat_metrics(path, engine, judge)
            if (path / "scores" / "summary.json").is_file()
            else {}
        )
        for path in repeat_dirs
    ]
    if any(per_repeat):
        lines += [
            "",
            "## FDB v1.5 results",
            "",
            "Latencies are pooled whole-file overlap intervals; "
            "label shares are over valid judge labels."
            + coverage_note("v1.5", repeat_dirs, per_repeat),
            "",
            *render_table(per_repeat, COLUMNS),
        ]
    else:
        pass
    per_repeat_semantic = [semantic_metrics(path) for path in repeat_dirs]
    if any(per_repeat_semantic):
        lines += [
            "",
            "## v1.5 semantic A/F/U judge",
            "",
            "quality = A/(A+F); coverage = (A+F)/N. "
            + "joint fails if any axis fails and accepts only if all three accept.",
            "",
            *render_table(per_repeat_semantic, SEMANTIC_COLUMNS),
        ]
    else:
        pass
    per_repeat_v10 = [v10_metrics(path) for path in repeat_dirs]
    if any(per_repeat_v10):
        relevance_judge = (
            "the local Qwen judge, non-official" if judge == "qwen" else "GPT"
        )
        lines += [
            "",
            "## FDB v1.0 results",
            "",
            "Pinned reference evaluation per subset; subsets are never pooled. "
            "Selected counts the recorded samples, Eligible the exports the "
            "evaluator can score, Evaluated the ones it scored; a subset that is "
            "not evaluated shows n/a and is listed under the table. Takeover rate "
            "should be low on pause handling and backchannel, and high on turn "
            "taking and interruption. Relevance is a 0-5 rating of the "
            f"interruptions the model took over, by {relevance_judge}; a nonzero "
            "Judge retries count inflates that subset's takeover rate."
            + coverage_note("v1.0", repeat_dirs, per_repeat_v10),
            "",
            *render_table(per_repeat_v10, V10_COLUMNS, V10_SUBSETS),
        ]
        unevaluated = [note for path in repeat_dirs for note in v10_unevaluated(path)]
        if unevaluated:
            lines += ["", "Not evaluated:", "", *unevaluated]
        else:
            pass
    else:
        pass
    return "\n".join(lines) + "\n"


def aggregate(settings: Settings) -> None:
    results = render(settings.run_root, ENGINE_LABEL, settings.judge)
    results_path = settings.run_root / "RESULTS.md"
    results_path.write_text(results, encoding="utf-8")
    log(results)
    log(f"Wrote {results_path}")
