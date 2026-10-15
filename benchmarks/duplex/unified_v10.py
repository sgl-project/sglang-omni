# SPDX-License-Identifier: Apache-2.0
"""System-agnostic Full-Duplex-Bench v1.0 metrics from reference transcripts and annotations."""

from __future__ import annotations

import bisect
import json
import math
import random
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from typing import TypeVar

from pydantic import JsonValue

from benchmarks.duplex.reference_core import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
)
from benchmarks.duplex.reference_v10 import MANIFEST_KIND
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v10_dataset import (
    ANNOTATION_FILES,
    DECLARED_COUNTS,
    SUBSET_TASKS,
    SUBSETS,
    Task,
)

KIND = "full-duplex-bench-v1.0-unified"
DEFINITIONS_VERSION = "fdb-v1.0-unified-3"
RESPONSE_WITHIN_S = (1, 2, 5)
TIME_DECIMALS = 6
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 0
Z_95 = NormalDist().inv_cdf(0.975)
MEDIAN = Fraction(1, 2)
P90 = Fraction(9, 10)
INTERVAL_TAILS = (Fraction(1, 40), Fraction(39, 40))
PAUSE_DIAGNOSTICS = ("inside_pause_onset", "pre_pause_speech", "carry_in", "any_speech")
Span = tuple[float, float]
Reference = list[Span] | float
Resampled = TypeVar("Resampled")


def seconds(value: float) -> float:
    """Microsecond resolution, so float noise in the files cannot move a time across a boundary."""
    return round(value, TIME_DECIMALS)


def word_spans(chunks: list[dict[str, JsonValue]]) -> list[Span]:
    """(start, end) per transcribed word; a null end makes the word zero-length at its start."""
    return [
        (seconds(start), seconds(start if end is None else end))
        for start, end in (chunk["timestamp"] for chunk in chunks)
    ]


def union(spans: list[Span]) -> list[Span]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def overlap_s(first: list[Span], second: list[Span]) -> float:
    """Seconds covered by both span sets."""
    return sum(
        (
            max(0.0, min(first_end, second_end) - max(first_start, second_start))
            for first_start, first_end in union(first)
            for second_start, second_end in union(second)
        ),
        0.0,
    )


def pause_sample(
    spans: list[Span], pauses: list[Span], window_s: float
) -> dict[str, JsonValue]:
    """Word starts in [first pause start, T] intrude, each on the pause that owns it.

    Pauses are sorted (see timestamp_problems). Pause i owns [start_i, start_i+1)
    and the last pause owns [start_last, T].
    """
    pause_starts = [start for start, _ in pauses]
    first_pause_s = pause_starts[0]
    starts = [start for start, _ in spans]
    intruding = [start for start in starts if first_pause_s <= start <= window_s]
    # note (luojiaxuan): bisect_right counts the pause starts at or before a word
    # start, so equal counts mean the same owning pause.
    return {
        "pauses": len(pauses),
        "pause_s": seconds(sum(end - start for start, end in union(pauses))),
        "intruded_pauses": len(
            {bisect.bisect_right(pause_starts, start) for start in intruding}
        ),
        "intrusion_words": len(intruding),
        "intrusion_s": seconds(overlap_s(spans, pauses)),
        "pause_intrusion": bool(intruding),
        "inside_pause_onset": any(
            pause_start <= start < pause_end
            for start in starts
            for pause_start, pause_end in pauses
        ),
        "pre_pause_speech": any(start < first_pause_s for start in starts),
        "carry_in": any(start < first_pause_s < end for start, end in spans),
        "any_speech": bool(spans),
    }


def response_sample(
    starts: list[float], reference_s: float, window_s: float
) -> dict[str, JsonValue]:
    """A response is timed from the reference to the first word; no word is no_response."""
    return {
        "outcome": "response" if starts else "no_response",
        "reference_s": reference_s,
        "observable_s": seconds(window_s - reference_s),
        "latency_s": seconds(min(starts) - reference_s) if starts else None,
    }


def turn_taking_sample(
    starts: list[float], turn_end_s: float, window_s: float
) -> dict[str, JsonValue]:
    """A word starting before the turn end makes an early_start, which has no latency."""
    early_words = sum(start < turn_end_s for start in starts)
    sample = {
        "early_words": early_words,
        **response_sample(starts, turn_end_s, window_s),
    }
    if early_words:
        sample.update(outcome="early_start", latency_s=None)
    else:
        pass
    return sample


def reference_times(task: Task, annotation: JsonValue) -> Reference:
    """The pauses for pause handling, E for turn-taking, R for user interruption."""
    if task == "pause_handling":
        return [
            (seconds(start), seconds(end))
            for start, end in (entry["timestamp"] for entry in annotation)
        ]
    elif task == "turn_taking":
        return seconds(annotation[0]["timestamp"][0])
    else:
        return seconds(annotation[0]["timestamp"][1])


def timestamp_problems(
    task: Task, spans: list[Span], reference: Reference, window_s: float
) -> list[str]:
    """Why a sample's times cannot be scored on the window [0, T]; empty when they can."""

    def outside(name: str, times: list[float]) -> list[str]:
        return [
            f"{name} {time} is outside [0, {window_s}]"
            for time in times
            if not 0 <= time <= window_s
        ]

    problems = outside("word start", [start for start, _ in spans])
    if task == "pause_handling":
        problems += ["pause.json lists no pause"] if not reference else []
        problems += [
            f"pause [{start}, {end}] ends before it starts"
            for start, end in reference
            if end < start
        ]
        problems += outside(
            "pause bound", [time for span in reference for time in span]
        )
        problems += [
            f"pause [{start}, {end}] starts before the previous pause "
            f"ends at {previous_end}"
            for (_, previous_end), (start, end) in zip(reference, reference[1:])
            if start < previous_end
        ]
    elif task == "turn_taking":
        problems += outside("turn end E", [reference])
    else:
        problems += outside("interruption end R", [reference])
        # note (luojiaxuan): reference-asr cuts the audio at R and adds R back to
        # every word, so an earlier word start means a broken transcript.
        problems += [
            f"word start {start} is before the interruption end R {reference}"
            for start, _ in spans
            if start < reference
        ]
    return problems


def score_sample(
    task: Task, spans: list[Span], reference: Reference, window_s: float
) -> dict[str, JsonValue]:
    starts = [start for start, _ in spans]
    if task == "pause_handling":
        return pause_sample(spans, reference, window_s)
    elif task == "turn_taking":
        return turn_taking_sample(starts, reference, window_s)
    else:
        return response_sample(starts, reference, window_s)


def score_eligible(
    tree: Path, row: dict[str, JsonValue], task: Task
) -> dict[str, JsonValue]:
    """Per-sample quantities; stops on a timestamp that timestamp_problems rejects."""
    sample_dir = tree / row["sample_id"]
    chunks = read_json(sample_dir / "output.json")["chunks"]
    spans = word_spans(chunks)
    reference = reference_times(task, read_json(sample_dir / ANNOTATION_FILES[task]))
    window_s = seconds(row["window"]["T_s"])
    problems = timestamp_problems(task, spans, reference, window_s)
    if problems:
        raise SystemExit(f"{row['sample_id']}: {'; '.join(problems)}")
    else:
        pass
    return {
        "words": len(chunks),
        "null_end_words": sum(chunk["timestamp"][1] is None for chunk in chunks),
        **score_sample(task, spans, reference, window_s),
    }


def wilson(count: int, n: int) -> list[float] | None:
    """95% Wilson score interval, with exact endpoints 0 at count 0 and 1 at count n."""
    if n == 0:
        return None
    else:
        pass
    proportion = count / n
    denominator = 1 + Z_95**2 / n
    center = (proportion + Z_95**2 / (2 * n)) / denominator
    margin = (
        Z_95
        * math.sqrt(proportion * (1 - proportion) / n + Z_95**2 / (4 * n**2))
        / denominator
    )
    return [
        0.0 if count == 0 else center - margin,
        1.0 if count == n else center + margin,
    ]


def rate(flags: list[bool], selected: int) -> dict[str, JsonValue]:
    """Share of the eligible samples (one flag each) out of `selected` samples.

    selected_bounds is the share over all selected samples with every
    ineligible outcome taken as False, then as True.
    """
    count, n = sum(flags), len(flags)
    return {
        "count": count,
        "n": n,
        "value": count / n if n else None,
        "ci95": wilson(count, n),
        "selected_bounds": [count / selected, (count + selected - n) / selected],
    }


def quantile(sorted_values: list[float], q: Fraction) -> float:
    """Q(q) = inf{t : F(t) >= q} of the values, the ceil(q*N)-th smallest one."""
    return sorted_values[math.ceil(q * len(sorted_values)) - 1]


def median(values: list[float]) -> float:
    return quantile(sorted(values), MEDIAN)


def finite(value: float) -> float | None:
    return None if math.isinf(value) else value


def bootstrap_ci(
    values: list[Resampled], statistic: Callable[[list[Resampled]], float | None]
) -> list[float]:
    """Percentile 95% interval over resamples; a resample where the statistic is None is skipped."""
    generator = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        estimate
        for estimate in (
            statistic(generator.choices(values, k=len(values)))
            for _ in range(BOOTSTRAP_REPLICATES)
        )
        if estimate is not None
    )
    return [quantile(estimates, tail) for tail in INTERVAL_TAILS]


def pooled_ratio(
    rows: list[dict[str, JsonValue]], numerator: str, denominator: str
) -> dict[str, JsonValue]:
    """Sum of numerators over sum of denominators (None for a zero sum), resampling whole samples."""
    pairs = [(row[numerator], row[denominator]) for row in rows]

    def statistic(sample: list[tuple[float, float]]) -> float | None:
        total = sum(bottom for _, bottom in sample)
        return sum(top for top, _ in sample) / total if total else None

    value = statistic(pairs)
    return {
        "numerator": sum(top for top, _ in pairs),
        "denominator": sum(bottom for _, bottom in pairs),
        "value": value,
        "ci95": None if value is None else bootstrap_ci(pairs, statistic),
    }


def latency_summary(latencies: list[float]) -> dict[str, JsonValue]:
    """Quantiles over all samples; math.inf marks no response, and a quantile on it is None."""
    if not latencies:
        return {
            "n": 0,
            "responses": 0,
            "median_s": None,
            "median_ci95": None,
            "p90_s": None,
        }
    else:
        pass
    values = sorted(latencies)
    return {
        "n": len(values),
        "responses": sum(math.isfinite(value) for value in values),
        "median_s": finite(quantile(values, MEDIAN)),
        "median_ci95": [finite(bound) for bound in bootstrap_ci(values, median)],
        "p90_s": finite(quantile(values, P90)),
    }


def subset_metrics(
    task: Task, rows: list[dict[str, JsonValue]], selected: int
) -> dict[str, JsonValue]:
    """Metrics over the eligible rows of a subset with `selected` samples."""
    if task == "pause_handling":
        return {
            "pause_intrusion_rate": rate(
                [row["pause_intrusion"] for row in rows], selected
            ),
            "pause_intrusion_per_pause": pooled_ratio(
                rows, "intruded_pauses", "pauses"
            ),
            "intrusion_s_per_pause_s": pooled_ratio(rows, "intrusion_s", "pause_s"),
            **{
                f"{flag}_rate": rate([row[flag] for row in rows], selected)
                for flag in PAUSE_DIAGNOSTICS
            },
        }
    else:
        pass
    latencies = [row["latency_s"] for row in rows]
    metrics = (
        {
            "early_start_rate": rate(
                [row["outcome"] == "early_start" for row in rows], selected
            )
        }
        if task == "turn_taking"
        else {}
    )
    for within_s in RESPONSE_WITHIN_S:
        metrics[f"response_rate_{within_s}s"] = rate(
            [latency is not None and latency <= within_s for latency in latencies],
            selected,
        )
    metrics["no_response_rate"] = rate(
        [row["outcome"] == "no_response" for row in rows], selected
    )
    metrics["response_latency_s"] = latency_summary(
        [math.inf if latency is None else latency for latency in latencies]
    )
    return metrics


def score_tree(tree: Path) -> dict[str, JsonValue]:
    """Write <tree>/unified.json and <tree>/unified-samples.jsonl; summary.json is never read."""
    manifest_path, asr_path = tree / "manifest.json", tree / "asr.json"
    if not asr_path.is_file():
        raise SystemExit(f"{asr_path} is missing; run reference-asr on this tree first")
    else:
        pass
    manifest, asr = read_json(manifest_path), read_json(asr_path)
    if manifest.get("kind") != MANIFEST_KIND:
        raise SystemExit(f"{manifest_path} is not a {MANIFEST_KIND} manifest")
    else:
        pass
    subsets, samples = {}, []
    for subset in SUBSETS:
        task = SUBSET_TASKS[subset]
        members = [row for row in manifest["samples"] if row["subset"] == subset]
        if task == "backchannel" or not members:
            continue
        else:
            pass
        eligible = [row for row in members if row["eligible"]]
        annotation_name = ANNOTATION_FILES[task]
        transcripts = {
            row["sample_id"].split("/")[1]: file_sha256(
                tree / row["sample_id"] / "output.json"
            )
            for row in eligible
        }
        recorded = (
            asr["subsets"][subset]["transcripts"] if subset in asr["subsets"] else {}
        )
        if transcripts != recorded or any(
            file_sha256(tree / row["sample_id"] / annotation_name)
            != row["files"][annotation_name]
            for row in eligible
        ):
            raise SystemExit(f"{tree / subset} differs from manifest.json or asr.json")
        else:
            pass
        scored = []
        for row in members:
            sample = {
                "sample_id": row["sample_id"],
                "subset": subset,
                "task": task,
                "eligible": row["eligible"],
            }
            if row["eligible"]:
                sample.update(score_eligible(tree, row, task))
                scored.append(sample)
            else:
                sample["reasons"] = row["reasons"]
            samples.append(sample)
        subsets[subset] = {
            "task": task,
            "counts": {
                "declared": DECLARED_COUNTS[subset],
                "selected": len(members),
                "eligible": len(eligible),
                "ineligible": len(members) - len(eligible),
                "ineligible_reasons": dict(
                    Counter(
                        reason
                        for row in members
                        if not row["eligible"]
                        for reason in row["reasons"]
                    )
                ),
            },
            "coverage": len(eligible) / len(members),
            "metrics": subset_metrics(task, scored, len(members)),
        }
        if task != "pause_handling":
            subsets[subset]["min_observable_s"] = min(
                (sample["observable_s"] for sample in scored), default=None
            )
        else:
            pass
    unified = {
        "schema_version": 1,
        "kind": KIND,
        "definitions_version": DEFINITIONS_VERSION,
        "manifest_sha256": file_sha256(manifest_path),
        "asr_sha256": file_sha256(asr_path),
        "intervals": {
            "confidence": 0.95,
            "sample_rates": "Wilson score",
            "other": "percentile bootstrap over samples",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "subsets": subsets,
    }
    atomic_write_json(tree / "unified.json", unified)
    atomic_write_bytes(
        tree / "unified-samples.jsonl",
        "".join(
            json.dumps(sample, sort_keys=True) + "\n" for sample in samples
        ).encode(),
    )
    return unified


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def interval(bounds: list[float] | None) -> str:
    return "" if bounds is None else f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def latency_text(value: float | None) -> str:
    return "not reached" if value is None else f"{value:.3f}"


def describe(metric: dict[str, JsonValue]) -> str:
    if "median_s" in metric and metric["n"] == 0:
        return "n/a 0/0 responded"
    elif "median_s" in metric:
        low, high = metric["median_ci95"]
        return (
            f"median {latency_text(metric['median_s'])} "
            f"[{latency_text(low)}, {latency_text(high)}] "
            f"p90 {latency_text(metric['p90_s'])} "
            f"{metric['responses']}/{metric['n']} responded"
        )
    elif "count" in metric:
        return (
            f"{number(metric['value'])} {interval(metric['ci95'])} "
            f"{metric['count']}/{metric['n']}, "
            f"all selected {interval(metric['selected_bounds'])}"
        )
    else:
        return (
            f"{number(metric['value'])} {interval(metric['ci95'])} "
            f"{metric['numerator']:g}/{metric['denominator']:g}"
        )


def reference_label(entry: dict[str, JsonValue]) -> str:
    """Official unless a non-official judge rated the subset; then name that judge."""
    judge = entry["judge"]
    if judge is None or judge["official"]:
        return "official"
    else:
        return f"reference (judge {'+'.join(judge['sent_models'])}, not official)"


def render(unified: dict[str, JsonValue], official: dict[str, JsonValue]) -> str:
    """Per-subset summary; official holds summary.json subsets for the reference column."""
    lines = [f"definitions {unified['definitions_version']}"]
    for subset, result in unified["subsets"].items():
        counts = result["counts"]
        header = (
            f"{subset}: {counts['eligible']} eligible of {counts['selected']} "
            f"selected (coverage {result['coverage']:.3f}), "
            f"{counts['declared']} declared"
        )
        if "min_observable_s" in result and result["min_observable_s"] is not None:
            header += f"; window after reference >= {result['min_observable_s']:.6f} s"
        else:
            pass
        lines.append(header)
        lines.extend(
            f"  {name:<26} {describe(metric)}"
            for name, metric in result["metrics"].items()
        )
        if subset in official:
            prefix = reference_label(official[subset])
            lines.extend(
                f"  {prefix} {label:<17} {value:.3f}"
                for label, value in official[subset]["result"].items()
            )
        else:
            pass
    return "\n".join(lines)
