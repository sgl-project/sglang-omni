# SPDX-License-Identifier: Apache-2.0
"""Summarize reference intervals and behavior with explicit coverage counts."""

from __future__ import annotations

import math
import random
import statistics
from argparse import Namespace
from collections import Counter
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.reference_behavior import behavior_units, build_request
from benchmarks.duplex.reference_core import (
    C_LABELS,
    EVENT_SELECTION_RULE,
    REFERENCE_REVISION,
    VARIANTS,
    Engine,
    atomic_write_json,
    canonical_hash,
    read_json,
    selected,
    utc_now,
)
from benchmarks.duplex.reference_source import ReferenceBehavior, load_official_behavior
from benchmarks.duplex.run_artifacts import file_sha256


def quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower_index, upper_index = math.floor(position), math.ceil(position)
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * (position - lower_index)


def cluster_bootstrap(
    clusters: list[list[float]], replicates: int, seed: str
) -> dict[str, JsonValue]:
    """Percentile CI of the pooled mean, resampling whole samples (clusters) with replacement."""
    cluster_sums = [sum(cluster) for cluster in clusters]
    cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_count = len(clusters)
    if cluster_count == 0 or sum(cluster_sizes) == 0:
        return {
            "unit": "sample",
            "unit_n": cluster_count,
            "replicates": 0,
            "ci95": None,
        }
    else:
        pass
    random_generator = random.Random(seed)
    sampled_means, empty_replicates = [], 0
    for _ in range(replicates):
        sample_indices = [
            random_generator.randrange(cluster_count) for _ in range(cluster_count)
        ]
        count = sum(cluster_sizes[i] for i in sample_indices)
        if count == 0:
            empty_replicates += 1
            continue
        else:
            pass
        sampled_means.append(sum(cluster_sums[i] for i in sample_indices) / count)
    sampled_means.sort()
    return {
        "unit": "sample",
        "unit_n": cluster_count,
        "replicates": replicates,
        "empty_replicates": empty_replicates,
        "seed": seed,
        "method": "percentile",
        "ci95": (
            [quantile(sampled_means, 0.025), quantile(sampled_means, 0.975)]
            if sampled_means
            else None
        ),
    }


def describe(
    clusters: list[list[float]], replicates: int, seed: str
) -> dict[str, JsonValue]:
    values = sorted(value for cluster in clusters for value in cluster)
    return {
        "interval_n": len(values),
        "samples": len(clusters),
        "zero_interval_samples": sum(1 for cluster in clusters if not cluster),
        "mean_s": statistics.fmean(values) if values else None,
        "median_s": statistics.median(values) if values else None,
        "bootstrap_pooled_mean": cluster_bootstrap(clusters, replicates, seed),
    }


def summarize_timing(
    engine: Engine, sids: list[str], variant: str, args: Namespace, group: str
) -> dict[str, JsonValue]:
    ledger = Counter()
    reasons = Counter()
    flags = Counter()
    stop_clusters, response_clusters, event_stop_clusters, event_response_clusters = (
        [],
        [],
        [],
        [],
    )
    for sample_id in sids:
        variant_state = engine.samples[sample_id]["variants"][variant]
        flags.update(variant_state.get("flags", []))
        if not variant_state["eligible"]:
            ledger["ineligible"] += 1
            reasons.update(variant_state["reasons"])
            continue
        else:
            pass
        path = engine.sample_dir(sample_id) / "receipts" / f"timing-{variant}.json"
        receipt = read_json(path) if path.exists() else {"status": "not_run"}
        ledger[receipt["status"]] += 1
        if receipt["status"] != "ok":
            continue
        else:
            pass
        folder = (
            engine.sample_dir(sample_id)
            if variant == "overlap"
            else engine.sample_dir(sample_id) / "clean"
        )
        intervals = folder / "latency_intervals.json"
        if file_sha256(intervals) != receipt["intervals_sha256"]:
            raise ValueError(f"Timing intervals changed after scoring: {intervals}")
        else:
            pass
        doc = read_json(intervals)
        for key, value in receipt["labels"].items():
            if value:
                ledger[f"label_{key}"] += 1
            else:
                pass
        stop_latencies_s = [
            end_s - start_s for start_s, end_s in doc["latency_stop_list"]
        ]
        response_latencies_s = [
            end_s - start_s for start_s, end_s in doc["latency_resp_list"]
        ]
        stop_clusters.append(stop_latencies_s)
        response_clusters.append(response_latencies_s)
        span = engine.samples[sample_id].get("event_span_s")
        if span is not None:
            event_stop_clusters.append(
                [
                    end_s - start_s
                    for start_s, end_s in doc["latency_stop_list"]
                    if start_s < span[1] and end_s > span[0]
                ]
            )
            first_response = next(
                (
                    [start_s, end_s]
                    for start_s, end_s in doc["latency_resp_list"]
                    if start_s >= span[0]
                ),
                None,
            )
            event_response_clusters.append(
                [first_response[1] - first_response[0]] if first_response else []
            )
        else:
            pass
    summary = {
        "population": len(sids),
        "status": dict(ledger),
        "ineligible_reasons": dict(reasons),
        "manifest_flags": dict(flags),
        "official_all_intervals": {
            "reduction": "campaign reduction of official per-sample intervals (end - start); "
            "official source defines no scalar aggregate",
            "stop": describe(
                stop_clusters, args.bootstrap, f"{args.seed}:{group}:stop"
            ),
            "response": describe(
                response_clusters, args.bootstrap, f"{args.seed}:{group}:resp"
            ),
        },
    }
    if event_stop_clusters:
        summary["event_selected_non_official"] = {
            "rule": EVENT_SELECTION_RULE,
            "stop": describe(
                event_stop_clusters, args.bootstrap, f"{args.seed}:{group}:ev_stop"
            ),
            "response": describe(
                event_response_clusters, args.bootstrap, f"{args.seed}:{group}:ev_resp"
            ),
        }
    else:
        pass
    return summary


def summarize_asr(engine: Engine, sids: list[str]) -> dict[str, JsonValue]:
    ledger = Counter()
    for sample_id in sids:
        for variant, files in VARIANTS.items():
            for file_name in files:
                if not engine.eligible(sample_id, variant):
                    ledger[f"{file_name}:ineligible"] += 1
                    continue
                else:
                    pass
                path = (
                    engine.sample_dir(sample_id)
                    / "receipts"
                    / f"asr-{file_name.rsplit('.', 1)[0]}.json"
                )
                receipt = (
                    read_json(path)
                    if path.exists()
                    else {"status": "not_run", "words": None}
                )
                ledger[f"{file_name}:{receipt['status']}"] += 1
                if receipt["status"] == "ok" and receipt["words"] == 0:
                    ledger[f"{file_name}:ok_empty_transcript"] += 1
                else:
                    pass
    return dict(sorted(ledger.items()))


def summarize_behavior(
    engine: Engine,
    sids: list[str],
    official: ReferenceBehavior,
    args: Namespace,
    group: str,
) -> dict[str, JsonValue]:
    ledger, labels, parsed = Counter(), Counter(), []
    valid_labels = []
    ready, blocked = behavior_units(engine, sids)
    for sample_id in sids:
        if sample_id in blocked:
            ledger[blocked[sample_id]] += 1
            continue
        else:
            pass
        judge = engine.sample_dir(sample_id) / "judge"
        if not (judge / "request.json").exists():
            ledger["not_prepared"] += 1
            continue
        else:
            pass
        prepared = read_json(judge / "request.json")
        if (
            build_request(official, engine.sample_dir(sample_id))["request_hash"]
            != prepared["request_hash"]
        ):
            ledger["stale_request"] += 1
            continue
        else:
            pass
        if not (judge / "result.json").exists():
            ledger["not_judged"] += 1
            continue
        else:
            pass
        result = read_json(judge / "result.json")
        if result["request_hash"] != prepared["request_hash"]:
            ledger["result_request_mismatch"] += 1
            continue
        else:
            pass
        ledger[result["status"]] += 1
        if result["parsed"] is not None:
            parsed.append(result["parsed"])
        else:
            pass
        if result["status"] == "valid":
            labels[result["label"]] += 1
            valid_labels.append(result["label"])
        else:
            pass
    try:
        _, totals, ratios = official.stats_by_axis(parsed)
        official_ratios = {
            axis: {k: round(value, 2) for k, value in sorted(ratios[axis].items())}
            for axis in ["C"]
        }
        official_ratios["C_total_tags"] = totals["C"]
    except Exception as exc:
        official_ratios = {"error": f"{type(exc).__name__}: {exc}"}
    proportions = {
        label: {
            "count": labels[label],
            "proportion": labels[label] / len(valid_labels) if valid_labels else None,
            "bootstrap": cluster_bootstrap(
                [[1.0 if x == label else 0.0] for x in valid_labels],
                args.bootstrap,
                f"{args.seed}:{group}:{label}",
            ),
        }
        for label in C_LABELS
    }
    return {
        "population": len(sids),
        "status": dict(ledger),
        "valid_n": len(valid_labels),
        "valid_label_proportions": proportions,
        "official_format_ratios": official_ratios,
        "official_format_note": "official stats_by_axis over every parsed response, rounded to 2 dp",
    }


def run_summarize(
    args: Namespace, engines: list[Engine], paths: dict[str, Path]
) -> dict[str, JsonValue]:
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    summary = {
        "generated_at": utc_now(),
        "reference_revision": REFERENCE_REVISION,
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "unit": "sample (one generation each)",
            "note": "sampling uncertainty over dataset samples; no repeated generations were run",
        },
        "engines": {},
        "populations": {},
    }
    for engine in engines:
        sample_ids = selected(engine, args.only)
        summary["populations"][engine.name] = {
            "manifest_sha256": canonical_hash(engine.manifest),
            "sample_ids": sample_ids,
        }
        groups = {"all": sample_ids}
        for sample_id in sample_ids:
            groups.setdefault(engine.samples[sample_id]["category"], []).append(
                sample_id
            )
        summary["engines"][engine.name] = {
            group: {
                "timing_official_overlap": summarize_timing(
                    engine, members, "overlap", args, f"{engine.name}:{group}:overlap"
                ),
                "timing_supplementary_clean": summarize_timing(
                    engine, members, "clean", args, f"{engine.name}:{group}:clean"
                ),
                "asr_files": summarize_asr(engine, members),
                "behavior": summarize_behavior(
                    engine, members, official, args, f"{engine.name}:{group}"
                ),
            }
            for group, members in sorted(groups.items())
        }
    atomic_write_json(args.out / "summary.json", summary)
    return summary
