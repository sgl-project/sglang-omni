#!/usr/bin/env python3
"""Extract startToFirstLiveASRMs baselines from Voxt session timing logs."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PREFIX = "Session timing summary. "


def percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * min(max(p, 0.0), 100.0) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def parse_records(text: str) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for line in text.splitlines():
        if PREFIX not in line:
            continue
        payload = line.split(PREFIX, 1)[1]
        head = payload.split(", firstLLM=", 1)[0]
        values: dict[str, str] = {}
        for part in head.split(", "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            values[key] = value
        first = values.get("startToFirstLiveASRMs")
        if first in (None, "n/a"):
            continue
        try:
            first_ms = int(first)
        except ValueError:
            continue
        records.append(
            {
                "output": values.get("output", ""),
                "pipeline": values.get("pipeline", ""),
                "asrModel": values.get("asrModel", ""),
                "asrProvider": values.get("asrProvider", ""),
                "startToFirstLiveASRMs": first_ms,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", default="transcription")
    parser.add_argument("--pipeline", default="liveDisplay")
    args = parser.parse_args()

    records: list[dict[str, str | int]] = []
    for path in args.logs:
        records.extend(parse_records(path.read_text(errors="ignore")))

    filtered = [
        record
        for record in records
        if record["output"] == args.output and record["pipeline"] == args.pipeline
    ]
    if not filtered:
        print("No matching first-live ASR samples.", file=sys.stderr)
        return 1

    by_model: dict[str, list[int]] = defaultdict(list)
    for record in filtered:
        by_model[str(record["asrModel"])].append(int(record["startToFirstLiveASRMs"]))

    print("asrModel\tn\tmean\tp50\tp95\tmin\tmax")
    for model, values in sorted(by_model.items()):
        print(
            "\t".join(
                [
                    model,
                    str(len(values)),
                    f"{statistics.mean(values):.1f}",
                    f"{percentile(values, 50):.1f}",
                    f"{percentile(values, 95):.1f}",
                    str(min(values)),
                    str(max(values)),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
