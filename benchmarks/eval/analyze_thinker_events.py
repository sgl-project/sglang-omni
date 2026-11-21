# SPDX-License-Identifier: Apache-2.0
"""Attribute thinker request time from profiler event JSONL (PD Phase 0).

Two outputs the passive stage table alone cannot give:

1. Per-request segment breakdown, matching the columns in
   ``docs/developer_reference/qwen3_asr_concurrency_profile.md`` so thinker
   numbers can be read against that ASR table.

2. The discriminating measurement: every inter-token gap on the thinker stage
   is labelled with how many prefills were in flight during that gap. If
   prefill admission perturbs running decode, gaps that overlap a prefill are
   longer than gaps that do not, and the excess grows with prompt length. If
   the queue is simply capacity-capped, the two populations coincide.

Usage:
    python analyze_thinker_events.py <event_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any

# A colocated run emits every thinker event under stage "thinker". A PD run splits
# the same work across "thinker_prefill" and "thinker_decode", so filtering on
# "thinker" alone yields ZERO events for a PD arm. Unioning the three names keeps
# one code path for both: prefill windows come from whichever stage ran the prefill,
# per-token emits from whichever stage streamed them, and the conditioned ITL keeps
# its meaning. In PD the prefill is on the OTHER card, so a ratio near 1.0 is the
# physically expected answer rather than a bug.
STAGES = {"thinker", "thinker_prefill", "thinker_decode"}

# (start_event, end_event, column name) -- same decomposition as the ASR profile.
SEGMENTS = [
    ("scheduler_request_build_start", "scheduler_request_build_end", "build"),
    ("scheduler_request_build_end", "scheduler_queue_enter", "build_to_queued"),
    ("scheduler_queue_enter", "scheduler_prefill_start", "queued_to_scheduled"),
    ("scheduler_prefill_start", "scheduler_prefill_end", "first_forward"),
    ("scheduler_prefill_end", "stage_complete", "decode_tail"),
    ("scheduler_prefill_start", "scheduler_first_emit", "prefill_to_first_emit"),
]


def load(event_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in glob.glob(os.path.join(event_dir, "*.jsonl")):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q / 100.0 * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_dir")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    rows = load(args.event_dir)
    if not rows:
        raise SystemExit(f"no events under {args.event_dir}")

    # request_id -> event_name -> first timestamp, thinker stage only
    first: dict[str, dict[str, int]] = defaultdict(dict)
    # request_id -> ordered per-token emit timestamps
    emits: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("stage") not in STAGES:
            continue
        rid = row["request_id"]
        name = row["event_name"]
        ts = row["timestamp_ns"]
        if name == "stage_stream_chunk_sent":
            emits[rid].append(ts)
        elif name not in first[rid]:
            first[rid][name] = ts

    # Prefill occupancy intervals, used to label each inter-token gap.
    prefill_windows = sorted(
        (ev["scheduler_prefill_start"], ev["scheduler_prefill_end"])
        for ev in first.values()
        if "scheduler_prefill_start" in ev and "scheduler_prefill_end" in ev
    )
    starts = [w[0] for w in prefill_windows]

    def prefills_overlapping(lo: int, hi: int) -> int:
        # Windows are short; scan from the first window that could overlap.
        idx = bisect.bisect_left(starts, lo)
        count = 0
        for j in range(max(0, idx - 64), len(prefill_windows)):
            s, e = prefill_windows[j]
            if s >= hi:
                break
            if e > lo:
                count += 1
        return count

    segments: dict[str, list[float]] = defaultdict(list)
    for ev in first.values():
        for start, end, label in SEGMENTS:
            if start in ev and end in ev:
                segments[label].append((ev[end] - ev[start]) / 1e6)

    clean_gaps: list[float] = []
    disturbed_gaps: list[float] = []
    for rid, stamps in emits.items():
        stamps.sort()
        for a, b in zip(stamps, stamps[1:]):
            gap_ms = (b - a) / 1e6
            # Exclude this request's own prefill; it precedes its first emit.
            if prefills_overlapping(a, b) > 0:
                disturbed_gaps.append(gap_ms)
            else:
                clean_gaps.append(gap_ms)

    report: dict[str, Any] = {
        "event_dir": args.event_dir,
        "requests": len(first),
        "segments_ms": {k: describe(v) for k, v in segments.items()},
        "itl_ms": {
            "all": describe(clean_gaps + disturbed_gaps),
            "no_concurrent_prefill": describe(clean_gaps),
            "with_concurrent_prefill": describe(disturbed_gaps),
        },
    }
    clean_p50 = report["itl_ms"]["no_concurrent_prefill"]["p50"]
    dirty_p50 = report["itl_ms"]["with_concurrent_prefill"]["p50"]
    if clean_p50 and dirty_p50:
        report["itl_ms"]["interference_ratio_p50"] = dirty_p50 / clean_p50

    print(f"requests: {report['requests']}")
    print("\nsegment (ms)        n     mean     p50     p95     p99     max")
    for _, _, label in SEGMENTS:
        d = report["segments_ms"].get(label)
        if not d or not d["n"]:
            continue

        def f(x):
            return f"{x:7.2f}" if x is not None else "      -"

        print(
            f"{label:<18}{d['n']:>5}{f(d['mean'])}{f(d['p50'])}"
            f"{f(d['p95'])}{f(d['p99'])}{f(d['max'])}"
        )

    print("\ninter-token gap (ms)")
    for key in ("all", "no_concurrent_prefill", "with_concurrent_prefill"):
        d = report["itl_ms"][key]

        def f(x):
            return f"{x:7.2f}" if x is not None else "      -"

        print(f"  {key:<26}{d['n']:>6}{f(d['mean'])}{f(d['p50'])}{f(d['p95'])}")
    if "interference_ratio_p50" in report["itl_ms"]:
        print(
            f"\n  p50 gap ratio (with prefill / without): "
            f"{report['itl_ms']['interference_ratio_p50']:.3f}"
        )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
