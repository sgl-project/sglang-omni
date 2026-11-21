"""Aggregate the PD-vs-colocated sweep into one table.

Arm C is two independent replicas, so a point's throughput is the pair's
combined rate and its latency percentiles come from the POOLED per-request
records -- averaging two replicas' p95s would not be a p95 of anything.

The conditioned inter-token quantities are the exception: they are averaged
across replicas rather than pooled, because the analyzer labels a decode gap by
whether a prefill was in flight in the SAME event stream. Pooling two replicas'
events into one directory would let a prefill on replica 1 mark a decode gap on
replica 2 as disturbed, which is a different GPU and no interference at all.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Any

COLO_RE = re.compile(
    r"^C(?P<pass>[ab])_(?P<wl>text|image)_r(?P<rate>[\d.]+)_rep(?P<rep>\d+)_rep(?P<replica>[12])$"
)
PD_RE = re.compile(r"^PD_(?P<wl>text|image)_r(?P<rate>[\d.]+)_rep(?P<rep>\d+)$")


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


def load_raw(raw_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in glob.glob(os.path.join(raw_dir, "*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return [r for r in rows if r.get("arm") != "warmup"]


def load_analysis(results_dir: str, tag: str) -> dict[str, Any] | None:
    matches = glob.glob(os.path.join(results_dir, f"analysis_{tag}__*.json"))
    if not matches:
        return None
    with open(matches[0]) as fh:
        return json.load(fh)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    failed = len(records) - len(ok)
    starts = [r["sent_at"] for r in records] or [0.0]
    ends = [
        r["sent_at"] + r["total_s"] for r in ok if r.get("total_s") is not None
    ] or starts
    wall = max(ends) - min(starts)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    itls = [g for r in ok for g in (r.get("itl_ms") or [])]
    ptoks = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]
    return {
        "requests_sent": len(records),
        "requests_ok": len(ok),
        "requests_failed": failed,
        "wall_s": wall,
        "achieved_rps": (len(ok) / wall) if wall > 0 else None,
        "ttft_p50_ms": (pct(ttfts, 50) or 0) * 1000 if ttfts else None,
        "ttft_p95_ms": (pct(ttfts, 95) or 0) * 1000 if ttfts else None,
        "itl_p50_ms": pct(itls, 50),
        "itl_p95_ms": pct(itls, 95),
        "prompt_tokens_mean": statistics.fmean(ptoks) if ptoks else None,
    }


def collect(root: str) -> dict[tuple, dict[str, Any]]:
    """key -> point, where key = (arm_label, workload, rate, repeat)."""
    raw_root = os.path.join(root, "raw")
    results_dir = os.path.join(root, "results")

    grouped: dict[tuple, list[str]] = defaultdict(list)
    for tag in sorted(os.listdir(raw_root)) if os.path.isdir(raw_root) else []:
        m = COLO_RE.match(tag)
        if m:
            key = (f"C{m['pass']}", m["wl"], float(m["rate"]), int(m["rep"]))
            grouped[key].append(tag)
            continue
        m = PD_RE.match(tag)
        if m:
            key = ("PD", m["wl"], float(m["rate"]), int(m["rep"]))
            grouped[key].append(tag)

    points: dict[tuple, dict[str, Any]] = {}
    for key, tags in grouped.items():
        records: list[dict[str, Any]] = []
        conditioned: list[dict[str, Any]] = []
        for tag in tags:
            records.extend(load_raw(os.path.join(raw_root, tag)))
            an = load_analysis(results_dir, tag)
            if an:
                conditioned.append(an)
        point = summarize(records)
        point["offered_rps"] = key[2]
        point["n_replicas"] = len(tags)

        clean = [
            a["itl_ms"]["no_concurrent_prefill"]["p50"]
            for a in conditioned
            if a.get("itl_ms", {}).get("no_concurrent_prefill", {}).get("p50")
        ]
        dirty = [
            a["itl_ms"]["with_concurrent_prefill"]["p50"]
            for a in conditioned
            if a.get("itl_ms", {}).get("with_concurrent_prefill", {}).get("p50")
        ]
        ratio = [
            a["itl_ms"]["interference_ratio_p50"]
            for a in conditioned
            if a.get("itl_ms", {}).get("interference_ratio_p50")
        ]
        n_dirty = [
            a["itl_ms"]["with_concurrent_prefill"]["n"]
            for a in conditioned
            if a.get("itl_ms", {}).get("with_concurrent_prefill", {}).get("n")
            is not None
        ]
        n_clean = [
            a["itl_ms"]["no_concurrent_prefill"]["n"]
            for a in conditioned
            if a.get("itl_ms", {}).get("no_concurrent_prefill", {}).get("n") is not None
        ]
        point["itl_clean_p50_ms"] = statistics.fmean(clean) if clean else None
        point["itl_disturbed_p50_ms"] = statistics.fmean(dirty) if dirty else None
        point["interference_ratio_p50"] = statistics.fmean(ratio) if ratio else None
        tot = sum(n_dirty) + sum(n_clean)
        point["disturbed_gap_share"] = (sum(n_dirty) / tot) if tot else None
        points[key] = point
    return points


def across_repeats(points: dict[tuple, dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Collapse the repeat axis to median + spread."""
    bykey: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for (arm, wl, rate, _rep), p in points.items():
        bykey[(arm, wl, rate)].append(p)

    out: dict[tuple, dict[str, Any]] = {}
    for key, ps in bykey.items():
        entry: dict[str, Any] = {"n_repeats": len(ps)}
        for field in (
            "achieved_rps",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "itl_p50_ms",
            "itl_p95_ms",
            "itl_clean_p50_ms",
            "itl_disturbed_p50_ms",
            "interference_ratio_p50",
            "disturbed_gap_share",
            "prompt_tokens_mean",
        ):
            vals = [p[field] for p in ps if p.get(field) is not None]
            entry[field] = statistics.median(vals) if vals else None
            entry[f"{field}_min"] = min(vals) if vals else None
            entry[f"{field}_max"] = max(vals) if vals else None
        entry["requests_ok"] = sum(p["requests_ok"] for p in ps)
        entry["requests_failed"] = sum(p["requests_failed"] for p in ps)
        out[key] = entry
    return out


def fmt(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./pdvc")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    points = collect(args.root)
    agg = across_repeats(points)

    print("\n=== per (arm x workload x offered rate), median over repeats ===")
    hdr = (
        f"{'arm':<5}{'workload':<8}{'offered':>8}{'achieved':>9}{'ok':>7}{'fail':>6}"
        f"{'ptok':>8}{'TTFTp50':>9}{'TTFTp95':>9}{'ITLp50':>8}{'ITLp95':>8}"
        f"{'ITLclean':>9}{'ITLdist':>8}{'ratio':>7}{'dist%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for wl in ("text", "image"):
        for arm in ("Ca", "Cb", "PD"):
            rows = sorted((k, v) for k, v in agg.items() if k[0] == arm and k[1] == wl)
            for (a, w, rate), v in rows:
                print(
                    f"{a:<5}{w:<8}{rate:>8.4g}{fmt(v['achieved_rps'], 2):>9}"
                    f"{v['requests_ok']:>7}{v['requests_failed']:>6}"
                    f"{fmt(v['prompt_tokens_mean'], 0):>8}"
                    f"{fmt(v['ttft_p50_ms']):>9}{fmt(v['ttft_p95_ms']):>9}"
                    f"{fmt(v['itl_p50_ms'], 2):>8}{fmt(v['itl_p95_ms'], 2):>8}"
                    f"{fmt(v['itl_clean_p50_ms'], 2):>9}"
                    f"{fmt(v['itl_disturbed_p50_ms'], 2):>8}"
                    f"{fmt(v['interference_ratio_p50'], 3):>7}"
                    f"{(fmt(v['disturbed_gap_share'] * 100, 1) if v['disturbed_gap_share'] is not None else '-'):>7}"
                )
        print()

    # ---- A/A noise band: pass a vs pass b of the SAME configuration ----------
    print("=== A/A noise band (arm C pass a vs pass b, identical config) ===")
    fields = [
        ("achieved_rps", "achieved rps"),
        ("ttft_p50_ms", "TTFT p50"),
        ("ttft_p95_ms", "TTFT p95"),
        ("itl_p50_ms", "ITL p50"),
        ("itl_p95_ms", "ITL p95"),
        ("itl_clean_p50_ms", "ITL clean p50"),
        ("interference_ratio_p50", "interference ratio"),
    ]
    band: dict[str, list[float]] = defaultdict(list)
    for wl in ("text", "image"):
        for key in sorted(k for k in agg if k[0] == "Ca" and k[1] == wl):
            _, w, rate = key
            a = agg.get(("Ca", w, rate))
            b = agg.get(("Cb", w, rate))
            if not a or not b:
                continue
            for field, _label in fields:
                va, vb = a.get(field), b.get(field)
                if va and vb and va > 0:
                    band[f"{wl}:{field}"].append(abs(vb - va) / va * 100.0)

    print(f"{'workload:metric':<34}{'n':>4}{'median %':>10}{'max %':>9}")
    print("-" * 57)
    for wl in ("text", "image"):
        for field, label in fields:
            vals = band.get(f"{wl}:{field}")
            if not vals:
                continue
            print(
                f"{wl + ':' + label:<34}{len(vals):>4}"
                f"{statistics.median(vals):>10.1f}{max(vals):>9.1f}"
            )

    if args.json_out:
        payload = {
            "points": {"|".join(str(x) for x in k): v for k, v in points.items()},
            "aggregated": {"|".join(str(x) for x in k): v for k, v in agg.items()},
            "aa_noise_pct": {k: sorted(v) for k, v in band.items()},
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
