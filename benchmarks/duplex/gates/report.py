"""Markdown tables from a gate results directory (standard library only; runs on the node or locally).

    python report.py run <run-dir>                       write <run-dir>/run.json, print one chain.log line
    python report.py ladder <results>                    per tree and session count: miss, lag, memory, startup, errors
    python report.py sweep <results>... [--miss-threshold X] [--budget-ms B]
                                                         per session count, the operating point, the current target level and its limiting stage
    python report.py stages [--budget-ms B] <run-dir>... per-stage busy share, calls over budget, worst call, gen-2 GC, peak memory
    python report.py perception <results>                perception-stage timing (batched and single calls, session opens, memory)
    python report.py serving <results>                   serving gate per concurrency and repetition
    python report.py agree <results> [--ref ref_units.json]
    python report.py unit <results>
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fdbconc_metrics import percentile, run_metrics  # noqa: E402

FIRST_WINDOW_S = 10.0
DEFAULT_BUDGET_MS = 1000.0


def read_text(path):
    with open(path) as f:
        return f.read()


def read_lines(path):
    return read_text(path).splitlines()


def write_json(obj, path, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


# ---------------------------------------------------------------- one run


def read_kv(path):
    values = {}
    if os.path.isfile(path):
        for line in read_lines(path):
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                values[key] = value
            else:
                pass
    else:
        pass
    return values


def stage_pids(run):
    names = {}
    path = os.path.join(run, "stage_pids.txt")
    if os.path.isfile(path):
        for line in read_lines(path):
            match = re.search(r"StageGroup (\w+): .*pids=\[(\d+)\]", line)
            if match:
                names[match.group(1)] = match.group(2)
            else:
                pass
    else:
        pass
    return names


def stage_rows(run, pid):
    path = os.path.join(run, "timing", f"{pid}.jsonl")
    if not os.path.isfile(path):
        return []
    else:
        return [json.loads(line) for line in read_lines(path) if line.strip()]


def union_seconds(intervals, lo, hi):
    total, end = 0.0, lo
    for start, stop in sorted(intervals):
        start, stop = max(start, end), min(stop, hi)
        if stop > start:
            total += stop - start
            end = stop
        else:
            pass
    return total


def primary_calls(rows):
    """The calls that carry a stage's work: session hooks, else the omni scheduler batch, else the engine batch."""
    for prefix in ("hooks:", "omni:run_batch:", "engine:run_batch:"):
        calls = [r for r in rows if r["kind"].startswith(prefix)]
        if calls:
            return calls
        else:
            pass
    return []


def stage_stats(run, t0, t1, budget_ms):
    stats = []
    for stage, pid in stage_pids(run).items():
        rows = stage_rows(run, pid)
        work = [(r["t"], r["t"] + r["dt"]) for r in rows if r["kind"] != "mem"]
        calls = [r for r in primary_calls(rows) if t0 <= r["t"] <= t1]
        gc2 = [
            r for r in rows if r["kind"] == "gc" and r["n"] == 2 and t0 <= r["t"] <= t1
        ]
        worst = max(calls, key=lambda r: r["dt"], default=None)
        first_hi = min(t0 + FIRST_WINDOW_S, t1)
        stats.append(
            {
                "stage": stage,
                "busy_first_pct": 100
                * union_seconds(work, t0, first_hi)
                / max(first_hi - t0, 1e-9),
                "busy_run_pct": 100 * union_seconds(work, t0, t1) / max(t1 - t0, 1e-9),
                "calls": len(calls),
                "mean_batch": st.mean(r["n"] for r in calls) if calls else float("nan"),
                "over_budget_pct": (
                    100 * sum(r["dt"] * 1000 > budget_ms for r in calls) / len(calls)
                    if calls
                    else float("nan")
                ),
                "call_p95_ms": 1000 * percentile([r["dt"] for r in calls], 0.95),
                "worst_ms": 1000 * worst["dt"] if worst else float("nan"),
                "worst_at_s": worst["t"] - t0 if worst else float("nan"),
                "worst_batch": worst["n"] if worst else None,
                "gc2_count": len(gc2),
                "gc2_max_ms": 1000 * max((r["dt"] for r in gc2), default=0.0),
                "peak_reserved_mib": max(
                    (r["max_reserved_mb"] for r in rows if r["kind"] == "mem"),
                    default=0,
                ),
                "peak_alloc_mib": max(
                    (r["max_alloc_mb"] for r in rows if r["kind"] == "mem"), default=0
                ),
            }
        )
    return sorted(stats, key=lambda s: (-s["busy_run_pct"], -s["busy_first_pct"]))


def workload_stats(run, units, t0, t1):
    """Thinker tokens per unit (context fill rate) and the perception time split per call, from the hook, between the first client send and the last receive."""
    pids = stage_pids(run)
    stats = {}

    def window_rows(stage):
        rows = stage_rows(run, pids[stage]) if stage in pids else []
        return [r for r in rows if t0 is not None and t0 <= r["t"] <= t1]

    thinker = window_rows("thinker")
    # forward modes are recorded by name (EXTEND, DECODE) or by their integer value (1, 2)
    extends = [
        r
        for r in thinker
        if r["kind"] in ("omni:next_batch:EXTEND", "omni:next_batch:1")
        and "tokens" in r
    ]
    decodes = [
        r
        for r in thinker
        if r["kind"] in ("omni:next_batch:DECODE", "omni:next_batch:2")
    ]
    if extends and units:
        prefill = sum(r["tokens"] for r in extends)
        decode = sum(r["n"] for r in decodes)
        stats["prefill_tokens_per_unit"] = prefill / units
        stats["decode_tokens_per_unit"] = decode / units
        stats["prefill_tokens_per_request"] = prefill / max(
            sum(r["n"] for r in extends), 1
        )
        stats["max_context"] = max((r.get("max_seq", 0) for r in thinker), default=0)
    else:
        pass
    perception = window_rows("perception")
    calls = [r for r in perception if r["kind"].startswith("hooks:")]
    if calls:
        total = sum(r["dt"] for r in calls)
        image = sum(r["dt"] for r in perception if r["kind"] == "perc:image")
        audio = sum(r["dt"] for r in perception if r["kind"] == "perc:audio")
        stats["perception_calls"] = len(calls)
        stats["perception_call_ms"] = 1000 * total / len(calls)
        stats["perception_image_ms"] = 1000 * image / len(calls)
        stats["perception_audio_ms"] = 1000 * audio / len(calls)
        stats["images_encoded"] = sum(
            1 for r in perception if r["kind"] == "perc:image"
        )
    else:
        pass
    return stats


def peak_card_mib(run):
    path = os.path.join(run, "mem.log")
    if not os.path.isfile(path):
        return 0
    else:
        values = [
            int(line.split(",")[0])
            for line in read_lines(path)
            if line.split(",")[0].strip().isdigit()
        ]
    return max(values, default=0)


def summarize_run(run, budget_ms=DEFAULT_BUDGET_MS):
    meta = read_kv(os.path.join(run, "meta.txt"))
    status = (
        read_text(os.path.join(run, "status")).strip()
        if os.path.isfile(os.path.join(run, "status"))
        else "RUNNING"
    )
    summary = {
        "run": os.path.basename(run.rstrip("/")),
        "tag": meta.get("tag"),
        "sha": meta.get("sha"),
        "sessions": int(meta.get("sessions", 0)),
        "speech": meta.get("speech"),
        "frames": int(meta.get("frames", 0)),
        "card": meta.get("card"),
        "status": status,
        "startup_s": int(meta["startup_s"]) if "startup_s" in meta else None,
    }
    if status != "READY":
        return summary
    else:
        metrics = run_metrics(run)
    tb_path = os.path.join(run, "tracebacks.txt")
    summary.update(metrics)
    summary["tracebacks"] = (
        int(read_text(tb_path).strip() or 0) if os.path.isfile(tb_path) else None
    )
    summary["peak_card_mib"] = peak_card_mib(run)
    t0, t1 = metrics["clients_first_send_s"], metrics["clients_last_receive_s"]
    summary["window_s"] = t1 - t0 if t0 is not None and t1 is not None else None
    summary["stages"] = (
        stage_stats(run, t0, t1, budget_ms) if summary["window_s"] else []
    )
    summary["budget_ms"] = budget_ms
    summary["workload"] = workload_stats(run, metrics["units"], t0, t1)
    return summary


def chain_line(s):
    if s["status"] != "READY":
        return f"{s['run']} card {s['card']} {s['status']}"
    else:
        top = s["stages"][0] if s["stages"] else None
        busiest = f" busiest={top['stage']} {top['busy_run_pct']:.0f}%" if top else ""
    return (
        f"{s['run']} card {s['card']} READY {s['startup_s']}s ms={s['sessions']} peakMiB={s['peak_card_mib']} tb={s['tracebacks']}: "
        f"samples={s['samples']} units={s['units']}/{s['expected_units']} miss={s['miss_pct']:.1f}% "
        f"lag p50={s['lag_p50_ms']:.0f} p95={s['lag_p95_ms']:.0f} ms errors={s['errors']}{busiest}"
    )


def load_runs(results, budget_ms=None):
    """Every run directory of a results directory except the warm-up runs, as summaries.

    Recomputed from the traces and the hook when the traces are there; a copy pulled without traces uses each run's run.json.
    """
    runs = []
    for meta in sorted(glob.glob(os.path.join(results, "*", "meta.txt"))):
        run = os.path.dirname(meta)
        if os.path.basename(run).startswith("warm-"):
            continue
        else:
            cached = os.path.join(run, "run.json")
        traces = glob.glob(os.path.join(run, "w*", "samples"))
        if os.path.isfile(cached) and (budget_ms is None or not traces):
            runs.append(json.loads(read_text(cached)))
        else:
            runs.append(summarize_run(run, budget_ms or DEFAULT_BUDGET_MS))
    return runs


# ---------------------------------------------------------------- formatting helpers


def fmt(value, digits=1, unit=""):
    if value is None or value != value:
        return "–"
    else:
        return f"{value:.{digits}f}{unit}"


def gib(mib):
    return f"{mib / 1024:.0f} GiB" if mib else "–"


def ok_runs(runs):
    return [r for r in runs if r["status"] == "READY"]


def miss_cell(runs):
    return " / ".join(
        fmt(r["miss_pct"]) if r["status"] == "READY" else r["status"] for r in runs
    )


def group_line(runs):
    good = ok_runs(runs)
    misses = [r["miss_pct"] for r in good]
    return {
        "miss_per_run": miss_cell(runs),
        "miss_stats": (
            f"{min(misses):.1f} / {st.median(misses):.1f} / {max(misses):.1f} / {st.mean(misses):.1f}"
            if misses
            else "–"
        ),
        "miss_mean": st.mean(misses) if misses else None,
        "miss_max": max(misses) if misses else None,
        "lag": (
            f"{st.median(r['lag_p50_ms'] for r in good):.0f} / {st.median(r['lag_p95_ms'] for r in good):.0f}"
            if good
            else "–"
        ),
        "lag_p95": st.median(r["lag_p95_ms"] for r in good) if good else None,
        "peak": gib(max((r["peak_card_mib"] for r in good), default=0)),
        "startup": " / ".join(
            str(r["startup_s"]) if r["startup_s"] is not None else "–" for r in runs
        ),
        "errors": f"{sum(r['sessions_with_error'] for r in good)} / {sum(r['errors'] for r in good)}",
        "tracebacks": str(sum(r["tracebacks"] or 0 for r in good)),
        "units": " / ".join(f"{r['units']}/{r['expected_units']}" for r in good) or "–",
        "cards": ",".join(f"c{r['card']}" for r in runs),
    }


def stage_aggregate(runs):
    """Mean of the per-stage numbers over the runs of one group (worst call: the maximum)."""
    by = collections.defaultdict(list)
    for run in ok_runs(runs):
        for s in run.get("stages", []):
            by[s["stage"]].append(s)
    rows = []
    for stage, items in by.items():
        worst = max(
            items, key=lambda s: s["worst_ms"] if s["worst_ms"] == s["worst_ms"] else -1
        )
        rows.append(
            {
                "stage": stage,
                "busy_first_pct": st.mean(s["busy_first_pct"] for s in items),
                "busy_run_pct": st.mean(s["busy_run_pct"] for s in items),
                "over_budget_pct": (
                    st.mean(
                        s["over_budget_pct"]
                        for s in items
                        if s["over_budget_pct"] == s["over_budget_pct"]
                    )
                    if any(s["calls"] for s in items)
                    else float("nan")
                ),
                "call_p95_ms": (
                    st.mean(
                        s["call_p95_ms"]
                        for s in items
                        if s["call_p95_ms"] == s["call_p95_ms"]
                    )
                    if any(s["calls"] for s in items)
                    else float("nan")
                ),
                "mean_batch": (
                    st.mean(
                        s["mean_batch"]
                        for s in items
                        if s["mean_batch"] == s["mean_batch"]
                    )
                    if any(s["calls"] for s in items)
                    else float("nan")
                ),
                "worst_ms": worst["worst_ms"],
                "worst_at_s": worst["worst_at_s"],
                "worst_batch": worst["worst_batch"],
                "gc2_count": sum(s["gc2_count"] for s in items),
                "gc2_max_ms": max(s["gc2_max_ms"] for s in items),
                "peak_reserved_mib": max(s["peak_reserved_mib"] for s in items),
                "runs": len(items),
            }
        )
    return sorted(rows, key=lambda s: (-s["busy_run_pct"], -s["busy_first_pct"]))


def print_stage_table(rows, budget_ms):
    print(
        f"| rank | stage | busy, first {FIRST_WINDOW_S:.0f} s | busy, whole run | calls over {budget_ms:.0f} ms | call p95 | worst call (at, batch) | mean batch | gen-2 GC (count, max) | peak reserved |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for rank, s in enumerate(rows, 1):
        print(
            f"| {rank} | {s['stage']} | {fmt(s['busy_first_pct'], 0, ' %')} | {fmt(s['busy_run_pct'], 0, ' %')} | {fmt(s['over_budget_pct'], 2, ' %')} | "
            f"{fmt(s['call_p95_ms'], 0, ' ms')} | {fmt(s['worst_ms'], 0, ' ms')} ({fmt(s['worst_at_s'], 1, ' s')}, {s['worst_batch']}) | {fmt(s['mean_batch'], 1)} | "
            f"{s['gc2_count']}, {fmt(s['gc2_max_ms'], 0, ' ms')} | {gib(s['peak_reserved_mib'])} |"
        )


# ---------------------------------------------------------------- gates


def report_ladder(results, budget_ms):
    runs = load_runs(results, budget_ms)
    groups = collections.OrderedDict()
    for run in sorted(runs, key=lambda r: (r["sessions"], r["tag"] or "", r["run"])):
        groups.setdefault((run["tag"], run["sessions"]), []).append(run)
    trees = sorted({"%s %s" % (r["tag"], (r["sha"] or "")[:9]) for r in runs})
    print(f"Results `{results}`; trees: {', '.join(trees)}")
    print()
    print(
        "| tree | sessions | runs (card) | miss per run | miss min / median / max / mean | lag p50 / p95 (median) ms | peak card memory | startup s | sessions with error / errors | tracebacks | units |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (tag, sessions), items in groups.items():
        g = group_line(items)
        print(
            f"| {tag} | {sessions} | {len(items)} ({g['cards']}) | {g['miss_per_run']} % | {g['miss_stats']} % | {g['lag']} | {g['peak']} | {g['startup']} | {g['errors']} | {g['tracebacks']} | {g['units']} |"
        )
    for (tag, sessions), items in groups.items():
        rows = stage_aggregate(items)
        if rows:
            print(
                f"\nStages, {tag} at {sessions} sessions (mean of {len(ok_runs(items))} runs):\n"
            )
            print_stage_table(rows, budget_ms)
        else:
            pass


def report_sweep(paths, threshold, budget_ms):
    """One sweep, or several sweeps of the same tree merged by session count (a zoom-in or repeat runs added to a sweep)."""
    results = paths[0]
    runs = [run for path in paths for run in load_runs(path, budget_ms)]
    gate = read_kv(os.path.join(results, "gate.txt"))
    threshold = float(gate.get("threshold", 1.0)) if threshold is None else threshold
    levels = sorted(
        {
            int(x)
            for path in paths
            for x in re.findall(
                r"sessions=([\d,]+)",
                read_kv(os.path.join(path, "gate.txt")).get("args", "sessions=0"),
            )[0].split(",")
        }
        - {0}
    )
    by_level = collections.defaultdict(list)
    for run in runs:
        by_level[run["sessions"]].append(run)
    tags = sorted({f"{r['tag']} {(r['sha'] or '')[:9]}" for r in runs})
    frames = sorted({r.get("frames", 0) for r in runs})
    print(
        f"Session sweep `{' + '.join(os.path.basename(p.rstrip('/')) for p in paths)}`: tree {', '.join(tags)}, frames per unit {frames}, started {gate.get('started', '?')}, miss threshold {threshold:g} %."
    )
    print()
    print(
        "| sessions | runs (card) | miss per run | miss mean / max | lag p95 (median) | peak card memory | startup s | sessions with error / errors | tracebacks | units | status |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    verdict = {}
    for level in levels or sorted(by_level):
        items = by_level.get(level, [])
        if not items:
            verdict[level] = "skipped"
            print(f"| {level} | 0 | – | – | – | – | – | – | – | – | skipped |")
            continue
        else:
            g = group_line(items)
        if len(ok_runs(items)) < len(items) or g["miss_mean"] is None:
            verdict[level] = "failed to start"
        elif any(r["sessions_with_error"] or r["tracebacks"] for r in items):
            verdict[level] = "session errors"
        elif g["miss_mean"] <= threshold:
            verdict[level] = "solved"
        else:
            verdict[level] = "not solved"
        print(
            f"| {level} | {len(items)} ({g['cards']}) | {g['miss_per_run']} % | {fmt(g['miss_mean'])} / {fmt(g['miss_max'])} % | {fmt(g['lag_p95'], 0, ' ms')} | "
            f"{g['peak']} | {g['startup']} | {g['errors']} | {g['tracebacks']} | {g['units']} | {verdict[level]} |"
        )
    print_workload_table(by_level, levels or sorted(by_level))
    ordered = sorted(verdict)
    target = next(
        (
            n
            for n in ordered
            if verdict[n] in ("not solved", "failed to start", "session errors")
        ),
        None,
    )
    solved_below = [
        n for n in ordered if verdict[n] == "solved" and (target is None or n < target)
    ]
    operating = solved_below[-1] if solved_below else None
    print()
    operating_text = f"{operating} sessions" if operating is not None else "none"
    print(
        f"**Operating point:** {operating_text} (largest level with mean miss ≤ {threshold:g} % and every smaller level solved)."
    )
    if target is None:
        print(
            "**Current target level:** none — every level is solved; extend `--sessions` upward."
        )
    else:
        print(
            f"**Current target level:** {target} sessions (smallest level not solved: {verdict[target]})."
        )
    stray = [
        n
        for n in ordered
        if verdict[n] == "solved" and target is not None and n > target
    ]
    if stray:
        print(
            f"Levels above the target that are solved again: {stray} (check the spread before trusting the operating point)."
        )
    else:
        pass
    print()
    print("Solved levels, busiest stage (mean over runs):")
    print()
    for n in solved_below:
        rows = stage_aggregate(by_level[n])
        top = rows[0] if rows else None
        line = (
            f"busiest {top['stage']} {top['busy_run_pct']:.0f} % over the run, {top['busy_first_pct']:.0f} % in the first {FIRST_WINDOW_S:.0f} s, worst call {fmt(top['worst_ms'], 0, ' ms')}"
            if top
            else "no stage timing"
        )
        print(f"- {n} sessions: {line}")
    if target is not None and verdict[target] in ("not solved", "session errors"):
        rows = stage_aggregate(by_level[target])
        print()
        print(
            f"### Limiting stage at the target level ({target} sessions, mean of {len(ok_runs(by_level[target]))} runs)"
        )
        print()
        print_stage_table(rows, budget_ms)
        if rows:
            top = rows[0]
            first = max(rows, key=lambda s: s["busy_first_pct"])
            print()
            print(
                f"**Limiting stage: {top['stage']}** — busy {top['busy_run_pct']:.0f} % over the run and {top['busy_first_pct']:.0f} % in the first {FIRST_WINDOW_S:.0f} s; "
                f"{fmt(top['over_budget_pct'], 2)} % of its calls exceed {budget_ms:.0f} ms; worst call {fmt(top['worst_ms'], 0, ' ms')} at {fmt(top['worst_at_s'], 1, ' s')} (batch {top['worst_batch']})."
            )
            stalls = [
                r
                for r in rows
                if r["over_budget_pct"] == r["over_budget_pct"]
                and r["over_budget_pct"] > 0
            ]
            for r in stalls:
                print(
                    f"Calls over {budget_ms:.0f} ms in {r['stage']}: {r['over_budget_pct']:.2f} % of its calls, worst {r['worst_ms']:.0f} ms at {r['worst_at_s']:.1f} s (batch {r['worst_batch']}); a single long call delays every unit queued behind it."
                )
            if first["stage"] != top["stage"]:
                print(
                    f"In the first {FIRST_WINDOW_S:.0f} s the busiest stage is {first['stage']} ({first['busy_first_pct']:.0f} %)."
                )
            else:
                pass
        else:
            print("No stage timing in these runs (was the hook on?).")
    elif target is not None:
        error = startup_error(results, by_level[target])
        print(f"\nThe target level failed to start: `{error}`")
        if operating is not None:
            rows = stage_aggregate(by_level[operating])
            print()
            print(
                f"### Stage ranking at the operating point ({operating} sessions, mean of {len(ok_runs(by_level[operating]))} runs), the nearest level that ran"
            )
            print()
            print_stage_table(rows, budget_ms)
        else:
            pass
    else:
        pass


def startup_error(results, runs):
    """The first error line of the server log of a run that did not start."""
    for run in runs:
        log = os.path.join(results, run["run"], "server.log")
        if run["status"] != "READY" and os.path.isfile(log):
            errors = [line.strip() for line in read_lines(log) if "Error" in line]
            return errors[-1][:300] if errors else "no error line in server.log"
        else:
            pass
    return "unknown"


def print_workload_table(by_level, levels):
    print()
    print(
        "| sessions | thinker prefill tokens per unit | per extend request | decode tokens per unit | units to fill 8192 tokens | largest context seen | perception call: audio / image / other ms | images encoded |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for level in levels:
        loads = [
            r["workload"] for r in ok_runs(by_level.get(level, [])) if r.get("workload")
        ]
        if not loads:
            continue
        else:
            pass

        def mean(key):
            values = [w[key] for w in loads if key in w]
            return st.mean(values) if values else float("nan")

        per_unit = mean("prefill_tokens_per_unit") + mean("decode_tokens_per_unit")
        other = (
            mean("perception_call_ms")
            - mean("perception_audio_ms")
            - mean("perception_image_ms")
        )
        print(
            f"| {level} | {fmt(mean('prefill_tokens_per_unit'), 1)} | {fmt(mean('prefill_tokens_per_request'), 1)} | {fmt(mean('decode_tokens_per_unit'), 1)} | "
            f"{fmt(8192 / per_unit if per_unit == per_unit and per_unit > 0 else None, 0)} | {max(w.get('max_context', 0) for w in loads)} | "
            f"{fmt(mean('perception_audio_ms'), 1)} / {fmt(mean('perception_image_ms'), 1)} / {fmt(other, 1)} | {fmt(mean('images_encoded'), 0)} |"
        )


def report_stages(runs, budget_ms):
    for run in runs:
        s = summarize_run(run, budget_ms)
        print(f"\n{chain_line(s)}; window {fmt(s.get('window_s'), 0, ' s')}\n")
        print_stage_table(s.get("stages", []), budget_ms)


def report_perception(results):
    """Perception numbers per run: batched calls (n >= 32) per session, single calls, the first call, the start-wave opens, busy share, memory."""
    rows_out = []
    for run in sorted(glob.glob(os.path.join(results, "*", "meta.txt"))):
        run = os.path.dirname(run)
        pid = stage_pids(run).get("perception")
        if os.path.basename(run).startswith("warm-") or pid is None:
            continue
        else:
            rows = sorted(stage_rows(run, pid), key=lambda r: r["t"])
        sessions = int(read_kv(os.path.join(run, "meta.txt")).get("sessions", 48))
        calls = [r for r in rows if r["kind"].startswith("perc:")] or [
            r for r in rows if r["kind"].startswith("hooks:")
        ]
        if not calls:
            continue
        else:
            opens = [r for r in rows if r["kind"].startswith("open:")][:sessions]
        big = [r["dt"] / r["n"] for r in calls if r["n"] >= 32]
        one = [r["dt"] for r in calls if r["n"] == 1]
        span = calls[-1]["t"] + calls[-1]["dt"] - calls[0]["t"]
        mem = [r for r in rows if r["kind"] == "mem"]
        d = {
            "run": os.path.basename(run),
            "big_ms": 1000 * st.median(big) if big else float("nan"),
            "n_big": len(big),
            "one_ms": 1000 * st.median(one) if one else float("nan"),
            "first_ms": 1000 * calls[0]["dt"],
            "first_n": calls[0]["n"],
            "opens": len(opens),
            "open_sum_ms": 1000 * sum(r["dt"] for r in opens),
            "open_span_ms": (
                1000 * (opens[-1]["t"] + opens[-1]["dt"] - opens[0]["t"])
                if opens
                else float("nan")
            ),
            "busy_pct": (
                100 * sum(r["dt"] for r in calls) / span if span > 0 else float("nan")
            ),
            "calls": len(calls),
            "mean_n": st.mean(r["n"] for r in calls),
            "reserved": max((r["max_reserved_mb"] for r in mem), default=0),
            "alloc": max((r["max_alloc_mb"] for r in mem), default=0),
        }
        rows_out.append(d)
    print(
        "| run | call at batch ≥32, p50 per session | call at batch 1, p50 | first call (batch) | start-wave opens: sum / span | busy share | calls / mean batch | peak reserved / allocated |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for d in rows_out:
        print(
            f"| {d['run']} | {fmt(d['big_ms'], 2, ' ms')} ({d['n_big']}) | {fmt(d['one_ms'], 1, ' ms')} | {d['first_ms']:.0f} ms ({d['first_n']}) | "
            f"{d['opens']}: {d['open_sum_ms']:.0f} / {fmt(d['open_span_ms'], 0)} ms | {fmt(d['busy_pct'], 1, ' %')} | {d['calls']} / {d['mean_n']:.2f} | {d['reserved']} / {d['alloc']} MiB |"
        )
    if len(rows_out) > 1:

        def mean(key):
            values = [d[key] for d in rows_out if d[key] == d[key]]
            return st.mean(values) if values else float("nan")

        print(
            f"| **mean** | {fmt(mean('big_ms'), 2, ' ms')} | {fmt(mean('one_ms'), 1, ' ms')} | {mean('first_ms'):.0f} ms | {mean('open_sum_ms'):.0f} / {fmt(mean('open_span_ms'), 0)} ms | "
            f"{fmt(mean('busy_pct'), 1, ' %')} | | {mean('reserved'):.0f} / {mean('alloc'):.0f} MiB |"
        )
    else:
        pass
    print()
    report_ladder(results, DEFAULT_BUDGET_MS)


def serving_rows(path, width):
    out = {}
    if os.path.isfile(path):
        for line in read_lines(path):
            fields = line.split()
            if len(fields) >= width and fields[0].isdigit() and "/" in fields[1]:
                out[int(fields[0])] = fields
            else:
                pass
    else:
        pass
    return out


def report_serving(results):
    """Columns are read from the tables benchmarks/duplex/serving.py prints (realtime: miss %, lag p95, reply p50, underrun; lockstep: units/s)."""
    meta = read_kv(os.path.join(results, "meta.txt"))
    reps = sorted(
        int(m.group(1))
        for m in (re.match(r"rep(\d+)-server\.out$", f) for f in os.listdir(results))
        if m
    )
    conc = [int(c) for c in meta.get("concurrencies", "").split(",") if c]
    realtime = [
        serving_rows(os.path.join(results, f"rep{r}-realtime.out"), 12) for r in reps
    ]
    lockstep = [
        serving_rows(os.path.join(results, f"rep{r}-lockstep.out"), 9) for r in reps
    ]
    startup = [
        read_text(os.path.join(results, f"rep{r}-server.out")).split()[-1] for r in reps
    ]
    tracebacks = [
        read_text(os.path.join(results, f"rep{r}-tracebacks.txt")).strip()
        for r in reps
        if os.path.isfile(os.path.join(results, f"rep{r}-tracebacks.txt"))
    ]
    print(
        f"Serving gate: {meta.get('tag')} {meta.get('sha', '')[:9]}, graphs={meta.get('graphs')}, {len(reps)} reps, startup {' / '.join(startup)}, tracebacks {' / '.join(tracebacks)}"
    )
    print()
    print(
        f"| c | miss % ({len(reps)} reps) | unit lag p95 ms | reply first-audio p50 ms | underrun % | lockstep units/s per session | sessions done |"
    )
    print("|---|---|---|---|---|---|---|")
    means = []
    for c in conc:

        def col(tables, i):
            return [t[c][i] if c in t else "–" for t in tables]

        miss = [x.rstrip("%") for x in col(realtime, 4)]
        lag, reply, under, ups, done = (
            col(realtime, 6),
            col(realtime, 9),
            [x.rstrip("%") for x in col(realtime, 11)],
            col(lockstep, 8),
            col(realtime, 1),
        )
        print(
            f"| {c} | {' / '.join(miss)} | {' / '.join(lag)} | {' / '.join(reply)} | {' / '.join(under)} | {' / '.join(ups)} | {' '.join(done)} |"
        )
        numeric = [float(x) for x in miss if x != "–"]
        means.append(
            (
                c,
                numeric,
                [float(x) for x in lag if x != "–"],
                [float(x) for x in ups if x != "–"],
            )
        )
    print()
    print(
        "Means (miss / lag p95 / lockstep units/s per session): "
        + "; ".join(
            f"c={c}: {fmt(st.mean(m) if m else None)} % / {fmt(st.mean(lag) if lag else None, 0)} ms / {fmt(st.mean(u) if u else None)}"
            for c, m, lag, u in means
        )
    )


def extract_units(results):
    """Per-unit speak flags from the recorder traces: a unit speaks when it carries output text or audio (the rule of the offline reference)."""
    units = {}
    for trace in sorted(
        glob.glob(
            os.path.join(results, "*", "samples", "*", "*", "input", "continuous.jsonl")
        )
    ):
        subset, name = trace.split("/samples/")[1].split("/")[:2]
        flags, speak = [], False
        for line in read_lines(trace):
            row = json.loads(line)
            kind = row["event"].get("type", "")
            if row["direction"] != "receive":
                continue
            elif kind in (
                "response.output_audio.delta",
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
            ):
                speak = speak or bool(row["event"].get("delta"))
            elif kind == "sglang.unit.done":
                flags.append(speak)
                speak = False
            else:
                pass
        units[f"{subset}/{name}"] = flags
    return units


def report_agree(results, ref_path):
    ours = extract_units(results)
    write_json(ours, os.path.join(results, "units.json"))
    ref = json.loads(read_text(ref_path))
    by = collections.defaultdict(collections.Counter)
    for key, flags in ref.items():
        subset = key.split("/")[0]
        if key not in ours:
            by[subset]["missing"] += 1
            continue
        else:
            got = ours[key]
        n = min(len(got), len(flags))
        agree = sum(x == y for x, y in zip(got, flags))
        c = by[subset]
        c["samples"] += 1
        c["units"] += n
        c["agree"] += agree
        c["exact"] += agree == n and len(got) == len(flags)
    total = collections.Counter()
    tb = os.path.join(results, "tracebacks.txt")
    print(
        f"Per-unit agreement with the offline reference ({os.path.basename(ref_path)}); recorded samples {len(ours)}, server tracebacks {read_text(tb).strip() if os.path.isfile(tb) else '?'}."
    )
    print()
    print("| subset | samples | units | agreement | exact samples | missing |")
    print("|---|---|---|---|---|---|")
    for subset, c in sorted(by.items()):
        total.update(c)
        print(
            f"| {subset} | {c['samples']} | {c['units']} | {100 * c['agree'] / max(c['units'], 1):.2f} % | {c['exact']} | {c['missing']} |"
        )
    print(
        f"| **total** | {total['samples']} | {total['units']} | **{100 * total['agree'] / max(total['units'], 1):.2f} %** | {total['exact']} | {total['missing']} |"
    )


def report_unit(results):
    for log in sorted(
        glob.glob(os.path.join(results, "*-cpu.log"))
        + glob.glob(os.path.join(results, "*-gpu.log"))
    ):
        lines = read_lines(log)
        summary = next(
            (
                line
                for line in reversed(lines)
                if re.search(r"\d+ (passed|failed)", line)
            ),
            "no summary",
        )
        failures = [line for line in lines if line.startswith(("FAILED ", "ERROR "))]
        print(f"**{os.path.basename(log)[:-4]}**: {summary.strip('= ')}")
        for line in failures:
            print(f"- {line}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            "run",
            "ladder",
            "sweep",
            "stages",
            "perception",
            "serving",
            "agree",
            "unit",
        ),
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--miss-threshold", type=float)
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--ref", default=os.path.join(HERE, "ref_units.json"))
    args = parser.parse_args()
    path = args.paths[0]
    if args.command == "run":
        summary = summarize_run(path, args.budget_ms)
        write_json(summary, os.path.join(path, "run.json"), indent=1)
        print(chain_line(summary))
    elif args.command == "ladder":
        report_ladder(path, args.budget_ms)
    elif args.command == "sweep":
        report_sweep(args.paths, args.miss_threshold, args.budget_ms)
    elif args.command == "stages":
        report_stages(args.paths, args.budget_ms)
    elif args.command == "perception":
        report_perception(path)
    elif args.command == "serving":
        report_serving(path)
    elif args.command == "agree":
        report_agree(path, args.ref)
    else:
        report_unit(path)


if __name__ == "__main__":
    main()
else:
    pass
