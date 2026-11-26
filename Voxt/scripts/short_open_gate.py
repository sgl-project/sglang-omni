#!/usr/bin/env python3
"""Gate a short-open latency run (mac-latency.sh): medians only when the
group is complete (mac_e2e_latency_test rows + environment evidence).

Input: a JSONL of latencyResult rows (VOXTNEW_LATENCY_OUT), a preflight
JSON {returncode, environment_reasons, peak_pressure}, and the rounds to
expect. Every round (attempt) must be kept: clean process, clean
environment, cold+warm scenarios, evidence gates, one shared text hash.
Only with three kept attempts are medians printed; otherwise the runner
exits 1 with the reason and no median is shown.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_loop as loop


def medians(rows, key):
    values = sorted(row[key] for row in rows if isinstance(row.get(key), (int, float)))
    if not values:
        return None
    return values[len(values) // 2]


def main(argv):
    if len(argv) != 4:
        print(
            "usage: short_open_gate.py <latency.jsonl> <preflight.json> <rounds>",
            file=sys.stderr,
        )
        return 2
    rows_path, preflight_path, rounds = Path(argv[1]), Path(argv[2]), int(argv[3])
    preflight = json.loads(preflight_path.read_text())
    rows = (
        [
            json.loads(line)
            for line in rows_path.read_text().splitlines()
            if line.strip()
        ]
        if rows_path.exists()
        else []
    )
    by_attempt: dict[int, list[dict]] = {}
    for row in rows:
        by_attempt.setdefault(int(row.get("attempt", row.get("round", 0))), []).append(
            row
        )
    reasons = list(preflight.get("environment_reasons", []))
    observations = []
    for attempt in range(1, rounds + 1):
        attempt_rows = by_attempt.get(attempt, [])
        (rows_path.parent / f"latency-attempt-{attempt}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in attempt_rows),
            encoding="utf-8",
        )
        pair = [row for row in attempt_rows if row.get("scenario") in ("cold", "warm")]
        pair.sort(key=lambda row: row.get("scenario", ""))
        kept = loop.short_open_attempt_kept(
            int(preflight.get("returncode", 1)),
            reasons,
            preflight.get("peak_pressure", ""),
            pair,
        )
        observations.append(
            {
                "attempt": attempt,
                "returncode": int(preflight.get("returncode", 1)),
                "environment_invalid_reason": reasons,
                "peak_memory_pressure": preflight.get("peak_pressure", ""),
                "kept": kept,
                "rows": attempt_rows,
            }
        )
    loop.write_json(rows_path.parent / "short-open-observations.json", observations)
    if not loop.short_open_group_complete(observations):
        detail = [
            {
                "attempt": o["attempt"],
                "kept": o["kept"],
                "scenarios": [
                    row.get("scenario")
                    for row in o["rows"]
                    if row.get("scenario") in ("cold", "warm")
                ],
            }
            for o in observations
        ]
        print(json.dumps({"complete": False, "attempts": detail}, indent=1))
        return 1
    out = {"complete": True, "attempts": rounds}
    for key in (
        "first_partial_ms",
        "stop_to_final_ms",
        "stop_to_insert_ms",
        "stop_to_done_ms",
    ):
        for scenario in ("cold", "warm"):
            med = medians(
                [
                    r
                    for o in observations
                    for r in o["rows"]
                    if r.get("scenario") == scenario
                ],
                key,
            )
            if med is not None:
                out[f"{scenario}_{key}_median"] = med
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
