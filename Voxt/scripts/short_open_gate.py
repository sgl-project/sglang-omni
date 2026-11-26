#!/usr/bin/env python3
"""Gate a short-open latency run (mac-latency.sh).

Each attempt has its own latency rows and before/during/after environment
record. Only three kept attempts with one shared text hash produce medians.
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
    if len(argv) != 3:
        print("usage: short_open_gate.py <build_dir> <rounds>", file=sys.stderr)
        return 2
    build, rounds = Path(argv[1]), int(argv[2])
    observations = []
    for attempt in range(1, rounds + 1):
        environment_path = build / f"latency-attempt-{attempt}.json"
        environment = (
            json.loads(environment_path.read_text())
            if environment_path.exists()
            else {
                "returncode": 1,
                "environment_reasons": ["no environment record"],
                "peak_pressure": "warn",
                "samples": [],
            }
        )
        rows_path = build / f"latency-round-{attempt}.jsonl"
        attempt_rows = (
            [
                json.loads(line)
                for line in rows_path.read_text().splitlines()
                if line.strip()
            ]
            if rows_path.exists()
            else []
        )
        (build / f"latency-attempt-{attempt}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in attempt_rows),
            encoding="utf-8",
        )
        pair = [row for row in attempt_rows if row.get("scenario") in ("cold", "warm")]
        pair.sort(key=lambda row: row.get("scenario", ""))
        kept = loop.short_open_attempt_kept(
            int(environment.get("returncode", 1)),
            list(environment.get("environment_reasons", [])),
            environment.get("peak_pressure", ""),
            pair,
        )
        observations.append(
            {
                "attempt": attempt,
                "returncode": int(environment.get("returncode", 1)),
                "environment_invalid_reason": environment.get(
                    "environment_reasons", []
                ),
                "peak_memory_pressure": environment.get("peak_pressure", ""),
                "kept": kept,
                "rows": attempt_rows,
            }
        )
    loop.write_json(build / "short-open-observations.json", observations)
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
