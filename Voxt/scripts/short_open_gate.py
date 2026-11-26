#!/usr/bin/env python3
"""Gate a short-open latency run (mac-latency.sh): medians only when every
per-attempt environment is clean and the group completes.

Each attempt runs its own go-test process with its own before/after
environment record (build/latency-attempt-N.json) and its own latency rows
(build/latency-round-N.jsonl), so a contender or pressure spike that
appears and disappears inside one attempt cannot pass as clean.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_loop as loop


def medians(rows, key):
    values = sorted(row[key] for row in rows if isinstance(row.get(key), (int, float)))
    return values[len(values) // 2] if values else None


def main(argv):
    if len(argv) != 3:
        print("usage: short_open_gate.py <build_dir> <rounds>", file=sys.stderr)
        return 2
    build, rounds = Path(argv[1]), int(argv[2])
    returncode = 0
    rc_file = build / "latency-returncode.txt"
    if rc_file.exists():
        returncode = int(rc_file.read_text().strip() or 0)
    observations = []
    for attempt in range(1, rounds + 1):
        env_path = build / f"latency-attempt-{attempt}.json"
        env = (
            json.loads(env_path.read_text())
            if env_path.exists()
            else {"environment_reasons": ["no environment record"], "peak_pressure": "warn"}
        )
        rows_path = build / f"latency-round-{attempt}.jsonl"
        rows = (
            [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            if rows_path.exists()
            else []
        )
        pair = [r for r in rows if r.get("scenario") in ("cold", "warm")]
        pair.sort(key=lambda r: r.get("scenario", ""))
        kept = loop.short_open_attempt_kept(
            returncode,
            list(env.get("environment_reasons", [])),
            env.get("peak_pressure", ""),
            pair,
        )
        observations.append(
            {
                "attempt": attempt,
                "kept": kept,
                "rows": pair,
                "environment_reasons": env.get("environment_reasons", []),
            }
        )
    if not loop.short_open_group_complete(observations):
        detail = [
            {
                "attempt": o["attempt"],
                "kept": o["kept"],
                "reasons": o["environment_reasons"],
                "scenarios": [r.get("scenario") for r in o["rows"]],
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
                [r for o in observations for r in o["rows"] if r.get("scenario") == scenario],
                key,
            )
            if med is not None:
                out[f"{scenario}_{key}_median"] = med
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
