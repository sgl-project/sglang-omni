#!/usr/bin/env python3
"""Run one short-open attempt and record its environment evidence.

Before, during, and after samples keep a contender or pressure spike that
starts and ends inside the Go command visible to the kept-attempt gate.
"""
import json
import subprocess
import sys
from pathlib import Path

CONTENDER_PATTERN = "qwen3_asr_server|silero_vad_server"
SAMPLE_INTERVAL_SECONDS = 0.1


def command_output(arguments: list[str], fallback: str = "") -> str:
    try:
        completed = subprocess.run(
            arguments, capture_output=True, text=True, check=False
        )
    except OSError:
        return fallback
    return completed.stdout.strip() or fallback


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(
            "usage: attempt_env.py <attempt.json> <samples.jsonl> <latency.log> <command...>",
            file=sys.stderr,
        )
        return 2
    environment_path = Path(argv[1])
    samples_path = Path(argv[2])
    log_path = Path(argv[3])
    command = argv[4:]
    samples_path.write_text("", encoding="utf-8")
    samples: list[dict[str, str]] = []

    def record(phase: str) -> None:
        sample = {
            "phase": phase,
            "others": command_output(["pgrep", "-fl", CONTENDER_PATTERN]),
            "power": command_output(["pmset", "-g", "ps"]),
            "pressure": command_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], "0"
            ),
        }
        samples.append(sample)
        with samples_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(sample) + "\n")
            output.flush()

    record("before")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while True:
            record("during")
            try:
                returncode = process.wait(timeout=SAMPLE_INTERVAL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
    record("after")

    reasons = []
    for sample in samples:
        phase = sample["phase"]
        if sample["others"]:
            reasons.append(f"contenders present {phase} run")
        if "AC Power" not in sample["power"]:
            reasons.append(f"not on AC power {phase} run")
    environment = {
        "returncode": returncode,
        "environment_reasons": list(dict.fromkeys(reasons)),
        "peak_pressure": (
            "normal" if all(sample["pressure"] == "1" for sample in samples) else "warn"
        ),
        "samples": samples,
    }
    environment_path.write_text(
        json.dumps(environment, indent=1) + "\n", encoding="utf-8"
    )
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
