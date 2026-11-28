#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "short_open_gate.py"


def row(attempt, scenario, **overrides):
    base = {
        "scenario": scenario,
        "attempt": attempt,
        "text": "fixed text",
        "text_sha256": "a" * 64,
        "min_preview_samples": 20000,
        "first_partial_ms": 1200.0,
        "stop_to_final_ms": 900.0,
        "stop_to_insert_ms": 1000.0,
        "stop_to_done_ms": 1100.0,
    }
    base.update(overrides)
    return base


class ShortOpenGateRunnerTests(unittest.TestCase):
    def run_gate(self, rows, preflight=None, rounds=3):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            environment = preflight or {
                "returncode": 0,
                "environment_reasons": [],
                "peak_pressure": "normal",
            }
            for attempt in range(1, rounds + 1):
                attempt_rows = [row for row in rows if row["attempt"] == attempt]
                (build / f"latency-round-{attempt}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in attempt_rows)
                )
                (build / f"latency-attempt-{attempt}.json").write_text(
                    json.dumps(environment)
                )
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(build), str(rounds)],
                capture_output=True,
                text=True,
            )

    def full_rows(self, bad_attempt=None, **bad):
        rows = []
        for attempt in (1, 2, 3):
            for scenario in ("cold", "warm"):
                rows.append(
                    row(attempt, scenario, **(bad if attempt == bad_attempt else {}))
                )
        return rows

    def test_complete_group_prints_medians(self):
        proc = self.run_gate(self.full_rows())
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = json.loads(proc.stdout)
        self.assertTrue(out["complete"])
        self.assertEqual(out["cold_first_partial_ms_median"], 1200.0)

    def test_one_invalid_attempt_means_no_median(self):
        rows = self.full_rows()
        for item in rows:
            if item["attempt"] == 2 and item["scenario"] == "warm":
                item["text_sha256"] = "b" * 64
        proc = self.run_gate(rows)
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("median", proc.stdout)
        self.assertIn('"kept": false', proc.stdout)

    def test_environment_reason_rejected(self):
        proc = self.run_gate(
            self.full_rows(),
            {
                "returncode": 0,
                "environment_reasons": ["contender: qwen3"],
                "peak_pressure": "normal",
            },
        )
        self.assertEqual(proc.returncode, 1)

    def test_pressure_rejected(self):
        proc = self.run_gate(
            self.full_rows(),
            {"returncode": 0, "environment_reasons": [], "peak_pressure": "warn"},
        )
        self.assertEqual(proc.returncode, 1)

    def test_failed_process_rejected(self):
        proc = self.run_gate(
            self.full_rows(),
            {"returncode": 1, "environment_reasons": [], "peak_pressure": "normal"},
        )
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
