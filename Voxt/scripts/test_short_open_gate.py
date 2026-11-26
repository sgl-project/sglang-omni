#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "short_open_gate.py"


def row(scenario, **overrides):
    base = {
        "scenario": scenario,
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
    def run_gate(self, build_rows, envs, returncode=0, rounds=3):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            for attempt, rows in enumerate(build_rows, 1):
                if rows is not None:
                    (build / f"latency-round-{attempt}.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in rows)
                    )
                (build / f"latency-attempt-{attempt}.json").write_text(
                    json.dumps(
                        envs.get(
                            attempt,
                            {"environment_reasons": [], "peak_pressure": "normal"},
                        )
                    )
                )
            (build / "latency-returncode.txt").write_text(str(returncode))
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(build), str(rounds)],
                capture_output=True,
                text=True,
            )

    def pair(self, **bad):
        return [row("cold", **bad), row("warm", **bad)]

    def test_complete_group_prints_medians(self):
        proc = self.run_gate([self.pair(), self.pair(), self.pair()], {})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            json.loads(proc.stdout)["cold_first_partial_ms_median"], 1200.0
        )

    def test_transient_contender_inside_one_attempt_rejects_it(self):
        envs = {
            1: {"environment_reasons": [], "peak_pressure": "normal"},
            2: {
                "environment_reasons": ["contender: qwen3_asr_server"],
                "peak_pressure": "normal",
            },
            3: {"environment_reasons": [], "peak_pressure": "normal"},
        }
        proc = self.run_gate([self.pair(), self.pair(), self.pair()], envs)
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("median", proc.stdout)
        detail = json.loads(proc.stdout)
        self.assertEqual(detail["attempts"][1]["kept"], False)
        self.assertEqual(detail["attempts"][0]["kept"], True)

    def test_pressure_spike_rejects_its_attempt(self):
        envs = {2: {"environment_reasons": [], "peak_pressure": "warn"}}
        proc = self.run_gate([self.pair(), self.pair(), self.pair()], envs)
        self.assertEqual(proc.returncode, 1)

    def test_missing_attempt_rows_rejects(self):
        proc = self.run_gate([self.pair(), None, self.pair()], {})
        self.assertEqual(proc.returncode, 1)

    def test_failed_process_rejects(self):
        proc = self.run_gate([self.pair(), self.pair(), self.pair()], {}, returncode=1)
        self.assertEqual(proc.returncode, 1)

    def test_text_drift_between_attempts_rejects(self):
        proc = self.run_gate(
            [
                self.pair(),
                [row("cold", text_sha256="b" * 64), row("warm", text_sha256="b" * 64)],
                self.pair(),
            ],
            {},
        )
        self.assertEqual(proc.returncode, 1)

    def test_no_partial_attempt_rejects(self):
        proc = self.run_gate(
            [self.pair(first_partial_ms=None), self.pair(), self.pair()], {}
        )
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
