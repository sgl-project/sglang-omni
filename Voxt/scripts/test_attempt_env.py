#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "attempt_env.py"
AC = "Now drawing from 'AC Power'"


class AttemptEnvTests(unittest.TestCase):
    def run_env(self, before_others, before_power, before_pressure, after_others, after_pressure):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "env.json"
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(out), before_others, before_power,
                 before_pressure, after_others, after_pressure],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return json.loads(out.read_text())

    def test_clean_attempt_has_no_reasons(self):
        env = self.run_env("", AC, "1", "", "1")
        self.assertEqual(env["environment_reasons"], [])
        self.assertEqual(env["peak_pressure"], "normal")

    def test_field_mapping_keeps_each_argument_in_place(self):
        env = self.run_env("12345 qwen3_asr_server", AC, "1", "67890 silero_vad_server", "2")
        self.assertIn("qwen3_asr_server", env["before"]["others"])
        self.assertIn("silero_vad_server", env["after"]["others"])
        self.assertEqual(env["before"]["pressure"], "1")
        self.assertEqual(env["after"]["pressure"], "2")
        self.assertEqual(env["peak_pressure"], "warn")
        self.assertEqual(len(env["environment_reasons"]), 2)
        self.assertNotIn("not on AC power", env["environment_reasons"])

    def test_no_ac_reports_in_both_samples(self):
        env = self.run_env("", "Now drawing from 'Battery Power'", "1", "", "1")
        self.assertEqual(env["environment_reasons"].count("not on AC power"), 2)

    def test_after_contender_is_a_reason(self):
        env = self.run_env("", AC, "1", "999 moss_transcribe_diarize_server", "1")
        self.assertTrue(any("moss_transcribe" in r for r in env["environment_reasons"]))


if __name__ == "__main__":
    unittest.main()
