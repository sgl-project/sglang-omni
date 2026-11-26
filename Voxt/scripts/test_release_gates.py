#!/usr/bin/env python3
"""Release gates for the short-open latency rows (review 138 Must 4).

Every kept group must show the native min_preview_samples it ran with
(the actual value, or null with an unavailable_reason; a numeric 0, a
missing field and a missing reason are all invalid) and the distinct
text hashes the kept decision was written against.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_loop as loop  # noqa: E402


def row(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "scenario": "cold",
        "text_sha256": "a" * 64,
        "min_preview_samples": 20000,
    }
    base.update(overrides)
    return base


class MinPreviewGateTests(unittest.TestCase):
    def test_numeric_zero_is_invalid(self):
        self.assertFalse(loop.min_preview_samples_valid(row(min_preview_samples=0)))

    def test_missing_field_is_invalid(self):
        broken = row()
        broken.pop("min_preview_samples")
        self.assertFalse(loop.min_preview_samples_valid(broken))

    def test_null_without_reason_is_invalid(self):
        self.assertFalse(loop.min_preview_samples_valid(row(min_preview_samples=None)))
        self.assertFalse(
            loop.min_preview_samples_valid(
                row(
                    min_preview_samples=None,
                    min_preview_samples_unavailable_reason="  ",
                )
            )
        )

    def test_legal_null_passes(self):
        self.assertTrue(
            loop.min_preview_samples_valid(
                row(
                    min_preview_samples=None,
                    min_preview_samples_unavailable_reason="no native moss_preview_timing event for this model or runtime",
                )
            )
        )

    def test_actual_value_passes(self):
        self.assertTrue(loop.min_preview_samples_valid(row(min_preview_samples=1250)))


class TextHashGateTests(unittest.TestCase):
    def test_rows_gate_requires_hashes(self):
        self.assertFalse(loop.latency_rows_valid([row(text_sha256="")]))
        self.assertTrue(loop.latency_rows_valid([row()]))

    def test_distinct_hashes_cold_warm_pair(self):
        same = "b" * 64
        rows = [
            row(scenario="cold", text_sha256=same),
            row(scenario="warm", text_sha256=same),
        ]
        self.assertEqual(loop.distinct_text_hashes(rows), [same])
        rows[1]["text_sha256"] = "c" * 64
        self.assertEqual(loop.distinct_text_hashes(rows), sorted({same, "c" * 64}))


class ShortOpenGateTests(unittest.TestCase):
    def observation(self, kept=True, scenario_hashes=None) -> dict[str, object]:
        rows = [
            row(scenario="cold", text_sha256=(scenario_hashes or ["a" * 64])[0]),
            row(
                scenario="warm",
                text_sha256=(scenario_hashes or ["a" * 64, "a" * 64])[-1],
            ),
        ]
        return {"kept": kept, "rows": rows}

    def test_attempt_gates(self):
        rows = self.observation()["rows"]
        self.assertTrue(loop.short_open_attempt_kept(0, [], "normal", rows))
        self.assertFalse(loop.short_open_attempt_kept(1, [], "normal", rows))
        self.assertFalse(loop.short_open_attempt_kept(0, ["load"], "normal", rows))
        self.assertFalse(loop.short_open_attempt_kept(0, [], "warning", rows))
        self.assertFalse(loop.short_open_attempt_kept(0, [], "normal", rows[:1]))
        self.assertFalse(
            loop.short_open_attempt_kept(
                0, [], "normal", [row(scenario="cold"), row(scenario="cold")]
            )
        )

    def test_attempt_rejects_cold_warm_hash_change(self):
        rows = self.observation()["rows"]
        rows[1]["text_sha256"] = "c" * 64
        self.assertEqual(len(loop.distinct_text_hashes(rows)), 2)
        self.assertFalse(loop.short_open_attempt_kept(0, [], "normal", rows))

    def test_group_complete_requires_three_kept_observations(self):
        self.assertFalse(loop.short_open_group_complete([self.observation()]))
        self.assertFalse(
            loop.short_open_group_complete([self.observation() for _ in range(4)])
        )
        complete = [self.observation() for _ in range(3)]
        self.assertTrue(loop.short_open_group_complete(complete))

    def test_group_complete_requires_every_kept(self):
        observations = [self.observation() for _ in range(2)]
        observations.append(self.observation(kept=False))
        # A third invalid attempt stops the group: no medians.
        self.assertFalse(loop.short_open_group_complete(observations))
        self.assertFalse(
            loop.short_open_group_complete(
                observations[:1] + observations[2:] + [self.observation()]
            )
        )

    def test_group_complete_rejects_cross_attempt_drift_and_writes_consistency(self):
        observations = []
        for letter in "abc":
            obs = self.observation(scenario_hashes=[letter * 64, letter * 64])
            self.assertTrue(loop.short_open_attempt_kept(0, [], "normal", obs["rows"]))
            observations.append(obs)
        self.assertFalse(loop.short_open_group_complete(observations))
        consistency = loop.text_consistency(observations)
        self.assertFalse(consistency["shared_text_hash"])
        self.assertEqual(
            consistency["scenarios"]["cold"], ["a" * 64, "b" * 64, "c" * 64]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text-consistency.json"
            loop.write_text_consistency(observations, path)
            self.assertEqual(json.loads(path.read_text()), consistency)

    def test_consistency_rejects_missing_hash(self):
        observations = [{"rows": [row(text_sha256=""), row(scenario="warm")]}]
        self.assertFalse(loop.text_consistency(observations)["shared_text_hash"])

    def test_switch_text_is_separate_from_short_open_consistency(self):
        observations = []
        for _ in range(3):
            obs = self.observation()
            obs["rows"].append(row(scenario="switch", text_sha256="z" * 64))
            observations.append(obs)
        self.assertTrue(loop.short_open_group_complete(observations))
        consistency = loop.text_consistency(observations)
        self.assertTrue(consistency["shared_text_hash"])
        self.assertEqual(consistency["scenarios"]["switch"], ["z" * 64] * 3)

    def test_group_complete_rejects_empty_hash(self):
        bad = self.observation()
        bad["rows"][1]["text_sha256"] = ""
        self.assertFalse(
            loop.short_open_group_complete(
                [bad, self.observation(), self.observation()]
            )
        )


class ShortOpenRunnerTests(unittest.TestCase):
    def run_latency_shell(
        self,
        drift=False,
        rounds=None,
        pressure_after=False,
        fail_before_rows=False,
        contender_after=False,
        contender_during=False,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            source = Path(__file__).resolve().parent
            for name in ("mac-latency.sh", "short_open_gate.py", "profile_loop.py"):
                shutil.copy2(source / name, scripts / name)
            optional_monitor = source / "attempt_env.py"
            if optional_monitor.exists():
                shutil.copy2(optional_monitor, scripts / optional_monitor.name)
            binaries = root / "bin"
            binaries.mkdir()

            def command(name, contents):
                path = binaries / name
                path.write_text(contents, encoding="utf-8")
                path.chmod(0o755)

            command("uname", "#!/bin/sh\nprintf 'Darwin\\n'\n")
            command("pmset", "#!/bin/sh\nprintf \"Now drawing from 'AC Power'\\n\"\n")
            command(
                "pgrep",
                "#!/bin/sh\nif [ -f build/fake-contender-active ]; then printf '123 /tmp/qwen3_asr_server --supervised\\n'; exit 0; fi\nif [ -f build/latency.log ] && [ \"${LATENCY_FAKE_CONTENDER_AFTER:-0}\" = 1 ]; then printf '123 /tmp/qwen3_asr_server --supervised\\n'; exit 0; fi\nexit 1\n",
            )
            command(
                "sysctl",
                "#!/bin/sh\nif [ -f build/latency.log ] && [ \"${LATENCY_FAKE_PRESSURE_AFTER:-0}\" = 2 ]; then printf '2\\n'; else printf '1\\n'; fi\n",
            )
            command(
                "go",
                """#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("LATENCY_FAKE_FAIL_BEFORE_ROWS") == "1":
    print("panic: fake latency setup failed", file=sys.stderr)
    sys.exit(1)

rounds = int(os.environ["VOXTNEW_LATENCY_ROUNDS"])
first_attempt = int(os.environ.get("VOXTNEW_LATENCY_ATTEMPT", "1"))
rows = []
for attempt in range(first_attempt, first_attempt + rounds):
    if os.environ.get("LATENCY_FAKE_CONTENDER_DURING") == "1" and attempt == 2:
        contender = Path("build/fake-contender-active")
        contender.touch()
        samples = Path(f"build/latency-attempt-{attempt}-samples.jsonl")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if samples.exists() and "qwen3_asr_server" in samples.read_text():
                break
            time.sleep(0.01)
        contender.unlink()
    for scenario in ("cold", "warm", "switch"):
        text_hash = "z" * 64 if scenario == "switch" else "a" * 64
        if os.environ.get("LATENCY_FAKE_DRIFT") == "1" and attempt == 2 and scenario == "warm":
            text_hash = "b" * 64
        rows.append({
            "attempt": attempt,
            "round": attempt,
            "scenario": scenario,
            "text": "fixed text",
            "text_sha256": text_hash,
            "min_preview_samples": 20000,
            "first_partial_ms": 1200.0,
            "stop_to_final_ms": 900.0,
            "stop_to_insert_ms": 1000.0,
            "stop_to_done_ms": 1100.0,
        })
Path(os.environ["VOXTNEW_LATENCY_OUT"]).write_text(
    "".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8"
)
print("    mac_e2e_latency_test.go:257: MEASURE latency cold n=3 valid")
""",
            )
            environment = os.environ.copy()
            environment["PATH"] = str(binaries) + os.pathsep + environment["PATH"]
            environment["VOXTNEW_MAIN_SHA"] = "test-source-sha"
            environment["VOXTNEW_HUB"] = str(root / "empty-hub")
            environment["LATENCY_FAKE_DRIFT"] = "1" if drift else "0"
            environment["LATENCY_FAKE_PRESSURE_AFTER"] = "2" if pressure_after else "1"
            environment["LATENCY_FAKE_FAIL_BEFORE_ROWS"] = (
                "1" if fail_before_rows else "0"
            )
            environment["LATENCY_FAKE_CONTENDER_AFTER"] = (
                "1" if contender_after else "0"
            )
            environment["LATENCY_FAKE_CONTENDER_DURING"] = (
                "1" if contender_during else "0"
            )
            arguments = [str(scripts / "mac-latency.sh"), "accurate", "fast"]
            if rounds is not None:
                arguments.append(str(rounds))
            process = subprocess.run(
                arguments,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
            )
            attempt_path = root / "build" / "latency-attempt-2.jsonl"
            environment_path = root / "build" / "latency-attempt-2.json"
            observations_path = root / "build" / "short-open-observations.json"
            return {
                "process": process,
                "attempt": attempt_path.read_text() if attempt_path.exists() else None,
                "environment": (
                    environment_path.read_text() if environment_path.exists() else None
                ),
                "observations": (
                    observations_path.read_text()
                    if observations_path.exists()
                    else None
                ),
            }

    def test_shell_emits_go_summary_and_attempt_evidence_for_complete_group(self):
        result = self.run_latency_shell()
        self.assertEqual(
            result["process"].returncode,
            0,
            result["process"].stdout + result["process"].stderr,
        )
        self.assertIn("MEASURE latency", result["process"].stdout)
        self.assertIsNotNone(result["attempt"])
        self.assertEqual(len(result["attempt"].splitlines()), 3)
        observations = json.loads(result["observations"])
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(observation["kept"] for observation in observations))

    def test_shell_rejects_invalid_attempt_without_measure_and_keeps_evidence(self):
        result = self.run_latency_shell(drift=True)
        self.assertEqual(result["process"].returncode, 1)
        self.assertNotIn("MEASURE latency", result["process"].stdout)
        self.assertIn('"complete": false', result["process"].stdout)
        self.assertIsNotNone(result["attempt"])
        observations = json.loads(result["observations"])
        self.assertFalse(observations[1]["kept"])

    def test_shell_rejects_pressure_that_rises_during_the_run(self):
        result = self.run_latency_shell(pressure_after=True)
        self.assertEqual(result["process"].returncode, 1)
        self.assertNotIn("MEASURE latency", result["process"].stdout)
        observations = json.loads(result["observations"])
        self.assertEqual(observations[0]["peak_memory_pressure"], "warn")

    def test_shell_rejects_contender_that_appears_during_the_run(self):
        result = self.run_latency_shell(contender_after=True)
        self.assertEqual(result["process"].returncode, 1)
        self.assertNotIn("MEASURE latency", result["process"].stdout)
        observations = json.loads(result["observations"])
        self.assertIn(
            "contenders present after run",
            observations[0]["environment_invalid_reason"],
        )

    def test_shell_rejects_contender_that_exits_before_the_after_check(self):
        result = self.run_latency_shell(contender_during=True)
        self.assertEqual(result["process"].returncode, 1)
        self.assertNotIn("MEASURE latency", result["process"].stdout)
        observations = json.loads(result["observations"])
        self.assertTrue(observations[0]["kept"])
        self.assertFalse(observations[1]["kept"])
        self.assertTrue(observations[2]["kept"])
        self.assertIn(
            "contenders present during run",
            observations[1]["environment_invalid_reason"],
        )
        environment = json.loads(result["environment"])
        self.assertEqual(
            [sample["phase"] for sample in environment["samples"][:2]],
            ["before", "during"],
        )
        self.assertEqual(environment["samples"][-1]["phase"], "after")

    def test_shell_preserves_empty_evidence_and_log_when_go_fails(self):
        result = self.run_latency_shell(fail_before_rows=True)
        output = result["process"].stdout + result["process"].stderr
        self.assertEqual(result["process"].returncode, 1)
        self.assertIn("panic: fake latency setup failed", output)
        self.assertIn('"complete": false', output)
        self.assertNotIn("Traceback", output)
        self.assertNotIn("FileNotFoundError", output)
        self.assertNotIn("MEASURE latency", output)
        self.assertEqual(result["attempt"], "")
        observations = json.loads(result["observations"])
        self.assertEqual(len(observations), 3)
        for observation in observations:
            self.assertEqual(observation["returncode"], 1)
            self.assertFalse(observation["kept"])
            self.assertEqual(observation["rows"], [])

    def test_shell_rejects_non_three_rounds(self):
        result = self.run_latency_shell(rounds=1)
        self.assertEqual(result["process"].returncode, 2)
        self.assertIn("ROUNDS must be 3", result["process"].stderr)


if __name__ == "__main__":
    unittest.main()
