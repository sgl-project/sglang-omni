#!/usr/bin/env python3
"""Release gates for the short-open latency rows (review 138 Must 4).

Every kept group must show the native min_preview_samples it ran with
(the actual value, or null with an unavailable_reason; a numeric 0, a
missing field and a missing reason are all invalid) and the distinct
text hashes the kept decision was written against.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

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

    def test_group_complete_requires_every_kept(self):
        observations = [self.observation() for _ in range(2)]
        observations.append(self.observation(kept=False))
        # A third invalid attempt stops the group: no medians.
        self.assertFalse(loop.short_open_group_complete(observations))
        self.assertFalse(
            loop.short_open_group_complete(observations[:2][:1] + observations[2:])
        )
        complete = [self.observation() for _ in range(3)]
        self.assertTrue(loop.short_open_group_complete(complete))

    def test_group_complete_rejects_empty_hash(self):
        bad = self.observation()
        bad["rows"][1]["text_sha256"] = ""
        self.assertFalse(loop.short_open_group_complete([bad, self.observation()]))


class EnvironmentProbeTests(unittest.TestCase):
    def test_ps_eperm_degrades_rss_to_null_and_preserves_host_environment(self):
        def output(arguments, **kwargs):
            if arguments[0] == "ps":
                raise OSError(1, "Operation not permitted")
            if arguments[:2] == ["/usr/bin/pmset", "-g"]:
                return subprocess.CompletedProcess(
                    args=arguments,
                    returncode=0,
                    stdout=(
                        "Now drawing from 'AC Power'\n"
                        if arguments[2] == "batt"
                        else "No thermal warning\n"
                    ),
                    stderr="",
                )
            if arguments[:2] == ["/usr/sbin/ioreg", "-c"]:
                return subprocess.CompletedProcess(
                    args=arguments,
                    returncode=0,
                    stdout='"HIDIdleTime" = 300000000000\n',
                    stderr="",
                )
            if arguments[:2] == ["/usr/bin/pgrep", "-fl"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="", stderr=""
                )
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="123\n", stderr=""
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            self.assertIsNone(loop.process_rss_bytes("qwen3_asr_server"))
            detail = loop.process_rss_detail("qwen3_asr_server")
            environment = loop.host_environment()
        self.assertIsNone(detail["value"])
        self.assertIn("Operation not permitted", str(detail["unavailable_reason"]))
        self.assertEqual(environment["power_source"], "AC Power")
        self.assertEqual(environment["idle_seconds"], 300)
        self.assertEqual(environment["contenders"], [])

    def test_rss_detail_distinguishes_missing_process_and_nonzero_exit(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                if arguments[2] == "qwen3_asr_server":
                    return subprocess.CompletedProcess(
                        args=arguments, returncode=1, stdout="", stderr=""
                    )
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="123\n", stderr=""
                )
            if arguments[0] == "ps":
                return subprocess.CompletedProcess(
                    args=arguments, returncode=1, stdout="", stderr="ps: illegal option"
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            missing = loop.process_rss_detail("qwen3_asr_server")
            failed = loop.process_rss_detail("sensevoice_server")
        self.assertIsNone(missing["value"])
        self.assertIn("no process named", str(missing["unavailable_reason"]))
        self.assertIsNone(failed["value"])
        self.assertIn("ps exited 1", str(failed["unavailable_reason"]))

    def test_rss_detail_reports_value_without_reason(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="123\n", stderr=""
                )
            if arguments[0] == "ps":
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout=" 2048 \n", stderr=""
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            detail = loop.process_rss_detail("sensevoice_server")
        self.assertEqual(detail["value"], 2048 * 1024)
        self.assertIsNone(detail["unavailable_reason"])

    def test_rss_detail_rejects_partial_pid_failure(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="123\n456\n", stderr=""
                )
            if arguments == ["ps", "-o", "rss=", "-p", "123"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="2048\n", stderr=""
                )
            if arguments == ["ps", "-o", "rss=", "-p", "456"]:
                raise OSError(1, "Operation not permitted")
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            detail = loop.process_rss_detail("sensevoice_server")
        self.assertIsNone(detail["value"])
        self.assertIn("456", str(detail["unavailable_reason"]))
        self.assertIn("Operation not permitted", str(detail["unavailable_reason"]))

    def test_rss_detail_rejects_partial_pid_timeout(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="123\n456\n", stderr=""
                )
            if arguments == ["ps", "-o", "rss=", "-p", "123"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="2048\n", stderr=""
                )
            if arguments == ["ps", "-o", "rss=", "-p", "456"]:
                raise subprocess.TimeoutExpired(arguments, 5)
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            detail = loop.process_rss_detail("sensevoice_server")
        self.assertIsNone(detail["value"])
        self.assertIn("456", str(detail["unavailable_reason"]))
        self.assertIn("timed out", str(detail["unavailable_reason"]))

    def test_scoped_processes_rejects_partial_rss_failure(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="100\n200\n", stderr=""
                )
            if arguments[:2] == ["pgrep", "-fl"]:
                return subprocess.CompletedProcess(
                    args=arguments,
                    returncode=0,
                    stdout=(
                        "100 /omni/bin/sensevoice_server --port 0\n"
                        "200 /omni/bin/sensevoice_server --port 1\n"
                    ),
                    stderr="",
                )
            if arguments[0] == "footprint":
                return subprocess.CompletedProcess(
                    args=arguments,
                    returncode=0,
                    stdout="Footprint: 100 MB\n",
                    stderr="",
                )
            if arguments == ["ps", "-o", "rss=", "-p", "100"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="2048\n", stderr=""
                )
            if arguments == ["ps", "-o", "rss=", "-p", "200"]:
                raise OSError(1, "Operation not permitted")
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            scoped = loop.scoped_processes("sensevoice_server", {100, 200}, 999, None)
        self.assertEqual(scoped["footprint_bytes"], 100 * 1024**2)
        self.assertIsNone(scoped["unavailable_reason"])
        self.assertIsNone(scoped["rss_bytes"])
        self.assertIn("200", str(scoped["rss_unavailable_reason"]))

    def test_scoped_processes_without_target_records_null_reasons(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="300\n", stderr=""
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            scoped = loop.scoped_processes("sensevoice_server", {999}, 999, None)
        self.assertEqual(scoped["pids"], [])
        self.assertIsNone(scoped["footprint_bytes"])
        self.assertIn("no target process", str(scoped["unavailable_reason"]))
        self.assertIsNone(scoped["rss_bytes"])
        self.assertIn("no target process", str(scoped["rss_unavailable_reason"]))
        self.assertEqual(scoped["outside_count"], 1)

    def test_scoped_processes_without_scope_records_null_reasons(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                raise OSError(1, "Operation not permitted")
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            scoped = loop.scoped_processes(
                "sensevoice_server", None, None, "no test process found"
            )
        self.assertEqual(scoped["pids"], [])
        self.assertEqual(scoped["argv"], [])
        self.assertIsNone(scoped["footprint_bytes"])
        self.assertIsNone(scoped["rss_bytes"])
        self.assertIn("no test process found", str(scoped["unavailable_reason"]))
        self.assertIn("no test process found", str(scoped["rss_unavailable_reason"]))
        self.assertIn("Operation not permitted", str(scoped["unavailable_reason"]))

    def test_process_discovery_failure_is_not_process_absence(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=2, stdout="", stderr="pgrep failed"
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            detail = loop.process_rss_detail("sensevoice_server")
            scoped = loop.scoped_processes("sensevoice_server", {999}, 999, None)
        self.assertIsNone(detail["value"])
        self.assertIn("pgrep exited 2", str(detail["unavailable_reason"]))
        self.assertNotIn("no process named", str(detail["unavailable_reason"]))
        self.assertIsNone(scoped["footprint_bytes"])
        self.assertIn("pgrep exited 2", str(scoped["unavailable_reason"]))
        self.assertNotIn("no target process", str(scoped["unavailable_reason"]))

    def test_scoped_processes_rejects_partial_footprint_failure(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="100\n200\n", stderr=""
                )
            if arguments[:2] == ["pgrep", "-fl"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="", stderr=""
                )
            if arguments == ["footprint", "-p", "100"]:
                return subprocess.CompletedProcess(
                    args=arguments,
                    returncode=0,
                    stdout="Footprint: 100 MB\n",
                    stderr="",
                )
            if arguments == ["footprint", "-p", "200"]:
                raise subprocess.TimeoutExpired(arguments, 20)
            if arguments[0] == "ps":
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="2048\n", stderr=""
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            scoped = loop.scoped_processes("sensevoice_server", {100, 200}, 999, None)
        self.assertIsNone(scoped["footprint_bytes"])
        self.assertIn("200", str(scoped["unavailable_reason"]))
        self.assertEqual(scoped["rss_bytes"], 2048 * 1024)

    def test_scoped_processes_rejects_failed_footprint_reads(self):
        def output(arguments, **kwargs):
            if arguments[:2] == ["pgrep", "-x"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="100\n200\n", stderr=""
                )
            if arguments[:2] == ["pgrep", "-fl"]:
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="", stderr=""
                )
            if arguments[0] == "footprint":
                return subprocess.CompletedProcess(
                    args=arguments,
                    returncode=1,
                    stdout="Footprint: 100 MB\n",
                    stderr="denied",
                )
            if arguments[0] == "ps":
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="2048\n", stderr=""
                )
            raise AssertionError(arguments)

        with mock.patch("profile_loop.subprocess.run", side_effect=output):
            scoped = loop.scoped_processes("sensevoice_server", {100, 200}, 999, None)
        self.assertIsNone(scoped["footprint_bytes"])
        self.assertIn("footprint", str(scoped["unavailable_reason"]))
        self.assertEqual(scoped["rss_bytes"], 2048 * 1024)
        self.assertIsNone(scoped["rss_unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
