import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import overlap_summary
from scripts import profile_loop as loop


class ParserTests(unittest.TestCase):
    def test_power_idle_and_contention_parsers(self):
        self.assertEqual(
            loop.parse_power_source("Now drawing from 'AC Power'\n"), "AC Power"
        )
        self.assertEqual(
            loop.parse_power_source("Now drawing from 'Battery Power'\n"),
            "Battery Power",
        )
        self.assertEqual(loop.parse_power_source("unavailable"), "")
        self.assertEqual(
            loop.parse_idle_seconds('    "HIDIdleTime" = 12345678901\n'), 12.345678901
        )
        with self.assertRaises(ValueError):
            loop.parse_idle_seconds("no idle counter")
        processes = (
            " 10 /tmp/qwen3_asr_server --supervised --model-directory /tmp/model\n"
            " 11 /usr/bin/python3\n"
            " 12 /opt/moss_transcribe_diarize_server --draft-model-path /tmp/draft --draft-bits 4\n"
            " 13 /opt/moss_transcribe_diarize_server --supervised\n"
        )
        self.assertEqual(
            loop.contending_processes(processes, {13}),
            ["10 qwen3_asr_server", "12 moss_transcribe_diarize_server"],
        )

    def test_contention_check_uses_pgrep_not_ps(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout="10 /tmp/moss_transcribe_diarize_server --supervised\n",
            stderr="",
        )
        with mock.patch(
            "scripts.profile_loop.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(
                loop.pgrep_contenders(), ["10 moss_transcribe_diarize_server"]
            )
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[:2], ["/usr/bin/pgrep", "-fl"])
        self.assertNotIn("/bin/ps", arguments)

    def test_pgrep_reads_full_arguments_on_this_mac(self):
        process = subprocess.Popen(
            ["/bin/sh", "-c", "sleep 0.2; :", "/tmp/moss_transcribe_diarize_server"]
        )
        try:
            self.assertEqual(
                loop.pgrep_contenders(),
                [f"{process.pid} moss_transcribe_diarize_server"],
            )
        finally:
            self.assertEqual(process.wait(), 0)

    def test_host_environment_uses_macos_tool_paths(self):
        def output(arguments):
            if arguments[:2] == ["/usr/bin/pmset", "-g"]:
                if arguments[2] == "batt":
                    return "Now drawing from 'AC Power'\n"
                return "CPU power status: 0\n"
            self.assertEqual(arguments[:2], ["/usr/sbin/ioreg", "-c"])
            return '"HIDIdleTime" = 300000000000\n'

        with (
            mock.patch("scripts.profile_loop.command_output", side_effect=output),
            mock.patch("scripts.profile_loop.pgrep_contenders", return_value=[]),
        ):
            environment = loop.host_environment()
        self.assertEqual(environment["power_source"], "AC Power")
        self.assertEqual(environment["idle_seconds"], 300)

    def test_median_requires_three_finite_numbers(self):
        self.assertEqual(loop.median([9, 1, 5]), 5)
        for values in ([1, 2], [1, float("nan"), 3], [1, True, 3]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                loop.median(values)


def raw_report(wall, status="success"):
    return {
        "status": status,
        "warnings": [],
        "errors": [],
        "total_wall_ms": wall,
        "total_rtf": {"value": wall / 1000},
        "first_final_text": {"value": wall / 2},
        "transcript": {"sha256": f"hash-{wall}"},
        "stages": [{"name": "decode", "index": 0, "wall_ms": wall / 10, "rtf": 0.1}],
        "resources": {
            "peak_rss_bytes": {"value": wall * 10},
            "peak_physical_footprint_bytes": {"value": wall * 20},
            "peak_memory_pressure": {"value": "normal"},
        },
    }


class SummaryTests(unittest.TestCase):
    def test_median_of_three_and_warning_rejection(self):
        reports = [raw_report(300), raw_report(100), raw_report(200)]
        for report in reports:
            report["stop_to_summary_ms"] = {"value": report["total_wall_ms"] * 0.8}
        summary = loop.summarize_reports(reports)
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["median"]["total_wall_ms"], 200)
        self.assertEqual(summary["median"]["stop_to_summary_ms"], {"value": 160})
        self.assertEqual(summary["median"]["stages"]["decode[0]"]["wall_ms"], 20)
        self.assertEqual(summary["median"]["peak_rss_bytes"], {"value": 2000})
        self.assertEqual(
            summary["median"]["peak_physical_footprint_bytes"], {"value": 4000}
        )
        with self.assertRaises(ValueError):
            loop.summarize_reports([raw_report(100), raw_report(200)])
        warning = raw_report(150)
        warning["warnings"] = ["draft refused"]
        with self.assertRaises(ValueError):
            loop.summarize_reports([raw_report(100), warning, raw_report(200)])

    def test_summarize_reports_rejects_short_dictation_reports_without_resources(self):
        reports = []
        for wall in (100, 200, 300):
            report = {
                "status": "success",
                "warnings": [],
                "errors": [],
                "total_wall_ms": wall,
                "stop_to_final_ms": {"value": wall / 4},
                "stages": [
                    {"name": "decode", "index": 0, "wall_ms": wall / 10, "rtf": 0.1}
                ],
            }
            reports.append(report)
        with self.assertRaises(ValueError):
            loop.summarize_reports(reports)

    def test_unavailable_metric_stays_null(self):
        reports = [raw_report(100), raw_report(200), raw_report(300)]
        reports[1]["resources"]["peak_rss_bytes"] = {
            "value": None,
            "unavailable_reason": "denied",
        }
        summary = loop.summarize_reports(reports)
        self.assertIsNone(summary["median"]["peak_rss_bytes"]["value"])
        self.assertTrue(summary["median"]["peak_rss_bytes"]["unavailable_reason"])

    def test_overlap_metrics_use_the_existing_median_summary(self):
        reports = [raw_report(300), raw_report(100), raw_report(200)]
        for report in reports:
            wall = report["total_wall_ms"]
            report["stop_to_final_ms"] = {"value": wall / 4}
            report["first_llm_text"] = {"value": wall / 5}
            report["timing"] = {
                "asr_total_ms": {"value": wall / 2},
                "asr_after_stop_ms": {"value": wall / 6},
                "llm_total_ms": {"value": wall / 3},
                "llm_after_stop_ms": {"value": wall / 7},
            }
            report["backlog_at_stop"] = {
                "asr_segments": {"value": wall / 100},
                "llm_segments": {"value": wall / 50},
            }
            report["stages"][0]["after_stop_ms"] = {"value": wall / 10}
            report["structure"] = {"sha256": f"structure-{wall}"}
            report["summary"] = {"sha256": f"summary-{wall}"}
        summary = loop.summarize_reports(reports)
        self.assertEqual(summary["median"]["stop_to_final_ms"], {"value": 50})
        self.assertEqual(summary["median"]["asr_after_stop_ms"], {"value": 200 / 6})
        self.assertEqual(summary["median"]["llm_after_stop_ms"], {"value": 200 / 7})
        self.assertEqual(summary["median"]["stop_backlog_asr_segments"], {"value": 2})
        self.assertEqual(
            summary["median"]["stages"]["decode[0]"]["after_stop_ms"], {"value": 20}
        )
        self.assertEqual(
            summary["structure_sha256"],
            ["structure-300", "structure-100", "structure-200"],
        )

    def test_overlap_summary_command_reads_three_run_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [root / f"run-{index}" for index in range(1, 4)]
            for run, wall in zip(runs, (300, 100, 200)):
                for method in overlap_summary.METHODS:
                    path = run / method / "result.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(raw_report(wall)), encoding="utf-8")
            summaries = overlap_summary.summarize_run_directories(runs, root / "median")
            self.assertEqual(summaries["overlap"]["median"]["total_wall_ms"], 200)
            self.assertTrue((root / "median" / "one_shot" / "summary.json").is_file())


class ValidationTests(unittest.TestCase):
    def test_unknown_sensitive_field_writes_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "name": "bad",
                                "audio": str(root / "missing.wav"),
                                "model": {"id": "asr.fast", "dir": "model"},
                                "language_model": {
                                    "provider": "custom",
                                    "key": "sentinel-must-not-escape",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            results = root / "results"
            with self.assertRaisesRegex(ValueError, "language_model"):
                loop.run_matrix(matrix, results, 0, 0.01, None, "1s")
            self.assertFalse(results.exists())
            self.assertEqual([path.name for path in root.iterdir()], ["matrix.json"])

    def test_prepare_rejects_unknown_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "audio": "missing.wav",
                "model": {"id": "asr.fast", "dir": "m"},
                "unexpected": "sentinel",
            }
            with self.assertRaises(ValueError):
                loop.prepare_config(config, root / "work")
            self.assertFalse((root / "work").exists())

    def test_unsafe_names_are_rejected_before_batch_creation(self):
        for name in (".", ".."):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                matrix = root / "matrix.json"
                matrix.write_text(
                    json.dumps(
                        [
                            {
                                "name": name,
                                "audio": "missing.wav",
                                "model": {"id": "asr.fast", "dir": "m"},
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                results = root / "results"
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    loop.run_matrix(matrix, results, 0, 0.01, None, "1s")
                self.assertFalse(results.exists())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = loop.resolve_child(root, "safe")
            self.assertEqual(child, root.resolve() / "safe")
            with self.assertRaises(ValueError):
                loop.resolve_child(root, "..")


class ProcessTests(unittest.TestCase):
    def test_midrun_violation_is_invalid_and_log_is_sanitised(self):
        endpoint = "http://127.0.0.1/private-endpoint"
        model = "private-model"
        key = "private-key"
        os.environ["PROFILE_LOOP_TEST_KEY"] = key
        self.addCleanup(os.environ.pop, "PROFILE_LOOP_TEST_KEY", None)
        normal = {"power_source": "AC Power", "idle_seconds": 1000, "contenders": []}
        abnormal = {
            "power_source": "Battery Power",
            "idle_seconds": 0,
            "contenders": [],
        }
        samples = iter(
            [
                normal,
                abnormal,
                normal,
                normal,
                normal,
                normal,
                normal,
                normal,
                normal,
                normal,
                normal,
            ]
        )

        def sample_environment():
            return next(samples, normal)

        command = [
            "/bin/sh",
            "-c",
            f"printf '%s %s %s\\n' '{endpoint}' '{model}' '{key}'; sleep 0.2",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "process.log"
            environment = os.environ.copy()
            returncode, during = loop.run_profile_process(
                command,
                environment,
                log_path,
                300,
                0.02,
                [endpoint, model, key],
                sample_environment,
            )
            self.assertEqual(returncode, 0)
            self.assertTrue(log_path.is_file())
            content = log_path.read_text(encoding="utf-8")
            for sentinel in (endpoint, model, key):
                self.assertNotIn(sentinel, content)
            before, after = normal, normal
            self.assertFalse(loop.environment_violations(before, 300))
            self.assertFalse(loop.environment_attempt_valid(during, after, 300))
            valid_reports = [raw_report(100), raw_report(200)]
            with self.assertRaises(ValueError):
                loop.summarize_reports(valid_reports)


class IsolationTests(unittest.TestCase):
    def test_prepare_copies_audio_and_hard_links_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            audio = source / "meeting.wav"
            audio.write_bytes(b"short fake PCM")
            mic_audio = source / "mic.wav"
            mic_audio.write_bytes(b"short fake mic PCM")
            model = source / "model"
            model.mkdir()
            (model / "weights").write_bytes(b"fake weights")
            original = {
                "audio": str(audio),
                "mic_audio": str(mic_audio),
                "model": {"id": "asr.fast", "dir": str(model)},
                "draft": {"enabled": False},
                "vad": {"enabled": False},
            }
            destination = root / "work"
            prepared = loop.prepare_config(original, destination)
            copied = Path(prepared["audio"])
            copied_mic = Path(prepared["mic_audio"])
            linked = Path(prepared["model"]["dir"]) / "weights"
            self.assertEqual(copied.read_bytes(), audio.read_bytes())
            self.assertNotEqual(copied, audio)
            self.assertEqual(copied.stat().st_mode & 0o777, 0o444)
            self.assertEqual(copied_mic.read_bytes(), mic_audio.read_bytes())
            self.assertNotEqual(copied_mic, mic_audio)
            self.assertEqual(copied_mic.stat().st_mode & 0o777, 0o444)
            self.assertEqual(linked.stat().st_ino, (model / "weights").stat().st_ino)
            self.assertEqual(linked.read_bytes(), b"fake weights")

    def test_valid_result_rejects_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            report = raw_report(100)
            report["warnings"] = ["not clean"]
            path.write_text(json.dumps(report), encoding="utf-8")
            valid, _ = loop.valid_result(path, 0)
            self.assertFalse(valid)

    def test_result_invalid_reasons_name_the_failing_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            reasons, report = loop.result_invalid_reasons(path, 1)
            self.assertEqual(reasons, ["returncode=1", "result_json_missing"])
            self.assertIsNone(report)

            report = raw_report(100)
            report["resources"]["peak_memory_pressure"] = {"value": "critical"}
            path.write_text(json.dumps(report), encoding="utf-8")
            reasons, loaded = loop.result_invalid_reasons(path, 0)
            self.assertEqual(reasons, ["peak_memory_pressure=critical"])
            self.assertEqual(loaded["status"], "success")

            report["warnings"] = ["not clean"]
            report["errors"] = ["failed once"]
            path.write_text(json.dumps(report), encoding="utf-8")
            reasons, _ = loop.result_invalid_reasons(path, 0)
            self.assertEqual(
                reasons,
                ["warnings=1", "errors=1", "peak_memory_pressure=critical"],
            )

            failed = raw_report(100, status="failed")
            path.write_text(json.dumps(failed), encoding="utf-8")
            reasons, _ = loop.result_invalid_reasons(path, 0)
            self.assertEqual(reasons, ["status=failed"])

            path.write_text(json.dumps(raw_report(100)), encoding="utf-8")
            reasons, loaded = loop.result_invalid_reasons(path, 0)
            self.assertEqual(reasons, [])
            self.assertIsNotNone(loaded)

    def test_environment_invalid_reasons_keep_midrun_violations(self):
        normal = {"power_source": "AC Power", "idle_seconds": 1000, "contenders": []}
        during = [
            {"environment": normal, "violations": []},
            {"environment": normal, "violations": ["user input during profiling"]},
            {"environment": normal, "violations": ["user input during profiling"]},
        ]
        self.assertEqual(
            loop.environment_invalid_reasons(during, normal, 300),
            ["user input during profiling"],
        )
        self.assertEqual(loop.environment_invalid_reasons([], normal, 300), [])
        after = {
            "power_source": "Battery Power",
            "idle_seconds": 1000,
            "contenders": ["10 qwen3_asr_server"],
        }
        self.assertEqual(
            loop.environment_invalid_reasons([], after, 300),
            ["left AC power", "contending native servers after the run"],
        )

    def test_valid_result_and_summary_reject_midrun_warning_or_missing_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            normal_report = raw_report(100)
            normal_report["resources"]["memory_start"] = {
                "pressure": {"value": "normal"}
            }
            normal_report["resources"]["memory_end"] = {"pressure": {"value": "normal"}}
            normal_report["resources"]["memory_pressure_warnings"] = []
            path.write_text(json.dumps(normal_report), encoding="utf-8")
            valid, _ = loop.valid_result(path, 0)
            self.assertTrue(valid)

            midrun_warning = raw_report(150)
            midrun_warning["resources"]["memory_start"] = {
                "pressure": {"value": "normal"}
            }
            midrun_warning["resources"]["memory_end"] = {
                "pressure": {"value": "normal"}
            }
            midrun_warning["resources"]["peak_memory_pressure"] = {"value": "warning"}
            midrun_warning["resources"]["memory_pressure_warnings"] = [
                {"at_ms": 2000.0, "pressure": "warning"}
            ]
            path.write_text(json.dumps(midrun_warning), encoding="utf-8")
            valid, _ = loop.valid_result(path, 0)
            self.assertFalse(valid)
            with self.assertRaises(ValueError):
                loop.summarize_reports(
                    [raw_report(100), midrun_warning, raw_report(200)]
                )

            missing_pressure = raw_report(150)
            del missing_pressure["resources"]["peak_memory_pressure"]
            path.write_text(json.dumps(missing_pressure), encoding="utf-8")
            valid, _ = loop.valid_result(path, 0)
            self.assertFalse(valid)
            with self.assertRaises(ValueError):
                loop.summarize_reports(
                    [raw_report(100), missing_pressure, raw_report(200)]
                )

            unavailable_pressure = raw_report(150)
            unavailable_pressure["resources"]["peak_memory_pressure"] = {
                "value": None,
                "unavailable_reason": "denied",
            }
            path.write_text(json.dumps(unavailable_pressure), encoding="utf-8")
            valid, _ = loop.valid_result(path, 0)
            self.assertFalse(valid)
            with self.assertRaises(ValueError):
                loop.summarize_reports(
                    [raw_report(100), unavailable_pressure, raw_report(200)]
                )


class AttemptsTests(unittest.TestCase):
    def test_single_attempt_diagnostic_succeeds_without_median(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            audio = source / "meeting.wav"
            audio.write_bytes(b"short fake PCM")
            model = source / "model"
            model.mkdir()
            (model / "weights").write_bytes(b"fake weights")
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    [
                        {
                            "name": "diag",
                            "audio": str(audio),
                            "model": {"id": "asr.fast", "dir": str(model)},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            environment = {
                "power_source": "AC Power",
                "idle_seconds": 1000,
                "contenders": [],
            }

            def fake_process(arguments, env, log_path, idle, sample, secrets, sampler=None):
                attempt_dir = Path(env["VOXTNEW_PROFILE_CONFIG"]).parent
                report = raw_report(100)
                attempt_dir.joinpath("result.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                return 0, []

            with (
                mock.patch.object(
                    loop, "wait_for_gate", return_value=environment
                ),
                mock.patch.object(
                    loop, "run_profile_process", side_effect=fake_process
                ),
                mock.patch.object(
                    loop, "host_environment", return_value=environment
                ),
            ):
                results = root / "results"
                self.assertEqual(
                    loop.run_matrix(matrix, results, 0, 0.01, None, "1s", max_attempts=1),
                    0,
                )
            batches = [path for path in results.iterdir() if path.is_dir()]
            self.assertEqual(len(batches), 1)
            summary = json.loads(
                batches[0].joinpath("diag", "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "diagnostic")
            self.assertEqual(summary["valid_runs"], 1)
            self.assertIn("no three-run median", summary["note"])
            self.assertNotIn("median", summary)
            self.assertEqual(
                summary["attempts"],
                [
                    {
                        "attempt": 1,
                        "returncode": 0,
                        "result_invalid_reason": [],
                        "environment_invalid_reason": [],
                        "valid": True,
                    }
                ],
            )
            batch = json.loads(
                batches[0].joinpath("summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(batch["status"], "diagnostic")

    def test_two_attempts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, r"1 \(diagnostic\) or 3"):
                loop.run_matrix(
                    root / "matrix.json",
                    root / "results",
                    0,
                    0.01,
                    None,
                    "1s",
                    max_attempts=2,
                )
        with (
            mock.patch.object(loop.sys, "platform", "darwin"),
            mock.patch.object(
                loop.sys,
                "argv",
                ["profile_loop.py", "matrix.json", "results", "--attempts", "2"],
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            loop.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
