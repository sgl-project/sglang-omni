#!/usr/bin/env python3
"""Run isolated long-recording profiles and retain median-of-3 reports."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

VOXT_ROOT = Path(__file__).resolve().parent.parent
SERVER_NAMES = {
    "qwen3_asr_server",
    "silero_vad_server",
    "whisper_server",
    "parakeet_server",
    "sensevoice_server",
    "cohere_transcribe_server",
    "nemotron_asr_server",
    "moss_transcribe_diarize_server",
}
PGREP_PATTERN = "/(" + "|".join(sorted(SERVER_NAMES)) + ")( |$)"
CONFIG_FIELDS = {
    "name",
    "mode",
    "audio",
    "mic_audio",
    "reference",
    "output_dir",
    "model",
    "draft",
    "decode_interval_ms",
    "segment_length_seconds",
    "cache_limit_mb",
    "vad",
    "language",
    "cleanup",
    "summary",
    "language_model",
}
NESTED_FIELDS = {
    "model": {"id", "dir"},
    "draft": {"enabled", "dir", "bits"},
    "vad": {"enabled", "model_dir"},
    "language_model": {"provider", "endpoint", "model", "key_env"},
}


def validate_config_schema(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("every matrix run must be an object")
    unknown = set(config) - CONFIG_FIELDS
    if unknown:
        raise ValueError(f"unsupported config fields: {', '.join(sorted(unknown))}")
    for field, allowed in NESTED_FIELDS.items():
        if field not in config:
            continue
        nested = config[field]
        if not isinstance(nested, dict):
            raise ValueError(f"{field} must be an object")
        unknown = set(nested) - allowed
        if unknown:
            raise ValueError(
                f"unsupported {field} fields: {', '.join(sorted(unknown))}"
            )


def load_configurations(matrix_path: Path) -> list[tuple[str, dict[str, object]]]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if isinstance(matrix, dict) and set(matrix) - {"runs"}:
        raise ValueError("matrix object accepts only the runs field")
    configurations = matrix.get("runs") if isinstance(matrix, dict) else matrix
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("matrix must be a non-empty list or an object with runs")
    named = []
    used_names = set()
    for index, config in enumerate(configurations):
        validate_config_schema(config)
        name = re.sub(
            r"[^A-Za-z0-9_.-]+", "-", str(config.get("name") or f"config-{index + 1}")
        )
        if name in ("", ".", "..") or name in used_names:
            raise ValueError(f"duplicate or unsafe configuration name: {name}")
        used_names.add(name)
        named.append((name, config))
    return named


def resolve_child(parent: Path, name: str) -> Path:
    root = parent.resolve(strict=False)
    child = (root / name).resolve(strict=False)
    if child == root or root not in child.parents:
        raise ValueError(f"path escapes its profiling root: {name}")
    return child


def parse_power_source(output: str) -> str:
    for line in output.splitlines():
        if "Now drawing from" not in line:
            continue
        match = re.search(r"'([^']+)'", line)
        if match:
            return match.group(1)
    return ""


def parse_idle_seconds(output: str) -> float:
    match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', output)
    if not match:
        raise ValueError("ioreg did not report HIDIdleTime")
    return int(match.group(1)) / 1_000_000_000


def contending_processes(
    pgrep_output: str, excluded_pids: set[int] | None = None
) -> list[str]:
    excluded = excluded_pids or set()
    contenders = []
    for line in pgrep_output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid_text, command = fields
        if not pid_text.isdigit() or int(pid_text) in excluded:
            continue
        for binary in SERVER_NAMES:
            if re.search(re.escape("/" + binary) + r"(?:\s|$)", command):
                contenders.append(f"{pid_text} {binary}")
                break
    return contenders


def pgrep_contenders(excluded_pids: set[int] | None = None) -> list[str]:
    result = subprocess.run(
        ["/usr/bin/pgrep", "-fl", PGREP_PATTERN],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"/usr/bin/pgrep failed: {result.stderr.strip()}")
    return contending_processes(result.stdout, excluded_pids)


def median(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("median requires exactly three valid runs")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("median inputs must be finite numbers")
    return sorted(values)[1]


def optional_median(
    reports: list[dict[str, object]], path: list[str]
) -> dict[str, object]:
    values = []
    for report in reports:
        current: object = report
        for component in path:
            if not isinstance(current, dict) or component not in current:
                return {
                    "value": None,
                    "unavailable_reason": "missing in at least one raw run",
                }
            current = current[component]
        if not isinstance(current, dict) or not isinstance(
            current.get("value"), (int, float)
        ):
            return {
                "value": None,
                "unavailable_reason": "unavailable in at least one raw run",
            }
        values.append(current["value"])
    return {"value": median(values)}


def numeric_path(report: dict[str, object], path: list[str]) -> float:
    current: object = report
    for component in path:
        if not isinstance(current, dict):
            raise ValueError(f"missing numeric field {'.'.join(path)}")
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"missing numeric field {'.'.join(path)}")
    return current


def report_peak_memory_pressure(report: dict[str, object]) -> str | None:
    resources = report.get("resources")
    if not isinstance(resources, dict):
        return None
    pressure = resources.get("peak_memory_pressure")
    if not isinstance(pressure, dict):
        return None
    value = pressure.get("value")
    return value if isinstance(value, str) else None


def report_memory_pressure_valid(report: dict[str, object]) -> bool:
    return report_peak_memory_pressure(report) == "normal"


def summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    if len(reports) != 3 or any(
        report.get("status") != "success"
        or report.get("warnings")
        or report.get("errors")
        or (
            ("resources" in report or "stop_to_final_ms" not in report)
            and not report_memory_pressure_valid(report)
        )
        for report in reports
    ):
        raise ValueError(
            "three successful, warning-free, normal-memory-pressure raw reports are required"
        )
    stage_keys = set()
    for report in reports:
        stages = report.get("stages")
        if not isinstance(stages, list):
            raise ValueError("raw report has no stages")
        for stage in stages:
            if isinstance(stage, dict):
                stage_keys.add(
                    (
                        str(stage.get("name")),
                        int(stage.get("index", 0)),
                        str(stage.get("track", "")),
                    )
                )
    stage_medians = {}
    for name, index, track in sorted(stage_keys):
        wall_values = []
        rtf_values = []
        for report in reports:
            stages = report["stages"]
            matches = [
                stage
                for stage in stages
                if stage.get("name") == name
                and int(stage.get("index", 0)) == index
                and str(stage.get("track", "")) == track
            ]
            if len(matches) != 1:
                raise ValueError(f"stage count differs for {name}[{index}]")
            wall_values.append(numeric_path(matches[0], ["wall_ms"]))
            if "rtf" in matches[0]:
                rtf_values.append(numeric_path(matches[0], ["rtf"]))
        key = f"{name}[{index}]"
        if track:
            key += f"({track})"
        stage_medians[key] = {"wall_ms": median(wall_values)}
        if len(rtf_values) == 3:
            stage_medians[key]["rtf"] = median(rtf_values)
        stage_matches = [
            next(
                stage
                for stage in report["stages"]
                if stage.get("name") == name
                and int(stage.get("index", 0)) == index
                and str(stage.get("track", "")) == track
            )
            for report in reports
        ]
        if "after_stop_ms" in stage_matches[0]:
            stage_medians[key]["after_stop_ms"] = optional_median(
                stage_matches, ["after_stop_ms"]
            )
    summary = {
        "status": "success",
        "valid_runs": 3,
        "text_sha256": [
            report.get("transcript", {}).get("sha256") for report in reports
        ],
        "median": {
            "total_wall_ms": median(
                [numeric_path(report, ["total_wall_ms"]) for report in reports]
            ),
            "total_rtf": optional_median(reports, ["total_rtf"]),
            "first_final_text_ms": optional_median(reports, ["first_final_text"]),
            "stop_to_summary_ms": optional_median(reports, ["stop_to_summary_ms"]),
            "peak_rss_bytes": optional_median(reports, ["resources", "peak_rss_bytes"]),
            "peak_physical_footprint_bytes": optional_median(
                reports, ["resources", "peak_physical_footprint_bytes"]
            ),
            "stages": stage_medians,
        },
    }
    if "stop_to_final_ms" in reports[0]:
        summary["median"].update(
            {
                "stop_to_final_ms": optional_median(reports, ["stop_to_final_ms"]),
                "first_llm_text_ms": optional_median(reports, ["first_llm_text"]),
                "asr_total_ms": optional_median(reports, ["timing", "asr_total_ms"]),
                "asr_after_stop_ms": optional_median(
                    reports, ["timing", "asr_after_stop_ms"]
                ),
                "llm_total_ms": optional_median(reports, ["timing", "llm_total_ms"]),
                "llm_after_stop_ms": optional_median(
                    reports, ["timing", "llm_after_stop_ms"]
                ),
                "stop_backlog_asr_segments": optional_median(
                    reports, ["backlog_at_stop", "asr_segments"]
                ),
                "stop_backlog_llm_segments": optional_median(
                    reports, ["backlog_at_stop", "llm_segments"]
                ),
            }
        )
        summary["structure_sha256"] = [
            report.get("structure", {}).get("sha256") for report in reports
        ]
        summary["summary_sha256"] = [
            (report.get("summary") or {}).get("sha256") for report in reports
        ]
    return summary


def command_output(arguments: list[str]) -> str:
    result = subprocess.run(arguments, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def host_environment() -> dict[str, object]:
    battery = command_output(["/usr/bin/pmset", "-g", "batt"])
    thermal = command_output(["/usr/bin/pmset", "-g", "therm"])
    idle = parse_idle_seconds(
        command_output(["/usr/sbin/ioreg", "-c", "IOHIDSystem", "-d", "4"])
    )
    contenders = pgrep_contenders()
    return {
        "power_source": parse_power_source(battery),
        "thermal": thermal,
        "idle_seconds": idle,
        "contenders": contenders,
    }


def wait_for_gate(idle_seconds: float, poll_seconds: float) -> dict[str, object]:
    while True:
        environment = host_environment()
        if (
            environment["power_source"] == "AC Power"
            and environment["idle_seconds"] >= idle_seconds
        ):
            if environment["contenders"]:
                raise RuntimeError(
                    f"contending native servers: {', '.join(environment['contenders'])}"
                )
            return environment
        print(
            f"waiting for AC power and {idle_seconds:g}s idle: {environment}",
            flush=True,
        )
        time.sleep(poll_seconds)


def copy_read_only(source: str, destination: Path) -> None:
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"not a file: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    destination.chmod(0o444)


def hard_link_tree(source: str, destination: Path) -> None:
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_dir():
        raise ValueError(f"not a model directory: {source_path}")
    for current, directories, files in os.walk(source_path):
        current_path = Path(current)
        relative = current_path.relative_to(source_path)
        (destination / relative).mkdir(parents=True, exist_ok=True)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ValueError(
                    f"symlinked model directories are unsupported: {current_path / directory}"
                )
        for filename in files:
            source_file = current_path / filename
            if source_file.is_symlink():
                source_file = source_file.resolve(strict=True)
            if not source_file.is_file():
                raise ValueError(f"not a regular model file: {source_file}")
            target = destination / relative / filename
            try:
                os.link(source_file, target)
            except OSError as error:
                raise RuntimeError(
                    f"hard link failed for {source_file}; choose --temp-root on the same filesystem: {error}"
                ) from error


def prepare_config(original: dict[str, object], directory: Path) -> dict[str, object]:
    validate_config_schema(original)
    config = copy.deepcopy(original)
    copy_read_only(
        str(config["audio"]), directory / "input" / Path(str(config["audio"])).name
    )
    config["audio"] = str(directory / "input" / Path(str(original["audio"])).name)
    if config.get("mic_audio"):
        mic_destination = (
            directory / "input" / ("mic-" + Path(str(config["mic_audio"])).name)
        )
        copy_read_only(str(config["mic_audio"]), mic_destination)
        config["mic_audio"] = str(mic_destination)
    if config.get("reference"):
        reference_name = Path(str(config["reference"])).name
        copy_read_only(str(config["reference"]), directory / "input" / reference_name)
        config["reference"] = str(directory / "input" / reference_name)
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    hard_link_tree(str(model["dir"]), directory / "model")
    model["dir"] = str(directory / "model")
    draft = config.get("draft")
    if isinstance(draft, dict) and draft.get("enabled"):
        hard_link_tree(str(draft["dir"]), directory / "draft")
        draft["dir"] = str(directory / "draft")
    vad = config.get("vad")
    if isinstance(vad, dict) and vad.get("enabled"):
        hard_link_tree(str(vad["model_dir"]), directory / "vad")
        vad["model_dir"] = str(directory / "vad")
    return config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def valid_result(path: Path, returncode: int) -> tuple[bool, dict[str, object] | None]:
    if returncode != 0 or not path.is_file():
        return False, None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    if (
        not isinstance(report, dict)
        or report.get("status") != "success"
        or report.get("warnings")
        or report.get("errors")
        or not report_memory_pressure_valid(report)
    ):
        return False, report if isinstance(report, dict) else None
    return True, report


def environment_violations(
    environment: dict[str, object], idle_seconds: float
) -> list[str]:
    violations = []
    if environment.get("power_source") != "AC Power":
        violations.append("left AC power")
    if environment.get("idle_seconds", 0) < idle_seconds:
        violations.append("user input during profiling")
    return violations


def environment_attempt_valid(
    during: list[dict[str, object]], after: dict[str, object], idle_seconds: float
) -> bool:
    if environment_violations(after, idle_seconds) or after.get("contenders"):
        return False
    return not any(sample.get("violations") for sample in during)


def redaction_values(config: dict[str, object]) -> list[str]:
    values = []
    language_model = config.get("language_model")
    if isinstance(language_model, dict):
        for field in ("endpoint", "model"):
            value = language_model.get(field)
            if isinstance(value, str) and value:
                values.append(value)
        key_env = language_model.get("key_env")
        if isinstance(key_env, str) and key_env and os.environ.get(key_env):
            values.append(os.environ[key_env])
    return sorted(values, key=len, reverse=True)


def sanitise_text(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


def run_profile_process(
    arguments: list[str],
    environment: dict[str, str],
    log_path: Path,
    idle_seconds: float,
    sample_seconds: float,
    secrets: list[str],
    environment_sampler=None,
) -> tuple[int, list[dict[str, object]]]:
    sampler = environment_sampler or host_environment
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    samples = []
    with os.fdopen(descriptor, "w", encoding="utf-8") as process_log:
        with subprocess.Popen(
            arguments,
            cwd=VOXT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:

            def drain_log() -> None:
                for line in process.stdout:
                    process_log.write(sanitise_text(line, secrets))
                    process_log.flush()

            reader = threading.Thread(target=drain_log, daemon=True)
            reader.start()
            while True:
                try:
                    returncode = process.wait(timeout=sample_seconds)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        environment_sample = sampler()
                        violations = environment_violations(
                            environment_sample, idle_seconds
                        )
                        samples.append(
                            {
                                "environment": environment_sample,
                                "violations": violations,
                            }
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        samples.append(
                            {"violations": [f"environment sample failed: {error}"]}
                        )
            reader.join()
    return returncode, samples


def run_matrix(
    matrix_path: Path,
    results_root: Path,
    idle_seconds: float,
    poll_seconds: float,
    temp_root: Path | None,
    go_timeout: str,
    environment_sample_seconds: float = 15,
) -> int:
    configurations = load_configurations(matrix_path)
    results_root.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch_dir = results_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    all_valid = True
    with tempfile.TemporaryDirectory(
        prefix="voxt-profile-loop-", dir=temp_root
    ) as temporary:
        temporary_path = Path(temporary)
        for name, original in configurations:
            config_dir = resolve_child(batch_dir, name)
            prepare_dir = resolve_child(temporary_path, name)
            reports = []
            raw_paths = []
            valid_count = 0
            config_error = ""
            try:
                prepared = prepare_config(original, prepare_dir)
                for attempt in range(1, 4):
                    attempt_dir = config_dir / f"run-{attempt}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    run_config = copy.deepcopy(prepared)
                    run_config["output_dir"] = str(attempt_dir)
                    config_path = attempt_dir / "config.json"
                    write_json(config_path, run_config)
                    before = wait_for_gate(idle_seconds, poll_seconds)
                    environment = os.environ.copy()
                    environment["VOXTNEW_PROFILE_CONFIG"] = str(config_path)
                    attempt_temp = prepare_dir / f"tmp-{attempt}"
                    attempt_temp.mkdir()
                    environment["TMPDIR"] = str(attempt_temp)
                    returncode, during = run_profile_process(
                        [
                            "go",
                            "test",
                            "-count=1",
                            "-run",
                            "^TestMeetingProfile$",
                            "-timeout",
                            go_timeout,
                            ".",
                        ],
                        environment,
                        attempt_dir / "process.log",
                        idle_seconds,
                        environment_sample_seconds,
                        redaction_values(run_config),
                    )
                    after = host_environment()
                    write_json(
                        attempt_dir / "environment.json",
                        {"before": before, "during": during, "after": after},
                    )
                    valid, report = valid_result(
                        attempt_dir / "result.json", returncode
                    )
                    if not environment_attempt_valid(during, after, idle_seconds):
                        valid = False
                    if valid and report is not None:
                        valid_count += 1
                        reports.append(report)
                        raw_paths.append(str(attempt_dir / "result.json"))
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                config_error = str(error)
            if valid_count == 3 and not config_error:
                try:
                    summary = summarize_reports(reports)
                except ValueError as error:
                    summary = {
                        "status": "failed",
                        "valid_runs": valid_count,
                        "error": str(error),
                    }
            else:
                summary = {
                    "status": "failed",
                    "valid_runs": valid_count,
                    "error": config_error or "fewer than three valid runs",
                }
            summary["raw_results"] = raw_paths
            write_json(config_dir / "summary.json", summary)
            summaries.append({"name": name, **summary})
            if summary["status"] != "success":
                all_valid = False
    batch_summary = {
        "batch": batch_id,
        "status": "success" if all_valid else "failed",
        "configurations": summaries,
    }
    write_json(batch_dir / "summary.json", batch_summary)
    with (results_root / "matrix-summary.jsonl").open(
        "a", encoding="utf-8"
    ) as aggregate:
        aggregate.write(json.dumps(batch_summary, ensure_ascii=False) + "\n")
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if all_valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--idle-seconds", type=float, default=300)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--go-timeout", default="6h")
    parser.add_argument("--environment-sample-seconds", type=float, default=15)
    arguments = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("profile-loop runs only on macOS")
    if (
        arguments.idle_seconds < 0
        or arguments.poll_seconds <= 0
        or arguments.environment_sample_seconds <= 0
    ):
        parser.error(
            "idle seconds must be nonnegative and poll/sample seconds positive"
        )
    try:
        return run_matrix(
            arguments.matrix.resolve(),
            arguments.results.resolve(),
            arguments.idle_seconds,
            arguments.poll_seconds,
            arguments.temp_root,
            arguments.go_timeout,
            arguments.environment_sample_seconds,
        )
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"profile-loop: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
