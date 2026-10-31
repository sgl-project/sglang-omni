# SPDX-License-Identifier: Apache-2.0
"""Measure Qwen3-ASR long recordings through its public HTTP and realtime APIs."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import platform
import selectors
import subprocess
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from sglang_omni_mlx.native.ci.long_audio_client import (
    HTTP_TRANSCRIPTION_FIELDS,
    MEMORY_SAMPLE_INTERVAL_SECONDS,
    MemorySampler,
    compare_cancellation_timing,
    score,
    stream_recording,
    transcribe_http,
)
from sglang_omni_mlx.native.ci.long_audio_corpus import build_recording, load_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--corpus-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--languages", nargs="+", choices=("en", "zh"), default=["en", "zh"]
    )
    parser.add_argument("--minutes", nargs="+", type=float, default=[5, 10, 30, 60])
    parser.add_argument("--mode", choices=("http", "realtime"), required=True)
    parser.add_argument(
        "--speed",
        type=float,
        default=1,
        help="Realtime audio seconds per wall second; >1 is accelerated, not live pacing",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300,
        help="HTTP request deadline and realtime wait after audio input ends",
    )
    parser.add_argument(
        "--max-footprint-gib",
        type=float,
        default=8,
        help="Sampled soft guard for the server footprint; allocations can overshoot between samples",
    )
    parser.add_argument("--seed", type=int, default=2888)
    arguments = parser.parse_args()
    if (
        platform.system() != "Darwin"
        or not all(
            math.isfinite(value)
            for value in [
                *arguments.minutes,
                arguments.speed,
                arguments.timeout_seconds,
                arguments.max_footprint_gib,
            ]
        )
        or min(arguments.minutes) <= 0
        or arguments.speed <= 0
        or arguments.timeout_seconds <= 0
        or arguments.max_footprint_gib <= 0
    ):
        parser.error(
            "Requires macOS and positive duration, speed, timeout and memory limit"
        )
    else:
        pass
    arguments.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(arguments).items()
        },
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "macos": platform.mac_ver()[0],
        "measurement_dependencies": {
            "rapidfuzz": version("rapidfuzz"),
            "websockets": version("websockets"),
        },
        "protocol": {
            "http_fields": HTTP_TRANSCRIPTION_FIELDS,
            "realtime_turn_detection": None,
            "realtime_audio_layout_and_stop_rules": "server defaults",
        },
        "memory_guard": {
            "kind": "sampled_soft",
            "sample_interval_seconds": MEMORY_SAMPLE_INTERVAL_SECONDS,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    metadata["git_status"] = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    metadata["binary_sha256"] = hashlib.sha256(
        arguments.server.read_bytes()
    ).hexdigest()
    metadata["model_files_sha256"] = {}
    for path in sorted(arguments.model_directory.iterdir()):
        if path.is_file():
            with path.open("rb") as model_file:
                metadata["model_files_sha256"][path.name] = hashlib.file_digest(
                    model_file, "sha256"
                ).hexdigest()
        else:
            pass
    (arguments.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    failures = 0
    for language in arguments.languages:
        clips = load_clips(arguments.corpus_directory, language)
        cancel_path = arguments.output / f"{language}-cancel.wav"
        cancellation_input = build_recording(clips, 300, arguments.seed, cancel_path)
        (arguments.output / f"{language}-cancel.input.json").write_text(
            json.dumps(cancellation_input, ensure_ascii=False, indent=2) + "\n"
        )
        for minutes in arguments.minutes:
            name = f"{language}-{minutes:g}m"
            wav_path = arguments.output / f"{name}.wav"
            recording = build_recording(clips, minutes * 60, arguments.seed, wav_path)
            (arguments.output / f"{name}.input.json").write_text(
                json.dumps(recording, ensure_ascii=False, indent=2) + "\n"
            )
            result = {
                "case": name,
                "status": "failed",
                "audio_seconds": recording["audio_seconds"],
            }
            with (arguments.output / f"{name}.server.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        str(arguments.server.resolve()),
                        "--supervised",
                        "--model-directory",
                        str(arguments.model_directory.resolve()),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                )
                sampler = MemorySampler(
                    process,
                    arguments.output / f"{name}.memory.jsonl",
                    int(arguments.max_footprint_gib * 1024**3),
                    60
                    + recording["audio_seconds"] / arguments.speed
                    + 4 * arguments.timeout_seconds,
                )
                sampler.thread.start()
                try:
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdout, selectors.EVENT_READ)
                        if not selector.select(timeout=60):
                            raise TimeoutError("Server startup deadline exceeded")
                        else:
                            ready = json.loads(process.stdout.readline())
                    port = ready["port"]
                    sampler.phase = "warmup"
                    warmup = transcribe_http(
                        port,
                        clips[0].path.read_bytes(),
                        language,
                        arguments.timeout_seconds,
                    )
                    time.sleep(2)
                    baseline_bytes = sampler.samples[-1]["footprint_bytes"]
                    measured_start = len(sampler.samples)
                    sampler.phase = "long_recording"
                    if arguments.mode == "http":
                        transcription = transcribe_http(
                            port,
                            wav_path.read_bytes(),
                            language,
                            arguments.timeout_seconds,
                        )
                    else:
                        transcription = stream_recording(
                            port,
                            wav_path,
                            language,
                            arguments.speed,
                            arguments.timeout_seconds,
                            arguments.output / f"{name}.events.jsonl",
                        )
                    sampler.phase = "post_session"
                    time.sleep(2)
                    result.update(transcription)
                    result.update(
                        score(
                            str(recording["reference"]), transcription["text"], language
                        )
                    )
                    result.update(
                        {
                            "baseline_footprint_bytes": baseline_bytes,
                            "peak_footprint_bytes": max(
                                sample["footprint_bytes"]
                                for sample in sampler.samples[measured_start:]
                            ),
                            "post_session_footprint_bytes": sampler.samples[-1][
                                "footprint_bytes"
                            ],
                            "rtf": transcription["wall_seconds"]
                            / recording["audio_seconds"],
                        }
                    )
                    sampler.phase = "cancellation_baseline"
                    natural = transcribe_http(
                        port,
                        cancel_path.read_bytes(),
                        language,
                        arguments.timeout_seconds,
                    )
                    sampler.phase = "cancellation"
                    cancellation = transcribe_http(
                        port,
                        cancel_path.read_bytes(),
                        language,
                        arguments.timeout_seconds,
                        cancel=True,
                    )
                    result["cancellation"] = cancellation
                    cancellation["natural_completion_seconds"] = natural["wall_seconds"]
                    if not cancellation["observed_running"]:
                        raise RuntimeError(
                            "Cancellation inconclusive: no running request observed at disconnect"
                        )
                    else:
                        pass
                    deadline = time.monotonic() + arguments.timeout_seconds
                    while True:
                        with closing(
                            http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                        ) as health:
                            health.request("GET", "/health")
                            states = json.loads(health.getresponse().read())[
                                "request_states"
                            ]
                        if not states:
                            cancellation["disconnect_to_idle_seconds"] = (
                                time.monotonic() - cancellation["disconnect_monotonic"]
                            )
                            break
                        elif time.monotonic() > deadline:
                            raise TimeoutError(
                                "Cancelled request did not return to idle"
                            )
                        else:
                            time.sleep(0.05)
                    sampler.phase = "recovery"
                    recovery = transcribe_http(
                        port,
                        clips[0].path.read_bytes(),
                        language,
                        arguments.timeout_seconds,
                    )
                    result["recovery_matches_warmup"] = (
                        recovery["text"] == warmup["text"]
                    )
                    time.sleep(2)
                    result["post_recovery_footprint_bytes"] = sampler.samples[-1][
                        "footprint_bytes"
                    ]
                    (
                        cancellation["paired_baseline_remaining_seconds"],
                        cancellation["idle_before_paired_baseline"],
                    ) = compare_cancellation_timing(
                        natural["heartbeat_to_completion_seconds"],
                        cancellation["heartbeat_to_disconnect_seconds"],
                        cancellation["disconnect_to_idle_seconds"],
                    )
                    if not result["recovery_matches_warmup"]:
                        raise RuntimeError(
                            "Post-cancellation transcript differs from warmup"
                        )
                    elif cancellation["idle_before_paired_baseline"] is not True:
                        result["status"] = "inconclusive"
                        result["error"] = (
                            "Paired timing does not establish an earlier return to idle"
                        )
                        failures += 1
                    else:
                        result["status"] = "measured"
                except Exception as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                    failures += 1
                finally:
                    sampler.done.set()
                    sampler.thread.join(timeout=5)
                    result["process_peak_footprint_bytes"] = max(
                        (sample["footprint_bytes"] for sample in sampler.samples),
                        default=0,
                    )
                    if process.poll() is None:
                        try:
                            process.stdin.write('{"command":"shutdown"}\n')
                            process.stdin.flush()
                            process.wait(timeout=10)
                        except (subprocess.TimeoutExpired, BrokenPipeError):
                            process.kill()
                            process.wait(timeout=10)
                    else:
                        pass
                    result["memory_limit_exceeded"] = sampler.exceeded
                    process.stdin.close()
                    process.stdout.close()
                    result["memory_sampling_error"] = sampler.error
                    if sampler.error or sampler.exceeded:
                        result["status"] = "failed"
                        failures += 1
                    else:
                        pass
                    (arguments.output / f"{name}.result.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in result.items()
                                if key != "text"
                            }
                        ),
                        flush=True,
                    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
