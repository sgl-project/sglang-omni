# SPDX-License-Identifier: Apache-2.0
"""Measure Qwen3-ASR long recordings through its public HTTP and realtime APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import selectors
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from sglang_omni_mlx.native.ci.long_audio_client import (
    HTTP_TRANSCRIPTION_FIELDS,
    MEMORY_SAMPLE_INTERVAL_SECONDS,
    MemorySampler,
    score,
    stream_recording,
    transcribe_chunks,
    transcribe_http,
)
from sglang_omni_mlx.native.ci.long_audio_corpus import (
    build_recording,
    chunk_frame_counts,
    load_clips,
)

SHUTDOWN_TIMEOUT_SECONDS = 10


def stop_server(process: subprocess.Popen[str]) -> str | None:
    """Stop the server and close its pipes; returns what went wrong, if anything."""
    if process.poll() is not None:
        process.communicate()
        return f"Server exited before shutdown with code {process.returncode}"
    else:
        pass
    try:
        process.communicate(
            '{"command":"shutdown"}\n', timeout=SHUTDOWN_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return f"Server did not shut down within {SHUTDOWN_TIMEOUT_SECONDS} seconds"
    if process.returncode != 0:
        return f"Server exited with code {process.returncode} during shutdown"
    else:
        return None


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
        "--chunk-seconds",
        type=float,
        help="HTTP only: send whole clips in requests of at most this many seconds, each with Voxt's token budget",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300,
        help="Deadline of each HTTP request and realtime wait after audio input ends",
    )
    parser.add_argument(
        "--max-footprint-gib",
        type=float,
        default=8,
        help="Sampled soft guard for the server footprint; allocations can overshoot between samples",
    )
    parser.add_argument("--seed", type=int, default=2888)
    parser.add_argument(
        "server_arguments",
        nargs="*",
        help="Server flags after --, such as -- --cache-limit-mb 512",
    )
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
    elif arguments.chunk_seconds is not None and (
        arguments.mode != "http" or not 0 < arguments.chunk_seconds < math.inf
    ):
        parser.error("--chunk-seconds needs --mode http and a positive duration")
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
                        *arguments.server_arguments,
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
                    + 3 * arguments.timeout_seconds,
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
                    baseline = sampler.samples[-1]
                    sampler.phase = "long_recording"
                    if arguments.chunk_seconds is not None:
                        transcription = transcribe_chunks(
                            port,
                            wav_path,
                            chunk_frame_counts(
                                recording["clip_frames"], arguments.chunk_seconds
                            ),
                            language,
                            arguments.timeout_seconds,
                        )
                    elif arguments.mode == "http":
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
                    session_end = sampler.samples[-1]
                    result.update(
                        {
                            "baseline_footprint_bytes": baseline["footprint_bytes"],
                            "warmup_peak_footprint_bytes": baseline[
                                "peak_footprint_bytes"
                            ],
                            "peak_footprint_bytes": session_end["peak_footprint_bytes"],
                            "post_session_footprint_bytes": session_end[
                                "footprint_bytes"
                            ],
                            "rtf": transcription["wall_seconds"]
                            / recording["audio_seconds"],
                        }
                    )
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
                    if not result["recovery_matches_warmup"]:
                        raise RuntimeError(
                            "Post-session transcript differs from warmup"
                        )
                    else:
                        result["status"] = "measured"
                except Exception as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                    failures += 1
                finally:
                    sampler.done.set()
                    sampler.thread.join(timeout=5)
                    result["process_peak_footprint_bytes"] = max(
                        (sample["peak_footprint_bytes"] for sample in sampler.samples),
                        default=0,
                    )
                    shutdown_error = stop_server(process)
                    result["server_exit_code"] = process.returncode
                    if shutdown_error is not None:
                        result["status"] = "failed"
                        if "error" in result:
                            result["error"] = f"{result['error']}; {shutdown_error}"
                        else:
                            result["error"] = shutdown_error
                        failures += 1
                    else:
                        pass
                    result["memory_limit_exceeded"] = sampler.exceeded
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
