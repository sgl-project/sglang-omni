# SPDX-License-Identifier: Apache-2.0
"""Measure Qwen3-ASR long recordings through its public HTTP and realtime APIs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from sglang_omni_mlx.native.ci.long_audio_client import (
    HTTP_TRANSCRIPTION_FIELDS,
    MEMORY_SAMPLE_INTERVAL_SECONDS,
    HttpTranscription,
    MemorySampler,
    RealtimeTranscription,
    TranscriptScore,
    score,
    stream_recording,
    transcribe_chunks,
    transcribe_http,
)
from sglang_omni_mlx.native.ci.long_audio_corpus import (
    Clip,
    build_recording,
    chunk_frame_counts,
    load_clips,
)
from sglang_omni_mlx.native.ci.provision import sha256
from sglang_omni_mlx.native.tests.native_server import (
    EVENT_TIMEOUT_SECONDS,
    read_supervised_event,
)

# The worker returns MLX's cache after a request, with no signal to wait for.
MEMORY_SETTLE_SECONDS = 2
SAMPLER_JOIN_TIMEOUT_SECONDS = 5
SHUTDOWN_TIMEOUT_SECONDS = 10


@dataclass(kw_only=True)
class CaseResult:
    case: str
    audio_seconds: float
    status: Literal["measured", "failed"] = "failed"
    error: str = ""
    transcription: HttpTranscription | RealtimeTranscription | None = None
    score: TranscriptScore | None = None
    real_time_factor: float | None = None
    recovery_matches_warmup: bool | None = None
    baseline_footprint_bytes: int | None = None
    warmup_peak_footprint_bytes: int | None = None
    peak_footprint_bytes: int | None = None
    post_session_footprint_bytes: int | None = None
    post_recovery_footprint_bytes: int | None = None
    process_peak_footprint_bytes: int = 0
    server_exit_code: int | None = None
    memory_limit_exceeded: bool = False
    memory_sampling_error: str = ""


def server_binary(arguments: argparse.Namespace) -> Path:
    return arguments.runtime_bin.resolve() / "qwen3_asr_server"


def model_directory(arguments: argparse.Namespace) -> Path:
    return arguments.data_root.resolve() / "models" / arguments.model.replace("/", "_")


def write_metadata(arguments: argparse.Namespace) -> None:
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
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "binary_sha256": sha256(server_binary(arguments)),
        "model_files_sha256": {
            path.name: sha256(path)
            for path in sorted(model_directory(arguments).iterdir())
            if path.is_file()
        },
    }
    (arguments.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def stop_server(process: subprocess.Popen[bytes]) -> str | None:
    """Stop the server and close its pipes; returns what went wrong, if anything."""
    if process.poll() is not None:
        process.communicate()
        return f"Server exited before shutdown with code {process.returncode}"
    else:
        pass
    try:
        process.communicate(
            b'{"command":"shutdown"}\n', timeout=SHUTDOWN_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return f"Server did not shut down within {SHUTDOWN_TIMEOUT_SECONDS} seconds"
    if process.returncode != 0:
        return f"Server exited with code {process.returncode} during shutdown"
    else:
        return None


def run_case(
    arguments: argparse.Namespace, clips: list[Clip], language: str, minutes: float
) -> bool:
    """Measure one recording on a dedicated server; returns whether it was measured."""
    name = f"{language}-{minutes:g}m"
    wav_path = arguments.output / f"{name}.wav"
    recording = build_recording(clips, minutes * 60, arguments.seed, wav_path)
    (arguments.output / f"{name}.input.json").write_text(
        json.dumps(asdict(recording), ensure_ascii=False, indent=2) + "\n"
    )
    warmup_wav = clips[0].path.read_bytes()
    result = CaseResult(case=name, audio_seconds=recording.audio_seconds)
    errors: list[str] = []
    with (arguments.output / f"{name}.server.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(server_binary(arguments)),
                "--supervised",
                "--model-directory",
                str(model_directory(arguments)),
                *arguments.server_arguments,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
        )
    sampler = MemorySampler(
        process,
        arguments.output / f"{name}.memory.jsonl",
        int(arguments.max_footprint_gib * 1024**3),
        EVENT_TIMEOUT_SECONDS
        + recording.audio_seconds / arguments.speed
        + 3 * arguments.timeout_seconds,
    )
    sampler.thread.start()
    try:
        ready = read_supervised_event(process)
        if ready["event"] != "ready":
            raise RuntimeError(f"Server failed to start: {ready['reason']}")
        else:
            port = ready["port"]
        sampler.phase = "warmup"
        warmup = transcribe_http(port, warmup_wav, language, arguments.timeout_seconds)
        time.sleep(MEMORY_SETTLE_SECONDS)
        baseline = sampler.samples[-1]
        sampler.phase = "long_recording"
        if arguments.chunk_seconds is not None:
            transcription = transcribe_chunks(
                port,
                wav_path,
                chunk_frame_counts(recording.clip_frames, arguments.chunk_seconds),
                language,
                arguments.timeout_seconds,
            )
        elif arguments.mode == "http":
            transcription = transcribe_http(
                port, wav_path.read_bytes(), language, arguments.timeout_seconds
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
        time.sleep(MEMORY_SETTLE_SECONDS)
        session_end = sampler.samples[-1]
        result.transcription = transcription
        result.score = score(recording.reference, transcription.text, language)
        result.baseline_footprint_bytes = baseline.footprint_bytes
        result.warmup_peak_footprint_bytes = baseline.peak_footprint_bytes
        result.peak_footprint_bytes = session_end.peak_footprint_bytes
        result.post_session_footprint_bytes = session_end.footprint_bytes
        if arguments.mode == "http":
            result.real_time_factor = (
                transcription.wall_seconds / recording.audio_seconds
            )
        else:
            pass
        sampler.phase = "recovery"
        recovery = transcribe_http(
            port, warmup_wav, language, arguments.timeout_seconds
        )
        result.recovery_matches_warmup = recovery.text == warmup.text
        time.sleep(MEMORY_SETTLE_SECONDS)
        result.post_recovery_footprint_bytes = sampler.samples[-1].footprint_bytes
        if recovery.text != warmup.text:
            raise RuntimeError("Post-session transcript differs from warmup")
        else:
            result.status = "measured"
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        sampler.done.set()
        sampler.thread.join(timeout=SAMPLER_JOIN_TIMEOUT_SECONDS)
        shutdown_error = stop_server(process)
    if shutdown_error is not None:
        errors.append(shutdown_error)
    else:
        pass
    if errors or sampler.error or sampler.exceeded:
        result.status = "failed"
    else:
        pass
    result.error = "; ".join(errors)
    result.process_peak_footprint_bytes = max(
        (sample.peak_footprint_bytes for sample in sampler.samples), default=0
    )
    result.server_exit_code = process.returncode
    result.memory_limit_exceeded = sampler.exceeded
    result.memory_sampling_error = sampler.error
    result_fields = asdict(result)
    (arguments.output / f"{name}.result.json").write_text(
        json.dumps(result_fields, ensure_ascii=False, indent=2) + "\n"
    )
    if result_fields["transcription"] is not None:
        result_fields["transcription"].pop("text")
    else:
        pass
    print(json.dumps(result_fields), flush=True)
    return result.status == "measured"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Repository of a provisioned checkpoint, such as mlx-community/Qwen3-ASR-0.6B-4bit",
    )
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
        help="Soft guard on the server's peak footprint; allocations can overshoot before the next read",
    )
    parser.add_argument("--seed", type=int, default=2888)
    parser.add_argument(
        "server_arguments",
        nargs="*",
        help="Server flags after --, such as -- --cache-limit-mb 512",
    )
    arguments = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Requires macOS for its process accounting")
    elif not all(
        0 < value < math.inf
        for value in [
            *arguments.minutes,
            arguments.speed,
            arguments.timeout_seconds,
            arguments.max_footprint_gib,
        ]
    ):
        parser.error("Duration, speed, timeout and memory limit must be positive")
    elif arguments.chunk_seconds is not None and (
        arguments.mode != "http" or not 0 < arguments.chunk_seconds < math.inf
    ):
        parser.error("--chunk-seconds needs --mode http and a positive duration")
    elif arguments.mode == "http" and arguments.speed != 1:
        parser.error("--speed needs --mode realtime")
    else:
        pass
    arguments.output.mkdir(parents=True, exist_ok=False)
    write_metadata(arguments)
    all_measured = True
    for language in arguments.languages:
        clips = load_clips(arguments.data_root / "corpus" / "v1", language)
        for minutes in arguments.minutes:
            is_measured = run_case(arguments, clips, language, minutes)
            all_measured = all_measured and is_measured
    raise SystemExit(0 if all_measured else 1)


if __name__ == "__main__":
    main()
