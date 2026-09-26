# SPDX-License-Identifier: Apache-2.0
"""Run a replayable continuous native session protocol case against a pinned endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import wave
from pathlib import Path

from benchmarks.duplex.artifacts import (
    add_server_identity_args,
    replay_run,
    server_identity,
    source_fingerprint,
)
from benchmarks.duplex.client import PACKET_MS, SAMPLE_RATE, TRANSPORT, run_session
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES

# Note (wenyao): Leave drain/close time before the server's 240 s deadline.
SESSION_LIMIT_S = 240.0
DRAIN_MARGIN_S = 10.0
MAX_TIMEOUT_S = SESSION_LIMIT_S - DRAIN_MARGIN_S


async def run(args: argparse.Namespace, pcm: bytes, server: dict) -> dict:
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "input.pcm").write_bytes(pcm)
    cases = [
        {
            "id": "continuous",
            "scenario": "continuous",
            "trace_file": "continuous.jsonl",
        },
    ]
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "source": source_fingerprint(),
        "server": server,
        "config": {
            "packet_ms": PACKET_MS,
            "paced": True,
            "timeout_s": args.timeout,
            "input_duration_s": len(pcm) / (SAMPLE_RATE * 2),
            "transport": TRANSPORT,
            "unsupported": [
                "response_cancel",
                "session_resume",
                "truncate",
            ],
            "unmeasured": [
                "automatic_speech_interruption",
                "concurrent_native_sessions",
                "semantic_quality",
                "acoustic_speech_onset",
                "audible_stop_time",
                "server_resource_release",
            ],
        },
        "input": {
            "file": "input.pcm",
            "sha256": hashlib.sha256(pcm).hexdigest(),
            "sample_rate": SAMPLE_RATE,
        },
        "cases": cases,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    for case in cases:
        await run_session(
            args.url,
            pcm,
            scenario=case["scenario"],
            trace_path=args.output / case["trace_file"],
            timeout_s=args.timeout,
            profile=args.profile,
        )
    report = replay_run(args.output)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="Native /v1/realtime WebSocket URL"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="PCM16 mono 16 kHz WAV with trailing silence",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable run directory"
    )
    add_server_identity_args(parser)
    parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="Whole-session deadline in seconds"
    )
    args = parser.parse_args()
    if not 0 < args.timeout <= MAX_TIMEOUT_S:
        parser.error(f"--timeout must be positive and at most {MAX_TIMEOUT_S} seconds")
    try:
        with wave.open(str(args.audio), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
                1,
                2,
                SAMPLE_RATE,
            ):
                parser.error("--audio must be PCM16 mono at 16000 Hz")
            pcm = audio.readframes(audio.getnframes())
    except (OSError, wave.Error) as exc:
        parser.error(f"--audio is unreadable: {exc}")
    duration_s = len(pcm) / (SAMPLE_RATE * 2)
    if not 0 < duration_s < MAX_TIMEOUT_S:
        parser.error(
            f"--audio duration must be positive and below {MAX_TIMEOUT_S} seconds"
        )
    if args.timeout <= duration_s:
        parser.error(
            f"--timeout must exceed the {duration_s:.2f} second paced input duration"
        )
    try:
        server = server_identity(
            args.url,
            revision=args.server_revision,
            model=args.model,
            model_revision=args.model_revision,
            runtime=args.runtime,
        )
    except ValueError as exc:
        parser.error(str(exc))
    report = asyncio.run(run(args, pcm, server))
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    raise SystemExit(
        0 if report["summary"]["passed"] == report["summary"]["selected"] else 1
    )


if __name__ == "__main__":
    main()
