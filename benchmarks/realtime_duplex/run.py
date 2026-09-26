# SPDX-License-Identifier: Apache-2.0
"""CLI for the duplex benchmark: run one session, then verdict it offline.

Live mode drives a session against a websocket endpoint, writes the
session log, and prints the replay report. Replay-only mode verdicts an
existing log without any server. Exit code is 0 only when every verdict
passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from benchmarks.realtime_duplex import schema
from benchmarks.realtime_duplex.harness import MODES, DuplexSessionDriver
from benchmarks.realtime_duplex.replay import replay
from benchmarks.realtime_duplex.session_log import read_session_log


def synthesize_audio(duration_s: float) -> bytes:
    """Deterministic stand-in input so runs are reproducible without a
    dataset download."""
    total = int(duration_s * schema.SAMPLE_RATE) * schema.BYTES_PER_SAMPLE
    return (b"\x55\xaa" * ((total + 1) // 2))[:total]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="websocket url of the realtime endpoint")
    parser.add_argument("--audio", type=Path, help="raw pcm16 16 kHz input file")
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="seconds of synthetic audio when --audio is absent",
    )
    parser.add_argument("--mode", choices=MODES, default="continuous")
    parser.add_argument("--frame-ms", type=int, default=schema.DEFAULT_FRAME_MS)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--out", type=Path, default=Path("duplex_session.jsonl"))
    parser.add_argument(
        "--replay-only", type=Path, help="verdict an existing session log and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.replay_only is not None:
        manifest, entries = read_session_log(args.replay_only)
        report = replay(manifest, entries)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if report.passed else 1

    if not args.url:
        print("--url is required unless --replay-only is given", file=sys.stderr)
        return 2
    if args.audio is not None:
        audio = args.audio.read_bytes()
    else:
        audio = synthesize_audio(args.duration)

    async def drive() -> object:
        driver = DuplexSessionDriver(
            args.url,
            audio,
            mode=args.mode,
            frame_ms=args.frame_ms,
            pace=args.pace,
            model=args.model,
        )
        return await driver.run()

    record = asyncio.run(drive())
    record.save(args.out)
    report = replay(record.manifest, record.entries)
    payload = report.to_dict()
    payload["client_error"] = record.client_error
    payload["session_log"] = str(args.out)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if report.passed and record.client_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
