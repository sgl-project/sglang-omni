# SPDX-License-Identifier: Apache-2.0
"""Record and transcribe Full-Duplex-Bench v1.5 paired sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from benchmarks.duplex.artifacts import add_server_identity_args, server_identity
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES
from benchmarks.duplex.run_artifacts import TIMELINES, accounting, load_run
from benchmarks.duplex.v15_runner import run_samples
from benchmarks.duplex.v15_transcribe import add_transcribe_arguments


def record(args: argparse.Namespace) -> int:
    run_manifest = asyncio.run(
        run_samples(
            args.dataset_root,
            url=args.url,
            output=args.output,
            server=server_identity(
                args.url,
                revision=args.server_revision,
                model=args.model,
                model_revision=args.model_revision,
                runtime=args.runtime,
            ),
            dataset_revision=args.dataset_revision,
            timeout_s=args.timeout,
            profile=args.profile,
            sample_ids=args.sample_id,
            max_per_subset=args.max_per_subset,
        )
    )
    manifest, _, _ = load_run(args.output)
    summary = accounting(manifest, run_manifest)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return (
        0 if summary["variant_status"] == {"pass": summary["selected_variants"]} else 1
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record_parser = commands.add_parser("record", help="Run overlap and clean sessions")
    record_parser.add_argument("--dataset-root", type=Path, required=True)
    record_parser.add_argument("--url", required=True, help="Native /v1/realtime URL")
    record_parser.add_argument(
        "--output", type=Path, required=True, help="New immutable run directory"
    )
    add_server_identity_args(record_parser)
    record_parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    record_parser.add_argument(
        "--dataset-revision", required=True, help="Dataset release or archive digest"
    )
    record_parser.add_argument(
        "--timeout", type=float, default=60.0, help="Per-session deadline in seconds"
    )
    selection = record_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-id",
        action="append",
        help="Explicit <subset>/<name> sample; repeatable. Default: full dataset",
    )
    selection.add_argument("--max-per-subset", type=int)
    record_parser.set_defaults(handler=record)

    timeline_help = "; ".join(
        f"{name}: {timeline['meaning']}" for name, timeline in TIMELINES.items()
    )

    transcribe_parser = commands.add_parser(
        "transcribe", help="Word-timestamped Whisper ASR of generated output audio"
    )
    add_transcribe_arguments(transcribe_parser, timeline_help, "whisper")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
