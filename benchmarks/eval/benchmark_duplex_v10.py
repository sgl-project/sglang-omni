# SPDX-License-Identifier: Apache-2.0
"""Record, transcribe and score Full-Duplex-Bench v1.0 turn-taking sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from benchmarks.duplex import v10_dataset
from benchmarks.duplex.artifacts import add_server_identity_args, server_identity
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES
from benchmarks.duplex.run_artifacts import TIMELINES, accounting, load_run
from benchmarks.duplex.v10_evaluation import RUN_KIND, score_run
from benchmarks.duplex.v15_runner import run_samples
from benchmarks.duplex.v15_transcribe import add_transcribe_arguments

VARIANTS = {"input": "input"}


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
            sample_ids=args.sample_id,
            max_per_subset=args.max_per_subset,
            dataset=v10_dataset,
            variants=VARIANTS,
            kind=RUN_KIND,
            profile=args.profile,
        )
    )
    manifest, _, _ = load_run(args.output)
    summary = accounting(manifest, run_manifest)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return (
        0 if summary["variant_status"] == {"pass": summary["selected_variants"]} else 1
    )


def score(args: argparse.Namespace) -> int:
    result = score_run(
        args.run,
        args.output,
        transcripts_dir=args.transcripts,
        timeline=args.timeline,
        backchannel_reference=args.backchannel_reference,
    )
    counts = result["accounting"]
    print(
        json.dumps(
            {
                "accounting": {
                    key: value for key, value in counts.items() if key != "failures"
                },
                "failures": len(counts["failures"]),
                "timeline": args.timeline,
                "tasks": result["scoring"]["tasks"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record_parser = commands.add_parser("record", help="Run one session per sample")
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
        "--timeout", type=float, default=120.0, help="Per-session deadline in seconds"
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
    # note (Jeffro): align with upstream which v1.0 uses Parakeet.
    add_transcribe_arguments(transcribe_parser, timeline_help, default_asr="parakeet")

    score_parser = commands.add_parser(
        "score", help="Offline takeover and latency scores from transcripts"
    )
    score_parser.add_argument("--run", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument(
        "--transcripts",
        type=Path,
        required=True,
        help="Directory written by the transcribe command",
    )
    score_parser.add_argument(
        "--timeline", choices=TIMELINES, default="simulated_playout", help=timeline_help
    )
    score_parser.add_argument(
        "--backchannel-reference",
        type=Path,
        help="Upstream icc_gt_distribution.json; without it backchannel timing is unscored",
    )
    score_parser.set_defaults(handler=score)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
