# SPDX-License-Identifier: Apache-2.0
"""Record, transcribe and score Full-Duplex-Bench v1.5 paired sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from benchmarks.duplex.artifacts import add_server_identity_args, server_identity
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES
from benchmarks.duplex.v15_evaluation import TIMELINES, accounting, load_run, score_run
from benchmarks.duplex.v15_runner import run_pairs
from benchmarks.duplex.v15_transcribe import transcribe_run


def record(args: argparse.Namespace) -> int:
    run = asyncio.run(
        run_pairs(
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
    summary = accounting(manifest, run)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return (
        0 if summary["variant_status"] == {"pass": summary["selected_variants"]} else 1
    )


def transcribe(args: argparse.Namespace) -> int:
    import whisper

    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"--model-path is not a local checkpoint: {args.model_path}"
        )
    model = whisper.load_model(str(args.model_path), device=args.device)
    result = transcribe_run(
        args.run,
        args.output,
        model=model,
        model_path=args.model_path,
        device=args.device,
        timeline=args.timeline,
    )
    statuses = [entry["status"] for entry in result["variants"]]
    print(
        json.dumps({status: statuses.count(status) for status in sorted(set(statuses))})
    )
    return 1 if "error" in statuses else 0


def score(args: argparse.Namespace) -> int:
    judge = None
    if args.judge_base_url is not None:
        judge = {
            "base_url": args.judge_base_url,
            "model": args.judge_model,
            "api_key_env": args.judge_api_key_env,
            "timeout_s": args.judge_timeout,
        }
    result = score_run(
        args.run,
        args.output,
        timeline=args.timeline,
        dataset_root=args.dataset_root,
        segments_path=args.segments,
        transcripts_dir=args.transcripts,
        judgements_path=args.judgements,
        judge=judge,
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
                "timing": {
                    category: {
                        key: summary[key]
                        for key in (
                            "selected",
                            "missing",
                            "eligible",
                            "paired_eligible",
                        )
                    }
                    for category, summary in result["timing"]["categories"].items()
                },
                "behavior": {
                    category: {
                        key: summary[key]
                        for key in ("selected", "scored", "label_counts")
                    }
                    for category, summary in result["behavior"]["summary"][
                        "categories"
                    ].items()
                },
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


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
        f"{name}: {spec['meaning']}" for name, spec in TIMELINES.items()
    )

    transcribe_parser = commands.add_parser(
        "transcribe", help="Word-timestamped Whisper ASR of generated output audio"
    )
    transcribe_parser.add_argument("--run", type=Path, required=True)
    transcribe_parser.add_argument("--output", type=Path, required=True)
    transcribe_parser.add_argument(
        "--model-path", type=Path, required=True, help="Local Whisper .pt checkpoint"
    )
    transcribe_parser.add_argument("--device", required=True)
    transcribe_parser.add_argument(
        "--timeline", choices=TIMELINES, default="simulated_playout", help=timeline_help
    )
    transcribe_parser.set_defaults(handler=transcribe)

    score_parser = commands.add_parser(
        "score", help="Offline timing and behavior scores"
    )
    score_parser.add_argument("--run", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument(
        "--timeline", choices=TIMELINES, default="simulated_playout", help=timeline_help
    )
    score_parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset copy holding input.json/clean_input.json; default: recorded root",
    )
    score_parser.add_argument(
        "--segments", type=Path, help="Supplied speech segments instead of Silero VAD"
    )
    score_parser.add_argument(
        "--transcripts", type=Path, help="Directory written by the transcribe command"
    )
    score_parser.add_argument("--judgements", type=Path, help="Offline judgement JSONL")
    score_parser.add_argument("--judge-base-url")
    score_parser.add_argument("--judge-model")
    score_parser.add_argument(
        "--judge-api-key-env", help="Name of the variable holding the judge API key"
    )
    score_parser.add_argument("--judge-timeout", type=float, default=60.0)
    score_parser.set_defaults(handler=score)

    args = parser.parse_args(argv)
    if args.command == "score":
        judge_args = (args.judge_base_url, args.judge_model, args.judge_api_key_env)
        if len({arg is None for arg in judge_args}) > 1:
            parser.error(
                "--judge-base-url, --judge-model and --judge-api-key-env go together"
            )
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
