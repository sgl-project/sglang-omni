# SPDX-License-Identifier: Apache-2.0
"""Record Full-Duplex-Bench v1.0 sessions and score them in the reference layout."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from openai import OpenAI

from benchmarks.duplex import v10_dataset
from benchmarks.duplex.artifacts import (
    add_protocol_args,
    add_server_identity_args,
    server_identity,
)
from benchmarks.duplex.reference_core import V10_REFERENCE_FILES, read_json
from benchmarks.duplex.reference_source import verify_reference
from benchmarks.duplex.reference_v10 import (
    JudgeLedger,
    evaluate,
    export,
    exported_subsets,
    transcribe,
)
from benchmarks.duplex.run_artifacts import accounting, load_run
from benchmarks.duplex.unified_v10 import render, score_tree
from benchmarks.duplex.v15_runner import run_samples

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
            kind=v10_dataset.RUN_KIND,
            profile=args.profile,
            concurrency=args.concurrency,
            turn_detection=args.turn_detection,
            legacy_tail_s=args.legacy_tail,
        )
    )
    manifest, _, _ = load_run(args.output)
    summary = accounting(manifest, run_manifest)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return (
        0 if summary["variant_status"] == {"pass": summary["selected_variants"]} else 1
    )


def reference_export(args: argparse.Namespace) -> int:
    manifest = export(args.run, args.dataset_root, args.out)
    print(json.dumps(manifest["counts"], indent=2))
    return 0


def reference_asr(args: argparse.Namespace) -> int:
    paths = verify_reference(args.reference_source, V10_REFERENCE_FILES)
    counts = transcribe(args.tree, paths, args.nemo, args.nemo_sha256, args.device)
    print(json.dumps(counts, indent=2))
    return 0


def reference_evaluate(args: argparse.Namespace) -> int:
    paths = verify_reference(args.reference_source, V10_REFERENCE_FILES)
    subsets = (
        [subset for subset in v10_dataset.SUBSETS if subset in args.subset]
        if args.subset
        else exported_subsets(args.tree)
    )
    if any(
        v10_dataset.SUBSET_TASKS[subset] == "user_interruption" for subset in subsets
    ):
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"interruption relevance needs an API key in ${args.api_key_env}"
            )
        else:
            pass
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        judge = JudgeLedger(
            client,
            args.tree / "judge" / f"user-interruption-{timestamp}.jsonl",
            str(client.base_url),
            args.served_model,
        )
    else:
        judge = None
    summary = evaluate(args.tree, paths, subsets, judge)
    print(
        json.dumps(
            {subset: summary["subsets"][subset]["result"] for subset in subsets},
            indent=2,
        )
    )
    return 0


def unified_score(args: argparse.Namespace) -> int:
    unified = score_tree(args.tree)
    summary_path = args.tree / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    changed = (
        []
        if summary is None
        else [
            name
            for name in ("manifest_sha256", "asr_sha256")
            if summary[name] != unified[name]
        ]
    )
    official = summary["subsets"] if summary is not None and not changed else {}
    print(render(unified, official))
    if changed:
        print(
            f"summary.json has a different {' and '.join(changed)} than this tree; "
            "official reference numbers omitted"
        )
    else:
        pass
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
    add_protocol_args(record_parser)
    record_parser.add_argument(
        "--dataset-revision", required=True, help="Dataset release or archive digest"
    )
    record_parser.add_argument(
        "--timeout", type=float, default=120.0, help="Per-session deadline in seconds"
    )
    record_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent variants; excess connections may receive "
        "HTTP 503 and fail if admission retries are exhausted",
    )
    selection = record_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-id",
        action="append",
        help="Explicit <subset>/<name> sample; repeatable. Default: full dataset",
    )
    selection.add_argument("--max-per-subset", type=int)
    record_parser.set_defaults(handler=record)

    export_parser = commands.add_parser(
        "reference-export",
        help="Fixed-window output.wav plus annotations in the reference layout",
    )
    export_parser.add_argument("--run", type=Path, action="append", required=True)
    export_parser.add_argument("--dataset-root", type=Path, required=True)
    export_parser.add_argument(
        "--out", type=Path, required=True, help="New reference tree directory"
    )
    export_parser.set_defaults(handler=reference_export)

    reference = argparse.ArgumentParser(add_help=False)
    reference.add_argument("--tree", type=Path, required=True)
    reference.add_argument("--reference-source", type=Path, required=True)

    asr_parser = commands.add_parser(
        "reference-asr", parents=[reference], help="Reference Parakeet transcripts"
    )
    asr_parser.add_argument(
        "--nemo", type=Path, required=True, help="Local parakeet-tdt-0.6b-v2.nemo"
    )
    asr_parser.add_argument("--nemo-sha256")
    asr_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    asr_parser.set_defaults(handler=reference_asr)

    evaluate_parser = commands.add_parser(
        "reference-evaluate",
        parents=[reference],
        help="Reference evaluation per subset into summary.json",
    )
    evaluate_parser.add_argument(
        "--subset",
        action="append",
        choices=v10_dataset.SUBSETS,
        help="Repeatable. Default: every exported subset",
    )
    evaluate_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    evaluate_parser.add_argument(
        "--base-url", help="Self-hosted OpenAI-compatible judge; non-official"
    )
    evaluate_parser.add_argument(
        "--served-model", help="Model name sent to --base-url in place of gpt-4-turbo"
    )
    evaluate_parser.set_defaults(handler=reference_evaluate)

    unified_parser = commands.add_parser(
        "unified-score",
        help="System-agnostic metrics from the reference transcripts and annotations",
    )
    unified_parser.add_argument(
        "--tree", type=Path, required=True, help="Tree transcribed by reference-asr"
    )
    unified_parser.set_defaults(handler=unified_score)

    args = parser.parse_args(argv)
    if args.command == "reference-evaluate" and (args.base_url is None) != (
        args.served_model is None
    ):
        parser.error("--base-url and --served-model go together")
    else:
        pass
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
