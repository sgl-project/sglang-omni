# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench v1.5 runbook commands.

Run from the repository root with the sglang-omni venv active:
    python -m benchmarks.duplex.fdb_v15 setup
    python -m benchmarks.duplex.fdb_v15 generate --run-name smoke --per-subset 1
    python -m benchmarks.duplex.fdb_v15 asr --run-name smoke
    python -m benchmarks.duplex.fdb_v15 judge --run-name smoke
    python -m benchmarks.duplex.fdb_v15 aggregate --run-name smoke
"""

from __future__ import annotations

import argparse
import shlex
import signal
import sys
from pathlib import Path
from types import FrameType

from benchmarks.duplex.fdb_v15.aggregate import aggregate
from benchmarks.duplex.fdb_v15.asr import asr
from benchmarks.duplex.fdb_v15.common import load_settings, log
from benchmarks.duplex.fdb_v15.generate import generate
from benchmarks.duplex.fdb_v15.judge import judge
from benchmarks.duplex.fdb_v15.selection import (
    SampleSelection,
    parse_count,
    parse_subset_count,
)
from benchmarks.duplex.fdb_v15.servers import serve_judge, serve_model
from benchmarks.duplex.fdb_v15.setup_assets import setup

SIGTERM_EXIT_CODE = 143
DEFAULT_RUN_NAME = "minicpmo-48"
DEFAULT_PER_SUBSET = 12


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_argument_group(
        "sample selection",
        "Each category takes its first N samples in sample-ID order. "
        "Default: 12 per category (48 pairs).",
    )
    exclusive = selection.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--per-subset",
        type=parse_count,
        default=DEFAULT_PER_SUBSET,
        metavar="N|all",
        help="Pairs per category; 'all' selects all 498 pairs",
    )
    exclusive.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        metavar="CATEGORY/ID",
        help="Explicit pair; repeatable",
    )
    exclusive.add_argument(
        "--sample-ids-file",
        type=Path,
        help="One CATEGORY/ID per line, such as another run's sample-ids.txt",
    )
    selection.add_argument(
        "--subset-count",
        dest="subset_counts",
        type=parse_subset_count,
        action="append",
        default=[],
        metavar="CATEGORY=N|all",
        help="Override --per-subset for one category; 0 skips it; repeatable",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.set_defaults(run_name=DEFAULT_RUN_NAME)
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Results go to $FDB_WORK/runs/RUN_NAME (default: %(default)s)",
    )
    repeat_options = argparse.ArgumentParser(add_help=False, parents=[run_options])
    repeat_options.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Which independent generation of the same pairs: 1, 2, 3, ... "
        "(default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Download and pin every input; safe to rerun")
    generate_parser = commands.add_parser(
        "generate",
        parents=[repeat_options],
        help="Step 1: record sessions, export scoring audio",
    )
    add_selection_arguments(generate_parser)
    generate_parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Parallel sessions against the one server (default: %(default)s)",
    )
    for name, help_text in (
        ("asr", "Step 2: Parakeet ASR and official timing"),
        ("judge", "Step 3: behavior and semantic judges, report"),
    ):
        commands.add_parser(
            name, parents=[repeat_options], help=help_text
        ).add_argument(
            "--retry-failed", action="store_true", help="Retry failed work units"
        )
    commands.add_parser(
        "aggregate",
        parents=[run_options],
        help="Combine finished repeats into RESULTS.md",
    )
    commands.add_parser(
        "serve-model", help="Serve the model under test in the foreground"
    )
    commands.add_parser("serve-judge", help="Serve the Qwen judge in the foreground")
    commands.add_parser("env", help="Print shell exports for the manual CLI commands")
    return parser


def sample_selection(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> SampleSelection:
    sample_ids = args.sample_ids
    if args.sample_ids_file is not None:
        sample_ids = args.sample_ids_file.read_text().split()
    else:
        pass
    if sample_ids is not None and args.subset_counts:
        parser.error("--subset-count cannot be combined with explicit sample IDs")
    else:
        pass
    return SampleSelection(
        per_subset=args.per_subset,
        subset_counts=dict(args.subset_counts),
        sample_ids=sample_ids,
    )


def exit_on_sigterm(signal_number: int, frame: FrameType | None) -> None:
    # Raising here runs the finally blocks that stop a running server.
    sys.exit(SIGTERM_EXIT_CODE)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, exit_on_sigterm)
    settings = load_settings(args.run_name)
    settings.apply_job_isolation()
    if args.command != "env":
        log(
            f"Job GPU {settings.gpu}: model port {settings.server_port}, "
            f"judge port {settings.judge_port} "
            f"(nccl {settings.judge_nccl_port}), "
            f"judge config {settings.judge_dir}"
        )
    else:
        pass
    if args.command == "setup":
        setup(settings)
    elif args.command == "generate":
        generate(settings, args.repeat, sample_selection(parser, args), args.num_shards)
    elif args.command == "asr":
        asr(settings, args.repeat, args.retry_failed)
    elif args.command == "judge":
        judge(settings, args.repeat, args.retry_failed)
    elif args.command == "aggregate":
        aggregate(settings)
    elif args.command == "serve-model":
        serve_model(settings)
    elif args.command == "serve-judge":
        serve_judge(settings)
    elif args.command == "env":
        for name, value in settings.shell_exports().items():
            log(f"export {name}={shlex.quote(value)}")
    else:
        raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    main()
else:
    pass
