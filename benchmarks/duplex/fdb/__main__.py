# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench runbook commands (v1.0 and v1.5).

Run from the repository root with the sglang-omni venv active:
    python -m benchmarks.duplex.fdb setup
    python -m benchmarks.duplex.fdb generate --run-name smoke --v15-per-subset 1 --v10-per-subset 3
    python -m benchmarks.duplex.fdb asr --run-name smoke
    python -m benchmarks.duplex.fdb judge --run-name smoke
    python -m benchmarks.duplex.fdb aggregate --run-name smoke

Each dataset has its own selection options; a count of 0 leaves it out.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import sys
from functools import partial
from pathlib import Path
from types import FrameType

from benchmarks.duplex.fdb.aggregate import aggregate
from benchmarks.duplex.fdb.asr import asr
from benchmarks.duplex.fdb.common import configure_logging, load_settings, log
from benchmarks.duplex.fdb.generate import generate
from benchmarks.duplex.fdb.judge import judge
from benchmarks.duplex.fdb.selection import (
    SampleSelection,
    parse_count,
    parse_subset_count,
)
from benchmarks.duplex.fdb.servers import serve_judge, serve_model
from benchmarks.duplex.fdb.setup_assets import setup
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS
from benchmarks.duplex.v15_dataset import SUBSETS as V15_SUBSETS

SIGTERM_EXIT_CODE = 143
DEFAULT_RUN_NAME = "minicpmo-48"
DEFAULT_PER_SUBSET = 12
# note (luojiaxuan): option prefix, name, subsets, unit and size of each dataset.
DATASETS = (
    ("v15", "v1.5", V15_SUBSETS, "pairs", 498),
    ("v10", "v1.0", V10_SUBSETS, "samples", 727),
)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_argument_group(
        "sample selection",
        "Each dataset takes the first N samples of every subset in sample-ID "
        "order; v1.5 counts pairs, v1.0 counts samples. Default: 12 per subset "
        "of both. A count of 0 leaves that dataset out of the run.",
    )
    for prefix, label, subsets, unit, total in DATASETS:
        exclusive = selection.add_mutually_exclusive_group()
        exclusive.add_argument(
            f"--{prefix}-per-subset",
            dest=f"{prefix}_per_subset",
            type=parse_count,
            default=DEFAULT_PER_SUBSET,
            metavar="N|all",
            help=f"{label} {unit} per subset; 'all' selects all {total}, 0 leaves "
            f"{label} out",
        )
        exclusive.add_argument(
            f"--{prefix}-sample-id",
            dest=f"{prefix}_sample_ids",
            action="append",
            metavar="SUBSET/ID",
            help=f"Exactly these {label} {unit}; repeatable",
        )
        exclusive.add_argument(
            f"--{prefix}-sample-ids-file",
            dest=f"{prefix}_sample_ids_file",
            type=Path,
            help=f"One SUBSET/ID per line, such as another run's {label} sample-ids.txt",
        )
        selection.add_argument(
            f"--{prefix}-subset-count",
            dest=f"{prefix}_subset_counts",
            type=partial(parse_subset_count, subsets=subsets),
            action="append",
            default=[],
            metavar="SUBSET=N|all",
            help=f"Overrides --{prefix}-per-subset for one {label} subset; 0 skips "
            "it; repeatable",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.set_defaults(run_name=DEFAULT_RUN_NAME)
    log_options = argparse.ArgumentParser(add_help=False)
    log_options.add_argument(
        "--log-level",
        choices=("debug", "info"),
        default="info",
        help="debug also streams the server and recorder logs this step writes "
        "(default: %(default)s)",
    )
    run_options = argparse.ArgumentParser(add_help=False, parents=[log_options])
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
    commands.add_parser(
        "setup",
        parents=[log_options],
        help="Download and pin every input; safe to rerun",
    )
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
        "serve-model",
        parents=[log_options],
        help="Serve the model under test in the foreground",
    )
    commands.add_parser(
        "serve-judge",
        parents=[log_options],
        help="Serve the Qwen judge in the foreground",
    )
    commands.add_parser(
        "env",
        parents=[log_options],
        help="Print shell exports for the manual CLI commands",
    )
    return parser


def sample_selection(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[SampleSelection, SampleSelection]:
    """The v1.5 pair selection and the v1.0 sample selection."""
    selections = []
    for prefix, label, _, _, _ in DATASETS:
        sample_ids = getattr(args, f"{prefix}_sample_ids")
        ids_file = getattr(args, f"{prefix}_sample_ids_file")
        if ids_file is not None:
            sample_ids = ids_file.read_text().split()
        else:
            pass
        subset_counts = getattr(args, f"{prefix}_subset_counts")
        if sample_ids is not None and subset_counts:
            parser.error(
                f"--{prefix}-subset-count cannot be combined with explicit {label} "
                "sample IDs"
            )
        else:
            pass
        selections.append(
            SampleSelection(
                per_subset=getattr(args, f"{prefix}_per_subset"),
                subset_counts=dict(subset_counts),
                sample_ids=sample_ids,
            )
        )
    v15_selection, v10_selection = selections
    return v15_selection, v10_selection


def exit_on_sigterm(signal_number: int, frame: FrameType | None) -> None:
    # Raising here runs the finally blocks that stop a running server.
    sys.exit(SIGTERM_EXIT_CODE)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
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
        generate(
            settings, args.repeat, *sample_selection(parser, args), args.num_shards
        )
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
