# SPDX-License-Identifier: Apache-2.0
"""Record one duplex system at several concurrency levels and report its serving timing."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.serving import serving_report
from benchmarks.duplex.v15_audio import write_json

logger = logging.getLogger(__name__)
RECORDERS = {
    "v1.0": "benchmarks.eval.benchmark_duplex_v10",
    "v1.5": "benchmarks.eval.benchmark_duplex_v15",
}
# Options the sweep sets on every recorder run itself.
SWEEP_OPTIONS = ("--concurrency", "--output")

# note (jeffro): this is a small default startup buffer, and other values can be passed to trace the underrun curve.
DEFAULT_PLAYBACK_RESERVE_MS = 80.0


def milliseconds(value_s: JsonValue) -> str:
    return "-" if value_s is None else f"{value_s * 1000:.0f}"


def ratio(value: JsonValue, form: str) -> str:
    return "-" if value is None else format(value, form)


def format_table(report: dict[str, JsonValue]) -> str:
    rows = [
        (
            "| N | mean open | sessions failed | 503 / in-session | rejected "
            "| affected | denial events | pacing violations "
            "| unit miss | unit lag p50 / p90 / p99 (ms) | first audio p50 / p90 (ms) "
            "| buffer needed p50 / p90 (ms) | underrun replies | RTF p90 "
            "| meets targets |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level in report["levels"]:
        real_time_factor = level["real_time_factor"]
        unit_lag = level["unit_lag_s"]
        first_audio = level["first_audio_latency_s"]
        buffer_needed = level["startup_buffer_needed_s"]
        rows.append(
            f"| {level['concurrency']} | {level['mean_open_sessions']:.1f} "
            f"| {level['sessions_failed']}/{level['sessions']} "
            f"| {level['admission_denials']} / {level['sessions_with_admission_rejections']} "
            f"| {level['sessions_rejected']} | {level['sessions_with_admission_events']} "
            f"| {level['admission_denial_events']} "
            f"| {level['input_pacing_violations']} "
            f"| {ratio(level['unit_miss_rate'], '.1%')} "
            f"| {milliseconds(unit_lag['p50'])} / {milliseconds(unit_lag['p90'])} / "
            f"{milliseconds(unit_lag['p99'])} "
            f"| {milliseconds(first_audio['p50'])} / {milliseconds(first_audio['p90'])} "
            f"| {milliseconds(buffer_needed['p50'])} / "
            f"{milliseconds(buffer_needed['p90'])} "
            f"| {level['replies_with_underrun']}/{level['replies']} "
            f"| {ratio(None if real_time_factor is None else real_time_factor['p90'], '.2f')} "
            f"| {'yes' if not level['unmet_targets'] else '; '.join(level['unmet_targets'])} |"
        )
    capacity = report["max_concurrency_meeting_targets"]
    if capacity is None:
        verdict = "No recorded concurrency meets the targets."
    elif report["capacity_bounded"]:
        verdict = f"Highest concurrency meeting the targets: {capacity}."
    else:
        verdict = (
            f"Every recorded concurrency meets the targets; capacity is at least "
            f"{capacity}."
        )
    return "\n".join(rows) + f"\n\n{verdict}\n"


def concurrency_levels(text: str) -> list[int]:
    """Distinct positive levels separated by commas, such as 1,2,4,8."""
    levels = [int(part) for part in text.split(",")]
    if min(levels) < 1 or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError(
            f"expected distinct positive levels such as 1,2,4,8, got {text!r}"
        )
    else:
        return sorted(levels)


def report(args: argparse.Namespace) -> int:
    serving = serving_report(args.run, args.playback_reserve_ms / 1000)
    write_json(args.output, serving)
    print(format_table(serving))
    return 0


def sweep(args: argparse.Namespace) -> int:
    recorder = import_module(RECORDERS[args.dataset]).main
    args.output.mkdir(parents=True, exist_ok=False)
    run_dirs = []
    for concurrency in args.concurrency:
        run_dir = args.output / f"concurrency-{concurrency}"
        logger.info(
            f"Recording {args.dataset} at concurrency {concurrency} -> {run_dir}"
        )
        # note (jeffro): the recorder exits nonzero when a session fails, which under
        # load is a result to report, not a reason to stop the sweep.
        recorder(
            [
                "record",
                *args.recorder_args,
                "--concurrency",
                str(concurrency),
                "--output",
                str(run_dir),
            ]
        )
        run_dirs.append(run_dir)
    serving = serving_report(run_dirs, args.playback_reserve_ms / 1000)
    write_json(args.output / "serving.json", serving)
    table = format_table(serving)
    (args.output / "SERVING.md").write_text(table)
    print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    playback = argparse.ArgumentParser(add_help=False)
    playback.add_argument(
        "--playback-reserve-ms",
        type=float,
        default=DEFAULT_PLAYBACK_RESERVE_MS,
        help="Simulated player startup buffer (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    report_parser = commands.add_parser(
        "report", parents=[playback], help="Report recorded runs"
    )
    report_parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Recorded run directory, one per concurrency level",
    )
    report_parser.add_argument(
        "--output", type=Path, required=True, help="serving.json path"
    )
    report_parser.set_defaults(handler=report)

    sweep_parser = commands.add_parser(
        "sweep",
        parents=[playback],
        help="Record every level against one running server, then report",
        description="Options after -- go to the recorder's record command unchanged "
        "(--url, --profile, --dataset-root, server identity, sample selection); the "
        "sweep sets --concurrency and --output per level.",
    )
    sweep_parser.add_argument("--dataset", choices=RECORDERS, required=True)
    sweep_parser.add_argument(
        "--concurrency",
        type=concurrency_levels,
        required=True,
        help="Levels to record, such as 1,2,4,8",
    )
    sweep_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the level runs, serving.json and SERVING.md",
    )
    sweep_parser.add_argument("recorder_args", nargs=argparse.REMAINDER)
    sweep_parser.set_defaults(handler=sweep)

    args = parser.parse_args(argv)
    if args.playback_reserve_ms < 0:
        parser.error("--playback-reserve-ms must be nonnegative")
    elif args.command == "sweep":
        args.recorder_args = [arg for arg in args.recorder_args if arg != "--"]
        if any(arg.split("=")[0] in SWEEP_OPTIONS for arg in args.recorder_args):
            parser.error(f"the sweep sets {' and '.join(SWEEP_OPTIONS)} itself")
        else:
            pass
    else:
        pass
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
