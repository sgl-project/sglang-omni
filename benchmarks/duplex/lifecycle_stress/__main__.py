# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle stress for /v1/realtime: how a server recovers from churn.

Run from the repository root:
    python -m benchmarks.duplex.lifecycle_stress mock --max-connections 128
    python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8765/v1/realtime --scenario churn --concurrency 64 --cycles 50 --output runs/churn
    python -m benchmarks.duplex.lifecycle_stress report runs/churn runs/drop --output results

mock serves the real realtime route over a model-free adapter and adds a debug route.
run drives one scenario against any realtime server and writes samples and a summary.
report pools several run directories into one summary and a markdown table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.duplex.lifecycle_stress.mock_server import MockSettings, serve_mock
from benchmarks.duplex.lifecycle_stress.records import SAMPLES_FILE, StressSamples
from benchmarks.duplex.lifecycle_stress.scenarios import (
    CLEANUP_WAIT_S,
    RunSettings,
    run_stress,
)
from benchmarks.duplex.lifecycle_stress.session_client import debug_url_of
from benchmarks.duplex.lifecycle_stress.summary import write_report, write_summary

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CONCURRENCY = 64
DEFAULT_CYCLES = 50
DEFAULT_APPENDS_PER_SESSION = 5


def add_mock_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    defaults = MockSettings()
    parser = subparsers.add_parser("mock", help="serve the model-free realtime route")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--mode", choices=("deployment", "legacy"), default=defaults.mode
    )
    parser.add_argument("--max-connections", type=int, default=defaults.max_connections)
    parser.add_argument("--idle-timeout-s", type=float, default=defaults.idle_timeout_s)
    parser.add_argument(
        "--session-update-timeout-s",
        type=float,
        default=defaults.session_update_timeout_s,
    )
    parser.add_argument(
        "--cleanup-timeout-s", type=float, default=defaults.cleanup_timeout_s
    )
    parser.add_argument("--reply-text", default=defaults.reply_text)
    parser.add_argument(
        "--process-delay-ms", type=float, default=defaults.process_delay_ms
    )
    parser.add_argument("--close-delay-ms", type=float, default=defaults.close_delay_ms)


def add_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "run", help="drive one scenario against a realtime server"
    )
    parser.add_argument("--url", required=True, help="ws://host:port/v1/realtime")
    parser.add_argument("--scenario", choices=("churn", "drop"), required=True)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument(
        "--appends-per-session", type=int, default=DEFAULT_APPENDS_PER_SESSION
    )
    parser.add_argument("--close-style", choices=("event", "frame"), default="event")
    parser.add_argument(
        "--drop-point",
        choices=("append", "update"),
        default="append",
        help="where the drop scenario leaves: after the last append, or right after session.update",
    )
    parser.add_argument(
        "--drop-style",
        choices=("abort", "close"),
        default="abort",
        help="how it leaves: a TCP reset, or a close frame sent without reading the server's replies",
    )
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--debug-url", default=None)
    parser.add_argument("--no-debug-route", action="store_true")
    parser.add_argument("--cleanup-wait-s", type=float, default=CLEANUP_WAIT_S)
    parser.add_argument("--output", type=Path, required=True)


def add_report_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("report", help="pool run directories into one table")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)


def run_settings(arguments: argparse.Namespace) -> RunSettings:
    if arguments.scenario == "drop":
        drop_point = arguments.drop_point
    else:
        drop_point = "none"
    if arguments.no_debug_route:
        debug_url = None
    elif arguments.debug_url is not None:
        debug_url = arguments.debug_url
    else:
        debug_url = debug_url_of(arguments.url)
    return RunSettings(
        url=arguments.url,
        scenario=arguments.scenario,
        concurrency=arguments.concurrency,
        cycles=arguments.cycles,
        appends_per_session=arguments.appends_per_session,
        close_style=arguments.close_style,
        drop_point=drop_point,
        drop_style=arguments.drop_style,
        max_connections=arguments.max_connections,
        server_pid=arguments.server_pid,
        debug_url=debug_url,
        cleanup_wait_s=arguments.cleanup_wait_s,
        output=arguments.output,
    )


def report(run_dirs: list[Path], output: Path) -> None:
    """Pool runs of one configuration (several client processes) into one row each."""
    grouped: dict[str, list[StressSamples]] = {}
    for run_dir in run_dirs:
        samples = StressSamples.model_validate_json(
            (run_dir / SAMPLES_FILE).read_text()
        )
        mode = "unknown" if samples.baseline is None else samples.baseline.mode
        leave = (
            f"{samples.drop_point}_{samples.drop_style}"
            if samples.scenario == "drop"
            else samples.close_style
        )
        grouped.setdefault(f"{samples.scenario}_{mode}_{leave}", []).append(samples)
    summaries = [
        write_summary(group, output / name) for name, group in sorted(grouped.items())
    ]
    print(write_report(summaries, output), end="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_mock_parser(subparsers)
    add_run_parser(subparsers)
    add_report_parser(subparsers)
    arguments = parser.parse_args()
    if arguments.command == "mock":
        settings = MockSettings(
            mode=arguments.mode,
            max_connections=arguments.max_connections,
            idle_timeout_s=arguments.idle_timeout_s,
            session_update_timeout_s=arguments.session_update_timeout_s,
            cleanup_timeout_s=arguments.cleanup_timeout_s,
            reply_text=arguments.reply_text,
            process_delay_ms=arguments.process_delay_ms,
            close_delay_ms=arguments.close_delay_ms,
        )
        asyncio.run(serve_mock(settings, arguments.host, arguments.port))
    elif arguments.command == "run":
        settings = run_settings(arguments)
        samples = asyncio.run(run_stress(settings))
        print(json.dumps(write_summary([samples], settings.output), indent=2))
    else:
        report(arguments.run_dirs, arguments.output)


if __name__ == "__main__":
    main()
