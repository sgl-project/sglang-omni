# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle stress for /v1/realtime: how a server recovers from churn.

Run from the repository root:
    python -m benchmarks.duplex.lifecycle_stress mock --max-connections 128
    python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8765/v1/realtime --server-revision "$(git rev-parse HEAD)" --model lifecycle-stress-mock --scenario churn --concurrency 64 --cycles 50 --output runs/churn
    python -m benchmarks.duplex.lifecycle_stress report runs/churn runs/drop --output results

mock serves the real realtime route over a model-free adapter and adds a debug route.
run drives one scenario against any realtime server and writes samples and a summary.
report pools several run directories into one summary and a markdown table.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from benchmarks.duplex.artifacts import add_server_identity_args, server_identity
from benchmarks.duplex.lifecycle_stress.mock_server import MockSettings, serve_mock
from benchmarks.duplex.lifecycle_stress.records import (
    REPORT_FILE,
    SAMPLES_FILE,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.scenarios import (
    CLEANUP_WAIT_S,
    RunSettings,
    run_stress,
)
from benchmarks.duplex.lifecycle_stress.session_client import (
    DEBUG_ROUTE_PATH,
    http_url_of,
)
from benchmarks.duplex.lifecycle_stress.summary import (
    REVISION_DISPLAY_LENGTH,
    leave_label,
    render_report,
    write_summary,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CONCURRENCY = 64
DEFAULT_CYCLES = 50
DEFAULT_APPENDS_PER_SESSION = 5
DEFAULT_RECONNECTS = 5


def mock(arguments: argparse.Namespace) -> None:
    settings = MockSettings(
        mode=arguments.mode,
        max_connections=arguments.max_connections,
        model_slots=arguments.model_slots,
        idle_timeout_s=arguments.idle_timeout_s,
        session_update_timeout_s=arguments.session_update_timeout_s,
        cleanup_timeout_s=arguments.cleanup_timeout_s,
        reply_text=arguments.reply_text,
        reply_chunks=arguments.reply_chunks,
        chunk_delay_ms=arguments.chunk_delay_ms,
        process_delay_ms=arguments.process_delay_ms,
        close_delay_ms=arguments.close_delay_ms,
    )
    asyncio.run(serve_mock(settings, arguments.host, arguments.port))


def run(arguments: argparse.Namespace) -> None:
    server = server_identity(
        arguments.url,
        revision=arguments.server_revision,
        model=arguments.model,
        model_revision=arguments.model_revision,
        runtime=arguments.runtime,
    )
    if arguments.no_debug_route:
        debug_url = None
    elif arguments.debug_url is not None:
        debug_url = arguments.debug_url
    else:
        debug_url = http_url_of(arguments.url, DEBUG_ROUTE_PATH)
    if arguments.scenario == "drop":
        drop_point = arguments.drop_point or "append"
    elif arguments.scenario in ("reconnect", "cancel"):
        drop_point = arguments.drop_point or "none"
    else:
        drop_point = "none"
    settings = RunSettings(
        url=arguments.url,
        server=server,
        scenario=arguments.scenario,
        concurrency=arguments.concurrency,
        cycles=arguments.cycles,
        appends_per_session=arguments.appends_per_session,
        close_style=arguments.close_style,
        drop_point=drop_point,
        drop_style=arguments.drop_style,
        idle_point=arguments.idle_point if arguments.scenario == "idle" else "none",
        reconnects=arguments.reconnects,
        max_connections=arguments.max_connections,
        server_pid=arguments.server_pid,
        debug_url=debug_url,
        cleanup_wait_s=arguments.cleanup_wait_s,
        output=arguments.output,
    )
    samples = asyncio.run(run_stress(settings))
    print(write_summary([samples], settings.output).model_dump_json(indent=2))


def report(arguments: argparse.Namespace) -> None:
    """Pool runs of one configuration against one server revision into one row each."""
    grouped: dict[str, list[StressSamples]] = {}
    for run_dir in arguments.run_dirs:
        samples = StressSamples.model_validate_json(
            (run_dir / SAMPLES_FILE).read_text()
        )
        mode = "unknown" if samples.baseline is None else samples.baseline.mode
        leave = leave_label(
            samples.idle_point,
            samples.drop_point,
            samples.drop_style,
            samples.close_style,
        ).replace(" ", "_")
        revision = str(samples.server["revision"])[:REVISION_DISPLAY_LENGTH]
        grouped.setdefault(f"{samples.scenario}_{mode}_{leave}_{revision}", []).append(
            samples
        )
    summaries = [
        write_summary(group, arguments.output / name)
        for name, group in sorted(grouped.items())
    ]
    report_text = render_report(summaries)
    (arguments.output / REPORT_FILE).write_text(report_text)
    print(report_text, end="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    mock_defaults = MockSettings()
    mock_parser = commands.add_parser(
        "mock", help="serve the model-free realtime route"
    )
    mock_parser.add_argument("--host", default=DEFAULT_HOST)
    mock_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    mock_parser.add_argument(
        "--mode", choices=("deployment", "legacy"), default=mock_defaults.mode
    )
    mock_parser.add_argument(
        "--max-connections", type=int, default=mock_defaults.max_connections
    )
    mock_parser.add_argument(
        "--model-slots",
        type=int,
        default=mock_defaults.model_slots,
        help="sessions the stand-in model holds at once; further ones get an admission_rejected error on session.update",
    )
    mock_parser.add_argument(
        "--idle-timeout-s", type=float, default=mock_defaults.idle_timeout_s
    )
    mock_parser.add_argument(
        "--session-update-timeout-s",
        type=float,
        default=mock_defaults.session_update_timeout_s,
    )
    mock_parser.add_argument(
        "--cleanup-timeout-s", type=float, default=mock_defaults.cleanup_timeout_s
    )
    mock_parser.add_argument("--reply-text", default=mock_defaults.reply_text)
    mock_parser.add_argument(
        "--reply-chunks",
        type=int,
        default=mock_defaults.reply_chunks,
        help="text deltas in each reply",
    )
    mock_parser.add_argument(
        "--chunk-delay-ms",
        type=float,
        default=mock_defaults.chunk_delay_ms,
        help="wait before each text delta, so a reply lasts long enough to be cancelled or left in progress",
    )
    mock_parser.add_argument(
        "--process-delay-ms", type=float, default=mock_defaults.process_delay_ms
    )
    mock_parser.add_argument(
        "--close-delay-ms", type=float, default=mock_defaults.close_delay_ms
    )
    mock_parser.set_defaults(handler=mock)

    run_parser = commands.add_parser(
        "run", help="drive one scenario against a realtime server"
    )
    run_parser.add_argument("--url", required=True, help="ws://host:port/v1/realtime")
    add_server_identity_args(run_parser)
    run_parser.add_argument(
        "--scenario",
        choices=("churn", "drop", "fill", "idle", "reconnect", "cancel"),
        required=True,
        help="churn: open, talk, close. drop: leave without reading replies. fill: hold --max-connections sessions open and try --concurrency more. idle: stop sending and wait for the server's timeout. reconnect: each client opens its next session the moment the last one ends. cancel: cancel the first reply while it is in progress",
    )
    run_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run_parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    run_parser.add_argument(
        "--appends-per-session", type=int, default=DEFAULT_APPENDS_PER_SESSION
    )
    run_parser.add_argument(
        "--close-style", choices=("event", "frame"), default="event"
    )
    run_parser.add_argument(
        "--drop-point",
        choices=("append", "update"),
        default=None,
        help="where a session leaves without reading replies: after the last append (the drop scenario's default), or right after session.update; reconnect and cancel close normally unless this is given",
    )
    run_parser.add_argument(
        "--drop-style",
        choices=("abort", "close"),
        default="abort",
        help="how it leaves: a TCP reset, or a close frame sent without reading the server's replies",
    )
    run_parser.add_argument(
        "--idle-point",
        choices=("created", "open"),
        default="open",
        help="where the idle scenario stops sending: before session.update, or after the last append",
    )
    run_parser.add_argument(
        "--reconnects",
        type=int,
        default=DEFAULT_RECONNECTS,
        help="sessions each client of the reconnect scenario opens after its first one in a cycle",
    )
    run_parser.add_argument("--max-connections", type=int, default=None)
    run_parser.add_argument("--server-pid", type=int, default=None)
    run_parser.add_argument("--debug-url", default=None)
    run_parser.add_argument("--no-debug-route", action="store_true")
    run_parser.add_argument("--cleanup-wait-s", type=float, default=CLEANUP_WAIT_S)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=run)

    report_parser = commands.add_parser(
        "report", help="pool run directories into one table"
    )
    report_parser.add_argument("run_dirs", nargs="+", type=Path)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.set_defaults(handler=report)

    arguments = parser.parse_args()
    if (
        arguments.command == "run"
        and arguments.scenario == "fill"
        and arguments.max_connections is None
    ):
        parser.error("--scenario fill needs --max-connections, the cap to fill")
    else:
        arguments.handler(arguments)


if __name__ == "__main__":
    main()
