# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle stress for /v1/realtime: a model-free mock server and a scenario runner."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.artifacts import add_server_identity_args, server_identity
from benchmarks.duplex.client import PACKET_BYTES, SAMPLE_RATE
from benchmarks.duplex.lifecycle_stress.mock_server import MockSettings, serve_mock
from benchmarks.duplex.lifecycle_stress.scenarios import RunSettings, run_stress
from benchmarks.duplex.lifecycle_stress.summary import write_summary
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v15_audio import normalize_audio

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# note (Yi Zhao): the top level the load generator was shown to pace on one host.
DEFAULT_CONCURRENCY = 64
DEFAULT_CYCLES = 50
# note (Yi Zhao): 0.4 s of input, the shortest session that still gets a reply from the mock before it leaves.
DEFAULT_PACKETS_PER_SESSION = 5
DEFAULT_RECONNECTS = 2
# note (Yi Zhao): the length of a Full-Duplex-Bench session, for the scenario that runs whole conversations.
DEFAULT_LONG_SECONDS = 15.0
# note (Yi Zhao): above the server's default cleanup timeout of 30 s, so a slow adapter close is not a timeout here.
DEFAULT_CLEANUP_WAIT_S = 35.0


def mock(arguments: argparse.Namespace) -> None:
    settings = MockSettings(
        protocol=arguments.protocol,
        max_connections=arguments.max_connections,
        model_slots=arguments.model_slots,
        idle_timeout_s=arguments.idle_timeout_s,
        session_update_timeout_s=arguments.session_update_timeout_s,
        reply_chunks=arguments.reply_chunks,
        chunk_delay_ms=arguments.chunk_delay_ms,
        close_delay_ms=arguments.close_delay_ms,
        unit_ms=arguments.unit_ms,
        supports_cancel=arguments.cancel,
        takes_turns=arguments.takes_turns,
    )
    asyncio.run(serve_mock(settings, arguments.host, arguments.port))


def run(arguments: argparse.Namespace) -> None:
    scenario = arguments.scenario
    leave_point = arguments.leave_point
    leave_style = arguments.leave_style
    if leave_point == "end" and arguments.protocol == "legacy":
        # note (Yi Zhao): the legacy protocol has no session.close; a close frame is its ordinary goodbye.
        leave_style = "close_frame"
    elif leave_point == "end":
        leave_style = "goodbye"
    elif leave_style == "goodbye":
        raise ValueError(
            "leaving before the end needs --leave-style close_frame or reset"
        )
    else:
        pass
    if scenario == "drop" and leave_point == "end":
        raise ValueError(
            "the drop scenario needs --leave-point update, append or reply"
        )
    elif scenario in ("churn", "idle", "long", "interrupt") and leave_point != "end":
        raise ValueError(f"the {scenario} scenario does not take --leave-point")
    elif scenario == "interrupt" and arguments.audio_file is None:
        raise ValueError(
            "the interrupt scenario needs --audio-file with speech that talks over the reply"
        )
    else:
        pass
    if arguments.audio_file is None:
        input_seconds = (
            DEFAULT_LONG_SECONDS
            if scenario == "long"
            else arguments.packets_per_session * PACKET_BYTES / (2 * SAMPLE_RATE)
        )
        pcm = bytes(int(input_seconds * SAMPLE_RATE) * 2)
        audio_sha256: str | None = None
        audio_conversion: dict[str, JsonValue] | None = None
    else:
        # note (Yi Zhao): the conversion the Full-Duplex-Bench recorder applies, so both send the same bytes for the same file.
        pcm, audio_conversion = normalize_audio(arguments.audio_file)
        audio_sha256 = file_sha256(arguments.audio_file)
    settings = RunSettings(
        url=arguments.url,
        server=server_identity(
            arguments.url,
            revision=arguments.server_revision,
            model=arguments.model,
            model_revision=arguments.model_revision,
            runtime=arguments.runtime,
        ),
        scenario=scenario,
        protocol=arguments.protocol,
        concurrency=arguments.concurrency,
        cycles=arguments.cycles,
        pcm=pcm,
        is_input_silence=arguments.audio_file is None,
        audio_file=None if arguments.audio_file is None else str(arguments.audio_file),
        audio_sha256=audio_sha256,
        audio_conversion=audio_conversion,
        packets_per_session=arguments.packets_per_session,
        leave_point=leave_point,
        leave_style=leave_style,
        idle_point=arguments.idle_point if scenario == "idle" else "none",
        reconnects=arguments.reconnects if scenario == "reconnect" else 0,
        max_connections=arguments.max_connections,
        server_pid=arguments.server_pid,
        has_debug_route=arguments.debug_route,
        has_device_memory=arguments.device_memory,
        server_log=arguments.server_log,
        cleanup_wait_s=arguments.cleanup_wait_s,
        output=arguments.output,
    )
    samples = asyncio.run(run_stress(settings))
    print(write_summary(samples, settings.output).model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    mock_defaults = MockSettings()
    mock_parser = commands.add_parser(
        "mock", help="serve the real realtime route over a model-free adapter"
    )
    mock_parser.add_argument("--host", default=DEFAULT_HOST)
    mock_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    mock_parser.add_argument(
        "--protocol",
        choices=("native", "legacy"),
        default=mock_defaults.protocol,
        help="native serves shared sessions under a connection cap; legacy serves the facade without one",
    )
    mock_parser.add_argument(
        "--max-connections", type=int, default=mock_defaults.max_connections
    )
    mock_parser.add_argument(
        "--model-slots",
        type=int,
        default=mock_defaults.model_slots,
        help="sessions the stand-in model holds; further ones are rejected on session.update",
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
        "--reply-chunks",
        type=int,
        default=mock_defaults.reply_chunks,
        help="text deltas in the reply to each packet",
    )
    mock_parser.add_argument(
        "--chunk-delay-ms",
        type=float,
        default=mock_defaults.chunk_delay_ms,
        help="wait before each delta, which keeps a reply in progress",
    )
    mock_parser.add_argument(
        "--close-delay-ms",
        type=float,
        default=mock_defaults.close_delay_ms,
        help="how long the stand-in model takes to close a session",
    )
    mock_parser.add_argument(
        "--unit-ms",
        type=int,
        default=mock_defaults.unit_ms,
        help="input the stand-in model takes per step, as the model it imitates publishes: 80, 1000, 200 or 20",
    )
    mock_parser.add_argument(
        "--cancel",
        action=argparse.BooleanOptionalAction,
        default=mock_defaults.supports_cancel,
        help="--no-cancel imitates a model that refuses response.cancel",
    )
    mock_parser.add_argument(
        "--takes-turns",
        action="store_true",
        help="answer speech like a turn-based pipeline, one delta per silent unit, and end a reply as interrupted when speech meets it",
    )
    mock_parser.set_defaults(handle=mock)

    run_parser = commands.add_parser("run", help="run one scenario against a server")
    run_parser.add_argument("--url", required=True, help="ws://host:port/v1/realtime")
    add_server_identity_args(run_parser)
    run_parser.add_argument(
        "--scenario",
        choices=("churn", "drop", "cancel", "idle", "reconnect", "long", "interrupt"),
        required=True,
    )
    run_parser.add_argument(
        "--protocol", choices=("native", "legacy"), default="native"
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="clients per cycle; above --max-connections the excess is expected to be denied",
    )
    run_parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    run_parser.add_argument(
        "--packets-per-session",
        type=int,
        default=DEFAULT_PACKETS_PER_SESSION,
        help="80 ms input packets a short session sends",
    )
    run_parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="speech to send instead of silence, which a model needs to reply; converted to mono 16 kHz as the Full-Duplex-Bench recorder does",
    )
    run_parser.add_argument(
        "--leave-point",
        choices=("end", "update", "append", "reply"),
        default="end",
        help="where the client leaves early: after session.update, after its packets, or while a reply is in progress",
    )
    run_parser.add_argument(
        "--leave-style",
        choices=("goodbye", "close_frame", "reset"),
        default="reset",
        help="how the client leaves early",
    )
    run_parser.add_argument(
        "--idle-point",
        choices=("created", "open"),
        default="open",
        help="idle scenario: stay silent before or after session.update",
    )
    run_parser.add_argument(
        "--reconnects",
        type=int,
        default=DEFAULT_RECONNECTS,
        help="reconnect scenario: sessions each client opens after its first",
    )
    run_parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="the server's connection cap; given, every cycle ends by filling it to prove the slots are back",
    )
    run_parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="server process on this host, for its resident memory when it has no debug route",
    )
    run_parser.add_argument(
        "--debug-route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read the session count from the mock's debug route; a real server has none",
    )
    run_parser.add_argument(
        "--device-memory",
        action="store_true",
        help="read the memory in use on this host's GPUs with nvidia-smi after every cycle; the server has to run on this host",
    )
    run_parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="log file of a server on this host; what it gains during the run is copied to the output and its tracebacks and error lines are counted",
    )
    run_parser.add_argument(
        "--cleanup-wait-s", type=float, default=DEFAULT_CLEANUP_WAIT_S
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handle=run)

    arguments = parser.parse_args()
    arguments.handle(arguments)


if __name__ == "__main__":
    main()
