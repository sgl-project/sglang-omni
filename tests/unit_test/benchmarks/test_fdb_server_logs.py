# SPDX-License-Identifier: Apache-2.0
"""The runbook's --log-level, the log followers behind it, and the handle that
stops a step when its server exited, with real subprocesses and no GPU."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from benchmarks.duplex.fdb import servers as servers_module
from benchmarks.duplex.fdb.__main__ import build_parser
from benchmarks.duplex.fdb.common import (
    FOLLOW_INTERVAL_S,
    LOGGER,
    configure_logging,
    follow_log,
    log,
)
from benchmarks.duplex.fdb.servers import RunningServer

WAIT_TIMEOUT_S = 10
SUBCOMMANDS = (
    "setup",
    "generate",
    "asr",
    "judge",
    "aggregate",
    "serve-model",
    "serve-judge",
    "env",
)


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.05)


def followed(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]


@pytest.fixture
def runbook_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let a test configure the runbook logger without leaking its handler."""
    monkeypatch.setattr(LOGGER, "handlers", [])
    yield
    LOGGER.setLevel(logging.NOTSET)


def test_every_subcommand_takes_a_log_level_that_defaults_to_info() -> None:
    parser = build_parser()
    # note (luojiaxuan): the usage line lists every subcommand, so a new one
    # fails here until it joins SUBCOMMANDS.
    assert "{" + ",".join(SUBCOMMANDS) + "}" in parser.format_usage()
    for name in SUBCOMMANDS:
        assert parser.parse_args([name]).log_level == "info"
        assert parser.parse_args([name, "--log-level", "debug"]).log_level == "debug"
    with pytest.raises(SystemExit):
        parser.parse_args(["judge", "--log-level", "warning"])


@pytest.mark.parametrize(
    ("level", "printed"),
    [
        ("info", "== Starting the judge server\n"),
        ("debug", "== Starting the judge server\n[judge-server] loading weights\n"),
    ],
)
def test_runbook_messages_print_as_they_are_and_debug_adds_followed_lines(
    level: str,
    printed: str,
    runbook_logger: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level)
    log("== Starting the judge server")
    LOGGER.debug("[judge-server] loading weights")
    assert capsys.readouterr().out == printed


def test_follower_streams_appended_lines_and_flushes_the_rest_on_exit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    log_file = tmp_path / "record-shard-0.log"
    with log_file.open("w") as log_handle, follow_log(log_file):
        writer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; print('session 1 pass', flush=True); sys.stdin.read(); "
                + "print('session 2 fail'); print('unterminated', end='')",
            ],
            stdin=subprocess.PIPE,
            stdout=log_handle,
        )
        wait_until(lambda: followed(caplog) == ["[record-shard-0] session 1 pass"])
        writer.communicate()
    assert followed(caplog) == [
        "[record-shard-0] session 1 pass",
        "[record-shard-0] session 2 fail",
        "[record-shard-0] unterminated",
    ]


def test_follower_holds_a_character_or_a_crlf_that_a_read_splits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    log_file = tmp_path / "record-shard-0.log"
    bar = "Loading: 50%|█████".encode()
    with log_file.open("wb", buffering=0) as log_handle, follow_log(log_file):
        # note (luojiaxuan): the first write ends inside the last character of the
        # bar and the second inside a CRLF; each is followed before the next lands.
        log_handle.write(b"first\n" + bar[:-1])
        wait_until(lambda: followed(caplog) == ["[record-shard-0] first"])
        log_handle.write(bar[-1:] + b"\nsecond\r")
        wait_until(lambda: len(followed(caplog)) == 2)
        log_handle.write(b"\nthird\n")
    assert followed(caplog) == [
        "[record-shard-0] first",
        "[record-shard-0] Loading: 50%|█████",
        "[record-shard-0] second",
        "[record-shard-0] third",
    ]


def test_follower_is_off_below_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER.name)
    log_file = tmp_path / "model-server.log"
    log_file.write_text("loading weights\n")
    threads = threading.active_count()
    with follow_log(log_file):
        assert threading.active_count() == threads
    assert followed(caplog) == []


def test_server_handle_stops_the_step_with_the_log_tail_once_the_server_exited(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "judge-server.log"
    with log_file.open("w") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "print('loading weights'); print('CUDA error: out of memory'); "
                + "raise SystemExit(3)",
            ],
            stdout=log_handle,
        )
    process.wait()
    server = RunningServer("judge server", process, log_file)
    with pytest.raises(SystemExit) as stopped:
        server.ensure_alive("during the v1.0 evaluation of synthetic_user_interruption")
    assert str(stopped.value) == (
        "ERROR: the judge server exited with code 3 during the v1.0 evaluation of "
        f"synthetic_user_interruption; last lines of {log_file}:\n"
        "loading weights\nCUDA error: out of memory"
    )


def test_running_server_hands_out_its_process_and_streams_only_its_own_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    log_file = tmp_path / "judge-server.log"
    log_file.write_text("a line of an earlier attempt\n")

    def start_server_from_cmd(
        command: list[str], log_file: Path, port: int, **_: object
    ) -> subprocess.Popen:
        # note (luojiaxuan): the server truncates its log after the follower's
        # first read, the order in which a stale log would be replayed.
        time.sleep(2 * FOLLOW_INTERVAL_S)
        with log_file.open("w") as log_handle:
            return subprocess.Popen(command, stdout=log_handle, start_new_session=True)

    monkeypatch.setattr(servers_module, "ensure_port_free", lambda port: None)
    monkeypatch.setattr(servers_module, "start_server_from_cmd", start_server_from_cmd)
    command = [
        sys.executable,
        "-c",
        "import time; print('server is ready', flush=True); time.sleep(60)",
    ]
    with servers_module.running_server(
        "judge server", command, "0", 30000, log_file, "/v1/models", "qwen", {}
    ) as server:
        wait_until(lambda: followed(caplog) == ["[judge-server] server is ready"])
        server.ensure_alive("during the v1.5 Qwen judge")
    assert server.process.poll() is not None
    assert followed(caplog) == ["[judge-server] server is ready"]
