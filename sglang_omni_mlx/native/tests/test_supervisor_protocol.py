# SPDX-License-Identifier: Apache-2.0
"""Voxt's supervisor protocol: the stdout events, stdin commands and command line every native server shares."""

from __future__ import annotations

import re
import signal
import subprocess
from pathlib import Path

import pytest
from native_server import EXIT_TIMEOUT_SECONDS, SupervisedServer

# Above any test's run time, so only a stop can end the load.
UNFINISHED_LOAD_MS = "600000"


def test_ready_event_names_the_listening_endpoint(
    stub_server_binary: Path, model_directory: Path
) -> None:
    # Voxt passes --startup-timeout-s to every server, which must accept it.
    server = SupervisedServer(
        stub_server_binary, "stub_asr", model_directory, ("--startup-timeout-s", "30")
    )
    ready = server.first_event
    assert ready["event"] == "ready"
    assert ready["host"] == "127.0.0.1"
    assert ready["server_pid"] == server.process.pid
    assert re.fullmatch(r"voxt-stub_asr-[0-9a-f]{12}", ready["model_name"])
    assert ready["startup_s"] >= 0
    health = server.request("GET", "/health")
    assert health.status == 200
    assert health.json() == {
        "status": "healthy",
        "running": True,
        "request_states": {},
    }
    models = server.request("GET", "/v1/models").json()
    assert models == {
        "object": "list",
        "data": [{"id": ready["model_name"], "object": "model"}],
    }
    server.stop()


def test_a_chosen_model_name_is_listed(
    stub_server_binary: Path, model_directory: Path
) -> None:
    server = SupervisedServer(
        stub_server_binary,
        "stub_speech",
        model_directory,
        ("--model-name", "voxt-speech"),
    )
    assert server.first_event["model_name"] == "voxt-speech"
    assert server.request("GET", "/v1/models").json()["data"][0]["id"] == "voxt-speech"
    server.stop()


def test_shutdown_reports_stopped_after_ignoring_malformed_lines(
    stub_server_binary: Path, model_directory: Path
) -> None:
    server = SupervisedServer(stub_server_binary, "stub_asr", model_directory)
    server.send_line("not json")
    server.send_line('{"command": "reload"}')
    assert server.request("GET", "/health").status == 200
    server.send_line('{"command": "shutdown"}')
    assert server.read_event() == {"event": "stopped"}
    assert server.wait() == 0


@pytest.mark.parametrize("stop", ["end of stdin", "SIGTERM"])
def test_other_stops_exit_without_an_event(
    stub_server_binary: Path, model_directory: Path, stop: str
) -> None:
    server = SupervisedServer(stub_server_binary, "stub_asr", model_directory)
    if stop == "end of stdin":
        server.process.stdin.close()
    else:
        server.process.send_signal(signal.SIGTERM)
    assert server.wait() == 0
    assert server.remaining_stdout() == b""


def test_shutdown_during_the_model_load_reports_stopped(
    stub_server_binary: Path, model_directory: Path
) -> None:
    """Voxt cancels a slow load with shutdown and waits for stopped."""
    process = subprocess.Popen(
        [
            str(stub_server_binary),
            "--supervised",
            "--model-kind",
            "stub_asr",
            "--model-directory",
            str(model_directory),
            "--load-delay-ms",
            UNFINISHED_LOAD_MS,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout, _ = process.communicate(
        b'{"command": "shutdown"}\n', timeout=EXIT_TIMEOUT_SECONDS
    )
    assert process.returncode == 0
    assert stdout == b'{"event":"stopped"}\n'


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--model-kind", "stub_asr"), "--model-path is required"),
        (("--model-directory", "x", "--model-kind", "stub_asr", "--no-such-flag", "1"), "unknown argument --no-such-flag"),
        (("--model-directory", "x", "--model-kind", "whisper"), "--model-kind must be stub_asr, stub_speech, stub_ming_tts or stub_chat"),
        (("--model-directory", "x", "--model-kind", "stub_asr", "--model-name"), "--model-name needs a value"),
        (("--model-directory", "x", "--model-kind", "stub_asr", "--max-queued-requests", "1"), "unknown argument --max-queued-requests"),
        (("--model-directory", "x", "--model-kind", "stub_speech", "--max-queued-requests", "-1"), "--max-queued-requests must not be negative"),
        (("--model-directory", "x", "--model-kind", "stub_speech", "--request-timeout-s", "0"), "--request-timeout-s must be above 0"),
    ],
)  # fmt: skip
def test_invalid_arguments_exit_before_any_event(
    stub_server_binary: Path, arguments: tuple[str, ...], message: str
) -> None:
    completed = subprocess.run(
        [str(stub_server_binary), "--supervised", *arguments],
        capture_output=True,
        text=True,
        timeout=EXIT_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert message in completed.stderr
