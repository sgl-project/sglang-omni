# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA MPS control-protocol parsing tests."""

from __future__ import annotations

import fcntl
import subprocess
from pathlib import Path

import pytest

from sglang_omni.mps import control
from sglang_omni.mps.manager import (
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
)


def test_snapshot_parses_driver_output_and_retains_server_client_pairs(
    monkeypatch, tmp_path
):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(client, "read_daemon_identity", lambda _pipe: 123)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)
    responses = {
        "get_server_list\n": "7000  8000\n",
        "get_client_list 7000\n": "101\n102\n",
        "get_client_list 8000\n": "909\n",
    }

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout=responses[kwargs["input"]],
            stderr="",
        )

    monkeypatch.setattr(control.subprocess, "run", run)

    assert client.snapshot(pipe_dir) == {
        MpsClientRef(7000, 101),
        MpsClientRef(7000, 102),
        MpsClientRef(8000, 909),
    }

    responses["get_client_list 7000\n"] = "101\nserver=202\n"
    with pytest.raises(MpsControlError, match="unexpected output"):
        client.snapshot(pipe_dir)


def test_snapshot_holds_one_control_lock_across_all_queries(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(client, "read_daemon_identity", lambda _pipe: 123)
    events: list[str] = []
    responses = {
        "get_server_list\n": "7000 8000\n",
        "get_client_list 7000\n": "101\n",
        "get_client_list 8000\n": "202\n",
    }

    def flock(_file, operation):
        if operation == fcntl.LOCK_EX:
            events.append("lock")

    def run(args, **kwargs):
        command = kwargs["input"].strip()
        events.append(command)
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout=responses[kwargs["input"]],
            stderr="",
        )

    monkeypatch.setattr(control.fcntl, "flock", flock)
    monkeypatch.setattr(control.subprocess, "run", run)

    assert client.snapshot(pipe_dir) == {
        MpsClientRef(7000, 101),
        MpsClientRef(8000, 202),
    }
    assert events == [
        "lock",
        "get_server_list",
        "get_client_list 7000",
        "get_client_list 8000",
    ]


def test_snapshot_retries_transient_query_failure_while_identity_is_stable(
    monkeypatch, tmp_path
):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(client, "read_daemon_identity", lambda _pipe: 123)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)
    calls: list[str] = []
    first_server_query = True

    def run(args, **kwargs):
        nonlocal first_server_query
        command = kwargs["input"]
        calls.append(command.strip())
        if command == "get_server_list\n" and first_server_query:
            first_server_query = False
            return subprocess.CompletedProcess(
                args,
                returncode=1,
                stdout="",
                stderr="Cannot send command to MPS control daemon process",
            )
        responses = {
            "get_server_list\n": "7000\n",
            "get_client_list 7000\n": "101\n",
        }
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout=responses[command],
            stderr="",
        )

    monkeypatch.setattr(control.subprocess, "run", run)

    assert client.snapshot(pipe_dir) == {MpsClientRef(7000, 101)}
    assert calls == ["get_server_list", "get_server_list", "get_client_list 7000"]


def test_snapshot_aborts_retry_when_daemon_identity_changes(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    identities = iter([123, 124])
    monkeypatch.setattr(client, "read_daemon_identity", lambda _pipe: next(identities))
    calls = 0

    def run(args, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return subprocess.CompletedProcess(
            args,
            returncode=1,
            stdout="",
            stderr="control failed",
        )

    monkeypatch.setattr(control.subprocess, "run", run)

    with pytest.raises(MpsControlError, match="identity changed.*123.*124"):
        client.snapshot(pipe_dir)
    assert calls == 1


def test_snapshot_rejects_nonzero_exit_and_timeout(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(client, "read_daemon_identity", lambda _pipe: 123)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)

    def nonzero(args, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            args,
            returncode=2,
            stdout="",
            stderr="control failed",
        )

    monkeypatch.setattr(control.subprocess, "run", nonzero)
    with pytest.raises(
        MpsControlError,
        match="snapshot failed after 3 attempts: .*control failed",
    ):
        client.snapshot(pipe_dir)

    def timeout(args, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(control.subprocess, "run", timeout)
    with pytest.raises(
        MpsControlError,
        match="snapshot failed after 3 attempts: .*timed out",
    ):
        client.snapshot(pipe_dir)


def test_mutating_control_query_is_serialized_without_retry(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    events: list[str] = []

    def flock(_file, operation):
        if operation == fcntl.LOCK_EX:
            events.append("lock")

    def run(args, **kwargs):
        events.append(kwargs["input"].strip())
        return subprocess.CompletedProcess(
            args,
            returncode=2,
            stdout="",
            stderr="control failed",
        )

    monkeypatch.setattr(control.fcntl, "flock", flock)
    monkeypatch.setattr(control.subprocess, "run", run)

    with pytest.raises(MpsControlError, match="control failed"):
        client.quit_daemon(pipe_dir)
    assert events == ["lock", "quit"]


def test_daemon_preexec_failure_is_distinct_from_ambiguous_start(monkeypatch):
    client = control.SubprocessMpsControlClient()

    def cannot_execute(*args, **kwargs):
        del args, kwargs
        raise PermissionError("not executable")

    monkeypatch.setattr(control.subprocess, "run", cannot_execute)

    with pytest.raises(MpsDaemonNotStartedError, match="failed to execute"):
        client.start_daemon(Path("/mps/pipe"), Path("/mps/log"), "GPU-abc")


def test_daemon_identity_requires_exact_binary_and_pipe_environment(monkeypatch):
    pipe_dir = Path("/mps/pipe")
    client = control.SubprocessMpsControlClient()
    environ = [b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe", b"PATH=/usr/bin", b""]

    def read_text(path):
        assert path == pipe_dir / "nvidia-cuda-mps-control.pid"
        return "123\n"

    def read_bytes(path):
        if path == Path("/proc/123/cmdline"):
            return b"/usr/bin/nvidia-cuda-mps-control\x00-d\x00"
        assert path == Path("/proc/123/environ")
        return b"\x00".join(environ)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(client, "daemon_process_alive", lambda pid: pid == 123)

    assert client.read_daemon_identity(pipe_dir) == 123

    environ[0] = b"CUDA_MPS_PIPE_DIRECTORY=/another/pipe"
    with pytest.raises(MpsControlError, match="exact pipe directory"):
        client.read_daemon_identity(pipe_dir)


def test_owner_liveness_comes_from_the_kernel_held_lease(tmp_path):
    lease_file = tmp_path / "owner"
    client = control.SubprocessMpsControlClient()

    with lease_file.open("w+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert client.owner_lease_held(lease_file)
        fcntl.flock(owner, fcntl.LOCK_UN)

    assert not client.owner_lease_held(lease_file)


def test_client_token_is_read_from_the_current_client_environment(monkeypatch):
    client = control.SubprocessMpsControlClient()
    environ = (
        b"PATH=/usr/bin\0"
        + f"{control.MPS_CLIENT_TOKEN_ENV}=owner-worker".encode()
        + b"\0"
    )

    monkeypatch.setattr(Path, "read_bytes", lambda path: environ)
    assert client.client_token(123) == "owner-worker"

    monkeypatch.setattr(Path, "read_bytes", lambda path: b"PATH=/usr/bin\0")
    assert client.client_token(123) is None
