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


def test_snapshot_parses_driver_output_and_retains_server_client_pairs(monkeypatch):
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

    assert control.SubprocessMpsControlClient().snapshot(Path("/mps/pipe")) == {
        MpsClientRef(7000, 101),
        MpsClientRef(7000, 102),
        MpsClientRef(8000, 909),
    }

    responses["get_client_list 7000\n"] = "101\nserver=202\n"
    with pytest.raises(MpsControlError, match="unexpected output"):
        control.SubprocessMpsControlClient().snapshot(Path("/mps/pipe"))


def test_control_query_rejects_nonzero_exit_and_timeout(monkeypatch):
    client = control.SubprocessMpsControlClient()

    def nonzero(args, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            args,
            returncode=2,
            stdout="",
            stderr="control failed",
        )

    monkeypatch.setattr(control.subprocess, "run", nonzero)
    with pytest.raises(MpsControlError, match="control failed"):
        client.snapshot(Path("/mps/pipe"))

    def timeout(args, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(control.subprocess, "run", timeout)
    with pytest.raises(MpsControlError, match="timed out"):
        client.snapshot(Path("/mps/pipe"))


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

    monkeypatch.setattr(Path, "read_bytes", lambda _path: environ)
    assert client.client_token(123) == "owner-worker"

    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"PATH=/usr/bin\0")
    assert client.client_token(123) is None
