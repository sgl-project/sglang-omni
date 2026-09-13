# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA MPS control-protocol parsing tests."""

from __future__ import annotations

import fcntl
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from queue import Empty

import pytest

from sglang_omni.mps import control
from sglang_omni.mps.manager import (
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
)
from sglang_omni.mps.state import state_root_lock


def test_snapshot_parses_driver_output_and_retains_server_client_pairs(
    monkeypatch, tmp_path
):
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

    assert control.SubprocessMpsControlClient().snapshot(
        tmp_path / "GPU-abc" / "pipe"
    ) == {
        MpsClientRef(7000, 101),
        MpsClientRef(7000, 102),
        MpsClientRef(8000, 909),
    }

    responses["get_client_list 7000\n"] = "101\nserver=202\n"
    with pytest.raises(MpsControlError, match="unexpected output"):
        control.SubprocessMpsControlClient().snapshot(tmp_path / "GPU-abc" / "pipe")


def test_control_query_rejects_nonzero_exit_and_timeout(monkeypatch, tmp_path):
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
        client.snapshot(tmp_path / "GPU-abc" / "pipe")

    def timeout(args, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(control.subprocess, "run", timeout)
    with pytest.raises(MpsControlError, match="timed out"):
        client.snapshot(tmp_path / "GPU-abc" / "pipe")


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


@pytest.mark.parametrize(
    "output", ["ACTIVE\n", "FAULT\n", "INITIALIZING\n", "", "Server not found\n"]
)
def test_get_server_status_uses_only_the_requested_native_command(
    monkeypatch, tmp_path, output
):
    pipe_dir = tmp_path / "GPU-abc" / "pipe"
    commands = []

    def run(args, **kwargs):
        commands.append(kwargs["input"])
        assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
        assert kwargs["timeout"] == control._QUERY_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(control.subprocess, "run", run)
    assert (
        control.SubprocessMpsControlClient().get_server_status(pipe_dir, 7000)
        == output.strip()
    )
    assert commands == ["get_server_status 7000\n"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "exec"])
def test_get_server_status_reports_native_failures(monkeypatch, tmp_path, failure):
    def run(args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 10)
        if failure == "exec":
            raise FileNotFoundError("missing control binary")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="query failed")

    monkeypatch.setattr(control.subprocess, "run", run)
    with pytest.raises(MpsControlError, match="get_server_status 7000.*failed"):
        control.SubprocessMpsControlClient().get_server_status(
            tmp_path / "GPU-abc" / "pipe", 7000
        )


def _control_worker(pipe_dir, operation, ready, start, commands):
    """Run production control methods with only native subprocess I/O replaced."""

    def run(args, **kwargs):
        with (pipe_dir.parent.parent / "native-active").open("w") as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = kwargs["input"].strip()
            commands.put((os.getpid(), command))
            with (pipe_dir.parent.parent / "commands.log").open("a") as command_log:
                command_log.write(f"{os.getpid()} {command}\n")
            time.sleep(0.01)
            output = {
                "get_server_list": "7000\n",
                "get_client_list 7000": "101\n",
                "get_server_status 7000": "ACTIVE\n",
                "terminate_client 7000 101": "0\n",
                "quit": "",
            }[command]
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    control.subprocess.run = run
    client = control.SubprocessMpsControlClient()
    ready.put(os.getpid())
    assert start.wait(10)
    for _ in range(4):
        if operation == "snapshot":
            assert client.snapshot(pipe_dir) == {MpsClientRef(7000, 101)}
        elif operation == "status":
            assert client.get_server_status(pipe_dir, 7000) == "ACTIVE"
        elif operation == "terminate":
            client.terminate_client(pipe_dir, MpsClientRef(7000, 101))
        else:
            client.quit_daemon(pipe_dir)


def test_native_queries_share_one_cross_process_lock(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    pipe_dir = tmp_path / "GPU-abc" / "pipe"
    start = ctx.Event()
    ready = ctx.Queue()
    commands = ctx.Queue()
    processes = [
        ctx.Process(
            target=_control_worker, args=(pipe_dir, operation, ready, start, commands)
        )
        for operation in ("snapshot", "status", "terminate", "quit")
    ]
    try:
        with state_root_lock(tmp_path, ".control-lock-GPU-abc"):
            for process in processes:
                process.start()
            for _ in processes:
                ready.get(timeout=10)
            start.set()
            # Note (kaige): a parent-held native lock must block every operation.
            with pytest.raises(Empty):
                commands.get(timeout=0.2)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        observed = [
            line.split(" ", 1)
            for line in (tmp_path / "commands.log").read_text().splitlines()
        ]
        assert len(observed) == 20
        for pid in {pid for pid, _ in observed}:
            worker_commands = [command for owner, command in observed if owner == pid]
            if worker_commands[0] == "get_server_list":
                assert (
                    worker_commands == ["get_server_list", "get_client_list 7000"] * 4
                )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
        ready.close()
        commands.close()


def test_failed_query_releases_control_lock(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "GPU-abc" / "pipe"

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(control.subprocess, "run", timeout)
    with pytest.raises(MpsControlError):
        control.SubprocessMpsControlClient().get_server_status(pipe_dir, 7000)
    with (tmp_path / ".control-lock-GPU-abc").open("r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_control_lock_survives_gpu_state_removal(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "GPU-abc" / "pipe"
    pipe_dir.mkdir(parents=True)
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(
        control.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="ACTIVE\n", stderr=""
        ),
    )
    assert client.get_server_status(pipe_dir, 7000) == "ACTIVE"
    inode = (tmp_path / ".control-lock-GPU-abc").stat().st_ino
    pipe_dir.rmdir()
    pipe_dir.parent.rmdir()
    assert client.get_server_status(pipe_dir, 7000) == "ACTIVE"
    assert (tmp_path / ".control-lock-GPU-abc").stat().st_ino == inode


def test_lock_failure_is_reported_as_control_error(monkeypatch, tmp_path):
    def denied(*args):
        raise PermissionError("control lock unavailable")

    monkeypatch.setattr(control, "state_root_lock", denied)
    with pytest.raises(
        MpsControlError, match="get_server_status 7000.*control lock unavailable"
    ):
        control.SubprocessMpsControlClient().get_server_status(
            tmp_path / "GPU-abc" / "pipe", 7000
        )
