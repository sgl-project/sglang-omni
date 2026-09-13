# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA MPS control-protocol parsing tests."""

from __future__ import annotations

import fcntl
import multiprocessing
import subprocess
import time
from pathlib import Path

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


def test_get_server_status_uses_only_the_requested_native_command(
    monkeypatch, tmp_path
):
    pipe_dir = tmp_path / "GPU-abc" / "pipe"
    commands = []

    def run(args, **kwargs):
        commands.append(kwargs["input"])
        assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
        assert kwargs["timeout"] == control._QUERY_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(args, 0, stdout=" ACTIVE\n", stderr="")

    monkeypatch.setattr(control.subprocess, "run", run)
    assert (
        control.SubprocessMpsControlClient().get_server_status(pipe_dir, 7000)
        == "ACTIVE"
    )
    assert commands == ["get_server_status 7000\n"]


@pytest.mark.parametrize(
    "failure,detail",
    [
        ("exit", "query failed"),
        ("timeout", "timed out"),
        ("exec", "missing control binary"),
    ],
)
def test_get_server_status_reports_native_failures_and_releases_lock(
    monkeypatch, tmp_path, failure, detail
):
    def run(args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 10)
        if failure == "exec":
            raise FileNotFoundError("missing control binary")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="query failed")

    monkeypatch.setattr(control.subprocess, "run", run)
    with pytest.raises(MpsControlError, match=f"get_server_status 7000.*{detail}"):
        control.SubprocessMpsControlClient().get_server_status(
            tmp_path / "GPU-abc" / "pipe", 7000
        )
    with (tmp_path / ".control-lock-GPU-abc").open("r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _control_worker(pipe_dir, operation, barrier, entered):
    """Run production control methods with only native subprocess I/O replaced."""

    def run(args, **kwargs):
        with (pipe_dir.parent.parent / "native-active").open("w") as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = kwargs["input"].strip()
            entered.set()
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
    barrier.wait(timeout=10)
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
    barrier = ctx.Barrier(5)
    entered = ctx.Event()
    processes = [
        ctx.Process(
            target=_control_worker, args=(pipe_dir, operation, barrier, entered)
        )
        for operation in ("snapshot", "status", "terminate", "quit")
    ]
    try:
        with state_root_lock(tmp_path, ".control-lock-GPU-abc"):
            for process in processes:
                process.start()
            barrier.wait(timeout=10)
            # Note (kaige): a parent-held native lock must block every operation.
            assert not entered.wait(timeout=0.2)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)


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


def test_daemon_liveness_rejects_zombie_proc_entries(monkeypatch):
    stats = {
        Path("/proc/430465/stat"): "430465 (nvidia-cuda-mps) Z 1 430465 0",
        Path("/proc/53748/stat"): "53748 (nvidia-cuda-mps-control) S 1 0 0",
        Path("/proc/7/stat"): "7 (weird) name) Z 1 0",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: stats[path])
    monkeypatch.setattr(control.os, "kill", lambda _pid, _signal: None)
    client = control.SubprocessMpsControlClient()

    assert not client.daemon_process_alive(430465)
    assert client.daemon_process_alive(53748)
    assert not client.daemon_process_alive(7)
