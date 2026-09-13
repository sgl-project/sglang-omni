# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA MPS control-protocol parsing tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sglang_omni.mps import control
from sglang_omni.mps.control import (
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
)


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


def test_terminate_client_does_not_require_a_success_response(monkeypatch, tmp_path):
    commands = []

    def run(args, **kwargs):
        commands.append(kwargs["input"])
        return subprocess.CompletedProcess(args, returncode=0, stdout="1\n", stderr="")

    monkeypatch.setattr(control.subprocess, "run", run)
    control.SubprocessMpsControlClient().terminate_client(
        tmp_path, MpsClientRef(7000, 101)
    )
    assert commands == ["terminate_client 7000 101\n"]


def test_daemon_preexec_failure_is_distinct_from_ambiguous_start(monkeypatch):
    client = control.SubprocessMpsControlClient()

    def cannot_execute(*args, **kwargs):
        del args, kwargs
        raise PermissionError("not executable")

    monkeypatch.setattr(control.subprocess, "run", cannot_execute)

    with pytest.raises(MpsDaemonNotStartedError, match="failed to execute"):
        client.start_daemon(Path("/mps/pipe"), Path("/mps/log"), ("GPU-abc",))


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
def test_get_server_status_reports_native_failures(
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


def test_daemon_start_uses_all_selected_gpu_uuids(monkeypatch):
    def run(args, **kwargs):
        assert args == ["nvidia-cuda-mps-control", "-d"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
        assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == "/mps/pipe"
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(control.subprocess, "run", run)
    control.SubprocessMpsControlClient().start_daemon(
        Path("/mps/pipe"), Path("/mps/log"), ("GPU-a", "GPU-b")
    )
