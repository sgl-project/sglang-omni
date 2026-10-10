# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA MPS control-protocol parsing tests."""

from __future__ import annotations

import errno
import fcntl
import subprocess
from pathlib import Path

import pytest

from sglang_omni.mps import control
from sglang_omni.mps.manager import (
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
    MpsRetryableControlError,
)


def test_snapshot_parses_driver_output_and_retains_server_client_pairs(
    monkeypatch, tmp_path
):
    pipe_dir = tmp_path / "gpu" / "pipe"
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

    assert control.SubprocessMpsControlClient().snapshot(pipe_dir) == {
        MpsClientRef(7000, 101),
        MpsClientRef(7000, 102),
        MpsClientRef(8000, 909),
    }

    responses["get_client_list 7000\n"] = "101\nserver=202\n"
    with pytest.raises(MpsControlError, match="unexpected output"):
        control.SubprocessMpsControlClient().snapshot(pipe_dir)


@pytest.mark.parametrize(
    "status", ["ACTIVE", "INITIALIZING", "FAULT", "", "Server not found"]
)
def test_get_server_status_queries_exact_pid_and_strips_output(
    monkeypatch, tmp_path, status
):
    pipe_dir = tmp_path / "gpu" / "pipe"

    def run(args, **kwargs):
        assert args == ["nvidia-cuda-mps-control"]
        assert kwargs["input"] == "get_server_status 7000\n"
        assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
        assert kwargs["timeout"] == 10
        return subprocess.CompletedProcess(
            args, returncode=0, stdout=f"  {status}\n", stderr=""
        )

    monkeypatch.setattr(control.subprocess, "run", run)
    assert (
        control.SubprocessMpsControlClient().get_server_status(pipe_dir, 7000) == status
    )


@pytest.mark.parametrize("operation", ["snapshot", "get_server_status"])
def test_control_query_rejects_nonzero_exit_and_timeout(
    monkeypatch, tmp_path, operation
):
    client = control.SubprocessMpsControlClient()
    pipe_dir = tmp_path / "gpu" / "pipe"

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
        if operation == "snapshot":
            client.snapshot(pipe_dir)
        else:
            client.get_server_status(pipe_dir, 7000)

    def timeout(args, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(control.subprocess, "run", timeout)
    with pytest.raises(MpsControlError, match="timed out"):
        if operation == "snapshot":
            client.snapshot(pipe_dir)
        else:
            client.get_server_status(pipe_dir, 7000)


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


@pytest.mark.parametrize(
    "failure", [FileNotFoundError(), PermissionError("unreadable"), "123 (mps)"]
)
def test_daemon_liveness_distinguishes_missing_stat_from_unreadable_or_malformed(
    monkeypatch, failure
):
    def read_stat(path):
        assert path == Path("/proc/123/stat")
        if isinstance(failure, OSError):
            raise failure
        else:
            return failure

    monkeypatch.setattr(Path, "read_text", read_stat)
    client = control.SubprocessMpsControlClient()
    if isinstance(failure, FileNotFoundError):
        assert not client.daemon_process_alive(123)
    else:
        with pytest.raises(MpsControlError, match="cannot inspect daemon pid"):
            client.daemon_process_alive(123)


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


@pytest.mark.parametrize(
    "failure, error_type",
    [
        (FileNotFoundError("missing"), MpsControlError),
        (PermissionError("not executable"), MpsRetryableControlError),
        (OSError(errno.ENOEXEC, "exec format error"), MpsControlError),
        (OSError(errno.EIO, "I/O failure"), MpsControlError),
        (OSError(errno.EINTR, "interrupted"), MpsRetryableControlError),
        (OSError(errno.EAGAIN, "try again"), MpsRetryableControlError),
        (OSError(errno.ETIMEDOUT, "timed out"), MpsRetryableControlError),
        (subprocess.TimeoutExpired("get_server_list", 10), MpsRetryableControlError),
        (subprocess.SubprocessError("invalid subprocess result"), MpsControlError),
    ],
)
def test_query_errors_preserve_native_cause(monkeypatch, failure, error_type):
    def cannot_execute(*args, **kwargs):
        raise failure

    monkeypatch.setattr(control.subprocess, "run", cannot_execute)
    with pytest.raises(MpsControlError) as exc_info:
        control.SubprocessMpsControlClient().query(Path("/mps/pipe"), "get_server_list")
    assert exc_info.value.__cause__ is failure
    assert type(exc_info.value) is error_type


@pytest.mark.parametrize("read", ["pid_file", "stat", "cmdline", "environ", "token"])
@pytest.mark.parametrize(
    "failure, error_type",
    [
        (PermissionError(errno.EACCES, "unreadable"), MpsRetryableControlError),
        (OSError(errno.EIO, "unreadable"), MpsControlError),
        (OSError(errno.EINTR, "interrupted"), MpsRetryableControlError),
        (OSError(errno.EAGAIN, "try again"), MpsRetryableControlError),
        (OSError(errno.ETIMEDOUT, "timed out"), MpsRetryableControlError),
    ],
)
def test_proc_io_failure_preserves_original_cause(
    monkeypatch, read, failure, error_type
):
    pipe_dir = Path("/mps/pipe")

    def read_text(path):
        if path == pipe_dir / "nvidia-cuda-mps-control.pid":
            if read == "pid_file":
                raise failure
            return "123\n"
        assert path == Path("/proc/123/stat")
        if read == "stat":
            raise failure
        return "123 (mps) S 1"

    def read_bytes(path):
        if path == Path("/proc/123/cmdline"):
            if read == "cmdline":
                raise failure
            return b"/usr/bin/nvidia-cuda-mps-control\0-d\0"
        assert path == Path("/proc/123/environ")
        if read in {"environ", "token"}:
            raise failure
        return b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe\0"

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    client = control.SubprocessMpsControlClient()
    with pytest.raises(MpsControlError) as exc_info:
        if read == "token":
            client.client_token(123)
        elif read == "stat":
            client.daemon_process_alive(123)
        else:
            client.read_daemon_identity(pipe_dir)
    assert exc_info.value.__cause__ is failure
    assert type(exc_info.value) is error_type


@pytest.mark.parametrize(
    "invalid",
    [
        "pid",
        "dead",
        "stat",
        "binary",
        "pipe",
        "empty_token",
        "duplicate_token",
        "non_ascii_token",
    ],
)
def test_confirmed_invalid_data_is_terminal(monkeypatch, invalid):
    pipe_dir = Path("/mps/pipe")
    token_prefix = f"{control.MPS_CLIENT_TOKEN_ENV}=".encode()

    def read_text(path):
        if path == pipe_dir / "nvidia-cuda-mps-control.pid":
            return "not-a-pid" if invalid == "pid" else "123\n"
        assert path == Path("/proc/123/stat")
        return {"dead": "123 (mps) Z 1", "stat": "123 (mps)"}.get(
            invalid, "123 (mps) S 1"
        )

    def read_bytes(path):
        if path == Path("/proc/123/cmdline"):
            return (
                b"other-process\0"
                if invalid == "binary"
                else b"nvidia-cuda-mps-control\0"
            )
        assert path == Path("/proc/123/environ")
        return {
            "pipe": b"CUDA_MPS_PIPE_DIRECTORY=/other/pipe\0",
            "empty_token": token_prefix + b"\0",
            "duplicate_token": token_prefix + b"a\0" + token_prefix + b"b\0",
            "non_ascii_token": token_prefix + b"\xff\0",
        }.get(invalid, b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe\0")

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    client = control.SubprocessMpsControlClient()
    with pytest.raises(MpsControlError):
        if invalid.endswith("token"):
            client.client_token(123)
        else:
            client.read_daemon_identity(pipe_dir)


def test_invalid_native_response_is_terminal(monkeypatch):
    def invalid_response(args, **kwargs):
        return subprocess.CompletedProcess(
            args, returncode=0, stdout="invalid", stderr=""
        )

    monkeypatch.setattr(control.subprocess, "run", invalid_response)
    client = control.SubprocessMpsControlClient()
    for operation in (
        lambda: client.snapshot(Path("/mps/pipe")),
        lambda: client.terminate_client(Path("/mps/pipe"), MpsClientRef(7000, 101)),
    ):
        with pytest.raises(MpsControlError):
            operation()


@pytest.mark.parametrize("boundary", ["native_query", "pid_file", "stat"])
def test_invalid_text_encoding_is_terminal_at_the_io_boundary(monkeypatch, boundary):
    error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid text")

    def invalid_text(*args, **kwargs):
        raise error

    client = control.SubprocessMpsControlClient()
    with pytest.raises(MpsControlError) as exc_info:
        if boundary == "native_query":
            monkeypatch.setattr(control.subprocess, "run", invalid_text)
            client.query(Path("/mps/pipe"), "get_server_list")
        elif boundary == "pid_file":
            monkeypatch.setattr(Path, "read_text", invalid_text)
            client.read_daemon_identity(Path("/mps/pipe"))
        else:
            monkeypatch.setattr(Path, "read_text", invalid_text)
            client.daemon_process_alive(123)
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("value", ["²", "١٢٣"])
def test_pid_protocol_rejects_non_ascii_digits(monkeypatch, value):
    with pytest.raises(MpsControlError):
        control.parse_pid_list(value, "get_server_list")
    monkeypatch.setattr(Path, "read_text", lambda _: value)
    with pytest.raises(MpsControlError):
        control.SubprocessMpsControlClient().read_daemon_identity(Path("/mps/pipe"))
