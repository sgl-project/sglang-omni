# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA MPS control-protocol parsing tests."""

from __future__ import annotations

import fcntl
import signal
import subprocess
from pathlib import Path

import pytest

from sglang_omni.mps import control
from sglang_omni.mps.manager import (
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
    MpsProcessIdentity,
    MpsRetryableControlError,
)
from tests.test_ci import test_mps_native as mps_ci


def proc_stat(pid: int, comm: str, state: str, starttime: int) -> str:
    fields = [state, "1"] + ["0"] * 17 + [str(starttime)]
    return f"{pid} ({comm}) " + " ".join(fields)


def test_snapshot_parses_driver_output_and_retains_server_client_pairs(
    monkeypatch, tmp_path
):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(
        client,
        "read_daemon_process_identity",
        lambda _pipe: MpsProcessIdentity(123, 1),
    )
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
    monkeypatch.setattr(
        client,
        "read_daemon_process_identity",
        lambda _pipe: MpsProcessIdentity(123, 1),
    )
    lock_path = tmp_path / ".control.lock"
    lock_path.touch()
    contender = lock_path.open("a+")
    real_flock = fcntl.flock
    events: list[str] = []
    exclusive_acquisitions = 0
    contended_commands: list[str] = []
    responses = {
        "get_server_list\n": "7000 8000\n",
        "get_client_list 7000\n": "101\n",
        "get_client_list 8000\n": "202\n",
    }

    def flock(file, operation):
        nonlocal exclusive_acquisitions
        if operation == fcntl.LOCK_EX:
            exclusive_acquisitions += 1
            events.append("lock")
        return real_flock(file, operation)

    def run(args, **kwargs):
        command = kwargs["input"].strip()
        events.append(command)
        try:
            real_flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            contended_commands.append(command)
        else:
            real_flock(contender, fcntl.LOCK_UN)
            pytest.fail(f"control lock was not held during {command!r}")
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout=responses[kwargs["input"]],
            stderr="",
        )

    try:
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
        assert exclusive_acquisitions == 1
        assert contended_commands == [
            "get_server_list",
            "get_client_list 7000",
            "get_client_list 8000",
        ]

        real_flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        real_flock(contender, fcntl.LOCK_UN)
    finally:
        contender.close()


def test_snapshot_rejects_daemon_identity_change(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    identities = iter(
        [
            MpsProcessIdentity(123, 1),
            MpsProcessIdentity(124, 1),
        ]
    )
    monkeypatch.setattr(
        client,
        "read_daemon_process_identity",
        lambda _pipe: next(identities),
    )

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout="7000\n" if kwargs["input"] == "get_server_list\n" else "101\n",
            stderr="",
        )

    monkeypatch.setattr(control.subprocess, "run", run)

    with pytest.raises(MpsControlError, match="identity changed.*123.*124"):
        client.snapshot(pipe_dir)


def test_snapshot_rejects_nonzero_exit_and_timeout(monkeypatch, tmp_path):
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(
        client,
        "read_daemon_process_identity",
        lambda _pipe: MpsProcessIdentity(123, 1),
    )

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
        client.snapshot(pipe_dir)

    def timeout(args, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(control.subprocess, "run", timeout)
    with pytest.raises(MpsRetryableControlError, match="timed out"):
        client.snapshot(pipe_dir)


def test_mutating_control_query_is_serialized(monkeypatch, tmp_path):
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


def test_transient_client_process_read_is_retryable(monkeypatch):
    def unavailable(_path):
        raise BlockingIOError("temporarily unavailable")

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    with pytest.raises(MpsRetryableControlError, match="cannot inspect client"):
        control.SubprocessMpsControlClient().client_token(123)


def test_daemon_preexec_failure_is_distinct_from_ambiguous_start(monkeypatch):
    client = control.SubprocessMpsControlClient()

    def cannot_execute(*args, **kwargs):
        del args, kwargs
        raise PermissionError("not executable")

    monkeypatch.setattr(control.subprocess, "run", cannot_execute)

    with pytest.raises(MpsDaemonNotStartedError, match="failed to execute"):
        client.start_daemon(Path("/mps/pipe"), Path("/mps/log"), "GPU-abc")


def test_daemon_process_identity_from_native_pid_file(monkeypatch):
    pipe_dir = Path("/mps/pipe")
    client = control.SubprocessMpsControlClient()

    def read_text(path):
        if path == pipe_dir / "nvidia-cuda-mps-control.pid":
            return "123\n"
        assert path == Path("/proc/123/stat")
        return proc_stat(123, "nvidia-cuda-mps-control", "T", 42)

    def read_bytes(path):
        if path == Path("/proc/123/cmdline"):
            return b"/usr/bin/nvidia-cuda-mps-control\x00-d\x00"
        assert path == Path("/proc/123/environ")
        return b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe\x00"

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    assert client.read_daemon_process_identity(pipe_dir) == MpsProcessIdentity(123, 42)


@pytest.mark.parametrize(
    ("stat_texts", "cmdline", "environ", "message"),
    [
        (
            (proc_stat(456, "nvidia-cuda-mps-server", "T", 84),) * 2,
            b"/usr/bin/nvidia-cuda-mps-server\x00",
            b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe\x00",
            None,
        ),
        ((), b"", b"", "does not exist"),
        (
            (proc_stat(456, "nvidia-cuda-mps-server", "Z", 84),) * 2,
            b"",
            b"",
            "zombie",
        ),
        (
            (proc_stat(456, "nvidia-cuda-mps-server", "S", 84),) * 2,
            b"/usr/bin/python\x00",
            b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe\x00",
            "not nvidia-cuda-mps-server",
        ),
        (
            (proc_stat(456, "nvidia-cuda-mps-server", "S", 84),) * 2,
            b"/usr/bin/nvidia-cuda-mps-server\x00",
            b"CUDA_MPS_PIPE_DIRECTORY=/other/pipe\x00",
            "exact pipe directory",
        ),
        (
            (
                proc_stat(456, "nvidia-cuda-mps-server", "S", 84),
                proc_stat(456, "nvidia-cuda-mps-server", "S", 85),
            ),
            b"/usr/bin/nvidia-cuda-mps-server\x00",
            b"CUDA_MPS_PIPE_DIRECTORY=/mps/pipe\x00",
            "changed identity during inspection",
        ),
    ],
)
def test_server_process_identity_contract(
    monkeypatch,
    stat_texts,
    cmdline,
    environ,
    message,
):
    pipe_dir = Path("/mps/pipe")
    client = control.SubprocessMpsControlClient()
    stats = iter(stat_texts)

    def read_text(_path):
        try:
            return next(stats)
        except StopIteration as exc:
            raise FileNotFoundError("gone") from exc

    monkeypatch.setattr(Path, "read_text", read_text)

    def read_bytes(path):
        if path == Path("/proc/456/cmdline"):
            return cmdline
        assert path == Path("/proc/456/environ")
        return environ

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    if message is None:
        assert client.read_server_process_identity(pipe_dir, 456) == (
            MpsProcessIdentity(456, 84)
        )
    else:
        with pytest.raises(MpsControlError, match=message):
            client.read_server_process_identity(pipe_dir, 456)


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


def make_pipe_dir(tmp_path):
    pipe_dir = tmp_path / "GPU-test" / "pipe"
    pipe_dir.mkdir(parents=True)
    return pipe_dir


def test_operator_cleanup_preserves_signal_and_control_order(tmp_path, monkeypatch):
    make_pipe_dir(tmp_path)
    events: list[tuple] = []
    live_sessions = {700}
    clients = {"owned-client"}

    class Control:
        daemon_alive = True

        def read_daemon_process_identity(self, selected_pipe_dir):
            events.append(("read", selected_pipe_dir))
            return MpsProcessIdentity(900, 1)

        def snapshot(self, selected_pipe_dir):
            events.append(("snapshot", selected_pipe_dir, frozenset(clients)))
            return set(clients)

        def quit_daemon(self, selected_pipe_dir):
            events.append(("quit", selected_pipe_dir))
            self.daemon_alive = False

        def daemon_process_alive(self, pid):
            events.append(("alive", pid, self.daemon_alive))
            return self.daemon_alive

    control_client = Control()

    def signal_sessions(session_ids, sig):
        matched = set(session_ids) & live_sessions
        events.append(("signal", sig, frozenset(matched)))
        if sig == signal.SIGKILL:
            live_sessions.difference_update(matched)
            clients.clear()
        return matched

    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        mps_ci,
        "list_daemon_pids",
        lambda: {900} if control_client.daemon_alive else set(),
    )
    monkeypatch.setattr(
        mps_ci,
        "mps_process_identities",
        lambda: (
            {mps_ci.ProcessIdentity(900, 1)} if control_client.daemon_alive else set()
        ),
    )
    monkeypatch.setattr(mps_ci, "signal_test_sessions", signal_sessions)
    monkeypatch.setattr(
        mps_ci,
        "assert_process_identities_gone",
        lambda identities: events.append(("gone", frozenset(identities))),
    )
    monkeypatch.setattr(control, "SubprocessMpsControlClient", lambda: control_client)

    mps_ci.operator_cleanup({700})

    assert not tmp_path.exists()
    assert [event[0] for event in events] == [
        "signal",
        "read",
        "snapshot",
        "signal",
        "snapshot",
        "quit",
        "alive",
        "gone",
    ]
    assert events[0][1] == signal.SIGSTOP
    assert events[3][1] == signal.SIGKILL

    mps_ci.operator_cleanup({700})
    assert not tmp_path.exists()


def test_operator_cleanup_preserves_state_until_mps_processes_are_gone(
    tmp_path,
    monkeypatch,
):
    make_pipe_dir(tmp_path)
    identity = mps_ci.ProcessIdentity(900, 1)

    def assert_processes_gone(_identities):
        raise AssertionError("still alive")

    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(mps_ci, "list_daemon_pids", lambda: set())
    monkeypatch.setattr(mps_ci, "mps_process_identities", lambda: {identity})
    monkeypatch.setattr(
        mps_ci,
        "signal_test_sessions",
        lambda session_ids, sig: set(session_ids),
    )
    monkeypatch.setattr(
        mps_ci,
        "assert_process_identities_gone",
        assert_processes_gone,
    )
    monkeypatch.setattr(control, "SubprocessMpsControlClient", lambda: object())

    with pytest.raises(AssertionError, match="still alive"):
        mps_ci.operator_cleanup({700})

    assert tmp_path.exists()


def test_operator_cleanup_preserves_state_on_snapshot_error(tmp_path, monkeypatch):
    make_pipe_dir(tmp_path)
    signals: list[int] = []

    class Control:
        def read_daemon_process_identity(self, _pipe_dir):
            return MpsProcessIdentity(900, 1)

        def snapshot(self, _pipe_dir):
            raise RuntimeError("snapshot failed")

    def signal_sessions(session_ids, sig):
        signals.append(sig)
        return set(session_ids)

    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(mps_ci, "list_daemon_pids", lambda: {900})
    monkeypatch.setattr(mps_ci, "mps_process_identities", lambda: set())
    monkeypatch.setattr(mps_ci, "signal_test_sessions", signal_sessions)
    monkeypatch.setattr(control, "SubprocessMpsControlClient", Control)

    with pytest.raises(RuntimeError, match="snapshot failed"):
        mps_ci.operator_cleanup({700})

    assert signals == [signal.SIGSTOP, signal.SIGKILL]
    assert tmp_path.exists()


def test_operator_cleanup_preserves_state_while_foreign_clients_remain(
    tmp_path,
    monkeypatch,
):
    make_pipe_dir(tmp_path)

    class Control:
        def read_daemon_process_identity(self, _pipe_dir):
            return MpsProcessIdentity(900, 1)

        def snapshot(self, _pipe_dir):
            return {"foreign-client"}

        def quit_daemon(self, _pipe_dir):
            raise AssertionError("daemon must not be quit")

    times = iter((0.0, 11.0))
    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(mps_ci, "list_daemon_pids", lambda: {900})
    monkeypatch.setattr(mps_ci, "mps_process_identities", lambda: set())
    monkeypatch.setattr(
        mps_ci,
        "signal_test_sessions",
        lambda session_ids, sig: set(session_ids),
    )
    monkeypatch.setattr(mps_ci.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(control, "SubprocessMpsControlClient", Control)

    with pytest.raises(AssertionError, match="clients remain"):
        mps_ci.operator_cleanup({700})

    assert tmp_path.exists()
