# SPDX-License-Identifier: Apache-2.0
"""Bounded read retries through acquired MPS leases and real filesystem locks."""

from __future__ import annotations

import errno
import fcntl
import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from sglang_omni.mps import manager as manager_module
from sglang_omni.mps.manager import (
    MpsControlError,
    MpsDirtyStateError,
    MpsError,
    MpsRetryableControlError,
)
from tests.unit_test.mps.test_mps_manager import (
    FakeControlClient,
    make_manager,
    start_serving,
)


@pytest.fixture
def short_root():
    with tempfile.TemporaryDirectory(prefix="mps-retry-", dir="/tmp") as root:
        yield Path(root)


@pytest.fixture
def serving(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)
    try:
        yield manager, client, lease
    finally:
        if lease.owner_fd >= 0:
            os.close(lease.owner_fd)
            lease.owner_fd = -1


class RetryClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def install_clock(manager, monkeypatch):
    clock = RetryClock()
    monkeypatch.setattr(manager_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(manager_module.time, "sleep", clock.sleep)
    manager.poll_interval = 0.6
    manager.start_timeout = 1.0
    manager.verify_timeout = 1.0
    manager.drain_timeout = 1.0
    manager.stop_timeout = 1.0
    return clock


def assert_locks_between_attempts(manager):
    with (manager.paths.state_root / f".lock-{manager.gpu_uuid}").open("r") as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with manager.owner_file.open("r") as owner:
        with pytest.raises(BlockingIOError):
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert manager.owner_file.read_text() == "active\n"


@pytest.mark.parametrize("read", ["read_daemon_identity", "snapshot", "client_token"])
@pytest.mark.parametrize("failure", ["interrupt", "again", "io_timeout", "timeout"])
def test_release_retries_temporary_read_failure_without_dropping_its_lease(
    serving, monkeypatch, read, failure
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    causes = {
        "interrupt": OSError(errno.EINTR, "interrupted"),
        "again": OSError(errno.EAGAIN, "try again"),
        "io_timeout": OSError(errno.ETIMEDOUT, "timed out"),
        "timeout": subprocess.TimeoutExpired("native query", 10),
    }
    original = getattr(client, read)
    attempts = []

    def fail_once(*args):
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise MpsRetryableControlError(failure) from causes[failure]
        return original(*args)

    def recovered(seconds):
        assert_locks_between_attempts(manager)
        assert manager.probe(lease) is None
        client.set_clients(manager.paths.pipe_dir, {})
        clock.sleep(seconds)

    monkeypatch.setattr(client, read, fail_once)
    monkeypatch.setattr(manager_module.time, "sleep", recovered)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)

    manager.release(lease)

    assert attempts[0] == 0
    assert clock.sleeps == [0.6]
    assert quit_daemon.call_count == 1
    assert lease.owner_fd == -1
    assert not manager.paths.state_dir.exists()
    assert client.terminated == client.unsafe_daemon_signals == []


@pytest.mark.parametrize("read", ["read_daemon_identity", "snapshot", "client_token"])
@pytest.mark.parametrize(
    "cause",
    [
        BlockingIOError(errno.EAGAIN, "temporarily unavailable"),
        PermissionError(errno.EACCES, "permission denied"),
    ],
)
def test_persistent_read_failure_preserves_dirty_state_and_blocks_join(
    serving, monkeypatch, read, cause
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    error = MpsRetryableControlError("read remains unavailable")
    error.__cause__ = cause
    original = getattr(client, read)
    failed = Mock(side_effect=error)
    monkeypatch.setattr(client, read, failed)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)

    with pytest.raises(MpsDirtyStateError, match="last control error") as exc_info:
        manager.release(lease)

    assert clock.now == 1.0
    assert clock.sleeps == [0.6, 0.4]
    assert failed.call_count == 3
    assert quit_daemon.call_count == 0
    assert client.terminated == client.unsafe_daemon_signals == []
    assert manager.owner_file.read_text() == "retained\n"
    assert lease.owner_fd == -1
    with manager.owner_file.open("r") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert manager.paths.state_dir.is_dir()
    causes = []
    cause = exc_info.value
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    assert error in causes
    monkeypatch.setattr(client, read, original)
    with pytest.raises(MpsError, match="retained"):
        make_manager(manager.paths.state_root, client).acquire({"peer": "peer-token"})


@pytest.mark.parametrize("operation", ["verify", "retire", "release"])
@pytest.mark.parametrize("read", ["read_daemon_identity", "snapshot", "client_token"])
@pytest.mark.parametrize("persistent", [False, True])
def test_permission_reads_recover_or_fail_at_the_phase_deadline(
    serving, monkeypatch, operation, read, persistent
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    error = MpsRetryableControlError("permission denied")
    cause = PermissionError(errno.EACCES, "permission denied")
    original = getattr(client, read)
    attempts = 0

    def permission_read(*args):
        nonlocal attempts
        attempts += 1
        if persistent or attempts == 1:
            raise error from cause
        else:
            return original(*args)

    def recovered(seconds):
        assert_locks_between_attempts(manager)
        if operation == "release" and not persistent:
            client.set_clients(manager.paths.pipe_dir, {})
        clock.sleep(seconds)

    monkeypatch.setattr(client, read, permission_read)
    monkeypatch.setattr(manager_module.time, "sleep", recovered)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)

    expected = (
        pytest.raises(MpsError, match="permission denied")
        if persistent
        else nullcontext()
    )
    with expected as exc_info:
        if operation == "verify":
            manager.verify(lease)
        elif operation == "retire":
            manager.retire_clients_for(lease, "worker")
        else:
            manager.release(lease)

    if persistent:
        observed_error = exc_info.value
        while observed_error is not None and observed_error is not error:
            observed_error = observed_error.__cause__
        assert observed_error is error
        assert clock.now == 1.0
        assert clock.sleeps == [0.6, 0.4]
        assert quit_daemon.call_count == 0
        assert client.terminated == []
        assert manager.paths.state_dir.is_dir()
        if operation == "release":
            assert lease.owner_fd == -1
            assert manager.owner_file.read_text() == "retained\n"
        else:
            assert lease.owner_fd >= 0
            assert manager.owner_file.read_text() == "active\n"
    else:
        assert clock.sleeps == [0.6]
        assert quit_daemon.call_count == (1 if operation == "release" else 0)
        assert client.terminated == (
            [manager_module.MpsClientRef(7000, 101)] if operation == "retire" else []
        )
        if operation == "release":
            assert lease.owner_fd == -1
            assert not manager.paths.state_dir.exists()
        else:
            assert lease.owner_fd >= 0


@pytest.mark.parametrize("pending", ["read_failure", "clients_remain"])
def test_deadline_prevents_a_fresh_attempt_even_if_it_would_now_succeed(
    serving, monkeypatch, pending
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    manager.poll_interval = 2.0
    if pending == "read_failure":
        monkeypatch.setattr(
            client,
            "snapshot",
            Mock(side_effect=MpsRetryableControlError("not ready")),
        )

    original_snapshot = FakeControlClient.snapshot.__get__(client)

    def become_ready(seconds):
        assert_locks_between_attempts(manager)
        clock.sleep(seconds)
        monkeypatch.setattr(client, "snapshot", original_snapshot)
        client.set_clients(manager.paths.pipe_dir, {})

    monkeypatch.setattr(manager_module.time, "sleep", become_ready)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)

    with pytest.raises(MpsDirtyStateError):
        manager.release(lease)

    assert clock.sleeps == [1.0]
    assert quit_daemon.call_count == 0
    assert manager.owner_file.read_text() == "retained\n"
    assert manager.paths.state_dir.is_dir()


@pytest.mark.parametrize("operation", ["verify", "retire", "release"])
@pytest.mark.parametrize("change", ["owner_replaced", "daemon_replaced"])
def test_retry_revalidates_lease_and_daemon_before_query_or_commit(
    serving, monkeypatch, operation, change
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    snapshot = Mock(side_effect=MpsRetryableControlError("try again"))
    monkeypatch.setattr(client, "snapshot", snapshot)

    def replace_session(seconds):
        assert_locks_between_attempts(manager)
        if change == "owner_replaced":
            manager.owner_file.unlink()
            manager.owner_file.write_text("active\n")
        else:
            new_pid = lease.daemon_pid + 100
            client.daemons[str(manager.paths.pipe_dir)] = new_pid
            client.alive_pids.add(new_pid)
            (manager.paths.pipe_dir / "nvidia-cuda-mps-control.pid").write_text(
                str(new_pid)
            )
        snapshot.side_effect = None
        snapshot.return_value = set()
        clock.sleep(seconds)

    monkeypatch.setattr(manager_module.time, "sleep", replace_session)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)
    match = "live MPS lease" if change == "owner_replaced" else "identity changed"
    with pytest.raises(MpsError, match=match):
        if operation == "verify":
            manager.verify(lease)
        elif operation == "retire":
            manager.retire_clients_for(lease, "worker")
        else:
            manager.release(lease)

    assert clock.sleeps == [0.6]
    assert quit_daemon.call_count == 0
    assert client.terminated == []
    assert snapshot.call_count == (2 if operation == "release" else 1)
    assert manager.paths.state_dir.is_dir()


@pytest.mark.parametrize("operation", ["verify", "retire"])
@pytest.mark.parametrize("read", ["snapshot", "client_token", "read_daemon_identity"])
def test_other_lifecycle_reads_recover_with_fresh_observations(
    serving, monkeypatch, operation, read
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    original = getattr(client, read)
    calls = 0

    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MpsRetryableControlError("read not available")
        return original(*args)

    def retry_sleep(seconds):
        assert_locks_between_attempts(manager)
        clock.sleep(seconds)

    monkeypatch.setattr(client, read, fail_once)
    monkeypatch.setattr(manager_module.time, "sleep", retry_sleep)
    expected = {manager_module.MpsClientRef(7000, 101)}
    if operation == "verify":
        assert manager.verify(lease) == expected
        assert client.terminated == []
    else:
        assert manager.retire_clients_for(lease, "worker") == expected
        assert set(client.terminated) == expected
    assert clock.sleeps == [0.6]


@pytest.mark.parametrize("operation", ["verify", "retire"])
def test_persistent_observation_failure_never_retires_or_changes_lease(
    serving, monkeypatch, operation
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    monkeypatch.setattr(
        client,
        "client_token",
        Mock(side_effect=MpsRetryableControlError("cannot inspect ownership")),
    )
    with pytest.raises(MpsError, match="last control error"):
        if operation == "verify":
            manager.verify(lease)
        else:
            manager.retire_clients_for(lease, "worker")
    assert clock.now == 1.0
    assert client.terminated == []
    assert lease.owner_fd >= 0
    assert manager.owner_file.read_text() == "active\n"


@pytest.mark.parametrize(
    "error_type, cause",
    [
        (MpsControlError, None),
        (MpsRetryableControlError, subprocess.TimeoutExpired("terminate_client", 10)),
        (MpsRetryableControlError, PermissionError(errno.EACCES, "permission denied")),
    ],
)
def test_partial_retirement_failure_is_not_retried(
    serving, monkeypatch, error_type, cause
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    client.set_clients(manager.paths.pipe_dir, {7000: [101, 102]})
    client.client_tokens[102] = "owner-worker"
    original = client.terminate_client

    def fail_second(pipe_dir, ref):
        if ref.client_pid == 102:
            raise error_type("termination failed after the first commit") from cause
        original(pipe_dir, ref)

    terminate = Mock(side_effect=fail_second)
    monkeypatch.setattr(client, "terminate_client", terminate)
    with pytest.raises(MpsControlError, match="termination failed"):
        manager.retire_clients_for(lease, "worker")
    assert terminate.call_count == 2
    assert [ref.client_pid for ref in client.terminated] == [101]
    assert clock.sleeps == []


@pytest.mark.parametrize("failure", ["quit", "exit_read", "daemon_survives"])
def test_quit_is_never_repeated_after_a_commit_or_a_lost_response(
    serving, monkeypatch, failure
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    client.set_clients(manager.paths.pipe_dir, {})
    if failure == "quit":
        client.quit_error = "response lost"
    elif failure == "exit_read":
        monkeypatch.setattr(
            client,
            "daemon_process_alive",
            Mock(side_effect=MpsRetryableControlError("read")),
        )
    else:
        client.quit_works = False
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)

    with pytest.raises(MpsDirtyStateError):
        manager.release(lease)
    assert quit_daemon.call_count == 1
    assert clock.sleeps == ([] if failure == "quit" else [0.6, 0.4])
    assert lease.owner_fd == -1
    assert manager.owner_file.read_text() == "retained\n"


def test_exit_read_recovers_under_the_quit_commit_barrier(serving, monkeypatch):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    client.set_clients(manager.paths.pipe_dir, {})
    alive = Mock(side_effect=[MpsRetryableControlError("stat not readable yet"), False])
    monkeypatch.setattr(client, "daemon_process_alive", alive)

    def confirm_under_lock(seconds):
        with (manager.paths.state_root / f".lock-{manager.gpu_uuid}").open("r") as gpu:
            with pytest.raises(BlockingIOError):
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock.sleep(seconds)

    monkeypatch.setattr(manager_module.time, "sleep", confirm_under_lock)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)
    manager.release(lease)
    assert quit_daemon.call_count == 1
    assert alive.call_count == 2
    assert not manager.paths.state_dir.exists()


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("cause", [None, PermissionError(errno.EACCES, "unreadable")])
def test_create_retries_only_readiness_and_rolls_back_without_reentrant_flock(
    short_root, monkeypatch, persistent, cause
):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    clock = install_clock(manager, monkeypatch)
    original = client.snapshot
    calls = 0

    def unavailable(pipe_dir):
        nonlocal calls
        calls += 1
        if persistent or calls == 1:
            raise MpsRetryableControlError("control not ready") from cause
        return original(pipe_dir)

    monkeypatch.setattr(client, "snapshot", unavailable)
    start = Mock(wraps=client.start_daemon)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "start_daemon", start)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)
    if persistent:
        with pytest.raises(MpsError, match="not ready") as exc_info:
            manager.acquire({"worker": "owner-worker"})
        assert isinstance(exc_info.value.__cause__, MpsDirtyStateError)
        assert manager.owner_file.read_text() == "retained\n"
        with manager.owner_file.open("r") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert quit_daemon.call_count == 0
    else:
        lease = manager.acquire({"worker": "owner-worker"})
        assert calls == 2
        manager.release(lease, clients_could_have_attached=False)
        assert quit_daemon.call_count == 1
        assert not manager.paths.state_dir.exists()
    assert start.call_count == 1


def test_zero_timeout_still_performs_one_immediate_observation(serving, monkeypatch):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    manager.verify_timeout = manager.drain_timeout = manager.stop_timeout = 0
    assert manager.verify(lease)
    client.set_clients(manager.paths.pipe_dir, {})
    manager.release(lease)
    assert clock.sleeps == []
    assert not manager.paths.state_dir.exists()


@pytest.mark.parametrize("operation", ["verify", "retire", "release"])
@pytest.mark.parametrize("read", ["snapshot", "client_token", "read_daemon_identity"])
@pytest.mark.parametrize("cause", [None, OSError(errno.EIO, "permanent I/O error")])
def test_terminal_read_failure_is_not_retried(
    serving, monkeypatch, operation, read, cause
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    error = MpsControlError("invalid result")
    error.__cause__ = cause
    failed = Mock(side_effect=error)
    monkeypatch.setattr(client, read, failed)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)
    with pytest.raises(MpsError, match="invalid result"):
        if operation == "verify":
            manager.verify(lease)
        elif operation == "retire":
            manager.retire_clients_for(lease, "worker")
        else:
            manager.release(lease)
    assert clock.sleeps == []
    assert failed.call_count == (2 if operation == "release" else 1)
    assert quit_daemon.call_count == 0
    assert client.terminated == []


@pytest.mark.parametrize("outcome", ["dead", "alive", "unavailable", "invalid"])
def test_lost_quit_response_only_retries_unavailable_exit_reads(
    serving, monkeypatch, outcome
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    client.set_clients(manager.paths.pipe_dir, {})
    error = MpsRetryableControlError("quit response lost")
    quit_daemon = Mock(side_effect=error)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)
    unavailable = MpsRetryableControlError("stat unreadable")
    replies = {
        "dead": [unavailable, False],
        "alive": [unavailable, True],
        "unavailable": [unavailable, unavailable],
        "invalid": [MpsControlError("invalid stat")],
    }
    alive = Mock(side_effect=replies[outcome])
    monkeypatch.setattr(client, "daemon_process_alive", alive)
    if outcome == "dead":
        manager.release(lease)
        assert not manager.paths.state_dir.exists()
    else:
        with pytest.raises(MpsDirtyStateError) as exc_info:
            manager.release(lease)
        assert manager.owner_file.read_text() == "retained\n"
        assert outcome != "alive" or exc_info.value.__cause__ is error
    assert quit_daemon.call_count == 1
    assert alive.call_count == (1 if outcome == "invalid" else 2)
    assert clock.sleeps == (
        []
        if outcome == "invalid"
        else [0.6, 0.4] if outcome == "unavailable" else [0.6]
    )
    assert lease.owner_fd == -1


def test_retirement_read_budget_uses_stop_timeout_without_reset(serving, monkeypatch):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    manager.stop_timeout = 0.4
    manager.drain_timeout = 10
    failed = Mock(side_effect=MpsRetryableControlError("unavailable"))
    monkeypatch.setattr(client, "snapshot", failed)
    with pytest.raises(MpsError, match="unavailable"):
        manager.retire_clients_for(lease, "worker")
    assert clock.sleeps == [0.4]
    assert failed.call_count == 1
    assert client.terminated == []
    assert lease.owner_fd >= 0


def test_release_read_failure_then_pending_clients_share_one_budget(
    serving, monkeypatch
):
    manager, client, lease = serving
    clock = install_clock(manager, monkeypatch)
    snapshot = Mock(
        side_effect=[
            MpsRetryableControlError("unavailable"),
            {manager_module.MpsClientRef(7000, 101)},
            {manager_module.MpsClientRef(7000, 101)},
        ]
    )
    monkeypatch.setattr(client, "snapshot", snapshot)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)
    with pytest.raises(MpsDirtyStateError, match="owned="):
        manager.release(lease)
    assert clock.now == 1.0
    assert clock.sleeps == [0.6, 0.4]
    assert quit_daemon.call_count == 0


def test_dirty_status_write_failure_still_closes_the_owner_fd(serving, monkeypatch):
    manager, client, lease = serving
    install_clock(manager, monkeypatch)
    monkeypatch.setattr(
        client, "snapshot", Mock(side_effect=MpsRetryableControlError("unreadable"))
    )
    monkeypatch.setattr(
        manager,
        "write_owner_status",
        Mock(side_effect=OSError(errno.EIO, "cannot persist")),
    )
    with pytest.raises(MpsDirtyStateError, match="unconfirmed"):
        manager.release(lease)
    assert lease.owner_fd == -1
    assert manager.owner_file.exists()
    with manager.owner_file.open("r") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert manager.paths.state_dir.exists()
    assert client.terminated == []
    with pytest.raises(MpsError, match="dirty state"):
        make_manager(manager.paths.state_root, client).acquire({"peer": "peer-token"})


def test_state_removal_failure_does_not_repeat_quit_or_close_spent_owner(
    serving, monkeypatch
):
    manager, client, lease = serving
    client.set_clients(manager.paths.pipe_dir, {})
    owner_fd = lease.owner_fd
    remove_error = OSError(errno.EIO, "cannot remove state")

    def fail_remove(path):
        assert path == manager.paths.state_dir
        raise remove_error

    monkeypatch.setattr(manager_module.shutil, "rmtree", fail_remove)
    quit_daemon = Mock(wraps=client.quit_daemon)
    monkeypatch.setattr(client, "quit_daemon", quit_daemon)

    with pytest.raises(MpsControlError, match="cannot remove state") as exc_info:
        manager.release(lease)

    assert exc_info.value.__cause__ is remove_error
    assert quit_daemon.call_count == 1
    assert not client.daemon_process_alive(lease.daemon_pid)
    assert lease.owner_fd == -1
    with pytest.raises(OSError) as closed_fd:
        os.fstat(owner_fd)
    assert closed_fd.value.errno == errno.EBADF
    assert not manager.owner_file.exists()
    assert manager.paths.state_dir.exists()
