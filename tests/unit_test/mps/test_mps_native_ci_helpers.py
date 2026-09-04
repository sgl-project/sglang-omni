# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the native-MPS hardware-test cleanup helper."""

from __future__ import annotations

import signal

import pytest

from sglang_omni.mps import control as mps_control
from tests.test_ci import test_mps_native as mps_ci


def _make_pipe_dir(tmp_path):
    pipe_dir = tmp_path / "GPU-test" / "pipe"
    pipe_dir.mkdir(parents=True)
    return pipe_dir


def test_operator_cleanup_preserves_signal_and_control_order(
    tmp_path,
    monkeypatch,
) -> None:
    pipe_dir = _make_pipe_dir(tmp_path)
    events: list[tuple] = []
    live_sessions = {700}
    clients = {"owned-client"}

    class Control:
        daemon_alive = True

        def read_daemon_identity(self, selected_pipe_dir):
            events.append(("read", selected_pipe_dir))
            return 900

        def snapshot(self, selected_pipe_dir):
            events.append(("snapshot", selected_pipe_dir, frozenset(clients)))
            return set(clients)

        def quit_daemon(self, selected_pipe_dir):
            events.append(("quit", selected_pipe_dir))
            self.daemon_alive = False

        def daemon_process_alive(self, pid):
            events.append(("alive", pid, self.daemon_alive))
            return self.daemon_alive

    control = Control()

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
        "_daemon_pids",
        lambda: {900} if control.daemon_alive else set(),
    )
    monkeypatch.setattr(
        mps_ci,
        "_mps_process_identities",
        lambda: {mps_ci._ProcessIdentity(900, 1)} if control.daemon_alive else set(),
    )
    monkeypatch.setattr(mps_ci, "_signal_test_sessions", signal_sessions)
    monkeypatch.setattr(
        mps_ci,
        "_assert_process_identities_gone",
        lambda identities: events.append(("gone", frozenset(identities))),
    )
    monkeypatch.setattr(mps_control, "SubprocessMpsControlClient", lambda: control)

    mps_ci._operator_cleanup({700})

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

    mps_ci._operator_cleanup({700})
    assert not tmp_path.exists()


def test_operator_cleanup_preserves_state_until_mps_processes_are_gone(
    tmp_path,
    monkeypatch,
) -> None:
    _make_pipe_dir(tmp_path)
    identity = mps_ci._ProcessIdentity(900, 1)

    def assert_processes_gone(_identities):
        raise AssertionError("still alive")

    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(mps_ci, "_daemon_pids", lambda: set())
    monkeypatch.setattr(mps_ci, "_mps_process_identities", lambda: {identity})
    monkeypatch.setattr(
        mps_ci,
        "_signal_test_sessions",
        lambda session_ids, sig: set(session_ids),
    )
    monkeypatch.setattr(
        mps_ci,
        "_assert_process_identities_gone",
        assert_processes_gone,
    )
    monkeypatch.setattr(
        mps_control,
        "SubprocessMpsControlClient",
        lambda: object(),
    )

    with pytest.raises(AssertionError, match="still alive"):
        mps_ci._operator_cleanup({700})

    assert tmp_path.exists()


def test_operator_cleanup_kills_owned_sessions_but_preserves_state_on_snapshot_error(
    tmp_path,
    monkeypatch,
) -> None:
    _make_pipe_dir(tmp_path)
    signals: list[int] = []

    class Control:
        def read_daemon_identity(self, _pipe_dir):
            return 900

        def snapshot(self, _pipe_dir):
            raise RuntimeError("snapshot failed")

    def signal_sessions(session_ids, sig):
        signals.append(sig)
        return set(session_ids)

    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(mps_ci, "_daemon_pids", lambda: {900})
    monkeypatch.setattr(mps_ci, "_mps_process_identities", lambda: set())
    monkeypatch.setattr(mps_ci, "_signal_test_sessions", signal_sessions)
    monkeypatch.setattr(mps_control, "SubprocessMpsControlClient", Control)

    with pytest.raises(RuntimeError, match="snapshot failed"):
        mps_ci._operator_cleanup({700})

    assert signals == [signal.SIGSTOP, signal.SIGKILL]
    assert tmp_path.exists()


def test_operator_cleanup_preserves_state_while_foreign_clients_remain(
    tmp_path,
    monkeypatch,
) -> None:
    _make_pipe_dir(tmp_path)

    class Control:
        def read_daemon_identity(self, _pipe_dir):
            return 900

        def snapshot(self, _pipe_dir):
            return {"foreign-client"}

        def quit_daemon(self, _pipe_dir):
            raise AssertionError("daemon must not be quit")

    times = iter((0.0, 11.0))
    monkeypatch.setattr(mps_ci, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(mps_ci, "_daemon_pids", lambda: {900})
    monkeypatch.setattr(mps_ci, "_mps_process_identities", lambda: set())
    monkeypatch.setattr(
        mps_ci,
        "_signal_test_sessions",
        lambda session_ids, sig: set(session_ids),
    )
    monkeypatch.setattr(mps_ci.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(mps_control, "SubprocessMpsControlClient", Control)

    with pytest.raises(AssertionError, match="clients remain"):
        mps_ci._operator_cleanup({700})

    assert tmp_path.exists()
