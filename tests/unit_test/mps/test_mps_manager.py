# SPDX-License-Identifier: Apache-2.0
"""Ownership-contract tests for the shared per-GPU MPS manager."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from pathlib import Path

import pytest

import sglang_omni.mps.manager as manager_module
from sglang_omni.mps.manager import (
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
    MpsDirtyStateError,
    MpsError,
    MpsLease,
    MpsManager,
)
from sglang_omni.mps.state import MpsGpuPaths


class FakeControlClient:
    """Strict, pipe-scoped stand-in for the MPS control interface."""

    def __init__(self):
        self.daemon_pid = 4242
        self._next_daemon_pid = 4242
        self.daemons: dict[str, int] = {}
        self.alive_pids: set[int] = set()
        self.held_owner_pids: set[int] = set()
        self.client_tokens: dict[int, str] = {}
        self.snapshots: dict[str, set[MpsClientRef]] = {}
        self.start_fails = False
        self.snapshot_error: str | None = None
        self.identity_error: str | None = None
        self.client_token_error: str | None = None
        self.quit_error: str | None = None
        self.quit_works = True
        self.unsafe_daemon_signals: list[tuple[int, bool]] = []
        self.terminated: list[MpsClientRef] = []
        self.terminate_error: str | None = None

    def start_daemon(self, pipe_dir, log_dir, gpu_uuid):
        del log_dir, gpu_uuid
        if self.start_fails:
            raise MpsControlError("spawn failed")
        self.daemon_pid = self._next_daemon_pid
        self._next_daemon_pid += 1
        self.daemons[str(pipe_dir)] = self.daemon_pid
        self.alive_pids.add(self.daemon_pid)
        (Path(pipe_dir) / "nvidia-cuda-mps-control.pid").write_text(
            str(self.daemon_pid)
        )

    def read_daemon_identity(self, pipe_dir):
        if self.identity_error is not None:
            raise MpsControlError(self.identity_error)
        pid_file = Path(pipe_dir) / "nvidia-cuda-mps-control.pid"
        try:
            pid = int(pid_file.read_text())
        except (OSError, ValueError) as exc:
            raise MpsControlError(f"cannot read native PID file: {exc}") from exc
        if self.daemons.get(str(pipe_dir)) != pid or pid not in self.alive_pids:
            raise MpsControlError(f"unverified daemon pid {pid}")
        return pid

    def snapshot(self, pipe_dir):
        if self.snapshot_error is not None:
            raise MpsControlError(self.snapshot_error)
        if str(pipe_dir) not in self.daemons:
            raise MpsControlError("control socket unavailable")
        return set(self.snapshots.get(str(pipe_dir), set()))

    def set_clients(self, pipe_dir, clients: dict[int, list[int]]) -> None:
        self.snapshots[str(pipe_dir)] = {
            MpsClientRef(server_pid, client_pid)
            for server_pid, client_pids in clients.items()
            for client_pid in client_pids
        }

    def terminate_client(self, pipe_dir, client):
        if self.terminate_error is not None:
            raise MpsControlError(self.terminate_error)
        self.terminated.append(client)
        remaining = self.snapshots.get(str(pipe_dir), set()) - {client}
        self.snapshots[str(pipe_dir)] = remaining

    def quit_daemon(self, pipe_dir):
        if self.quit_error is not None:
            raise MpsControlError(self.quit_error)
        if not self.quit_works:
            return
        pid = self.daemons.get(str(pipe_dir))
        if pid is not None:
            self.alive_pids.discard(pid)
        self.snapshots.pop(str(pipe_dir), None)

    def daemon_process_alive(self, pid):
        return pid in self.alive_pids

    def terminate_daemon_process(self, pid, force=False):
        self.unsafe_daemon_signals.append((pid, force))
        self.alive_pids.discard(pid)

    def client_token(self, pid):
        if self.client_token_error is not None:
            raise MpsControlError(self.client_token_error)
        return self.client_tokens.get(pid)

    def owner_lease_held(self, lease_file):
        try:
            return int(lease_file.name) in self.held_owner_pids
        except ValueError as exc:
            raise MpsControlError(f"malformed owner file {lease_file}") from exc


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def daemon_pid_file(paths: MpsGpuPaths) -> Path:
    return paths.pipe_dir / "nvidia-cuda-mps-control.pid"


def owner_marker(manager: MpsManager, pid: int | None = None) -> Path:
    return manager.paths.owners_dir / str(os.getpid() if pid is None else pid)


@pytest.fixture
def short_root():
    root = Path(tempfile.mkdtemp(prefix="mps-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def make_manager(root, client, gpu_uuid=GPU_UUID):
    return MpsManager(
        paths=MpsGpuPaths(state_root=root, gpu_uuid=gpu_uuid),
        gpu_uuid=gpu_uuid,
        client=client,
        poll_interval=0.0,
        start_timeout=0.02,
        verify_timeout=0.02,
        drain_timeout=0.02,
        stop_timeout=0.02,
    )


def seed_shared_dir(
    root,
    client,
    *,
    daemon_pid,
    owners: dict[int, bool] | None = None,
    clients: dict[int, list[int]] | None = None,
):
    paths = MpsGpuPaths(state_root=root, gpu_uuid=GPU_UUID)
    paths.pipe_dir.mkdir(parents=True)
    paths.log_dir.mkdir()
    paths.owners_dir.mkdir()
    daemon_pid_file(paths).write_text(str(daemon_pid))
    client.daemons[str(paths.pipe_dir)] = daemon_pid
    client.alive_pids.add(daemon_pid)
    for owner, held in (owners or {}).items():
        (paths.owners_dir / str(owner)).write_text("active\n")
        if held:
            client.held_owner_pids.add(owner)
    client.set_clients(paths.pipe_dir, clients or {})
    return paths


def start_serving(root, client, client_pid=101):
    manager = make_manager(root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.set_clients(manager.paths.pipe_dir, {7000: [client_pid]})
    client.client_tokens[client_pid] = "owner-worker"
    manager.verify(lease)
    return manager, lease


def test_first_owner_uses_native_pid_file_and_returns_one_lease(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)

    lease = manager.acquire({"worker": "owner-worker"})

    assert lease.daemon_pid == client.daemon_pid
    owner_stat = os.stat(manager.paths.owners_dir / str(os.getpid()))
    fd_stat = os.fstat(lease.owner_fd)
    assert (fd_stat.st_dev, fd_stat.st_ino) == (owner_stat.st_dev, owner_stat.st_ino)
    assert owner_marker(manager).read_text() == "active\n"
    assert daemon_pid_file(manager.paths).read_text() == str(client.daemon_pid)
    assert not (manager.paths.state_dir / "manifest").exists()


def test_second_owner_joins_only_when_every_existing_lease_is_held(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    manager = make_manager(short_root, client)

    lease = manager.acquire({"worker": "owner-worker"})
    assert lease.daemon_pid == 999

    manager.release(lease)
    assert paths.state_dir.is_dir()
    assert (paths.owners_dir / "888").exists()
    assert client.daemon_process_alive(999)


def test_dead_co_owner_is_preserved_and_blocks_join(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True, 777: False},
    )

    with pytest.raises(MpsError, match="777"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})

    assert (paths.owners_dir / "777").exists()


def test_dirty_state_never_claims_ambiguous_clients_are_safe_to_terminate(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={777: False},
        clients={7000: [101], 8000: [202]},
    )

    with pytest.raises(MpsError) as exc_info:
        make_manager(short_root, client).acquire({"worker": "owner-worker"})

    message = str(exc_info.value)
    control = f"CUDA_MPS_PIPE_DIRECTORY={paths.pipe_dir} nvidia-cuda-mps-control"
    assert "terminate_client 7000 101" not in message
    assert "terminate_client 8000 202" not in message
    assert "not proven to belong" in message
    quit_daemon = message.index(f"printf '%s\\n' quit | {control}")
    remove_state = message.index(f"rm -rf {paths.state_dir}")
    assert quit_daemon < remove_state


def test_idle_daemon_is_not_adopted(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={777: False},
    )

    with pytest.raises(MpsError, match="dirty state"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})

    assert paths.state_dir.is_dir()
    assert (paths.owners_dir / "777").exists()
    assert client.daemon_process_alive(999)


def test_missing_native_pid_file_fails_even_when_control_responds(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    daemon_pid_file(paths).unlink()

    with pytest.raises(MpsError, match="native PID file"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})

    assert paths.state_dir.is_dir()


def test_malformed_owner_entry_fails_without_deletion(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    malformed = paths.owners_dir / "not-a-pid"
    malformed.write_text("")

    with pytest.raises(MpsError, match="malformed owner lease"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})

    assert malformed.exists()


def test_unresponsive_new_daemon_is_persisted_without_process_signals(short_root):
    client = FakeControlClient()
    client.snapshot_error = "control unavailable"
    manager = make_manager(short_root, client)

    with pytest.raises(MpsError, match="control daemon"):
        manager.acquire({"worker": "owner-worker"})

    assert client.unsafe_daemon_signals == []
    assert manager.paths.state_dir.is_dir()
    assert owner_marker(manager).read_text() == "retained\n"


def test_ambiguous_start_failure_without_native_identity_preserves_state(short_root):
    client = FakeControlClient()
    client.start_fails = True
    manager = make_manager(short_root, client)

    with pytest.raises(MpsError, match="spawn failed") as exc_info:
        manager.acquire({"worker": "owner-worker"})

    assert manager.paths.state_dir.is_dir()
    assert client.unsafe_daemon_signals == []
    assert owner_marker(manager).read_text() == "retained\n"
    assert isinstance(exc_info.value.__cause__, MpsDirtyStateError)
    assert "lock is released" in str(exc_info.value.__cause__)


def test_preexec_daemon_failure_removes_unstarted_state(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)

    def cannot_execute(pipe_dir, log_dir, gpu_uuid):
        del pipe_dir, log_dir, gpu_uuid
        raise MpsDaemonNotStartedError("control binary was not executed")

    client.start_daemon = cannot_execute

    with pytest.raises(MpsDaemonNotStartedError, match="not executed"):
        manager.acquire({"worker": "owner-worker"})

    assert not manager.paths.state_dir.exists()
    assert client.unsafe_daemon_signals == []


def test_startup_rollback_never_signals_an_unverified_pid(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)

    def lose_identity(pipe_dir):
        del pipe_dir
        client.identity_error = "native identity changed"
        raise MpsControlError("control unavailable")

    client.snapshot = lose_identity

    with pytest.raises(MpsError, match="control daemon"):
        manager.acquire({"worker": "owner-worker"})

    assert manager.paths.state_dir.is_dir()
    assert client.unsafe_daemon_signals == []
    assert owner_marker(manager).read_text() == "retained\n"


def test_verify_returns_current_exact_client_refs(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"a": "owner-a", "b": "owner-b"})
    client.set_clients(manager.paths.pipe_dir, {7000: [101, 102], 8000: [909]})
    client.client_tokens.update({101: "owner-a", 102: "owner-b", 909: "foreign"})

    attached = manager.verify(lease)

    assert attached == {MpsClientRef(7000, 101), MpsClientRef(7000, 102)}


def test_verify_matches_inherited_process_token_on_cuda_client(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.set_clients(manager.paths.pipe_dir, {7000: [200]})
    client.client_tokens[200] = "owner-worker"

    assert manager.verify(lease) == {MpsClientRef(7000, 200)}


def test_verify_does_not_accumulate_clients_across_snapshots(short_root, monkeypatch):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"a": "owner-a", "b": "owner-b"})
    snapshots = iter(
        [
            {MpsClientRef(7000, 101)},
            {MpsClientRef(7000, 202)},
        ]
    )
    client.client_tokens.update({101: "owner-a", 202: "owner-b"})
    last = {MpsClientRef(7000, 202)}
    monkeypatch.setattr(client, "snapshot", lambda _pipe: next(snapshots, last))

    with pytest.raises(MpsError, match=r"stage process\(es\) \['a'\]"):
        manager.verify(lease)

    assert lease.owner_fd >= 0


def test_probe_allows_a_verified_client_to_exit(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)
    assert manager.probe(lease) is None

    client.set_clients(manager.paths.pipe_dir, {})
    assert manager.probe(lease) is None


def test_probe_distinguishes_identity_and_snapshot_failures(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)

    client.identity_error = "native PID unavailable"
    assert manager.probe(lease) == (
        "daemon identity query failed: native PID unavailable"
    )

    client.identity_error = None
    replacement_pid = lease.daemon_pid + 1
    client.daemons[str(manager.paths.pipe_dir)] = replacement_pid
    client.alive_pids.add(replacement_pid)
    daemon_pid_file(manager.paths).write_text(str(replacement_pid))
    assert manager.probe(lease) == (
        f"daemon identity changed from {lease.daemon_pid} to {replacement_pid}"
    )

    client.daemons[str(manager.paths.pipe_dir)] = lease.daemon_pid
    daemon_pid_file(manager.paths).write_text(str(lease.daemon_pid))
    client.snapshot_error = "control socket unavailable"
    assert manager.probe(lease) == (
        "client snapshot query failed: control socket unavailable"
    )


def test_dead_root_with_live_descendant_persists_dirty_and_reports_cleanup(
    short_root,
):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.set_clients(manager.paths.pipe_dir, {7000: [200]})
    client.client_tokens[200] = "owner-worker"
    manager.verify(lease)

    with pytest.raises(MpsDirtyStateError, match="owned=") as exc_info:
        manager.release(lease)

    assert lease.owner_fd == -1
    assert manager.paths.state_dir.is_dir()
    assert owner_marker(manager).read_text() == "retained\n"
    assert client.daemon_process_alive(lease.daemon_pid)
    assert client.unsafe_daemon_signals == []
    assert f"Owner PID {os.getpid()}" in str(exc_info.value)
    assert "Current MPS client refs" in str(exc_info.value)
    assert "terminate_client 7000 200" in str(exc_info.value)
    assert "lock is released" in str(exc_info.value)

    owner_fd = os.open(owner_marker(manager), os.O_RDWR)
    try:
        fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(owner_fd)

    with pytest.raises(MpsError, match="dirty state"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})


def test_dirty_owner_guidance_never_targets_a_coowners_clients(short_root):
    client = FakeControlClient()
    seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    foreign = MpsClientRef(8000, 202)
    client.set_clients(manager.paths.pipe_dir, {7000: [101], 8000: [202]})
    client.client_tokens.update({101: "owner-worker", 202: "coowner-worker"})
    manager.verify(lease)

    with pytest.raises(MpsDirtyStateError) as exc_info:
        manager.release(lease)

    message = str(exc_info.value)
    assert "terminate_client 7000 101" in message
    assert "terminate_client 8000 202" not in message
    assert f"not proven to belong to this lease: [{foreign!r}]" in message
    assert lease.owner_fd == -1


def test_verified_owner_release_ignores_only_coowner_clients(
    short_root,
):
    client = FakeControlClient()
    seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.set_clients(manager.paths.pipe_dir, {7000: [101], 8000: [202]})
    client.client_tokens.update({101: "owner-worker", 202: "coowner-worker"})
    manager.verify(lease)
    client.set_clients(manager.paths.pipe_dir, {8000: [202]})

    manager.release(lease)

    assert lease.owner_fd == -1
    assert not owner_marker(manager).exists()
    assert client.daemon_process_alive(lease.daemon_pid)
    assert client.snapshot(manager.paths.pipe_dir) == {MpsClientRef(8000, 202)}


def test_client_created_after_verify_prevents_shared_owner_release(short_root):
    client = FakeControlClient()
    seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.set_clients(manager.paths.pipe_dir, {7000: [101]})
    client.client_tokens[101] = "owner-worker"
    manager.verify(lease)

    client.set_clients(manager.paths.pipe_dir, {7000: [303], 8000: [202]})
    client.client_tokens.update({303: "owner-worker", 202: "coowner-worker"})

    with pytest.raises(MpsDirtyStateError) as exc_info:
        manager.release(lease)

    assert "terminate_client 7000 303" in str(exc_info.value)
    assert "terminate_client 8000 202" not in str(exc_info.value)
    assert owner_marker(manager).read_text() == "retained\n"
    assert (manager.paths.owners_dir / "888").read_text() == "active\n"
    assert client.daemon_process_alive(999)


def test_unattributable_orphan_persists_dirty_and_blocks_join(short_root):
    client = FakeControlClient()
    seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
        clients={7000: [200], 8000: [202]},
    )
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})

    with pytest.raises(MpsDirtyStateError) as exc_info:
        manager.release(lease)

    assert "terminate_client 7000 200" not in str(exc_info.value)
    assert owner_marker(manager).read_text() == "retained\n"
    assert lease.owner_fd == -1
    with pytest.raises(MpsError, match="retained"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})


def test_dirty_owner_blocks_join_without_interrupting_clean_coowner(
    short_root,
    monkeypatch,
):
    current_pid = 1001
    monkeypatch.setattr(manager_module.os, "getpid", lambda: current_pid)
    client = FakeControlClient()

    manager_a = make_manager(short_root, client)
    lease_a = manager_a.acquire({"worker-a": "owner-a"})
    client.held_owner_pids.add(1001)

    current_pid = 1002
    manager_b = make_manager(short_root, client)
    lease_b = manager_b.acquire({"worker-b": "owner-b"})
    client.held_owner_pids.add(1002)

    client.set_clients(manager_a.paths.pipe_dir, {7000: [101], 8000: [202]})
    client.client_tokens.update({101: "owner-a", 202: "owner-b"})
    current_pid = 1001
    manager_a.verify(lease_a)
    current_pid = 1002
    manager_b.verify(lease_b)

    current_pid = 1001
    with pytest.raises(MpsDirtyStateError) as exc_info:
        manager_a.release(lease_a)

    message = str(exc_info.value)
    assert "terminate_client 7000 101" in message
    assert "terminate_client 8000 202" not in message
    assert owner_marker(manager_a, 1001).read_text() == "retained\n"
    assert lease_a.owner_fd == -1
    current_pid = 1002
    assert manager_b.probe(lease_b) is None
    assert client.daemon_process_alive(lease_b.daemon_pid)

    current_pid = 1003
    with pytest.raises(MpsError, match="retained") as join_error:
        make_manager(short_root, client).acquire({"worker": "owner-worker"})
    assert "terminate_client 7000 101" not in str(join_error.value)
    assert "terminate_client 8000 202" not in str(join_error.value)

    client.set_clients(manager_b.paths.pipe_dir, {})
    current_pid = 1002
    manager_b.release(lease_b)

    assert lease_b.owner_fd == -1
    assert client.daemon_process_alive(lease_b.daemon_pid)
    assert manager_b.paths.state_dir.is_dir()
    assert (manager_b.paths.owners_dir / "1001").read_text() == "retained\n"
    assert not (manager_b.paths.owners_dir / "1002").exists()


def test_last_owner_preserves_unknown_clients_instead_of_quitting(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.set_clients(manager.paths.pipe_dir, {7000: [909]})

    with pytest.raises(MpsDirtyStateError, match="unattributable"):
        manager.release(lease)

    assert lease.owner_fd == -1
    assert owner_marker(manager).read_text() == "retained\n"
    assert manager.paths.state_dir.is_dir()
    assert client.daemon_process_alive(lease.daemon_pid)


def test_happy_path_detaches_releases_and_quits_last_owner(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)
    client.set_clients(manager.paths.pipe_dir, {})

    manager.release(lease)

    assert lease.owner_fd == -1
    assert not manager.paths.state_dir.exists()
    assert not client.daemon_process_alive(lease.daemon_pid)


def test_dead_daemon_during_service_is_preserved_as_dirty(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)
    client.set_clients(manager.paths.pipe_dir, {})
    client.alive_pids.discard(lease.daemon_pid)

    with pytest.raises(MpsDirtyStateError, match="unverified daemon") as exc_info:
        manager.release(lease)

    assert lease.owner_fd == -1
    assert owner_marker(manager).read_text() == "retained\n"
    assert manager.paths.state_dir.is_dir()
    assert f"Owner PID {os.getpid()}" in str(exc_info.value)
    assert "Current MPS client refs: unavailable" in str(exc_info.value)
    assert "lock is released" in str(exc_info.value)


def test_dirty_coowner_does_not_block_clean_owner_exit(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    client.held_owner_pids.discard(888)

    manager.release(lease)

    assert lease.owner_fd == -1
    assert not owner_marker(manager).exists()
    assert (paths.owners_dir / "888").exists()
    assert paths.state_dir.is_dir()
    assert client.daemon_process_alive(lease.daemon_pid)

    with pytest.raises(MpsError, match="dirty state"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})


def test_corrupt_coowner_does_not_block_clean_owner_exit(short_root):
    client = FakeControlClient()
    paths = seed_shared_dir(
        short_root,
        client,
        daemon_pid=999,
        owners={888: True},
    )
    manager = make_manager(short_root, client)
    lease = manager.acquire({"worker": "owner-worker"})
    corrupt_owner = paths.owners_dir / "888"
    corrupt_owner.write_text("broken\n")

    manager.release(lease)

    assert lease.owner_fd == -1
    assert not owner_marker(manager).exists()
    assert corrupt_owner.read_text() == "broken\n"
    assert paths.state_dir.is_dir()
    assert client.daemon_process_alive(lease.daemon_pid)

    with pytest.raises(MpsError, match="invalid status"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})


def test_daemon_refusing_quit_persists_dirty_state(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)
    client.set_clients(manager.paths.pipe_dir, {})
    client.quit_works = False

    with pytest.raises(MpsDirtyStateError, match="did not exit") as exc_info:
        manager.release(lease)

    assert lease.owner_fd == -1
    assert owner_marker(manager).read_text() == "retained\n"
    assert manager.paths.state_dir.is_dir()
    assert f"Owner PID {os.getpid()}" in str(exc_info.value)
    assert client.unsafe_daemon_signals == []

    with pytest.raises(MpsError, match="retained"):
        make_manager(short_root, client).acquire({"worker": "owner-worker"})


def test_quit_control_error_persists_dirty_and_releases_authority(short_root):
    client = FakeControlClient()
    manager, lease = start_serving(short_root, client)
    client.set_clients(manager.paths.pipe_dir, {})
    client.quit_error = "quit control failed"

    with pytest.raises(
        MpsDirtyStateError,
        match="quit control failed",
    ) as exc_info:
        manager.release(lease)

    assert lease.owner_fd == -1
    assert owner_marker(manager).read_text() == "retained\n"
    assert manager.paths.state_dir.is_dir()
    assert "lock is released" in str(exc_info.value)
    assert client.unsafe_daemon_signals == []


def test_release_requires_the_acquisition_token(short_root, tmp_path):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    foreign_fd = os.open(
        tmp_path / "foreign-owner",
        os.O_CREAT | os.O_RDWR,
        0o600,
    )

    try:
        with pytest.raises(MpsError, match="live MPS lease"):
            manager.release(
                MpsLease(
                    daemon_pid=123,
                    owner_fd=foreign_fd,
                    client_tokens={"worker": "owner-worker"},
                )
            )
    finally:
        os.close(foreign_fd)


def test_daemon_liveness_rejects_zombie_proc_entries(monkeypatch):
    from sglang_omni.mps import control

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


def test_retire_targets_only_the_named_process_clients(short_root):
    """Retirement must reach this process's clients and nobody else's."""

    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"a": "token-a", "b": "token-b"})
    client.set_clients(manager.paths.pipe_dir, {7000: [201, 202, 301, 999]})
    client.client_tokens.update(
        {201: "token-a", 202: "token-a", 301: "token-b", 999: "other-serve"}
    )

    retired = manager.retire_clients_for(lease, "a")

    assert {ref.client_pid for ref in retired} == {201, 202}
    assert {ref.client_pid for ref in client.terminated} == {201, 202}


def test_retire_is_a_noop_for_an_unmanaged_process(short_root):
    client = FakeControlClient()
    manager = make_manager(short_root, client)
    lease = manager.acquire({"a": "token-a"})
    client.set_clients(manager.paths.pipe_dir, {7000: [201]})
    client.client_tokens[201] = "token-a"

    assert manager.retire_clients_for(lease, "not-managed") == set()
    assert client.terminated == []
