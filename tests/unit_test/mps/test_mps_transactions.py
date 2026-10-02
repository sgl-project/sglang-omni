# SPDX-License-Identifier: Apache-2.0
"""Manager transactions with native I/O stand-ins and real process-held locks."""

from __future__ import annotations

import fcntl
import multiprocessing
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.synchronize import Event
from pathlib import Path
from unittest.mock import patch

import pytest

from sglang_omni.mps import control
from sglang_omni.mps import manager as manager_module
from sglang_omni.mps.manager import MpsClientRef, MpsControlError, MpsDirtyStateError
from tests.unit_test.mps.test_mps_manager import GPU_UUID, make_manager

CLIENT_PIDS = {"a": (101, 102), "b": (202,)}


@pytest.fixture
def short_root():
    with tempfile.TemporaryDirectory(prefix="mps-txn-", dir="/tmp") as root:
        yield Path(root)


class NativeMpsBackend:
    """Only the NVIDIA subprocess and /proc boundary are substituted."""

    def __init__(self, root: Path, actor: str):
        self.root = root
        self.actor = actor
        self.daemon_file = root / "daemon"
        self.lock_path = root / f".lock-{GPU_UUID}"
        self.commands: list[str] = []
        self.pause_command: str | None = None
        self.pause_events: dict[str, Event] = {}

    def attach_clients(self):
        for client_pid in CLIENT_PIDS[self.actor]:
            (self.root / f"client-{client_pid}").touch()

    def detach_clients(self):
        for client_pid in CLIENT_PIDS[self.actor]:
            (self.root / f"client-{client_pid}").unlink(missing_ok=True)

    def read_daemon_identity(self, pipe_dir):
        if not self.daemon_file.exists():
            raise MpsControlError("native daemon is absent")
        return int((pipe_dir / "nvidia-cuda-mps-control.pid").read_text())

    def daemon_process_alive(self, daemon_pid):
        return daemon_pid == 4242 and self.daemon_file.exists()

    def client_token(self, client_pid):
        return next(
            f"owner-{actor}"
            for actor, client_pids in CLIENT_PIDS.items()
            if client_pid in client_pids
        )

    def run(self, arguments, **options):
        with self.lock_path.open("r") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = "-d" if len(arguments) == 2 else options["input"].strip()
        self.commands.append(command)
        if (self.root / "record").exists():
            with (self.root / "commands").open("a") as log:
                log.write(f"{self.actor}:{command}\n")
        if command == self.pause_command:
            self.pause_events["paused"].set()
            assert self.pause_events["continue"].wait(10)
            self.pause_command = None
        client_pids = sorted(
            int(path.name.removeprefix("client-"))
            for path in self.root.glob("client-*")
        )
        if command == "-d":
            self.daemon_file.touch()
            pipe_dir = Path(options["env"]["CUDA_MPS_PIPE_DIRECTORY"])
            (pipe_dir / "nvidia-cuda-mps-control.pid").write_text("4242")
            output = ""
        elif command == "get_server_list":
            output = "7000\n" if client_pids else ""
        elif command == "get_client_list 7000":
            output = "\n".join(map(str, client_pids))
        elif command == "get_server_status 7000":
            output = "ACTIVE\n" if client_pids else "Server not found\n"
        elif command.startswith("terminate_client 7000 "):
            (self.root / f"client-{command.split()[-1]}").unlink()
            output = "0\n"
        else:
            assert command == "quit"
            self.daemon_file.unlink()
            output = ""
        return subprocess.CompletedProcess(
            arguments, returncode=0, stdout=output, stderr=""
        )


@contextmanager
def native_client(
    root: Path, actor: str = "a"
) -> Iterator[tuple[control.SubprocessMpsControlClient, NativeMpsBackend]]:
    backend = NativeMpsBackend(root, actor)
    client = control.SubprocessMpsControlClient()
    with (
        patch.object(control.subprocess, "run", backend.run),
        patch.object(client, "read_daemon_identity", backend.read_daemon_identity),
        patch.object(client, "daemon_process_alive", backend.daemon_process_alive),
        patch.object(client, "client_token", backend.client_token),
    ):
        yield client, backend


def serve_process(root: Path, actor: str, operation: str, events: dict[str, Event]):
    original_flock = fcntl.flock
    original_sleep = manager_module.time.sleep

    def observed_flock(descriptor, lock_mode):
        if lock_mode == fcntl.LOCK_EX:
            events["attempted"].set()
        return original_flock(descriptor, lock_mode)

    def drain_sleep(seconds):
        events["paused"].set()
        assert events["continue"].wait(10)
        original_sleep(seconds)

    with (
        patch.object(fcntl, "flock", observed_flock),
        native_client(root, actor) as (
            client,
            backend,
        ),
    ):
        manager = make_manager(root, client)
        manager.poll_interval = 0.02
        manager.drain_timeout = 10
        lease = manager.acquire({actor: f"owner-{actor}"})
        try:
            backend.attach_clients()
            manager.verify(lease)
            events["ready"].set()
            assert events["start"].wait(10)
            backend.pause_events = events
            if operation == "verify":
                backend.pause_command = "get_server_list"
                manager.verify(lease)
            elif operation == "retire":
                backend.pause_command = "terminate_client 7000 101"
                manager.retire_clients_for(lease, actor)
            elif operation == "release":
                backend.detach_clients()
                backend.pause_command = "quit"
                manager.release(lease)
            elif operation == "drain":
                with patch.object(manager_module.time, "sleep", drain_sleep):
                    manager.release(lease)
            else:
                assert operation == "probe"
                assert manager.probe(lease) is None
            events["done"].set()
            assert events["finish"].wait(10)
        finally:
            backend.detach_clients()
            if lease.owner_fd >= 0:
                manager.release(lease)


def make_events(context):
    return {
        name: context.Event()
        for name in (
            "ready",
            "start",
            "attempted",
            "paused",
            "continue",
            "done",
            "finish",
        )
    }


def finish_processes(processes, event_groups):
    for events in event_groups:
        events["start"].set()
        events["continue"].set()
        events["finish"].set()
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)


@pytest.mark.parametrize("operation", ["verify", "retire"])
def test_manager_transaction_cannot_interleave_with_another_serve_probe(
    short_root, operation
):
    context = multiprocessing.get_context("spawn")
    actor_a, actor_b = make_events(context), make_events(context)
    process_a = context.Process(
        target=serve_process, args=(short_root, "a", operation, actor_a)
    )
    process_b = context.Process(
        target=serve_process, args=(short_root, "b", "probe", actor_b)
    )
    process_a.start()
    process_b.start()
    try:
        assert actor_a["ready"].wait(5)
        assert actor_b["ready"].wait(5)
        actor_b["attempted"].clear()
        (short_root / "record").touch()
        actor_a["start"].set()
        assert actor_a["paused"].wait(5)
        actor_b["start"].set()
        assert actor_b["attempted"].wait(5)
        assert not actor_b["done"].wait(0.2)
        actor_a["continue"].set()
        assert actor_a["done"].wait(5)
        assert actor_b["done"].wait(5)
        (short_root / "record").unlink()
        expected = ["a:get_server_list", "a:get_client_list 7000"]
        if operation == "retire":
            expected += ["a:terminate_client 7000 101", "a:terminate_client 7000 102"]
        assert (short_root / "commands").read_text().splitlines() == expected + [
            "b:get_server_status 7000"
        ]
    finally:
        finish_processes([process_a, process_b], [actor_a, actor_b])
    assert process_a.exitcode == process_b.exitcode == 0
    assert not (short_root / GPU_UUID).exists()


def test_drain_keeps_owner_lease_but_allows_a_new_serve_to_join(short_root):
    context = multiprocessing.get_context("spawn")
    actor_a, actor_b = make_events(context), make_events(context)
    process_a = context.Process(
        target=serve_process, args=(short_root, "a", "drain", actor_a)
    )
    process_b = context.Process(
        target=serve_process, args=(short_root, "b", "probe", actor_b)
    )
    process_a.start()
    processes = [process_a]
    try:
        assert actor_a["ready"].wait(5)
        actor_a["start"].set()
        assert actor_a["paused"].wait(5)
        owner_a = short_root / GPU_UUID / "owners" / str(process_a.pid)
        assert control.SubprocessMpsControlClient().owner_lease_held(owner_a)
        process_b.start()
        processes.append(process_b)
        assert actor_b["ready"].wait(5)
        actor_b["start"].set()
        assert actor_b["done"].wait(5)
        for client_pid in CLIENT_PIDS["a"]:
            (short_root / f"client-{client_pid}").unlink()
        actor_a["continue"].set()
        assert actor_a["done"].wait(5)
        assert not owner_a.exists()
        assert (short_root / "daemon").exists()
        owner_b = short_root / GPU_UUID / "owners" / str(process_b.pid)
        assert control.SubprocessMpsControlClient().owner_lease_held(owner_b)
    finally:
        finish_processes(processes, [actor_a, actor_b])
    assert process_a.exitcode == process_b.exitcode == 0
    assert not (short_root / GPU_UUID).exists()


def test_last_owner_quit_finishes_before_a_new_serve_can_acquire(short_root):
    context = multiprocessing.get_context("spawn")
    actor_a, actor_b = make_events(context), make_events(context)
    process_a = context.Process(
        target=serve_process, args=(short_root, "a", "release", actor_a)
    )
    process_b = context.Process(
        target=serve_process, args=(short_root, "b", "probe", actor_b)
    )
    process_a.start()
    processes = [process_a]
    try:
        assert actor_a["ready"].wait(5)
        (short_root / "record").touch()
        actor_a["start"].set()
        assert actor_a["paused"].wait(5)
        process_b.start()
        processes.append(process_b)
        assert actor_b["attempted"].wait(5)
        assert not actor_b["ready"].wait(0.2)
        actor_a["continue"].set()
        assert actor_a["done"].wait(5)
        assert actor_b["ready"].wait(5)
        actor_b["start"].set()
        assert actor_b["done"].wait(5)
        commands = (short_root / "commands").read_text().splitlines()
        assert commands.index("a:quit") < commands.index("b:-d")
        (short_root / "record").unlink()
    finally:
        finish_processes(processes, [actor_a, actor_b])
    assert process_a.exitcode == process_b.exitcode == 0


def test_native_manager_lifecycle_reuses_one_persistent_gpu_lock(short_root):
    with native_client(short_root) as (client, backend):
        manager = make_manager(short_root, client)
        lock_inode = None
        for _ in range(2):
            lease = manager.acquire({"worker": "owner-a"})
            backend.attach_clients()
            assert manager.verify(lease) == {
                MpsClientRef(7000, 101),
                MpsClientRef(7000, 102),
            }
            assert manager.probe(lease) is None
            assert manager.retire_clients_for(lease, "worker") == {
                MpsClientRef(7000, 101),
                MpsClientRef(7000, 102),
            }
            manager.release(lease)
            assert not manager.paths.state_dir.exists()
            if lock_inode is None:
                lock_inode = backend.lock_path.stat().st_ino
            else:
                assert backend.lock_path.stat().st_ino == lock_inode
        assert not list(short_root.glob(".control-lock-*"))


@pytest.mark.parametrize("failure", ["flock", "state_root_mode"])
def test_transaction_failure_reaches_probe_and_releases_owner_on_close(
    short_root, monkeypatch, failure
):
    with native_client(short_root) as (client, backend):
        manager = make_manager(short_root, client)
        lease = manager.acquire({"worker": "owner-a"})
        backend.attach_clients()
        manager.verify(lease)
        original_flock = fcntl.flock

        def cannot_lock(descriptor, lock_mode):
            if lock_mode == fcntl.LOCK_EX:
                raise PermissionError("cannot acquire GPU lock")
            return original_flock(descriptor, lock_mode)

        if failure == "flock":
            monkeypatch.setattr(fcntl, "flock", cannot_lock)
        else:
            short_root.chmod(0o755)
        command_count = len(backend.commands)
        try:
            assert "MPS GPU transaction failed" in manager.probe(lease)
            with pytest.raises(MpsDirtyStateError, match="could not persist"):
                manager.release(lease)
            assert lease.owner_fd == -1
            assert len(backend.commands) == command_count
        finally:
            short_root.chmod(0o700)
            monkeypatch.setattr(fcntl, "flock", original_flock)
            backend.detach_clients()
        assert manager.owner_file.exists()
        assert not client.owner_lease_held(manager.owner_file)


def test_release_failure_persists_dirty_under_gpu_lock(short_root, monkeypatch):
    with native_client(short_root) as (client, backend):
        manager = make_manager(short_root, client)
        lease = manager.acquire({"worker": "owner-a"})
        backend.attach_clients()
        manager.verify(lease)
        error = MpsControlError("snapshot unavailable")
        write_owner_status = manager.write_owner_status

        def fail_snapshot(_):
            raise error

        def write_status_under_lock(owner_fd, status):
            with backend.lock_path.open("r") as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_owner_status(owner_fd, status)

        with monkeypatch.context() as patcher:
            patcher.setattr(manager, "write_owner_status", write_status_under_lock)
            patcher.setattr(client, "snapshot", fail_snapshot)
            with pytest.raises(MpsDirtyStateError) as exc_info:
                manager.release(lease)

        assert exc_info.value.__cause__ is error
        assert lease.owner_fd == -1
        assert manager.owner_file.read_text() == "retained\n"
        assert not client.owner_lease_held(manager.owner_file)
        assert manager.paths.state_dir.exists()
        assert backend.daemon_file.exists()
        with backend.lock_path.open("r") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_native_probe_error_releases_the_gpu_transaction(
    short_root, monkeypatch, failure
):
    with native_client(short_root) as (client, backend):
        manager = make_manager(short_root, client)
        lease = manager.acquire({"worker": "owner-a"})
        backend.attach_clients()
        manager.verify(lease)

        def failing_run(arguments, **options):
            if options.get("input") == "get_server_status 7000\n":
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(arguments, 10)
                return subprocess.CompletedProcess(
                    arguments, returncode=2, stdout="", stderr="failed"
                )
            return backend.run(arguments, **options)

        monkeypatch.setattr(control.subprocess, "run", failing_run)
        try:
            assert "status query failed" in manager.probe(lease)
            with backend.lock_path.open("r") as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            backend.detach_clients()
            manager.release(lease)
