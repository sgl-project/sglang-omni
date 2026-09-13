# SPDX-License-Identifier: Apache-2.0
"""Serve-local MPS placement, lifecycle, and cancellation contracts."""

from __future__ import annotations

import asyncio
import stat
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from sglang_omni.mps.control import (
    MPS_CLIENT_TOKEN_ENV,
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
    MpsDirtyStateError,
    MpsError,
)
from sglang_omni.mps.decision import MpsDecisionError
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.runtime import create_for_pipeline
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec

_FACTORY = f"{__name__}.unused_factory"


def proc(name, gpu_id, tp_size=1):
    return StageWorkerProcessSpec(
        process_name=name,
        stage_specs=[
            StageLaunchConfig(
                stage_name=name,
                factory=_FACTORY,
                gpu_id=gpu_id,
                placement_gpu_id=gpu_id,
                tp_size=tp_size,
            )
        ],
    )


def gpu_uuid(index: int) -> str:
    return f"GPU-aaaaaaaa-bbbb-cccc-dddd-{index:012d}"


class FakeControlClient:
    """Native I/O fake; lifecycle state remains in the production runtime."""

    def __init__(self):
        self.daemons = {}
        self.alive_pids = set()
        self.snapshots = {}
        self.client_tokens = {}
        self.server_statuses = {}
        self.calls = []

    def start_daemon(self, pipe_dir, log_dir, gpu_uuids):
        pid = 4242 + len(self.daemons)
        self.daemons[str(pipe_dir)] = pid
        self.alive_pids.add(pid)
        (pipe_dir / "nvidia-cuda-mps-control.pid").write_text(str(pid))
        self.calls.append(("start", pipe_dir, gpu_uuids))

    def read_daemon_identity(self, pipe_dir):
        try:
            pid = int((pipe_dir / "nvidia-cuda-mps-control.pid").read_text())
        except (OSError, ValueError) as exc:
            raise MpsControlError("cannot read daemon identity") from exc
        if pid not in self.alive_pids or self.daemons.get(str(pipe_dir)) != pid:
            raise MpsControlError("unverified daemon identity")
        return pid

    def snapshot(self, pipe_dir):
        self.calls.append(("snapshot", pipe_dir))
        return set(self.snapshots.get(str(pipe_dir), set()))

    def set_clients(self, pipe_dir, clients):
        self.snapshots[str(pipe_dir)] = {
            MpsClientRef(server, pid)
            for server, pids in clients.items()
            for pid in pids
        }
        for server in clients:
            self.server_statuses.setdefault((str(pipe_dir), server), "ACTIVE")

    def client_token(self, pid):
        return self.client_tokens.get(pid)

    def get_server_status(self, pipe_dir, server_pid):
        self.calls.append(("status", pipe_dir, server_pid))
        return self.server_statuses.get((str(pipe_dir), server_pid), "Server not found")

    def terminate_client(self, pipe_dir, client):
        self.calls.append(("terminate", pipe_dir, client))
        self.snapshots[str(pipe_dir)].remove(client)

    def quit_daemon(self, pipe_dir):
        self.calls.append(("quit", pipe_dir))
        self.alive_pids.discard(self.daemons[str(pipe_dir)])

    def daemon_process_alive(self, pid):
        return pid in self.alive_pids


class FakeDeviceInfo:
    def __init__(
        self,
        unsupported: dict[int, str] | None = None,
        physical_ids: dict[int, int] | None = None,
        resolution_errors: dict[int, str] | None = None,
    ):
        self.unsupported = unsupported or {}
        self.physical_ids = physical_ids or {}
        self.resolution_errors = resolution_errors or {}

    def inspect(self, gpu_ids):
        return {
            gpu_id: (
                MpsPhysicalDevice(None, self.resolution_errors[gpu_id])
                if gpu_id in self.resolution_errors
                else MpsPhysicalDevice(
                    gpu_uuid(self.physical_ids.get(gpu_id, gpu_id)),
                    self.unsupported.get(gpu_id),
                )
            )
            for gpu_id in gpu_ids
        }


@pytest.fixture
def short_root():
    with tempfile.TemporaryDirectory(prefix="mpsr-", dir="/tmp") as root:
        yield Path(root)


def colocated():
    return [proc("a", 0), proc("b", 0), proc("solo", 1)]


def create(
    short_root,
    mode="auto",
    procs=None,
    unsupported=None,
    physical_ids=None,
    resolution_errors=None,
    client=None,
    state_root=None,
):
    with (
        patch("sglang_omni.platforms.current_platform.is_cuda", return_value=True),
        patch("sglang_omni.mps.runtime.shutil.which", return_value="/fake/mps-control"),
    ):
        runtime, devices = create_for_pipeline(
            mode=mode,
            process_specs=procs if procs is not None else colocated(),
            device_info=FakeDeviceInfo(unsupported, physical_ids, resolution_errors),
            client=client or FakeControlClient(),
            state_root=short_root if state_root is None else state_root,
        )
    if runtime is not None:
        runtime.poll_interval = 0
        runtime.start_timeout = 0.02
        runtime.verify_timeout = 0.02
        runtime.stop_timeout = 0.02
    return runtime, devices


@pytest.mark.parametrize("mode,gpu_ids", [("off", [0, 0]), ("auto", [0, 1])])
def test_disabled_mps_creates_nothing(short_root, mode, gpu_ids):
    processes = [proc(name, gpu_id) for name, gpu_id in zip(("a", "b"), gpu_ids)]
    assert create(short_root, mode=mode, procs=processes) == (None, {})
    assert not list(short_root.iterdir())


@pytest.mark.parametrize("mode", ["auto", "on"])
@pytest.mark.parametrize(
    "extra_processes,device_options,unresolved_ordinals",
    [
        pytest.param([], {}, [], id="multiple-physical-gpus"),
        pytest.param(
            [],
            {"unsupported": {1: "NVML capability query failed"}},
            [],
            id="nvml-failure",
        ),
        pytest.param(
            [proc("multi", 9)],
            {"resolution_errors": {9: "CUDA_ERROR_INVALID_DEVICE"}},
            [9],
            id="same-process-resolution-failure",
        ),
        pytest.param(
            [proc("broken", 9)],
            {"resolution_errors": {9: "CUDA_ERROR_INVALID_DEVICE"}},
            [],
            id="unrelated-process-resolution-failure",
        ),
    ],
)
def test_multi_physical_process_rejected_before_device_errors(
    short_root, mode, extra_processes, device_options, unresolved_ordinals
):
    client = FakeControlClient()
    with pytest.raises(MpsError) as exc_info:
        create(
            short_root,
            mode=mode,
            procs=[proc("multi", 0), proc("multi", 1), *extra_processes],
            client=client,
            **device_options,
        )

    message = str(exc_info.value)
    assert "process 'multi'" in message
    assert f"0: '{gpu_uuid(0)}'" in message
    assert f"1: '{gpu_uuid(1)}'" in message
    if unresolved_ordinals:
        assert f"unresolved CUDA ordinals: {unresolved_ordinals}" in message
    assert "Use mps=off" in message
    assert list(short_root.iterdir()) == []
    assert client.daemons == {}


@pytest.mark.parametrize("source", ["factory_kwargs", "typed_kwargs"])
@pytest.mark.parametrize("mode", ["auto", "on"])
def test_cuda_zero_uses_the_narrowed_worker_namespace(short_root, mode, source):
    processes = [proc("a", 1), proc("b", 1)]
    setattr(processes[0].stage_specs[0], source, {"device": "cuda:0"})

    runtime, devices = create(short_root, mode=mode, procs=processes)

    assert sorted(set(devices.values())) == [gpu_uuid(1)]
    assert devices["a"] == gpu_uuid(1)
    assert devices["b"] == gpu_uuid(1)


def test_nonzero_cuda_device_is_rejected_before_mps_acquisition(short_root):
    client = FakeControlClient()
    processes = [proc("a", 1)]
    processes[0].stage_specs[0].typed_kwargs = {"device": "cuda:1"}

    with pytest.raises((MpsError, MpsDecisionError)) as exc_info:
        create(
            short_root,
            mode="on",
            procs=processes,
            client=client,
        )

    message = str(exc_info.value)
    assert "process 'a'" in message
    assert "explicit CUDA ordinal(s) [1]" in message
    assert "cuda:0" in message
    assert "mps=off" in message
    assert list(short_root.iterdir()) == []
    assert client.daemons == {}


def test_pipeline_edge_to_another_gpu_does_not_change_mps_process_planning(
    short_root,
):
    processes = [proc("a", 0), proc("b", 0), proc("remote", 1)]
    source = processes[0].stage_specs[0]
    source.next_stages = "remote"
    source.stage_gpu_ids = {"remote": (1,)}

    runtime, devices = create(short_root, procs=processes)

    assert sorted(set(devices.values())) == [gpu_uuid(0)]


def test_tp_ranks_do_not_block_an_eligible_group_on_another_gpu(short_root):
    processes = [
        proc("thinker_tp0", 0, tp_size=2),
        proc("thinker_tp1", 1, tp_size=2),
        proc("a", 2),
        proc("b", 2),
    ]

    runtime, devices = create(short_root, procs=processes)

    assert sorted(set(devices.values())) == [gpu_uuid(2)]
    assert runtime.env_for_process("thinker_tp0") == {}
    assert runtime.env_for_process("thinker_tp1") == {}


def test_unsupported_gpu_under_auto_downgrades_to_off(short_root):
    assert create(short_root, unsupported={0: "MIG enabled"}) == (None, {})


def test_unsupported_gpu_under_on_raises(short_root):
    with pytest.raises(MpsError, match="MIG"):
        create(short_root, mode="on", unsupported={0: "MIG enabled"})


def test_native_mps_rejects_cuda_alike_non_nvidia_platform(monkeypatch):
    class NonNvidiaPlatform:
        @staticmethod
        def is_cuda() -> bool:
            return False

        @staticmethod
        def is_cuda_alike() -> bool:  # pragma: no cover - must not be consulted
            raise AssertionError("NVIDIA MPS must not use is_cuda_alike()")

    platforms = ModuleType("sglang_omni.platforms")
    platforms.current_platform = NonNvidiaPlatform()
    monkeypatch.setitem(sys.modules, "sglang_omni.platforms", platforms)

    assert create_for_pipeline("auto", []) == (None, {})
    with pytest.raises(MpsError, match="requires an NVIDIA CUDA platform"):
        create_for_pipeline("on", [])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CUDA_MPS_PIPE_DIRECTORY", "/parent/mps"),
        ("SGLANG_OMNI_WEIGHT_SHARE", "invalid-but-enabled"),
    ],
)
def test_parent_mps_conflict_is_reported_before_state_creation(
    short_root,
    monkeypatch,
    name,
    value,
):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("SGLANG_OMNI_MPS_STATE_ROOT", str(short_root))

    with pytest.raises(MpsError) as exc_info:
        create_for_pipeline("on", colocated())

    message = str(exc_info.value)
    assert "parent" in message
    assert f"{name}={value!r}" in message
    assert "mps=off" in message
    assert list(short_root.iterdir()) == []


@pytest.mark.asyncio
async def test_one_private_daemon_for_multiple_gpus_and_workers(short_root):
    client = FakeControlClient()
    runtime, devices = create(
        short_root, mode="on", client=client, procs=(spec for spec in colocated())
    )
    assert not list(short_root.iterdir())
    await runtime.start(devices.values())
    run_dir = runtime.run_dir
    assert run_dir.name.startswith("run-")
    assert devices == {"a": gpu_uuid(0), "b": gpu_uuid(0), "solo": gpu_uuid(1)}
    assert client.calls[0] == ("start", runtime.pipe_dir, (gpu_uuid(0), gpu_uuid(1)))
    assert len(client.daemons) == 1
    assert all(
        stat.S_IMODE(p.stat().st_mode) == 0o700
        for p in [run_dir, runtime.pipe_dir, runtime.log_dir]
    )
    assert runtime.server_pid is None
    for name in devices:
        assert runtime.env_for_process(name)["CUDA_MPS_PIPE_DIRECTORY"] == str(
            runtime.pipe_dir
        )
    assert runtime.env_for_process("cpu") == {}
    client.set_clients(runtime.pipe_dir, {7000: [101, 102, 103]})
    client.client_tokens.update(
        {
            pid: runtime.env_for_process(name)[MPS_CLIENT_TOKEN_ENV]
            for pid, name in zip([101, 102, 103], devices)
        }
    )
    await runtime.verify()
    assert runtime.server_pid == 7000
    client.set_clients(runtime.pipe_dir, {})
    await runtime.close()
    assert not runtime.has_resources
    assert not run_dir.exists()
    assert not client.alive_pids


@pytest.mark.asyncio
async def test_two_serves_never_join_or_clean_each_others_run(short_root):
    client = FakeControlClient()
    a, devices = create(short_root, client=client)
    b, _ = create(short_root, client=client)
    stale = short_root / "run-stale"
    stale.mkdir()
    (stale / "evidence").write_text("old run")
    await a.start(devices.values())
    await b.start(devices.values())
    assert a.run_dir != b.run_dir
    assert a.daemon_pid != b.daemon_pid
    b_dir = b.run_dir
    await a.close()
    assert b_dir.is_dir()
    assert b.daemon_pid in client.alive_pids
    assert (stale / "evidence").read_text() == "old run"
    await b.close()
    assert list(short_root.iterdir()) == [stale]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clients,tokens,match",
    [
        ({}, {}, "never attached"),
        ({7000: [101, 102]}, {101: "a", 102: "a"}, "'b'"),
        ({7000: [101], 8000: [102]}, {101: "a", 102: "b"}, "share one server"),
        ({7000: [101, 102]}, {101: "a"}, "'b'"),
    ],
)
async def test_verify_requires_all_worker_tokens_on_one_server(
    short_root, clients, tokens, match
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    client.set_clients(runtime.pipe_dir, clients)
    client.client_tokens.update(
        {
            pid: runtime.env_for_process(name)[MPS_CLIENT_TOKEN_ENV]
            for pid, name in tokens.items()
        }
    )
    with pytest.raises(MpsError, match=match):
        await runtime.verify()
    assert runtime.server_pid is None
    client.set_clients(runtime.pipe_dir, {})
    await runtime.close()


@pytest.mark.asyncio
async def test_verify_accepts_cuda_descendants_but_does_not_accumulate_snapshots(
    short_root, monkeypatch
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    client.client_tokens.update(
        {
            101: runtime.env_for_process("a")[MPS_CLIENT_TOKEN_ENV],
            102: runtime.env_for_process("b")[MPS_CLIENT_TOKEN_ENV],
        }
    )
    snapshots = iter([{MpsClientRef(7000, 101)}, {MpsClientRef(7000, 102)}])
    with monkeypatch.context() as patch:
        patch.setattr(
            client, "snapshot", lambda _: next(snapshots, {MpsClientRef(7000, 102)})
        )
        with pytest.raises(MpsError, match="'a'"):
            await runtime.verify()
    client.set_clients(runtime.pipe_dir, {7000: [101, 102, 103]})
    client.client_tokens[103] = runtime.env_for_process("a")[MPS_CLIENT_TOKEN_ENV]
    await runtime.verify()
    assert runtime.server_pid == 7000
    await runtime.retire_process_clients("a")
    assert client.snapshot(runtime.pipe_dir) == {MpsClientRef(7000, 102)}
    client.set_clients(runtime.pipe_dir, {})
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["client_token", "terminate_client"])
async def test_retirement_continues_after_one_client_fails(
    short_root, monkeypatch, failure
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    client.set_clients(runtime.pipe_dir, {7000: [101, 102, 103]})
    for pid, name in [(101, "a"), (102, "a"), (103, "b")]:
        client.client_tokens[pid] = runtime.env_for_process(name)[MPS_CLIENT_TOKEN_ENV]
    original = getattr(client, failure)

    def fail_first(*args):
        pid = args[0] if failure == "client_token" else args[1].client_pid
        if pid == 101:
            raise MpsControlError("client unavailable")
        return original(*args)

    monkeypatch.setattr(client, failure, fail_first)
    await runtime.retire_process_clients("a")
    assert client.snapshot(runtime.pipe_dir) == {
        MpsClientRef(7000, 101),
        MpsClientRef(7000, 103),
    }
    await runtime.close()
    assert not runtime.has_resources
    assert not client.alive_pids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["ACTIVE", "FAULT", "", "Server not found", MpsControlError("query timed out")],
)
async def test_watchdog_only_queries_verified_server_even_after_clients_exit(
    short_root, monkeypatch, status
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    assert "not verified" in await runtime.probe()
    client.set_clients(runtime.pipe_dir, {7000: [101, 102]})
    for pid, name in [(101, "a"), (102, "b")]:
        client.client_tokens[pid] = runtime.env_for_process(name)[MPS_CLIENT_TOKEN_ENV]
    await runtime.verify()
    client.set_clients(runtime.pipe_dir, {8000: [101, 102]})
    if isinstance(status, MpsControlError):

        def fail(pipe_dir, server_pid):
            client.calls.append(("status", pipe_dir, server_pid))
            raise status

        monkeypatch.setattr(client, "get_server_status", fail)
    else:
        client.server_statuses[(str(runtime.pipe_dir), 7000)] = status
    client.calls.clear()
    reason = await runtime.probe()
    assert (reason is None) == (status == "ACTIVE")
    if isinstance(status, MpsControlError):
        assert "query timed out" in reason
    assert client.calls == [("status", runtime.pipe_dir, 7000)]
    assert runtime.server_pid == 7000
    client.set_clients(runtime.pipe_dir, {})
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("executed", [False, True])
async def test_start_failure_distinguishes_unexecuted_and_ambiguous_daemon(
    short_root, monkeypatch, executed
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)

    def fail(*args):
        error = MpsControlError if executed else MpsDaemonNotStartedError
        raise error("start failed")

    monkeypatch.setattr(client, "start_daemon", fail)
    with pytest.raises(MpsControlError, match="start failed") as exc:
        await runtime.start(devices.values())
    assert runtime.has_resources == executed
    assert bool(list(short_root.iterdir())) == executed
    if executed:
        assert isinstance(exc.value.__cause__, MpsDirtyStateError)


@pytest.mark.asyncio
async def test_failed_start_with_verified_daemon_uses_normal_cleanup(
    short_root, monkeypatch
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    start = client.start_daemon

    def fail(*args):
        start(*args)
        raise MpsControlError("lost start response")

    monkeypatch.setattr(client, "start_daemon", fail)
    with pytest.raises(MpsControlError, match="lost start response"):
        await runtime.start(devices.values())
    assert not runtime.has_resources
    assert not client.alive_pids


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["identity", "quit", "alive"])
async def test_cleanup_preserves_directory_when_resource_release_is_uncertain(
    short_root, monkeypatch, failure
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    run_dir = runtime.run_dir

    def fail(*args):
        raise MpsControlError("unavailable")

    if failure == "quit":
        monkeypatch.setattr(client, "quit_daemon", fail)
    elif failure == "identity":
        client.daemons[str(runtime.pipe_dir)] = 999
        client.alive_pids.add(999)
        (runtime.pipe_dir / "nvidia-cuda-mps-control.pid").write_text("999")
    else:
        monkeypatch.setattr(client, "quit_daemon", lambda _: None)
    with pytest.raises(MpsDirtyStateError, match="preserved"):
        await runtime.close()
    assert run_dir.is_dir()
    assert runtime.has_resources
    if failure == "identity":
        assert not any(call[0] == "quit" for call in client.calls)


@pytest.mark.asyncio
async def test_lost_quit_response_is_allowed_only_after_verified_daemon_exit(
    short_root, monkeypatch
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    quit_daemon = client.quit_daemon

    def fail(pipe):
        quit_daemon(pipe)
        raise MpsControlError("lost quit response")

    monkeypatch.setattr(client, "quit_daemon", fail)
    await runtime.close()
    assert not runtime.has_resources


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["start", "close"])
async def test_cancellation_waits_for_native_operation_before_releasing_lock(
    short_root, monkeypatch, operation
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    entered, release = threading.Event(), threading.Event()
    method = "start_daemon" if operation == "start" else "quit_daemon"
    original = getattr(client, method)
    if operation != "start":
        await runtime.start(devices.values())

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(client, method, blocked)
        call = (
            runtime.start(devices.values()) if operation == "start" else runtime.close()
        )
        task = asyncio.create_task(call)
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert runtime._operation_lock.locked()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not runtime.has_resources
    assert not client.alive_pids


@pytest.mark.asyncio
async def test_cancelled_probe_finishes_before_concurrent_close(
    short_root, monkeypatch
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    client.set_clients(runtime.pipe_dir, {7000: [101, 102]})
    for pid, name in [(101, "a"), (102, "b")]:
        client.client_tokens[pid] = runtime.env_for_process(name)[MPS_CLIENT_TOKEN_ENV]
    await runtime.verify()
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return "ACTIVE"

    monkeypatch.setattr(client, "get_server_status", blocked)
    client.set_clients(runtime.pipe_dir, {})
    cleanup_entered = threading.Event()
    quit_daemon = client.quit_daemon

    def record_cleanup(pipe_dir):
        cleanup_entered.set()
        return quit_daemon(pipe_dir)

    monkeypatch.setattr(client, "quit_daemon", record_cleanup)
    client.calls.clear()
    probe = asyncio.create_task(runtime.probe())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        probe.cancel()
        close = asyncio.create_task(runtime.close())
        assert not await asyncio.to_thread(cleanup_entered.wait, 0.1)
        assert not close.done()
        assert not client.calls
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await probe
    await close
    assert not runtime.has_resources


@pytest.mark.asyncio
async def test_private_root_permissions_and_socket_limit(short_root):
    root = short_root / "new"
    runtime, devices = create(short_root, state_root=root)
    await runtime.start(devices.values())
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    await runtime.close()
    root.chmod(0o755)
    with pytest.raises(ValueError, match="expected 0o700"):
        await runtime.start(devices.values())
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    root.chmod(0o700)
    link = short_root / "link"
    link.symlink_to(root, target_is_directory=True)
    runtime, devices = create(short_root, state_root=link)
    with pytest.raises(ValueError, match="symlink"):
        await runtime.start(devices.values())
    long_root = short_root / ("x" * 100)
    runtime, devices = create(short_root, state_root=long_root)
    with pytest.raises(ValueError, match="sun_path"):
        await runtime.start(devices.values())
    assert list(long_root.iterdir()) == []


@pytest.mark.asyncio
async def test_state_root_owned_by_another_uid_is_rejected(short_root, monkeypatch):
    import sglang_omni.mps.state as state

    runtime, devices = create(short_root)
    uid = state.os.getuid()
    monkeypatch.setattr(state.os, "getuid", lambda: uid + 1)
    with pytest.raises(ValueError, match="not current uid"):
        await runtime.start(devices.values())
    assert not list(short_root.iterdir())
    assert not runtime.has_resources


@pytest.mark.asyncio
async def test_repeated_close_cancellation_preserves_native_failure(
    short_root, monkeypatch
):
    client = FakeControlClient()
    runtime, devices = create(short_root, client=client)
    await runtime.start(devices.values())
    entered, release = threading.Event(), threading.Event()

    def blocked_quit(pipe_dir):
        entered.set()
        assert release.wait(5)
        raise MpsControlError("quit response unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(client, "quit_daemon", blocked_quit)
        closing = asyncio.create_task(runtime.close())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError) as exc:
            await closing
    assert isinstance(exc.value.__cause__, MpsDirtyStateError)
    assert "quit response unavailable" in str(exc.value.__cause__)
    assert runtime.run_dir.is_dir()
    await runtime.close()
