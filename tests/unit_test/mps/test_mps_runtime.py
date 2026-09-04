# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level MPS acquisition, routing, and rollback tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from sglang_omni.mps.decision import MpsDecisionError
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.manager import (
    MPS_CLIENT_TOKEN_ENV,
    MpsClientRef,
    MpsDirtyStateError,
    MpsError,
)
from sglang_omni.mps.runtime import MpsPipelineRuntime, create_for_pipeline
from tests.unit_test.mps.test_mps_manager import FakeControlClient

_FACTORY = f"{__name__}.unused_factory"


@dataclass
class ResolvedStageLaunch:
    """Minimal resolved launch record consumed by MPS planning."""

    stage_name: str
    gpu_id: int | None
    tp_size: int = 1
    placement_gpu_id: int | None = None
    factory: str = _FACTORY
    factory_kwargs: dict = field(default_factory=dict)
    typed_kwargs: dict = field(default_factory=dict)
    factory_arg_defaults: dict = field(default_factory=dict)
    env_defaults: dict = field(default_factory=dict)
    next_stages: str | list[str] | None = None
    stage_gpu_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class ResolvedProcessSpec:
    process_name: str
    stage_specs: list[ResolvedStageLaunch] = field(default_factory=list)


def proc(name, gpu_id, tp_size=1):
    return ResolvedProcessSpec(
        process_name=name,
        stage_specs=[
            ResolvedStageLaunch(
                stage_name=name,
                gpu_id=gpu_id,
                placement_gpu_id=gpu_id,
                tp_size=tp_size,
            )
        ],
    )


def gpu_uuid(index: int) -> str:
    return f"GPU-aaaaaaaa-bbbb-cccc-dddd-{index:012d}"


def manager_on(runtime: MpsPipelineRuntime, physical_index: int):
    return runtime.managers[gpu_uuid(physical_index)]


def owner_marker(manager) -> Path:
    return manager.paths.owners_dir / str(os.getpid())


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
    root = Path(tempfile.mkdtemp(prefix="mpsr-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


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
    process_specs = procs if procs is not None else colocated()
    runtime = MpsPipelineRuntime.create(
        mode=mode,
        process_specs=process_specs,
        device_info=FakeDeviceInfo(
            unsupported,
            physical_ids,
            resolution_errors,
        ),
        client=client or FakeControlClient(),
        state_root=short_root if state_root is None else state_root,
    )
    if runtime is not None:
        for manager in runtime.managers.values():
            manager.poll_interval = 0.0
            manager.drain_timeout = 0.02
            manager.stop_timeout = 0.02
    return runtime


def detach_all(runtime, client):
    for manager in runtime.managers.values():
        client.set_clients(manager.paths.pipe_dir, {})


def test_off_does_not_inspect_external_process_pipe(short_root):
    processes = colocated()
    processes[0].stage_specs[0].env_defaults = {
        "CUDA_MPS_PIPE_DIRECTORY": "/external/mps"
    }

    assert create(short_root, mode="off", procs=processes) is None


def test_auto_without_colocation_creates_nothing(short_root):
    assert create(short_root, procs=[proc("a", 0), proc("b", 1)]) is None


@pytest.mark.asyncio
async def test_env_only_for_acquired_client_processes(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    await runtime.start()

    env = runtime.env_for_process("a")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000000"
    assert "CUDA_MPS_PIPE_DIRECTORY" in env
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"
    assert env[MPS_CLIENT_TOKEN_ENV]
    assert runtime.env_for_process("solo") == {}

    detach_all(runtime, client)
    await runtime.close()


@pytest.mark.asyncio
async def test_parent_visible_ordinal_selects_the_physical_uuid(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        client=client,
        physical_ids={0: 1},
    )

    await runtime.start()

    assert runtime.env_for_process("a")["CUDA_VISIBLE_DEVICES"] == (
        "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000001"
    )
    assert runtime.env_for_process("b")["CUDA_VISIBLE_DEVICES"] == (
        "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000001"
    )

    detach_all(runtime, client)
    await runtime.close()


@pytest.mark.asyncio
async def test_logical_gpu_aliases_coalesce_by_physical_uuid(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        mode="auto",
        procs=[proc("a", 0), proc("b", 1)],
        client=client,
        physical_ids={0: 1, 1: 1},
    )

    assert runtime is not None
    assert list(runtime.managers) == [gpu_uuid(1)]

    await runtime.start()

    assert runtime.env_for_process("a")["CUDA_VISIBLE_DEVICES"] == gpu_uuid(1)
    assert runtime.env_for_process("b")["CUDA_VISIBLE_DEVICES"] == gpu_uuid(1)
    manager = manager_on(runtime, 1)
    daemon_pid = client.daemons[str(manager.paths.pipe_dir)]
    assert set(client.daemons) == {str(manager.paths.pipe_dir)}
    client.set_clients(manager.paths.pipe_dir, {7000: [11, 12]})
    client.client_tokens.update(
        {
            11: runtime.env_for_process("a")[MPS_CLIENT_TOKEN_ENV],
            12: runtime.env_for_process("b")[MPS_CLIENT_TOKEN_ENV],
        }
    )
    await runtime.verify()
    client.set_clients(manager.paths.pipe_dir, {7000: [11]})
    assert await runtime.probe_failures() == {}

    client.set_clients(manager.paths.pipe_dir, {})
    await runtime.close()

    assert not runtime.has_leases
    assert not client.daemon_process_alive(daemon_pid)
    assert not manager.paths.state_dir.exists()


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_one_process_cannot_resolve_to_multiple_physical_gpus(
    short_root,
    mode,
):
    client = FakeControlClient()
    with pytest.raises(MpsError) as exc_info:
        create(
            short_root,
            mode=mode,
            procs=[proc("duplicate", 0), proc("duplicate", 1)],
            client=client,
        )

    message = str(exc_info.value)
    assert "process 'duplicate'" in message
    assert f"0: '{gpu_uuid(0)}'" in message
    assert f"1: '{gpu_uuid(1)}'" in message
    assert "Use mps=off" in message
    assert list(short_root.iterdir()) == []
    assert client.daemons == {}


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_nvml_failure_does_not_hide_driver_proven_multi_physical_process(
    short_root,
    mode,
):
    client = FakeControlClient()

    with pytest.raises(MpsError) as exc_info:
        create(
            short_root,
            mode=mode,
            procs=[proc("multi", 0), proc("multi", 1)],
            unsupported={1: "NVML capability query failed"},
            client=client,
        )

    message = str(exc_info.value)
    assert "process 'multi'" in message
    assert f"0: '{gpu_uuid(0)}'" in message
    assert f"1: '{gpu_uuid(1)}'" in message
    assert "Use mps=off" in message
    assert list(short_root.iterdir()) == []
    assert client.daemons == {}


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_known_multi_physical_subset_precedes_driver_resolution_error(
    short_root,
    mode,
):
    client = FakeControlClient()

    with pytest.raises(MpsError) as exc_info:
        create(
            short_root,
            mode=mode,
            procs=[proc("multi", 0), proc("multi", 1), proc("multi", 9)],
            resolution_errors={9: "CUDA_ERROR_INVALID_DEVICE"},
            client=client,
        )

    message = str(exc_info.value)
    assert "process 'multi'" in message
    assert f"0: '{gpu_uuid(0)}'" in message
    assert f"1: '{gpu_uuid(1)}'" in message
    assert "unresolved CUDA ordinals: [9]" in message
    assert "Use mps=off" in message
    assert list(short_root.iterdir()) == []
    assert client.daemons == {}


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_unrelated_resolution_error_does_not_hide_multi_physical_process(
    short_root,
    mode,
):
    client = FakeControlClient()

    with pytest.raises(MpsError) as exc_info:
        create(
            short_root,
            mode=mode,
            procs=[proc("multi", 0), proc("multi", 1), proc("broken", 9)],
            resolution_errors={9: "CUDA_ERROR_INVALID_DEVICE"},
            client=client,
        )

    message = str(exc_info.value)
    assert "process 'multi'" in message
    assert f"0: '{gpu_uuid(0)}'" in message
    assert f"1: '{gpu_uuid(1)}'" in message
    assert "Use mps=off" in message
    assert list(short_root.iterdir()) == []
    assert client.daemons == {}


@pytest.mark.parametrize("source", ["factory_kwargs", "typed_kwargs"])
@pytest.mark.parametrize("mode", ["auto", "on"])
def test_cuda_zero_uses_the_narrowed_worker_namespace(short_root, mode, source):
    processes = [proc("a", 1), proc("b", 1)]
    setattr(processes[0].stage_specs[0], source, {"device": "cuda:0"})

    runtime = create(short_root, mode=mode, procs=processes)

    assert list(runtime.managers) == [gpu_uuid(1)]
    assert runtime.env_for_process("a")["CUDA_VISIBLE_DEVICES"] == gpu_uuid(1)
    assert runtime.env_for_process("b")["CUDA_VISIBLE_DEVICES"] == gpu_uuid(1)


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

    runtime = create(short_root, procs=processes)

    assert list(runtime.managers) == [gpu_uuid(0)]


def test_tp_ranks_do_not_block_an_eligible_group_on_another_gpu(short_root):
    processes = [
        ResolvedProcessSpec(
            "thinker_tp0",
            [
                ResolvedStageLaunch(
                    stage_name="thinker",
                    gpu_id=0,
                    placement_gpu_id=0,
                    tp_size=2,
                )
            ],
        ),
        ResolvedProcessSpec(
            "thinker_tp1",
            [
                ResolvedStageLaunch(
                    stage_name="thinker",
                    gpu_id=1,
                    placement_gpu_id=1,
                    tp_size=2,
                )
            ],
        ),
        proc("a", 2),
        proc("b", 2),
    ]

    runtime = create(short_root, procs=processes)

    assert list(runtime.managers) == [gpu_uuid(2)]
    assert runtime.env_for_process("thinker_tp0") == {}
    assert runtime.env_for_process("thinker_tp1") == {}


@pytest.mark.asyncio
async def test_cancelled_start_rolls_back_before_any_client_can_attach(
    short_root,
    monkeypatch,
):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    manager = manager_on(runtime, 0)
    original_acquire = manager.acquire
    entered = threading.Event()
    release = threading.Event()

    def blocked_acquire(client_tokens):
        entered.set()
        assert release.wait(timeout=5)
        return original_acquire(client_tokens)

    monkeypatch.setattr(manager, "acquire", blocked_acquire)
    start_task = asyncio.create_task(runtime.start())
    assert await asyncio.to_thread(entered.wait, 1)
    start_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert not runtime.has_leases
    assert not manager.paths.state_dir.exists()


@pytest.mark.asyncio
async def test_new_state_root_is_created_private(short_root):
    client = FakeControlClient()
    state_root = short_root / "new-state"
    runtime = create(
        short_root,
        client=client,
        state_root=state_root,
    )

    await runtime.start()

    assert stat.S_IMODE(state_root.stat().st_mode) == 0o700
    detach_all(runtime, client)
    await runtime.close()


@pytest.mark.asyncio
async def test_existing_nonprivate_state_root_is_rejected_without_chmod(
    short_root,
):
    state_root = short_root / "shared"
    state_root.mkdir(mode=0o755)
    state_root.chmod(0o755)
    runtime = create(short_root, state_root=state_root)

    with pytest.raises(MpsError, match="expected 0o700"):
        await runtime.start()

    assert stat.S_IMODE(state_root.stat().st_mode) == 0o755
    assert list(state_root.iterdir()) == []


@pytest.mark.asyncio
async def test_symlink_state_root_is_rejected_without_mutating_target(short_root):
    target = short_root / "target"
    target.mkdir(mode=0o700)
    state_root = short_root / "state-link"
    state_root.symlink_to(target, target_is_directory=True)
    runtime = create(short_root, state_root=state_root)

    with pytest.raises(MpsError, match="must not be a symlink"):
        await runtime.start()

    assert state_root.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert list(target.iterdir()) == []


@pytest.mark.asyncio
async def test_state_root_owned_by_another_uid_is_rejected(
    short_root,
    monkeypatch,
):
    import sglang_omni.mps.state as mps_state

    runtime = create(short_root)
    actual_uid = os.getuid()
    monkeypatch.setattr(mps_state.os, "getuid", lambda: actual_uid + 1)

    with pytest.raises(MpsError, match="not current uid"):
        await runtime.start()

    assert list(short_root.iterdir()) == []


def test_unsupported_gpu_under_auto_downgrades_to_off(short_root):
    assert create(short_root, unsupported={0: "MIG enabled"}) is None


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

    assert create_for_pipeline("auto", []) is None
    with pytest.raises(MpsError, match="requires an NVIDIA CUDA platform"):
        create_for_pipeline("on", [])


@pytest.mark.asyncio
async def test_close_releases_all_acquired_leases(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    await runtime.start()
    manager = manager_on(runtime, 0)
    client.set_clients(manager.paths.pipe_dir, {7000: [11, 12]})
    client.client_tokens.update(
        {
            11: runtime.env_for_process("a")[MPS_CLIENT_TOKEN_ENV],
            12: runtime.env_for_process("b")[MPS_CLIENT_TOKEN_ENV],
        }
    )
    await runtime.verify()
    client.set_clients(manager.paths.pipe_dir, {})

    await runtime.close()

    assert not runtime.has_leases
    assert not manager.paths.state_dir.exists()


def test_process_pipe_dir_is_rejected_before_state_creation(short_root):
    processes = colocated()
    processes[0].stage_specs[0].env_defaults = {
        "CUDA_MPS_PIPE_DIRECTORY": "/external/mps"
    }

    with pytest.raises(MpsError) as exc_info:
        create(short_root, procs=processes)

    message = str(exc_info.value)
    assert "process 'a'" in message
    assert "CUDA_MPS_PIPE_DIRECTORY='/external/mps'" in message
    assert "mps=off" in message
    assert list(short_root.iterdir()) == []


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
async def test_multi_gpu_start_rolls_back_only_successful_acquisitions(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        mode="on",
        procs=[proc("a", 0), proc("b", 1)],
        client=client,
    )
    assert list(runtime.managers) == [gpu_uuid(0), gpu_uuid(1)]
    dirty = manager_on(runtime, 1).paths
    dirty.pipe_dir.mkdir(parents=True)
    dirty.log_dir.mkdir()
    dirty.owners_dir.mkdir()
    (dirty.owners_dir / "777").write_text("")

    with pytest.raises(MpsError, match="dirty state"):
        await runtime.start()

    assert not runtime.has_leases
    assert not manager_on(runtime, 0).paths.state_dir.exists()
    assert (dirty.owners_dir / "777").exists()


@pytest.mark.asyncio
async def test_multi_gpu_pre_spawn_rollback_leaves_shared_owner_clean(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        mode="on",
        procs=[proc("a", 0), proc("b", 1)],
        client=client,
    )
    shared = manager_on(runtime, 0)
    shared.paths.pipe_dir.mkdir(parents=True)
    shared.paths.log_dir.mkdir()
    shared.paths.owners_dir.mkdir()
    (shared.paths.pipe_dir / "nvidia-cuda-mps-control.pid").write_text("9000")
    (shared.paths.owners_dir / "888").write_text("active\n")
    client.daemons[str(shared.paths.pipe_dir)] = 9000
    client.alive_pids.add(9000)
    client.held_owner_pids.add(888)
    foreign = MpsClientRef(7000, 101)
    client.set_clients(shared.paths.pipe_dir, {7000: [101]})

    dirty = manager_on(runtime, 1).paths
    dirty.pipe_dir.mkdir(parents=True)
    dirty.log_dir.mkdir()
    dirty.owners_dir.mkdir()
    (dirty.owners_dir / "777").write_text("retained\n")

    with pytest.raises(MpsError, match="dirty state"):
        await runtime.start()

    assert not runtime.has_leases
    assert not owner_marker(shared).exists()
    assert (shared.paths.owners_dir / "888").read_text() == "active\n"
    assert client.snapshot(shared.paths.pipe_dir) == {foreign}
    assert client.daemon_process_alive(9000)


@pytest.mark.asyncio
async def test_multi_gpu_close_persists_dirty_gpu_and_releases_clean_gpu(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        procs=[proc("a", 0), proc("b", 0), proc("c", 1), proc("d", 1)],
        client=client,
    )
    await runtime.start()
    detach_all(runtime, client)
    dirty_manager = manager_on(runtime, 1)
    clean_manager = manager_on(runtime, 0)
    client.set_clients(dirty_manager.paths.pipe_dir, {7000: [30]})
    client.client_tokens[30] = runtime.env_for_process("c")[MPS_CLIENT_TOKEN_ENV]

    with pytest.raises(MpsDirtyStateError, match="owned="):
        await runtime.close()

    assert not runtime.has_leases
    assert dirty_manager.paths.state_dir.is_dir()
    assert owner_marker(dirty_manager).read_text() == "retained\n"
    assert not clean_manager.paths.state_dir.exists()
    assert client.unsafe_daemon_signals == []


@pytest.mark.asyncio
async def test_start_attempts_are_classified_per_physical_gpu(short_root):
    client = FakeControlClient()
    runtime = create(
        short_root,
        mode="on",
        procs=[proc("attempted", 0), proc("not-started", 1)],
        client=client,
    )
    attempted = manager_on(runtime, 0)
    not_started = manager_on(runtime, 1)
    foreign_clients = {}

    for index, manager in enumerate((attempted, not_started)):
        paths = manager.paths
        paths.pipe_dir.mkdir(parents=True)
        paths.log_dir.mkdir()
        paths.owners_dir.mkdir()
        daemon_pid = 9000 + index
        owner_pid = 8000 + index
        (paths.pipe_dir / "nvidia-cuda-mps-control.pid").write_text(str(daemon_pid))
        (paths.owners_dir / str(owner_pid)).write_text("active\n")
        client.daemons[str(paths.pipe_dir)] = daemon_pid
        client.alive_pids.add(daemon_pid)
        client.held_owner_pids.add(owner_pid)
        client.set_clients(paths.pipe_dir, {7000 + index: [200 + index]})
        client.client_tokens[200 + index] = f"foreign-owner-{index}"
        foreign_clients[manager.gpu_uuid] = client.snapshot(paths.pipe_dir)

    await runtime.start()

    with pytest.raises(MpsDirtyStateError, match="ownership is incomplete"):
        await runtime.close(process_start_attempts={"attempted"})

    current_owner = str(os.getpid())
    assert (attempted.paths.owners_dir / current_owner).read_text() == ("retained\n")
    assert not (not_started.paths.owners_dir / current_owner).exists()
    for index, manager in enumerate((attempted, not_started)):
        assert (manager.paths.owners_dir / str(8000 + index)).read_text() == (
            "active\n"
        )
        assert (
            client.snapshot(manager.paths.pipe_dir) == foreign_clients[manager.gpu_uuid]
        )
        assert client.daemon_process_alive(9000 + index)

    later_lease = not_started.acquire({"later": "later-owner"})
    not_started.release(later_lease, clients_could_have_attached=False)
    assert not (not_started.paths.owners_dir / current_owner).exists()
    with pytest.raises(MpsError, match="retained"):
        attempted.acquire({"later": "later-owner"})


@pytest.mark.asyncio
async def test_preverify_clients_are_preserved_without_guessing_ownership(short_root):
    client = FakeControlClient()
    runtime = create(short_root, client=client)
    await runtime.start()
    manager = manager_on(runtime, 0)
    client.set_clients(manager.paths.pipe_dir, {7000: [200], 8000: [909]})
    client.client_tokens.update(
        {
            200: runtime.env_for_process("a")[MPS_CLIENT_TOKEN_ENV],
            909: "foreign-owner",
        }
    )

    with pytest.raises(MpsDirtyStateError) as exc_info:
        await runtime.close()

    message = str(exc_info.value)
    assert not runtime.has_leases
    assert "terminate_client 7000 200" in message
    assert "terminate_client 8000 909" not in message
    assert client.unsafe_daemon_signals == []
    assert owner_marker(manager).read_text() == "retained\n"
