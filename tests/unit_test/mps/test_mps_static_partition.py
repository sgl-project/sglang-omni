# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sglang_omni.mps import control
from sglang_omni.mps.manager import (
    MpsControlError,
    MpsDirtyStateError,
    MpsError,
    MpsGpuPartitions,
    MpsSmPartition,
)
from tests.unit_test.mps.test_mps_manager import (
    FakeControlClient,
    make_manager,
    seed_shared_dir,
)
from tests.unit_test.mps.test_mps_runtime import create, gpu_uuid, proc

PARTITION = "D/79e/DgwYMDAAAAAAAAAAAAAAAAAAAAAAAA"
HEADER = "GPU Partition free used free used clients\nchunks chunks SM SM\n"


def query_client(monkeypatch, responses):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, responses[kwargs["input"].strip()], ""
        )

    monkeypatch.setattr(control.subprocess, "run", run)
    return control.SubprocessMpsControlClient()


def test_partition_id_retains_all_slashes(monkeypatch):
    client = query_client(
        monkeypatch,
        {
            f"sm_partition add {gpu_uuid(0)} 10": f"Partition {gpu_uuid(0)}/{PARTITION} created\n"
        },
    )
    assert client.create_partition(Path("/pipe"), gpu_uuid(0), 10) == PARTITION


def test_add_rejects_failed_creation(monkeypatch):
    output = (
        "Failed to fulfill the requested SM partition of 10 chunks, "
        "error CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION"
    )
    client = query_client(monkeypatch, {f"sm_partition add {gpu_uuid(0)} 10": output})
    with pytest.raises(MpsControlError, match="unexpected output"):
        client.create_partition(Path("/pipe"), gpu_uuid(0), 10)


def test_static_daemon_launch_scopes_gpu_and_passes_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        control.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    control.SubprocessMpsControlClient().start_daemon(
        Path("/pipe"), Path("/log"), gpu_uuid(0), static_partitioning=True
    )
    args, kwargs = calls[0]
    assert args == ["nvidia-cuda-mps-control", "-d", "-S"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == gpu_uuid(0)
    assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(Path("/pipe"))


@pytest.mark.parametrize(
    "flag,enabled", [(b"-S", True), (b"--static-partitioning", True), (b"-d", False)]
)
def test_daemon_mode_reads_native_process_arguments(monkeypatch, flag, enabled):
    client = control.SubprocessMpsControlClient()
    monkeypatch.setattr(client, "read_daemon_identity", lambda pipe_dir: 123)
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: b"nvidia-cuda-mps-control\0" + flag + b"\0"
    )
    assert client.static_partitioning_enabled(Path("/pipe")) is enabled


@pytest.mark.parametrize(
    "rows,used", [("", 0), (f"GPU-aa {PARTITION} 10 80 Yes\n", 10)]
)
def test_lspart_parses_summary_and_optional_partition(monkeypatch, rows, used):
    client = query_client(
        monkeypatch,
        {"lspart": HEADER + f"GPU-aa {15-used} {used} {132-used*8} {used*8}\n" + rows},
    )
    (gpu,) = client.list_partitions(Path("/pipe"))
    assert (gpu.free_chunks, gpu.used_chunks, gpu.free_sm, gpu.used_sm) == (
        15 - used,
        used,
        132 - used * 8,
        used * 8,
    )
    assert gpu.partitions == (
        (MpsSmPartition(PARTITION, 10, 80, True),) if rows else ()
    )


def test_remove_rejects_in_use_partition_even_on_zero_exit(monkeypatch):
    client = query_client(
        monkeypatch,
        {
            f"sm_partition rm {gpu_uuid(0)} {PARTITION}": "Partition %s in use. Terminate all clients before removing.",
            "lspart": HEADER + f"GPU-aa 5 10 52 80\nGPU-aa {PARTITION} 10 80 Yes\n",
        },
    )
    with pytest.raises(MpsControlError, match="in use"):
        client.remove_partition(Path("/pipe"), gpu_uuid(0), PARTITION)


class StaticControlClient(FakeControlClient):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.static = {}
        self.partitions = {}
        self.created = []
        self.removed = []
        self.remove_error = False

    def start_daemon(self, pipe_dir, log_dir, gpu_uuid, *, static_partitioning=False):
        super().start_daemon(pipe_dir, log_dir, gpu_uuid)
        self.static[str(pipe_dir)] = static_partitioning

    def static_partitioning_enabled(self, pipe_dir):
        return self.static.get(str(pipe_dir), False)

    def create_partition(self, pipe_dir, gpu_uuid, chunks):
        assert self.static_partitioning_enabled(pipe_dir)
        pid = f"D/{len(self.created)}/AAAAAAAAAA"
        self.partitions.setdefault(str(pipe_dir), {})[pid] = chunks
        self.created.append((gpu_uuid, chunks, pid))
        return pid

    def list_partitions(self, pipe_dir):
        parts = self.partitions.get(str(pipe_dir), {})
        used = sum(parts.values())
        return (
            MpsGpuPartitions(
                "GPU-aa",
                15 - used,
                used,
                (15 - used) * self.chunk_size + 12,
                used * self.chunk_size,
                tuple(
                    MpsSmPartition(
                        pid,
                        chunks,
                        chunks * self.chunk_size,
                        False,
                    )
                    for pid, chunks in parts.items()
                ),
            ),
        )

    def remove_partition(self, pipe_dir, gpu_uuid, partition_id):
        if self.remove_error:
            raise MpsControlError("partition in use")
        assert not self.snapshots.get(str(pipe_dir))
        del self.partitions[str(pipe_dir)][partition_id]
        self.removed.append(partition_id)


def capped(name, cap, gpu=0, logical=None):
    spec = proc(name, gpu)
    spec.sm_cap = cap
    spec.logical_process_name = logical or name
    return spec


def test_mixed_caps_rejected_on_physical_aliases(short_root):
    with pytest.raises(MpsError, match="every process.*sm_cap"):
        create(
            short_root,
            procs=[capped("a", 80), capped("b", None, 1)],
            physical_ids={0: 0, 1: 0},
        )


def test_runtime_off_cannot_ignore_cap(short_root):
    with pytest.raises(ValueError, match="sm_cap.*requires MPS"):
        create(short_root, mode="off", procs=[capped("a", 80)])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caps,chunk_size,match",
    [
        ([80, 48], 8, "requested 128 SM.*available 120 SM"),
        ([79, 40], 8, "multiples of 8.*8, 16"),
        ([8], 6, "multiples of 6.*6, 12"),
    ],
)
async def test_invalid_budgets_rollback_before_workers(
    short_root, caps, chunk_size, match
):
    client = StaticControlClient(chunk_size)
    runtime = create(
        short_root,
        procs=[capped(str(i), cap) for i, cap in enumerate(caps)],
        client=client,
    )
    with pytest.raises(MpsError, match=match):
        await runtime.start()
    assert not runtime.has_leases
    assert not any(client.partitions.values())
    assert not client.alive_pids


@pytest.mark.asyncio
async def test_replicas_share_partition_and_receive_local_ordinal(short_root):
    client = StaticControlClient()
    runtime = create(
        short_root,
        procs=[capped(f"audio@r{i}", 80, logical="audio") for i in range(4)]
        + [capped("engine", 40)],
        client=client,
    )
    await runtime.start()
    try:
        envs = [runtime.env_for_process(f"audio@r{i}") for i in range(4)]
        assert len({env["CUDA_MPS_SM_PARTITION"] for env in envs}) == 1
        assert envs[0]["CUDA_MPS_SM_PARTITION"] == f"{gpu_uuid(0)}/D/1/AAAAAAAAAA"
        assert all(env["CUDA_VISIBLE_DEVICES"] == "0" for env in envs)
        assert all(env["CUDA_MPS_PIPE_DIRECTORY"] for env in envs)
        assert len({env["SGLANG_OMNI_MPS_CLIENT_TOKEN"] for env in envs}) == 4
        assert [chunks for _, chunks, _ in client.created] == [1, 10, 5]
    finally:
        await runtime.close()
    assert len(client.removed) == 3


@pytest.mark.asyncio
async def test_partition_cleanup_failure_is_reported_and_state_retained(short_root):
    client = StaticControlClient()
    runtime = create(short_root, procs=[capped("a", 80)], client=client)
    await runtime.start()
    client.remove_error = True
    with pytest.raises(MpsDirtyStateError, match="partition in use"):
        await runtime.close()
    assert any(client.partitions.values())
    assert client.alive_pids
    assert not runtime.has_leases


def test_static_request_refuses_running_dynamic_daemon(short_root):
    client = StaticControlClient()
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners={888: True})
    manager = make_manager(short_root, client)
    manager.static_partitioning = True
    with pytest.raises(MpsError, match="static partitioning") as error:
        manager.acquire({"a": "token"}, sm_caps={"a": 80})
    assert "quit" in str(error.value)
    assert paths.state_dir.exists()
    assert client.alive_pids == {999}


@pytest.mark.parametrize(
    "device_kwargs",
    [{"unsupported": {0: "unsupported"}}, {"resolution_errors": {0: "missing UUID"}}],
)
def test_auto_cannot_fall_back_when_sm_cap_is_requested(short_root, device_kwargs):
    with pytest.raises(MpsError, match="sm_cap|physical GPU"):
        create(short_root, procs=[capped("a", 80)], **device_kwargs)


def test_tp_resident_cannot_be_omitted_from_mixed_cap_validation(short_root):
    with pytest.raises(MpsError, match="every process.*sm_cap"):
        create(short_root, procs=[capped("a", 80), proc("tp", 0, tp_size=2)])


def test_uncapped_tp_placement_does_not_require_extra_gpu_inspection(short_root):
    runtime = create(
        short_root, procs=[proc("a", 0), proc("b", 0), proc("tp", 2, tp_size=2)]
    )
    assert runtime is not None
    assert runtime.env_for_process("tp") == {}


@pytest.mark.asyncio
async def test_replicas_on_different_gpus_reserve_once_on_each(short_root):
    client = StaticControlClient()
    runtime = create(
        short_root,
        procs=[capped("audio@r0", 80, 0, "audio"), capped("audio@r1", 80, 1, "audio")],
        client=client,
    )
    await runtime.start()
    try:
        assert runtime.env_for_process("audio@r0")["CUDA_MPS_SM_PARTITION"].startswith(
            gpu_uuid(0) + "/"
        )
        assert runtime.env_for_process("audio@r1")["CUDA_MPS_SM_PARTITION"].startswith(
            gpu_uuid(1) + "/"
        )
        assert [chunks for _, chunks, _ in client.created] == [1, 10, 1, 10]
    finally:
        await runtime.close()


def test_owned_clients_drain_before_partitions_are_removed(short_root):
    client = StaticControlClient()
    manager = make_manager(short_root, client)
    manager.static_partitioning = True
    lease = manager.acquire({"a": "owned"}, sm_caps={"a": 80})
    client.set_clients(manager.paths.pipe_dir, {7000: [101]})
    client.client_tokens[101] = "owned"
    snapshot = client.snapshot
    calls = []

    def detach_after_first_snapshot(pipe_dir):
        value = snapshot(pipe_dir)
        calls.append(value)
        client.set_clients(pipe_dir, {})
        return value

    client.snapshot = detach_after_first_snapshot
    manager.release(lease)
    assert calls[0]
    assert not calls[1]
    assert len(client.removed) == 2


def test_shared_static_daemon_accounts_for_foreign_partitions(short_root):
    client = StaticControlClient()
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners={888: True})
    client.static[str(paths.pipe_dir)] = True
    client.partitions[str(paths.pipe_dir)] = {"foreign/a/b": 6}
    manager = make_manager(short_root, client)
    manager.static_partitioning = True
    with pytest.raises(MpsError, match="requested 80 SM.*available 72 SM"):
        manager.acquire({"a": "owned"}, sm_caps={"a": 80})
    assert client.partitions[str(paths.pipe_dir)] == {"foreign/a/b": 6}
    assert client.alive_pids == {999}


def test_dynamic_request_refuses_running_static_daemon(short_root):
    client = StaticControlClient()
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners={888: True})
    client.static[str(paths.pipe_dir)] = True
    with pytest.raises(MpsError, match="static partitioning mode mismatch"):
        make_manager(short_root, client).acquire({"a": "owned"})


@pytest.mark.parametrize("failed_chunks", [1, 10])
def test_lost_create_response_retains_dirty_owner(short_root, failed_chunks):
    import os

    client = StaticControlClient()
    paths = seed_shared_dir(short_root, client, daemon_pid=999, owners={888: True})
    client.static[str(paths.pipe_dir)] = True
    create_partition = client.create_partition

    def lose_response(pipe_dir, gpu_uuid, chunks):
        partition_id = create_partition(pipe_dir, gpu_uuid, chunks)
        if chunks == failed_chunks:
            raise MpsControlError("lost creation response")
        return partition_id

    client.create_partition = lose_response
    manager = make_manager(short_root, client)
    manager.static_partitioning = True
    with pytest.raises(
        MpsDirtyStateError, match="partition creation outcome is uncertain"
    ):
        manager.acquire({"a": "owned"}, sm_caps={"a": 80})
    assert (paths.owners_dir / str(os.getpid())).read_text() == "retained\n"
    assert client.partitions[str(paths.pipe_dir)]
    assert client.alive_pids == {999}


def test_compiled_caps_reach_all_spawn_specs(short_root):
    from sglang_omni.config.schema import (
        EndpointsConfig,
        PipelineConfig,
        ProcessConfig,
        StageConfig,
    )
    from sglang_omni.pipeline.mp_runner import _build_stage_groups
    from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime

    config = PipelineConfig(
        model_path="dummy",
        mps="auto",
        endpoints=EndpointsConfig(base_path=str(short_root)),
        stages=[
            StageConfig(
                name="audio_stage",
                process="audio",
                factory_path="unused.factory",
                gpu=0,
                gpu_memory_fraction=0.1,
                terminal=True,
            )
        ],
        processes={
            "audio": ProcessConfig(
                num_replicas=4, replica_devices=[0, 0, 0, 0], sm_cap=80
            )
        },
    )
    prep = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
            replica_topology=prep.replica_topology,
            logical_process_plan=prep.logical_process_plan,
        )
        specs = [spec for group in groups for spec in group.process_specs]
        assert [spec.process_name for spec in specs] == [
            f"audio@r{i}" for i in range(4)
        ]
        assert all(
            spec.sm_cap == 80 and spec.logical_process_name == "audio" for spec in specs
        )
    finally:
        prep.runtime_dir.close()


@pytest.mark.parametrize("cap", [None, 56])
def test_worker_gate_runs_only_for_capped_process_before_model(monkeypatch, cap):
    from sglang_omni.mps import numerics
    from sglang_omni.pipeline import stage_workers

    events = []
    monkeypatch.setattr(
        stage_workers, "_prepare_accelerator_environment", lambda *args: None
    )
    monkeypatch.setattr(stage_workers, "apply_gpu_compat_env_defaults", lambda: None)
    monkeypatch.setattr(
        stage_workers, "prepare_weight_share_process_compat", lambda: None
    )
    monkeypatch.setattr(
        numerics,
        "validate_sm_partition",
        lambda sm_cap: events.append(("gate", sm_cap)),
    )
    monkeypatch.setattr(
        stage_workers, "_run_process", lambda *args: events.append(("model", None))
    )
    spec = stage_workers.StageWorkerProcessSpec(
        "audio", [stage_workers.StageLaunchConfig(stage_name="audio")], sm_cap=cap
    )
    stage_workers.stage_process_main(spec, None)
    assert events == (
        [("gate", 56), ("model", None)] if cap is not None else [("model", None)]
    )


def test_numerical_gate_failure_reaches_startup_error_channel(monkeypatch):
    from types import SimpleNamespace

    from sglang_omni.mps import numerics
    from sglang_omni.pipeline import stage_workers

    errors = []
    models = []
    monkeypatch.setattr(
        stage_workers, "_prepare_accelerator_environment", lambda *args: None
    )
    monkeypatch.setattr(stage_workers, "apply_gpu_compat_env_defaults", lambda: None)
    monkeypatch.setattr(
        stage_workers, "prepare_weight_share_process_compat", lambda: None
    )
    monkeypatch.setattr(
        stage_workers, "_destroy_torch_distributed_process_group", lambda *args: None
    )
    monkeypatch.setattr(
        stage_workers, "_reclaim_process_cuda_memory", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        stage_workers, "_run_process", lambda *args: models.append(True)
    )

    def fail(sm_cap):
        raise RuntimeError(f"{sm_cap} SM unsafe; choose a different sm_cap")

    monkeypatch.setattr(numerics, "validate_sm_partition", fail)
    spec = stage_workers.StageWorkerProcessSpec(
        "audio", [stage_workers.StageLaunchConfig(stage_name="audio")], sm_cap=56
    )
    with pytest.raises(SystemExit) as error:
        stage_workers.stage_process_main(spec, None, SimpleNamespace(put=errors.append))
    assert error.value.code == 1
    assert not models
    assert "56 SM unsafe; choose a different sm_cap" in errors[0]
