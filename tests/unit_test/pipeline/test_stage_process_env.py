# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    patched_spawn_env,
)
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from sglang_omni.utils.gpu_memory import get_gpu_startup_lock_path
from tests.unit_test.fixtures import affinity_probe
from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler, fake_factory_path

cuda_platform = CUDAOmniPlatform()


@pytest.fixture(autouse=True)
def force_cuda_device(monkeypatch):
    """These tests assert the CUDA TP env/device contract (CUDA_VISIBLE_DEVICES,
    torch.cuda.set_device). Pin the device layer to CUDA so they exercise that
    path regardless of the test host's real accelerator; otherwise an XPU host
    takes the ZE_AFFINITY_MASK / all-cards-visible branch and the assertions fail.
    """
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )


def tp_spec(*, gpu_id: int) -> StageLaunchConfig:
    return StageLaunchConfig(
        stage_name="thinker",
        role="leader",
        tp_rank=0,
        tp_size=2,
        gpu_id=gpu_id,
    )


def worker_spec(*stage_specs: StageLaunchConfig) -> StageWorkerProcessSpec:
    return StageWorkerProcessSpec(
        process_name="worker",
        stage_specs=list(stage_specs),
    )


@pytest.mark.parametrize("parent_threads", [None, "4"])
def test_spawn_env_applies_cpu_plan_and_respects_parent(
    monkeypatch: pytest.MonkeyPatch, parent_threads: str | None
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_OMP_FROM_CPU_PLAN", raising=False)
    if parent_threads is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", parent_threads)
    else:
        pass
    spec = worker_spec(StageLaunchConfig(stage_name="preprocess"))
    spec.cpu_threads = 9

    with patched_spawn_env(spec):
        assert os.environ["OMP_NUM_THREADS"] == (parent_threads or "9")
        assert os.environ.get("SGLANG_OMNI_OMP_FROM_CPU_PLAN") == (
            "1" if parent_threads is None else None
        )

    assert os.environ.get("OMP_NUM_THREADS") == parent_threads
    assert "SGLANG_OMNI_OMP_FROM_CPU_PLAN" not in os.environ


@pytest.mark.parametrize(
    "env_defaults,extra_env,expected_threads",
    [
        ({"OMP_NUM_THREADS": "1"}, {}, "1"),
        ({"OMP_NUM_THREADS": "1"}, {"OMP_NUM_THREADS": "2"}, "2"),
    ],
)
def test_spawn_env_cpu_plan_preserves_configured_omp(
    monkeypatch: pytest.MonkeyPatch,
    env_defaults: dict[str, str],
    extra_env: dict[str, str],
    expected_threads: str,
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_OMP_FROM_CPU_PLAN", raising=False)
    spec = worker_spec(
        StageLaunchConfig(stage_name="preprocess", env_defaults=env_defaults)
    )
    spec.cpu_threads = 37

    with patched_spawn_env(spec, extra_env=extra_env):
        assert os.environ["OMP_NUM_THREADS"] == expected_threads
        assert "SGLANG_OMNI_OMP_FROM_CPU_PLAN" not in os.environ

    assert "OMP_NUM_THREADS" not in os.environ
    assert "SGLANG_OMNI_OMP_FROM_CPU_PLAN" not in os.environ


class StartupClock:
    """A fake monotonic clock that a rank script advances one step per poll."""

    def __init__(self, steps: Callable[[float], None]) -> None:
        self.now = 0.0
        self.steps = steps

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float | None) -> bool:
        self.now += seconds or 0.0
        self.steps(self.now)
        return False


class ScriptedReady:
    def __init__(self, clock: StartupClock) -> None:
        self.clock = clock
        self.ready = False

    def is_set(self) -> bool:
        return self.ready

    def wait(self, timeout: float | None = None) -> bool:
        return self.clock.advance(timeout)


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("slow_start", None, None),
        ("stalled_sibling", TimeoutError, "rank1 made no startup progress"),
        ("dead_after_ready", RuntimeError, "rank1 died during startup"),
    ],
)
def test_wait_ready_watches_every_rank_on_its_own_progress_deadline(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error: type[Exception] | None,
    message: str | None,
) -> None:
    alive = [True, True]
    counters = [SimpleNamespace(value=0), SimpleNamespace(value=0)]

    def steps(now: float) -> None:
        # Rank 0 makes progress every second and is ready at 30s, three
        # timeouts in. Rank 1 depends on the case.
        if now < 30:
            counters[0].value += 1
        else:
            ready[0].ready = True
        if case == "slow_start":
            counters[1].value += 1
            ready[1].ready = now >= 25
        elif case == "dead_after_ready":
            ready[1].ready = True
            alive[1] = now < 3
        else:
            pass

    clock = StartupClock(steps)
    ready = [ScriptedReady(clock), ScriptedReady(clock)]
    monkeypatch.setattr(
        stage_workers, "time", SimpleNamespace(monotonic=clock.monotonic)
    )
    group = stage_workers.StageGroup(
        "ranks",
        [
            StageWorkerProcessSpec(process_name="rank0", stage_specs=[]),
            StageWorkerProcessSpec(process_name="rank1", stage_specs=[]),
        ],
    )
    group._processes = [  # noqa: leading-underscore  # production name
        SimpleNamespace(is_alive=lambda i=i: alive[i], exitcode=None) for i in range(2)
    ]
    group.ready_events = ready
    group.startup_error_channels = [queue.Queue(), queue.Queue()]
    group.startup_progress = counters

    if error is None:
        asyncio.run(group.wait_ready(10))
        assert clock.now >= 30
    else:
        with pytest.raises(error, match=message):
            asyncio.run(group.wait_ready(10))
        assert clock.now < 30
        assert not ready[0].ready


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="needs control over the CPU mask of two or more CPUs",
)
def test_spawned_process_threads_start_on_the_planned_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_cpus = os.sched_getaffinity(0)
    spec = worker_spec(StageLaunchConfig(stage_name="preprocess"))
    spec.cpu_affinity = frozenset({min(launcher_cpus)})
    monkeypatch.setattr(
        stage_workers, "stage_process_main", affinity_probe.report_thread_cpus
    )
    monkeypatch.setenv(affinity_probe.SPAWNED_ENV, "1")
    group = stage_workers.StageGroup("probe", [spec])

    group.spawn(multiprocessing.get_context("spawn"))
    import_thread_cpus, thread_cpus = group.startup_error_channels[0].get(timeout=120)
    group.processes[0].join(timeout=30)

    assert os.sched_getaffinity(0) == launcher_cpus
    assert import_thread_cpus == sorted(spec.cpu_affinity)
    assert all(cpus == sorted(spec.cpu_affinity) for cpus in thread_cpus)


def test_tp_process_env_maps_logical_gpu_through_visible_devices() -> None:
    env = cuda_platform.get_stage_process_env(
        tp_spec(gpu_id=1), {"CUDA_VISIBLE_DEVICES": "3,4"}
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


def test_tp_process_env_turns_nccl_nvls_off() -> None:
    env = cuda_platform.get_stage_process_env(tp_spec(gpu_id=0), {})

    assert env["NCCL_NVLS_ENABLE"] == "0"


def test_tp_process_env_leaves_an_operator_nvls_value_alone() -> None:
    env = cuda_platform.get_stage_process_env(
        tp_spec(gpu_id=0), {"NCCL_NVLS_ENABLE": "1"}
    )

    assert "NCCL_NVLS_ENABLE" not in env


def test_spawn_env_maps_the_planned_gpu_even_with_a_configured_visibility(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(stage_workers, "current_platform", cuda_platform)
    stage_spec = tp_spec(gpu_id=1)
    stage_spec.env_defaults = {"CUDA_VISIBLE_DEVICES": "2,3"}

    with patched_spawn_env(worker_spec(stage_spec)):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_spawn_env_keeps_a_configured_stage_nvls_value(monkeypatch) -> None:
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    monkeypatch.setattr(stage_workers, "current_platform", cuda_platform)
    stage_spec = tp_spec(gpu_id=1)
    stage_spec.env_defaults = {"NCCL_NVLS_ENABLE": "1"}

    with patched_spawn_env(worker_spec(stage_spec)):
        assert os.environ["NCCL_NVLS_ENABLE"] == "1"

    assert "NCCL_NVLS_ENABLE" not in os.environ


def test_non_tp_stage_gets_no_cuda_process_env() -> None:
    spec = StageLaunchConfig(stage_name="thinker", tp_size=1, gpu_id=0)

    assert cuda_platform.get_stage_process_env(spec, {}) == {}


def test_tp_process_env_rejects_single_visible_device_for_second_gpu() -> None:
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES only exposes"):
        cuda_platform.get_stage_process_env(
            tp_spec(gpu_id=1), {"CUDA_VISIBLE_DEVICES": "0"}
        )


def test_tp_process_env_requires_gpu_id() -> None:
    with pytest.raises(ValueError, match="requires a GPU id"):
        cuda_platform.get_stage_process_env(
            StageLaunchConfig(stage_name="thinker", tp_size=2), {}
        )


def test_xpu_tp_process_env_emits_no_visibility_variable() -> None:
    """XPU TP must keep every card visible: ZE_AFFINITY_MASK isolation hides peers
    and hangs XCCL discovery, unlike CUDA_VISIBLE_DEVICES with NCCL. With nothing
    inherited the hook narrows nothing and emits no mask at all -- an empty mask
    would hide every card."""
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    env = XPUOmniPlatform().get_stage_process_env(tp_spec(gpu_id=1), {})

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env == {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"}


def test_xpu_tp_preserves_a_mask_that_covers_the_whole_group() -> None:
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    env = XPUOmniPlatform().get_stage_process_env(
        tp_spec(gpu_id=1), {"ZE_AFFINITY_MASK": "4,5"}
    )

    assert "ZE_AFFINITY_MASK" not in env
    assert env == {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"}


def test_xpu_tp_rejects_a_mask_too_small_for_the_group() -> None:
    """A single-card mask cannot host a 2-rank stage. Dropping it would silently
    move the stage to physical 0..1, so fail loudly instead."""
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    with pytest.raises(ValueError, match="exposes 1"):
        XPUOmniPlatform().get_stage_process_env(
            tp_spec(gpu_id=1), {"ZE_AFFINITY_MASK": "3"}
        )


def test_xpu_tp_rejects_a_gpu_id_outside_the_mask() -> None:
    """gpu_id indexes into the mask's numbering, not the host's."""
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    with pytest.raises(ValueError, match="exposes only 2 cards"):
        XPUOmniPlatform().get_stage_process_env(
            tp_spec(gpu_id=2), {"ZE_AFFINITY_MASK": "4,5"}
        )


def test_xpu_tp_process_env_requires_gpu_id() -> None:
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    with pytest.raises(ValueError, match="requires a GPU id"):
        XPUOmniPlatform().get_stage_process_env(
            StageLaunchConfig(stage_name="thinker", tp_size=2), {}
        )


def test_xpu_tp_rank_keeps_its_card_despite_an_inherited_cuda_marker(
    monkeypatch,
) -> None:
    """SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS is a CUDA placement marker: only
    CUDAOmniPlatform sets it, and pinned SGLang honors it solely under
    is_cuda_alike(). An XPU child that inherited it must not take the fast path --
    that normalized every rank to gpu_id=0, so all ranks would bind xpu:0 and XPU's
    own visibility policy would never run.
    """
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    monkeypatch.setattr(stage_workers, "current_platform", XPUOmniPlatform())
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    spec = StageLaunchConfig(
        stage_name="thinker",
        role="follower",
        tp_rank=1,
        tp_size=2,
        gpu_id=1,
        factory_arg_defaults={"gpu_id": 1},
        comm_config={"gpu_id": 1},
    )

    stage_workers.prepare_accelerator_environment(spec, RecordingLog())

    assert spec.gpu_id == 1
    assert spec.factory_arg_defaults["gpu_id"] == 1
    assert spec.comm_config["gpu_id"] == 1
    assert spec.placement_gpu_id is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_tp_child_keeps_parent_mapped_visible_device(monkeypatch) -> None:
    """Child startup normalizes the already-mapped TP device to local cuda:0."""
    monkeypatch.setattr(stage_workers, "current_platform", cuda_platform)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    spec = StageLaunchConfig(
        stage_name="thinker",
        role="follower",
        tp_rank=1,
        tp_size=2,
        gpu_id=1,
        factory_kwargs={"gpu_id": 1},
        typed_kwargs={"gpu_id": 1},
        factory_arg_defaults={"gpu_id": 1},
        comm_config={"gpu_id": 1},
    )

    stage_workers.prepare_accelerator_environment(spec, RecordingLog())

    assert spec.gpu_id == 0
    assert spec.placement_gpu_id == 1
    assert spec.typed_kwargs["gpu_id"] == 0
    assert spec.factory_arg_defaults["gpu_id"] == 0
    assert spec.comm_config["gpu_id"] == 0
    # The pipeline author's own channel is not placement owned, so narrowing
    # the device must leave it alone.
    assert spec.factory_kwargs["gpu_id"] == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"


def test_rocm_tp_child_keeps_parent_mapped_hip_visible_device(monkeypatch) -> None:
    """ROCm TP children normalize the single HIP-visible card to local cuda:0."""
    monkeypatch.setattr(stage_workers, "current_platform", ROCMOmniPlatform())
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    spec = StageLaunchConfig(
        stage_name="thinker",
        role="follower",
        tp_rank=1,
        tp_size=2,
        gpu_id=1,
        factory_arg_defaults={"gpu_id": 1},
        comm_config={"gpu_id": 1},
    )
    log = RecordingLog()

    stage_workers.prepare_accelerator_environment(spec, log)

    assert spec.gpu_id == 0
    assert spec.placement_gpu_id == 1
    assert spec.factory_arg_defaults["gpu_id"] == 0
    assert spec.comm_config["gpu_id"] == 0
    assert os.environ["HIP_VISIBLE_DEVICES"] == "5"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert any("CUDA_VISIBLE_DEVICES=5" in message for message in log.messages)
    assert get_gpu_startup_lock_path(spec.gpu_id).name.endswith("_gpu_5_startup.lock")


def test_spawn_env_applies_stage_defaults_before_child_start(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_TEST_STAGE_ENV", raising=False)
    spec = StageLaunchConfig(
        stage_name="thinker",
        env_defaults={"SGLANG_TEST_STAGE_ENV": "default"},
    )

    with patched_spawn_env(worker_spec(spec)):
        assert os.environ["SGLANG_TEST_STAGE_ENV"] == "default"

    assert "SGLANG_TEST_STAGE_ENV" not in os.environ


def test_spawn_env_preserves_operator_stage_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_TEST_STAGE_ENV", "operator")
    spec = StageLaunchConfig(
        stage_name="thinker",
        env_defaults={"SGLANG_TEST_STAGE_ENV": "default"},
    )

    with patched_spawn_env(worker_spec(spec)):
        assert os.environ["SGLANG_TEST_STAGE_ENV"] == "operator"

    assert os.environ["SGLANG_TEST_STAGE_ENV"] == "operator"


def test_spawn_env_combines_stage_defaults_with_tp_visible_device(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_TEST_STAGE_ENV", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    monkeypatch.setattr(stage_workers, "current_platform", cuda_platform)
    stage_spec = tp_spec(gpu_id=1)
    stage_spec.env_defaults = {"SGLANG_TEST_STAGE_ENV": "default"}

    with patched_spawn_env(worker_spec(stage_spec)):
        assert os.environ["SGLANG_TEST_STAGE_ENV"] == "default"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
        assert os.environ["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"

    assert "SGLANG_TEST_STAGE_ENV" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,4"


def test_rocm_spawn_env_maps_rank_through_hip_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,5")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(stage_workers, "current_platform", ROCMOmniPlatform())

    with patched_spawn_env(worker_spec(tp_spec(gpu_id=1))):
        assert os.environ["HIP_VISIBLE_DEVICES"] == "5"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        assert os.environ["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"

    assert os.environ["HIP_VISIBLE_DEVICES"] == "3,5"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("platform", [cuda_platform, ROCMOmniPlatform()])
def test_native_child_owner_sees_only_its_gpus(monkeypatch, platform) -> None:
    """A stage that owns native children starts narrowed to the GPUs its
    engine owns and hands that engine local ids in their declared order."""
    variable = (
        "HIP_VISIBLE_DEVICES"
        if isinstance(platform, ROCMOmniPlatform)
        else "CUDA_VISIBLE_DEVICES"
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(variable, "4,5,6,7")
    monkeypatch.setattr(stage_workers, "current_platform", platform)
    spec = StageLaunchConfig(
        stage_name="generation",
        gpu_id=1,
        allow_child_processes=True,
        factory_kwargs={"runtime_gpu_ids": [1, 3]},
        factory_arg_defaults={"gpu_id": 1},
        comm_config={"gpu_id": 1},
    )

    with patched_spawn_env(worker_spec(spec)):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,7"
        assert os.environ[variable] == "5,7"
        # The child starts with this environment.
        stage_workers.prepare_accelerator_environment(spec, RecordingLog())

    assert (spec.gpu_id, spec.placement_gpu_id) == (0, 1)
    assert spec.factory_kwargs["runtime_gpu_ids"] == [0, 1]
    assert spec.factory_arg_defaults["gpu_id"] == spec.comm_config["gpu_id"] == 0
    assert os.environ[variable] == "4,5,6,7"


class RecordingLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        if args:
            message = message % args
        self.messages.append(message)


def test_scheduler_applies_child_defaults_without_overriding_explicit_args(
    monkeypatch,
) -> None:
    seen_gpu_ids: list[int] = []

    @contextmanager
    def fake_lock(gpu_id: int):
        seen_gpu_ids.append(gpu_id)
        yield Path("/tmp/test.lock")

    monkeypatch.setattr(stage_workers, "gpu_startup_lock", fake_lock)
    spec = StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path("runtime_factory"),
        factory_kwargs={
            "model_path": "runtime-model",
            "thinker_max_seq_len": 128,
        },
        factory_arg_defaults={
            "model_path": "global-model",
            "gpu_id": 3,
            "total_gpu_memory_fraction": 0.25,
        },
    )

    result = stage_workers.construct_scheduler(spec, 3, RecordingLog())

    assert result["model_path"] == "runtime-model"
    assert result["gpu_id"] == 3
    assert result["thinker_max_seq_len"] == 128
    assert result["total_gpu_memory_fraction"] == 0.25
    assert seen_gpu_ids == [3]


def test_scheduler_rejects_replica_device_factory_without_gpu_id() -> None:
    spec = StageLaunchConfig(
        stage_name="legacy@r0",
        factory=fake_factory_path("runtime_factory_with_device"),
        factory_kwargs={"device": "cuda:0"},
        factory_arg_defaults={"model_path": "model", "gpu_id": 1},
        require_factory_gpu_id=True,
    )

    with pytest.raises(
        ValueError,
        match="legacy@r0.*replica_devices.*does not declare a gpu_id parameter",
    ):
        stage_workers.construct_scheduler(spec, 1, RecordingLog())


def test_construct_stage_uses_placement_gpu_id_for_device_and_startup_lock(
    monkeypatch,
) -> None:
    """Placement-owned gpu_id must drive device setup and startup lock."""

    class FakeStage:
        def __init__(self, **kwargs):
            self.scheduler = kwargs["scheduler"]

    set_device_calls: list[int] = []
    seen_gpu_ids: list[int] = []

    @contextmanager
    def fake_lock(gpu_id: int):
        seen_gpu_ids.append(gpu_id)
        yield Path("/tmp/test.lock")

    monkeypatch.setattr(
        torch.get_device_module(platforms.current_platform.device_type),
        "set_device",
        lambda gpu_id: set_device_calls.append(int(gpu_id)),
    )
    monkeypatch.setattr(stage_workers, "gpu_startup_lock", fake_lock)
    monkeypatch.setattr(stage_workers, "Stage", FakeStage)
    monkeypatch.setattr(stage_workers, "current_platform", cuda_platform)

    specs = [
        StageLaunchConfig(
            stage_name=f"gpu_stage_{idx}",
            factory=fake_factory_path("make_scheduler_accepting_gpu_id"),
            factory_arg_defaults={"gpu_id": idx},
            gpu_id=idx,
            placement_gpu_id=idx + 2,
        )
        for idx in range(2)
    ]

    stages = [stage_workers.construct_stage(spec, RecordingLog()) for spec in specs]

    assert [stage.scheduler.gpu_id for stage in stages] == [0, 1]
    assert set_device_calls == [0, 1]
    assert seen_gpu_ids == [0, 1]


def test_construct_stage_skips_device_pinning_for_child_process_engines(
    monkeypatch,
) -> None:
    """A host that only owns a native engine's subprocesses pins no device."""

    class _FakeStage:
        def __init__(self, **kwargs):
            self.scheduler = kwargs["scheduler"]
            self.allow_child_processes = kwargs["allow_child_processes"]

    set_device_calls: list[int] = []

    @contextmanager
    def _fake_lock(gpu_id: int):
        yield Path("/tmp/test.lock")

    monkeypatch.setattr(
        torch.get_device_module(platforms.current_platform.device_type),
        "set_device",
        lambda gpu_id: set_device_calls.append(int(gpu_id)),
    )
    monkeypatch.setattr(stage_workers, "gpu_startup_lock", _fake_lock)
    monkeypatch.setattr(stage_workers, "Stage", _FakeStage)
    monkeypatch.setattr(stage_workers, "current_platform", cuda_platform)

    spec = StageLaunchConfig(
        stage_name="native_stage",
        factory=fake_factory_path("make_scheduler_accepting_gpu_id"),
        factory_arg_defaults={"gpu_id": 1},
        gpu_id=1,
        allow_child_processes=True,
    )

    stage = stage_workers.construct_stage(spec, RecordingLog())

    assert stage.scheduler.gpu_id == 1
    assert stage.allow_child_processes is True
    assert set_device_calls == []


def test_narrowed_tp_child_locks_its_local_device(monkeypatch) -> None:
    """A TP child narrowed to one card locks through its local gpu_id.

    The lock path resolves the physical card via CUDA_VISIBLE_DEVICES, so the
    pre-narrowing placement id must not be used: it would index past the
    single visible device.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    seen_gpu_ids: list[int] = []

    @contextmanager
    def fake_lock(gpu_id: int):
        seen_gpu_ids.append(gpu_id)
        yield get_gpu_startup_lock_path(gpu_id)

    monkeypatch.setattr(stage_workers, "gpu_startup_lock", fake_lock)
    spec = StageLaunchConfig(
        stage_name="thinker",
        role="follower",
        tp_rank=1,
        tp_size=2,
        placement_gpu_id=1,
        gpu_id=0,
        factory=fake_factory_path("make_scheduler_accepting_gpu_id"),
        factory_arg_defaults={"gpu_id": 0},
    )

    scheduler = stage_workers.construct_scheduler(spec, 0, RecordingLog())

    assert scheduler.gpu_id == 0
    assert seen_gpu_ids == [0]
    assert get_gpu_startup_lock_path(0).name.endswith("_gpu_4_startup.lock")


def test_cpu_scheduler_construction_skips_startup_lock(monkeypatch) -> None:
    def unexpected_lock(gpu_id: int):
        raise AssertionError(f"unexpected GPU lock for {gpu_id}")

    monkeypatch.setattr(stage_workers, "gpu_startup_lock", unexpected_lock)
    spec = StageLaunchConfig(
        stage_name="decode",
        factory=fake_factory_path("make_scheduler"),
    )

    scheduler = stage_workers.construct_scheduler(spec, None, RecordingLog())

    assert isinstance(scheduler, FakeScheduler)


@pytest.mark.parametrize(
    ("platform_type", "expected"),
    [
        (platforms.CPUOmniPlatform, []),
        (
            platforms.XPUOmniPlatform,
            ["set_device:xpu:1", "synchronize", "empty_cache"],
        ),
        (
            CUDAOmniPlatform,
            ["set_device:cuda:1", "synchronize", "empty_cache", "ipc_collect"],
        ),
    ],
)
def test_stage_teardown_reclaims_through_the_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform_type: type[platforms.OmniPlatform],
    expected: list[str],
) -> None:
    """A stage that dies on a non-CUDA accelerator still has to give its memory back,
    which the torch.cuda.is_available guard used to skip entirely."""
    calls: list[str] = []
    platform = platform_type()
    monkeypatch.setattr(stage_workers, "current_platform", platform)
    monkeypatch.setattr(
        platform_type,
        "set_device",
        lambda self, device: calls.append(f"set_device:{device}"),
    )
    monkeypatch.setattr(
        platform_type, "synchronize", lambda self: calls.append("synchronize")
    )
    monkeypatch.setattr(
        platform_type, "empty_cache", lambda self: calls.append("empty_cache")
    )
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: calls.append("ipc_collect"))

    stage_workers.reclaim_process_gpu_memory(
        [1], logging.getLogger(__name__), reason="test"
    )

    assert calls == expected


STAGE_PROCESS_WITH_LIVE_PARENT = """
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from sglang_omni.pipeline import stage_workers

def block_after_registration() -> None:
    Path(marker_dir, "registered").touch()
    Event().wait()

stage_workers.configure_dependency_loggers = block_after_registration
stage_workers.stage_process_main(
    SimpleNamespace(log_level=20, stage_specs=[], process_name="p"), None
)
"""


def process_has_exited(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return True
    else:
        return stat_path.read_text().rsplit(") ", 1)[1].split()[0] == "Z"


def test_stage_process_asks_to_die_with_its_parent(tmp_path: Path) -> None:
    spawn_and_wait = (
        "import multiprocessing, sys, time\n"
        "child = multiprocessing.get_context('spawn').Process(target=exec, args=("
        "sys.argv[1], {'marker_dir': sys.argv[2]}))\n"
        "child.start()\n"
        "open(sys.argv[2] + '/child.pid', 'w').write(str(child.pid))\n"
        "time.sleep(600)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", spawn_and_wait, STAGE_PROCESS_WITH_LIVE_PARENT]
        + [str(tmp_path)]
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 300
        while not (tmp_path / "registered").exists():
            assert parent.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        child_pid = int((tmp_path / "child.pid").read_text())

        parent.kill()
        parent.wait(timeout=30)

        deadline = time.monotonic() + 30
        while not process_has_exited(child_pid):
            assert time.monotonic() < deadline, "stage process outlived its parent"
            time.sleep(0.05)
    finally:
        parent.kill()
        if child_pid is not None and not process_has_exited(child_pid):
            os.kill(child_pid, signal.SIGKILL)
        else:
            pass


# note (Richard Wang): the worker spawned below runs this once its parent is
# gone, so the parent dies before the worker registers its death signal. The
# worker inherits the output pipes, so the run returns once it exits.
STAGE_PROCESS_AFTER_PARENT_DIED = """
import multiprocessing, time
from pathlib import Path
from types import SimpleNamespace
from sglang_omni.pipeline import stage_workers
parent = multiprocessing.parent_process()
while parent.is_alive():
    time.sleep(0.01)
try:
    stage_workers.stage_process_main(
        SimpleNamespace(log_level=20, stage_specs=[], process_name="p"), None
    )
except ValueError:
    Path(marker_dir, "built").touch()
finally:
    Path(marker_dir, "exited").touch()
"""


def test_stage_process_exits_when_its_parent_died_before_it_started(
    tmp_path: Path,
) -> None:
    spawn_and_die = (
        "import multiprocessing, os, sys\n"
        "multiprocessing.get_context('spawn').Process(target=exec, args=("
        "sys.argv[1], {'marker_dir': sys.argv[2]})).start()\n"
        "os._exit(0)\n"
    )
    command = [sys.executable, "-c", spawn_and_die, STAGE_PROCESS_AFTER_PARENT_DIED]

    subprocess.run([*command, str(tmp_path)], capture_output=True, timeout=300)

    assert (tmp_path / "exited").exists()
    assert not (tmp_path / "built").exists()


# note (Richard Wang): run_process stands in for a stage that stopped or was
# interrupted. The object in the script's globals reports interpreter teardown.
STAGE_PROCESS_THAT_STOPS = """
import atexit, sys, threading, time
from types import SimpleNamespace
from sglang_omni.pipeline import stage_workers

marker, interrupted = sys.argv[1], sys.argv[2] == "1"

def note(text):
    with open(marker, "a") as fh:
        fh.write(text)

class Teardown:
    # Module globals are cleared before this runs at teardown.
    def __del__(self, open=open, marker=marker):
        with open(marker, "a") as fh:
            fh.write("teardown,")

def run_process(spec, ready_event, log):
    threading.Thread(target=lambda: (time.sleep(0.2), note("thread,"))).start()
    atexit.register(note, "atexit,")
    sys.stdout.write("unflushed")
    if interrupted:
        raise KeyboardInterrupt

teardown = Teardown()
stage_workers.prepare_accelerator_environment = lambda stage_spec, log: None
stage_workers.run_process = run_process
stage = SimpleNamespace(gpu_id=None, allow_child_processes=False)
spec = SimpleNamespace(log_level=20, stage_specs=[stage], process_name="p")
stage_workers.stage_process_main(spec, None)
"""


@pytest.mark.parametrize("interrupted, code", [(False, 0), (True, 1)])
def test_stage_process_exits_without_interpreter_teardown(
    tmp_path: Path, interrupted: bool, code: int
) -> None:
    marker = tmp_path / "marker.txt"
    command = [sys.executable, "-c", STAGE_PROCESS_THAT_STOPS, str(marker)]

    done = subprocess.run(
        [*command, str(int(interrupted))], capture_output=True, text=True, timeout=300
    )

    assert done.returncode == code, done.stderr
    assert "unflushed" in done.stdout
    # Threads finish before the atexit callbacks, as at a normal exit.
    assert marker.read_text() == "thread,atexit,"
