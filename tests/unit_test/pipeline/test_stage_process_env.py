# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    _patched_spawn_env,
    get_stage_process_env,
)
from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler, fake_factory_path


def _tp_spec(*, gpu_id: int) -> StageLaunchConfig:
    return StageLaunchConfig(
        stage_name="thinker",
        role="leader",
        tp_rank=0,
        tp_size=2,
        gpu_id=gpu_id,
    )


def _worker_spec(*stage_specs: StageLaunchConfig) -> StageWorkerProcessSpec:
    return StageWorkerProcessSpec(
        process_name="worker",
        stage_specs=list(stage_specs),
    )


def test_tp_process_env_maps_logical_gpu_through_visible_devices() -> None:
    env = get_stage_process_env(_tp_spec(gpu_id=1), {"CUDA_VISIBLE_DEVICES": "3,4"})

    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


def test_tp_process_env_rejects_single_visible_device_for_second_gpu() -> None:
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES only exposes"):
        get_stage_process_env(_tp_spec(gpu_id=1), {"CUDA_VISIBLE_DEVICES": "0"})


def test_tp_process_env_requires_gpu_id() -> None:
    with pytest.raises(ValueError, match="requires a GPU id"):
        get_stage_process_env(StageLaunchConfig(stage_name="thinker", tp_size=2), {})


def test_tp_child_keeps_parent_mapped_visible_device(monkeypatch) -> None:
    """Child startup normalizes the already-mapped TP device to local cuda:0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    spec = StageLaunchConfig(
        stage_name="thinker",
        role="follower",
        tp_rank=1,
        tp_size=2,
        gpu_id=1,
        factory_arg_defaults={"gpu_id": 1},
        relay_config={"gpu_id": 1},
    )

    stage_workers._prepare_cuda_environment(spec, _RecordingLog())

    assert spec.gpu_id == 0
    assert spec.factory_arg_defaults["gpu_id"] == 0
    assert spec.relay_config["gpu_id"] == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"


def test_spawn_env_applies_stage_defaults_before_child_start(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_TEST_STAGE_ENV", raising=False)
    spec = StageLaunchConfig(
        stage_name="thinker",
        env_defaults={"SGLANG_TEST_STAGE_ENV": "default"},
    )

    with _patched_spawn_env(_worker_spec(spec)):
        assert os.environ["SGLANG_TEST_STAGE_ENV"] == "default"

    assert "SGLANG_TEST_STAGE_ENV" not in os.environ


def test_spawn_env_preserves_operator_stage_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_TEST_STAGE_ENV", "operator")
    spec = StageLaunchConfig(
        stage_name="thinker",
        env_defaults={"SGLANG_TEST_STAGE_ENV": "default"},
    )

    with _patched_spawn_env(_worker_spec(spec)):
        assert os.environ["SGLANG_TEST_STAGE_ENV"] == "operator"

    assert os.environ["SGLANG_TEST_STAGE_ENV"] == "operator"


def test_spawn_env_combines_stage_defaults_with_tp_visible_device(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_TEST_STAGE_ENV", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    stage_spec = _tp_spec(gpu_id=1)
    stage_spec.env_defaults = {"SGLANG_TEST_STAGE_ENV": "default"}

    with _patched_spawn_env(_worker_spec(stage_spec)):
        assert os.environ["SGLANG_TEST_STAGE_ENV"] == "default"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
        assert os.environ["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"

    assert "SGLANG_TEST_STAGE_ENV" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,4"


class _RecordingLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        if args:
            message = message % args
        self.messages.append(message)


def test_gpu_scheduler_construction_uses_startup_lock(monkeypatch) -> None:
    """GPU stage factory construction is serialized per visible device."""
    seen_gpu_ids: list[int] = []

    @contextmanager
    def _fake_lock(gpu_id: int):
        seen_gpu_ids.append(gpu_id)
        yield Path("/tmp/test.lock")

    monkeypatch.setattr(stage_workers, "gpu_startup_lock", _fake_lock)
    spec = StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path("make_scheduler"),
    )

    scheduler = stage_workers._construct_scheduler(spec, 0, _RecordingLog())

    assert isinstance(scheduler, FakeScheduler)
    assert seen_gpu_ids == [0]


def test_scheduler_applies_child_defaults_without_overriding_explicit_args(
    monkeypatch,
) -> None:
    seen_gpu_ids: list[int] = []

    @contextmanager
    def _fake_lock(gpu_id: int):
        seen_gpu_ids.append(gpu_id)
        yield Path("/tmp/test.lock")

    monkeypatch.setattr(stage_workers, "gpu_startup_lock", _fake_lock)
    spec = StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path("runtime_factory"),
        factory_args={
            "model_path": "runtime-model",
            "thinker_max_seq_len": 128,
        },
        factory_arg_defaults={
            "model_path": "global-model",
            "gpu_id": 3,
            "total_gpu_memory_fraction": 0.25,
        },
    )

    result = stage_workers._construct_scheduler(spec, 3, _RecordingLog())

    assert result["model_path"] == "runtime-model"
    assert result["gpu_id"] == 3
    assert result["thinker_max_seq_len"] == 128
    assert result["total_gpu_memory_fraction"] == 0.25
    assert seen_gpu_ids == [3]


def test_construct_stage_uses_placement_gpu_id_for_device_and_startup_lock(
    monkeypatch,
) -> None:
    """Placement-owned gpu_id must drive device setup and startup lock."""
    import torch

    class _FakeStage:
        def __init__(self, **kwargs):
            self.scheduler = kwargs["scheduler"]

    set_device_calls: list[int] = []
    seen_gpu_ids: list[int] = []

    @contextmanager
    def _fake_lock(gpu_id: int):
        seen_gpu_ids.append(gpu_id)
        yield Path("/tmp/test.lock")

    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda gpu_id: set_device_calls.append(int(gpu_id)),
    )
    monkeypatch.setattr(stage_workers, "gpu_startup_lock", _fake_lock)
    monkeypatch.setattr(stage_workers, "Stage", _FakeStage)

    specs = [
        StageLaunchConfig(
            stage_name=f"gpu_stage_{idx}",
            factory=fake_factory_path("make_scheduler_accepting_gpu_id"),
            factory_arg_defaults={"gpu_id": 0},
            gpu_id=0,
        )
        for idx in range(2)
    ]

    stages = [stage_workers._construct_stage(spec, _RecordingLog()) for spec in specs]

    assert [stage.scheduler.gpu_id for stage in stages] == [0, 0]
    assert set_device_calls == [0, 0]
    assert seen_gpu_ids == [0, 0]


def test_cpu_scheduler_construction_skips_startup_lock(monkeypatch) -> None:
    def _unexpected_lock(gpu_id: int):
        raise AssertionError(f"unexpected GPU lock for {gpu_id}")

    monkeypatch.setattr(stage_workers, "gpu_startup_lock", _unexpected_lock)
    spec = StageLaunchConfig(
        stage_name="decode",
        factory=fake_factory_path("make_scheduler"),
    )

    scheduler = stage_workers._construct_scheduler(spec, None, _RecordingLog())

    assert isinstance(scheduler, FakeScheduler)
