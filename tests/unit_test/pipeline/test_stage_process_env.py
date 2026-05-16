# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from sglang_omni.pipeline import stage_process
from sglang_omni.pipeline.stage_process import StageProcessSpec, get_stage_process_env
from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler, fake_factory_path


def _tp_spec(*, gpu_id: int) -> StageProcessSpec:
    return StageProcessSpec(
        stage_name="thinker",
        role="leader",
        tp_rank=0,
        tp_size=2,
        gpu_id=gpu_id,
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
        get_stage_process_env(StageProcessSpec(stage_name="thinker", tp_size=2), {})


def test_tp_child_keeps_parent_mapped_visible_device(monkeypatch) -> None:
    """Child startup normalizes the already-mapped TP device to local cuda:0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    spec = StageProcessSpec(
        stage_name="thinker",
        role="follower",
        tp_rank=1,
        tp_size=2,
        gpu_id=1,
        factory_args={"gpu_id": 1},
        relay_config={"gpu_id": 1},
    )

    stage_process._prepare_cuda_environment(spec, _RecordingLog())

    assert spec.gpu_id == 0
    assert spec.factory_args["gpu_id"] == 0
    assert spec.relay_config["gpu_id"] == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"


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

    monkeypatch.setattr(stage_process, "gpu_startup_lock", _fake_lock)
    spec = StageProcessSpec(
        stage_name="thinker",
        factory=fake_factory_path("make_scheduler"),
    )

    scheduler = stage_process._construct_scheduler(spec, 0, _RecordingLog())

    assert isinstance(scheduler, FakeScheduler)
    assert seen_gpu_ids == [0]


def test_cpu_scheduler_construction_skips_startup_lock(monkeypatch) -> None:
    def _unexpected_lock(gpu_id: int):
        raise AssertionError(f"unexpected GPU lock for {gpu_id}")

    monkeypatch.setattr(stage_process, "gpu_startup_lock", _unexpected_lock)
    spec = StageProcessSpec(
        stage_name="decode",
        factory=fake_factory_path("make_scheduler"),
    )

    scheduler = stage_process._construct_scheduler(spec, None, _RecordingLog())

    assert isinstance(scheduler, FakeScheduler)
