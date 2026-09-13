# SPDX-License-Identifier: Apache-2.0
"""Spawn-time MPS environment injection tests."""

from __future__ import annotations

import logging
import os

import pytest

from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    _patched_spawn_env,
    _prepare_accelerator_environment,
)

_FACTORY = f"{__name__}.unused_factory"


def _launch_stage(
    stage_name: str = "thinker",
    *,
    gpu_id: int | None = 0,
    env_defaults: dict[str, str] | None = None,
) -> StageLaunchConfig:
    return StageLaunchConfig(
        stage_name=stage_name,
        factory=_FACTORY,
        gpu_id=gpu_id,
        placement_gpu_id=gpu_id,
        env_defaults=env_defaults or {},
    )


def _process_spec(stage: StageLaunchConfig) -> StageWorkerProcessSpec:
    return StageWorkerProcessSpec(process_name=stage.stage_name, stage_specs=[stage])


@pytest.fixture(autouse=True)
def _no_gpu_compat_probe(monkeypatch):
    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(stage_workers, "get_gpu_compat_env_defaults", lambda _env: {})


def test_mps_overlay_is_visible_only_during_spawn(monkeypatch):
    spec = _process_spec(_launch_stage())
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with _patched_spawn_env(
        spec,
        extra_env={
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe",
            "CUDA_VISIBLE_DEVICES": "GPU-abc",
        },
    ):
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps/pipe"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc"

    assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_no_mps_overlay_keeps_existing_stage_default_behavior(monkeypatch):
    spec = _process_spec(_launch_stage(env_defaults={"WORKER_DEFAULT": "stage-value"}))
    monkeypatch.delenv("WORKER_DEFAULT", raising=False)

    with _patched_spawn_env(spec):
        assert os.environ["WORKER_DEFAULT"] == "stage-value"

    assert "WORKER_DEFAULT" not in os.environ


def test_cpu_stage_keeps_none_gpu_id_under_single_device_marker(monkeypatch):
    spec = _launch_stage("preprocessing", gpu_id=None)
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    _prepare_accelerator_environment(spec, logging.getLogger("test"))

    assert spec.gpu_id is None
