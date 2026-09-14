# SPDX-License-Identifier: Apache-2.0
"""Spawn-time MPS environment injection tests."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.mps.manager import MpsError
from sglang_omni.mps.runtime import MpsPipelineRuntime
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    _patched_spawn_env,
    _prepare_accelerator_environment,
)
from tests.unit_test.mps.test_mps_manager import FakeControlClient

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


@pytest.fixture
def short_root():
    root = Path(tempfile.mkdtemp(prefix="mpsenv-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


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


def _resolved_config_process(*, pipeline_env: dict, stage_env: dict):
    with tempfile.TemporaryDirectory(prefix="mps-env-", dir="/tmp") as base_path:
        config = PipelineConfig(
            model_path="model",
            name="mps-env",
            entry_stage="worker",
            endpoints=EndpointsConfig(base_path=base_path),
            env_defaults=pipeline_env,
            stages=[
                StageConfig(
                    name="worker",
                    process="pipeline",
                    factory_path=f"{__name__}.unused_factory",
                    gpu=0,
                    terminal=True,
                    env=stage_env,
                )
            ],
        )
        prep = prepare_pipeline_runtime(config)
        try:
            return _build_stage_groups(
                config,
                stages_cfg=prep.stages_cfg,
                endpoints=prep.endpoints,
                placement_plan=prep.placement_plan,
                process_plan=prep.process_plan,
                replica_topology=prep.replica_topology,
            )[0].process_specs[0]
        finally:
            prep.runtime_dir.close()


class _DeviceInfoMustNotRun:
    def inspect(self, _gpu_ids):  # pragma: no cover - contract assertion
        raise AssertionError("process env conflicts must fail before device inspection")


@pytest.mark.parametrize("mode", ["auto", "on"])
@pytest.mark.parametrize("source", ["pipeline", "stage"])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CUDA_VISIBLE_DEVICES", "1"),
        ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        ("CUDA_MPS_PIPE_DIRECTORY", "/external/mps"),
        ("SGLANG_OMNI_WEIGHT_SHARE", "leader:/tmp/weights"),
    ],
)
def test_mps_rejects_worker_gpu_environment_overrides_before_acquire(
    short_root,
    mode,
    source,
    name,
    value,
):
    process_spec = _resolved_config_process(
        pipeline_env={name: value} if source == "pipeline" else {},
        stage_env={name: value} if source == "stage" else {},
    )

    with pytest.raises(MpsError) as exc_info:
        MpsPipelineRuntime.create(
            mode=mode,
            process_specs=[process_spec],
            device_info=_DeviceInfoMustNotRun(),
            client=FakeControlClient(),
            state_root=short_root,
        )

    message = str(exc_info.value)
    assert "process 'pipeline'" in message
    assert "stage 'worker'" in message
    assert f"{name}={value!r}" in message
    assert "mps=off" in message
    assert list(short_root.iterdir()) == []


def test_cpu_stage_keeps_none_gpu_id_under_single_device_marker(monkeypatch):
    spec = _launch_stage("preprocessing", gpu_id=None)
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    _prepare_accelerator_environment(spec, logging.getLogger("test"))

    assert spec.gpu_id is None
