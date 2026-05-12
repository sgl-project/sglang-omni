# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)

import sglang_omni_v1.model_runner.sglang_model_runner as runner_mod
import sglang_omni_v1.models.qwen3_omni.stages as qwen_stages


def _runner(*, total_gpu_memory_fraction: float | None):
    runner = runner_mod.SGLModelRunner.__new__(runner_mod.SGLModelRunner)
    runner.gpu_id = 0
    runner.mem_fraction_static = 0.9
    runner._total_gpu_memory_fraction = total_gpu_memory_fraction
    return runner


def test_colocated_ar_budget_uses_stage_total_fraction(monkeypatch) -> None:
    runner = _runner(total_gpu_memory_fraction=0.4)
    monkeypatch.setattr(
        runner_mod,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: 30 * 1024**3,
    )
    monkeypatch.setattr(
        runner_mod,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=100 * 1024**3),
    )

    available = runner_mod.SGLModelRunner._profile_available_bytes(runner, 0)

    assert available == 10 * 1024**3


@pytest.mark.parametrize("process_memory", [None, 0])
def test_colocated_ar_budget_requires_process_memory(
    monkeypatch,
    process_memory,
) -> None:
    runner = _runner(total_gpu_memory_fraction=0.4)
    monkeypatch.setattr(
        runner_mod,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: process_memory,
    )
    monkeypatch.setattr(
        runner_mod,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=100 * 1024**3),
    )

    with pytest.raises(RuntimeError, match="requires NVML process memory"):
        runner_mod.SGLModelRunner._profile_available_bytes(runner, 0)


def test_non_colocated_ar_uses_upstream_sglang_profile(monkeypatch) -> None:
    runner = _runner(total_gpu_memory_fraction=None)

    def _fake_upstream(self, pre_model_load_memory):
        assert pre_model_load_memory == 123
        return 456

    monkeypatch.setattr(
        ModelRunnerKVCacheMixin,
        "_profile_available_bytes",
        _fake_upstream,
    )

    assert runner_mod.SGLModelRunner._profile_available_bytes(runner, 123) == 456


def test_qwen_ar_factory_derives_mem_fraction_from_total_budget() -> None:
    overrides = {"disable_cuda_graph": False}

    qwen_stages._apply_colocated_ar_memory_contract(
        overrides,
        stage_name="thinker",
        total_gpu_memory_fraction=0.78,
    )

    assert overrides["mem_fraction_static"] == 0.78


def test_qwen_ar_factory_rejects_conflicting_memory_contract() -> None:
    with pytest.raises(ValueError, match="conflicting colocated memory contracts"):
        qwen_stages._apply_colocated_ar_memory_contract(
            {"mem_fraction_static": 0.7},
            stage_name="thinker",
            total_gpu_memory_fraction=0.78,
        )
