# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.llada2_uni import stages


def _capture_server_args_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    server_args_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    bootstrap_module = ModuleType("sglang_omni.models.llada2_uni.bootstrap")

    def fake_create_scheduler(
        server_args,
        gpu_id,
        *,
        tp_rank,
        nccl_port,
    ):
        return "scheduler"

    bootstrap_module.create_dllm_thinker_scheduler = fake_create_scheduler
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.llada2_uni.bootstrap",
        bootstrap_module,
    )

    backend_module = ModuleType("sglang_omni.scheduling.sglang_backend")

    def fake_build_server_args(model_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            dllm_algorithm=kwargs["dllm_algorithm"],
            mem_fraction_static=kwargs.get("mem_fraction_static"),
        )

    backend_module.build_sglang_server_args = fake_build_server_args
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        backend_module,
    )

    result = stages.create_sglang_dllm_thinker_executor_from_config(
        "model",
        server_args_overrides=server_args_overrides,
    )

    assert result == "scheduler"
    return captured


def test_thinker_enables_cuda_graph_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_kwargs = _capture_server_args_kwargs(monkeypatch)

    assert "disable_cuda_graph" not in build_kwargs


def test_thinker_preserves_explicit_cuda_graph_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_kwargs = _capture_server_args_kwargs(
        monkeypatch,
        server_args_overrides={"disable_cuda_graph": True},
    )

    assert build_kwargs["disable_cuda_graph"] is True
