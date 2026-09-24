# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.hardware_backend.mlx import runtime as mlx_runtime

from sglang_omni.models.chatterbox.engine_builder import ChatterboxT3EngineBuilder


def test_generation_defaults_cuda_defaults_to_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = ChatterboxT3EngineBuilder(max_new_tokens=1024)
    builder.device = None
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 64


def test_generation_defaults_mlx_single_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    builder = ChatterboxT3EngineBuilder(max_new_tokens=1024)
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 1
    assert defaults["mlx_enable_sampling"] is True


def test_generation_defaults_mps_single_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = ChatterboxT3EngineBuilder(max_new_tokens=1024)
    builder.device = "mps:0"
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 1
    assert defaults["attention_backend"] == "torch_native"


def test_validate_mps_rejects_multiple_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = ChatterboxT3EngineBuilder(max_new_tokens=1024)
    builder.device = "mps:0"

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(
            SimpleNamespace(max_running_requests=4)
        )


def test_validate_non_mps_allows_multiple_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = ChatterboxT3EngineBuilder(max_new_tokens=1024)
    builder.device = "cuda:0"

    builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=4))
