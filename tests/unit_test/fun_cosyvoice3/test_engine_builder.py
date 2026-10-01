# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.hardware_backend.mlx import runtime as mlx_runtime

from sglang_omni.models.fun_cosyvoice3 import engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def enable_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    monkeypatch.setattr(
        engine_builder_module.current_platform,
        "is_mps",
        lambda: True,
    )


def valid_mlx_server_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_running_requests=1,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        disable_overlap_schedule=True,
        enable_priority_scheduling=False,
        mlx_enable_sampling=True,
    )


def test_mlx_engine_profile_disables_incompatible_scheduler_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_mlx(monkeypatch)
    builder = FunCosyVoice3EngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["max_running_requests"] == 1
    assert defaults["disable_radix_cache"] is True
    assert defaults["disable_overlap_schedule"] is True
    assert defaults["chunked_prefill_size"] == -1
    assert defaults["mlx_enable_sampling"] is True
    assert builder.extra_scheduler_kwargs() == {
        "enable_async_decode": True,
        "async_decode_min_batch_size": 1,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_running_requests", 2, "max_running_requests=1"),
        ("disable_radix_cache", False, "disable_radix_cache=True"),
        ("chunked_prefill_size", 128, "chunked_prefill_size=-1"),
        ("disable_overlap_schedule", False, "disable_overlap_schedule=True"),
        ("enable_priority_scheduling", True, "priority preemption"),
        ("mlx_enable_sampling", False, "mlx_enable_sampling=True"),
    ],
)
def test_mlx_engine_rejects_unsafe_overrides(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    enable_mlx(monkeypatch)
    server_args = valid_mlx_server_args()
    setattr(server_args, field, value)

    with pytest.raises(ValueError, match=message):
        FunCosyVoice3EngineBuilder().validate_before_infrastructure(server_args)


def test_mlx_engine_passes_distinct_native_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_mlx(monkeypatch)
    builder = FunCosyVoice3EngineBuilder(
        mlx_model_path="mlx-org/model",
        mlx_model_revision="mlx-revision",
    )
    builder.checkpoint_root = "/official/model"

    assert builder.infra_kwargs() == {
        "mlx_model_path": "mlx-org/model",
        "mlx_model_revision": "mlx-revision",
    }


def test_torch_mps_uses_single_request_native_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "mps:0"

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["max_running_requests"] == 1

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=2))


def cuda_overrides(
    monkeypatch: pytest.MonkeyPatch, **server_args_overrides: object
) -> tuple[FunCosyVoice3EngineBuilder, dict[str, object]]:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "cuda:0"
    overrides = build_generation_batch_overrides(
        server_args_overrides=server_args_overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(overrides)
    return builder, overrides


def test_cuda_engine_caps_the_kv_pool_at_the_running_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, overrides = cuda_overrides(monkeypatch)

    assert overrides["max_total_tokens"] == 32 * builder.context_length


def test_kv_pool_cap_follows_an_operator_max_running_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, overrides = cuda_overrides(monkeypatch, max_running_requests=64)

    assert overrides["max_running_requests"] == 64
    assert overrides["max_total_tokens"] == 64 * builder.context_length


def test_operator_max_total_tokens_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _, overrides = cuda_overrides(monkeypatch, max_total_tokens=5000)

    assert overrides["max_total_tokens"] == 5000
