# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.utils import tensor_bridge

from sglang_omni.models.fun_cosyvoice3 import engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder


def _enable_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    monkeypatch.setattr(
        engine_builder_module.current_platform,
        "is_mps",
        lambda: True,
    )


def _valid_mlx_server_args() -> SimpleNamespace:
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
    _enable_mlx(monkeypatch)
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
    _enable_mlx(monkeypatch)
    server_args = _valid_mlx_server_args()
    setattr(server_args, field, value)

    with pytest.raises(ValueError, match=message):
        FunCosyVoice3EngineBuilder().validate_before_infrastructure(server_args)


def test_mlx_engine_passes_distinct_native_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_mlx(monkeypatch)
    builder = FunCosyVoice3EngineBuilder(
        mlx_model_path="mlx-org/model",
        mlx_model_revision="mlx-revision",
    )
    builder._checkpoint_root = "/official/model"

    assert builder.infra_kwargs() == {
        "mlx_model_path": "mlx-org/model",
        "mlx_model_revision": "mlx-revision",
    }


def test_torch_mps_uses_single_request_native_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "mps:0"

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["max_running_requests"] == 1

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=2))
