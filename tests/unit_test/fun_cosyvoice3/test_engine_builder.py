# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs

import sglang_omni.models.fun_cosyvoice3.engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    FULL_PREFILL_ATTENTION_BACKENDS,
    CudaGraphBackend,
    build_generation_batch_overrides,
)
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from tests.unit_test.fixtures.mini_checkpoint import write_mini_llama_checkpoint


def test_torch_mps_uses_single_request_native_attention() -> None:
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "mps:0"

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["max_running_requests"] == 1

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=2))


def cuda_overrides(
    **server_args_overrides: object,
) -> tuple[FunCosyVoice3EngineBuilder, dict[str, object]]:
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "cuda:0"
    overrides = build_generation_batch_overrides(
        server_args_overrides=server_args_overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(overrides)
    return builder, overrides


def test_cuda_engine_caps_the_kv_pool_at_the_running_requests() -> None:
    builder, overrides = cuda_overrides()

    assert overrides["max_total_tokens"] == 32 * builder.context_length


def test_kv_pool_cap_follows_an_operator_max_running_requests() -> None:
    builder, overrides = cuda_overrides(max_running_requests=64)

    assert overrides["max_running_requests"] == 64
    assert overrides["max_total_tokens"] == 64 * builder.context_length


def test_operator_max_total_tokens_is_kept() -> None:
    _, overrides = cuda_overrides(max_total_tokens=5000)

    assert overrides["max_total_tokens"] == 5000


def resolved_cuda_server_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_args_overrides: dict[str, object],
) -> ServerArgs:
    monkeypatch.setattr(
        engine_builder_module, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    checkpoint = write_mini_llama_checkpoint(tmp_path)
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "cuda:0"
    builder.server_args_overrides = server_args_overrides
    overrides = build_generation_batch_overrides(
        server_args_overrides=server_args_overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(overrides)
    server_args = build_sglang_server_args(
        checkpoint, context_length=builder.context_length, device="cuda", **overrides
    )
    builder.customize_server_args(server_args)
    return server_args


@pytest.mark.parametrize(
    "declared_ladder",
    [
        {"cuda_graph_bs_prefill": [4, 8], "cuda_graph_max_bs_prefill": 8},
        {"cuda_graph_config": {"prefill": {"bs": [4, 8], "max_bs": 8}}},
    ],
    ids=["flat", "nested"],
)
def test_a_declared_prefill_ladder_is_served_without_an_added_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_ladder: dict[str, object],
) -> None:
    server_args = resolved_cuda_server_args(
        tmp_path, monkeypatch, {"attention_backend": "fa3", **declared_ladder}
    )
    prefill = resolved_view(server_args).cuda_graph_config.prefill

    assert prefill.backend == CudaGraphBackend.FULL
    assert prefill.bs == [4, 8]


@pytest.mark.parametrize(
    ("server_args_overrides", "platform"),
    [
        ({"attention_backend": "triton"}, {}),
        ({}, {"is_sm100": True, "is_hopper_with_cuda_12_3": False}),
    ],
    ids=["operator_triton", "sm100_default"],
)
def test_prefill_stays_eager_where_the_attention_backend_cannot_capture_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_args_overrides: dict[str, object],
    platform: dict[str, bool],
) -> None:
    """An attention backend that cannot capture a full prefill graph, the operator's or
    SGLang's default on SM100, leaves the prefill eager instead of the breakable graph
    this model refuses."""
    with override_platform(**platform):
        server_args = resolved_cuda_server_args(
            tmp_path, monkeypatch, server_args_overrides
        )
    resolved = resolved_view(server_args)

    assert attention_backends_of(resolved)[0] not in FULL_PREFILL_ATTENTION_BACKENDS
    assert resolved.cuda_graph_config.prefill.backend == CudaGraphBackend.DISABLED
