# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_prefill_cuda_graph_batch_sizes,
)
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.server_args import ServerArgs

import sglang_omni.models.fun_cosyvoice3.engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
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


@pytest.mark.parametrize("declared_ladder", [None, [4, 8]])
def test_a_cached_prompts_one_token_prefill_has_a_graph_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared_ladder: list[int] | None
) -> None:
    server_args = resolved_cuda_server_args(
        tmp_path,
        monkeypatch,
        {
            "attention_backend": "fa3",
            **(
                {}
                if declared_ladder is None
                else {
                    "cuda_graph_bs_prefill": declared_ladder,
                    "cuda_graph_max_bs_prefill": 8,
                }
            ),
        },
    )
    resolved = resolved_view(server_args)

    assert resolved.cuda_graph_config.prefill.backend == CudaGraphBackend.FULL
    assert resolved.cuda_graph_config.prefill.bs == (
        [1, *generate_prefill_cuda_graph_batch_sizes(resolved.chunked_prefill_size)]
        if declared_ladder is None
        else declared_ladder
    )


def test_prefill_stays_eager_where_the_attention_backend_cannot_capture_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an attention backend that cannot capture a full prefill graph, the default
    full graph fell back to the breakable one this model refuses, and startup failed."""
    server_args = resolved_cuda_server_args(
        tmp_path, monkeypatch, {"attention_backend": "triton"}
    )

    assert (
        resolved_view(server_args).cuda_graph_config.prefill.backend
        == CudaGraphBackend.DISABLED
    )
