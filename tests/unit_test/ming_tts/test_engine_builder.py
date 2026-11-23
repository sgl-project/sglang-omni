# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import sglang_omni.models.ming_tts.engine_builder as engine_builder_module
from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder
from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner


def adjust_overrides(key: str, value: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        key: value,
    }
    MingTtsEngineBuilder().adjust_overrides(overrides)
    return overrides


@pytest.mark.parametrize(
    ("is_cuda", "expected_backend"), [(True, "breakable"), (False, "disabled")]
)
def test_ming_tts_default_prefill_backend_is_declared_supported(
    monkeypatch: pytest.MonkeyPatch, is_cuda: bool, expected_backend: str
) -> None:
    """Non-CUDA platforms bypass the graphed prefill forward, so they must not request one."""
    from sglang_omni.models.ming_tts import CAPABILITIES
    from sglang_omni.scheduling.generation_batch_policy import (
        build_default_prefill_cuda_graph_bs,
    )

    monkeypatch.setattr(
        engine_builder_module,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: is_cuda),
    )
    defaults = MingTtsEngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["cuda_graph_backend_prefill"] == expected_backend
    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert MingTtsEngineBuilder.supports_breakable_prefill_cuda_graph is True
    assert defaults["cuda_graph_bs_prefill"] == build_default_prefill_cuda_graph_bs(512)


def resolved_server_args(overrides: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        max_running_requests=overrides["max_running_requests"],
        attention_backend="fa3",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        disable_cuda_graph=overrides["disable_cuda_graph"],
        enable_torch_compile=overrides["enable_torch_compile"],
        torch_compile_max_bs=overrides["torch_compile_max_bs"],
        chunked_prefill_size=overrides["chunked_prefill_size"],
        max_prefill_tokens=overrides["max_prefill_tokens"],
        tp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        lora_paths=None,
        enable_lora=None,
        moe_a2a_backend="none",
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"], bs=overrides["cuda_graph_bs"]
            ),
            prefill=SimpleNamespace(
                backend=overrides["cuda_graph_backend_prefill"],
                bs=overrides.get("cuda_graph_bs_prefill"),
                max_bs=overrides.get("cuda_graph_max_bs_prefill"),
            ),
        ),
        _cuda_graph_config_locked=frozenset(),
    )


@pytest.mark.parametrize(
    ("server_args_overrides", "expected_backend"),
    [({}, "disabled"), ({"disable_cuda_graph": False}, "breakable")],
)
def test_ming_tts_default_policy_passes_validation_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
    server_args_overrides: dict[str, Any],
    expected_backend: str,
) -> None:
    """The eager default must not request a prefill graph, and enabling graphs restores it."""
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )

    monkeypatch.setattr(
        engine_builder_module, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    builder = MingTtsEngineBuilder()
    builder.context_length = 8192
    overrides = build_generation_batch_overrides(
        server_args_overrides=server_args_overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(overrides)

    assert overrides["cuda_graph_backend_prefill"] == expected_backend
    validate_generation_batch_policy(
        model_name=builder.model_name,
        server_args=resolved_server_args(overrides),
        allowed_prefill_backends=builder.allowed_prefill_cuda_graph_backends(),
    )


def test_ming_tts_abort_callback_resets_runner_state() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner.request_states = {"req-ming-tts": object()}
    builder = object.__new__(MingTtsEngineBuilder)
    builder.model_runner = runner

    abort_callback = builder.make_abort_callback()
    abort_callback("req-ming-tts")
    abort_callback("req-ming-tts")

    assert runner.request_states == {}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("disable_overlap_schedule", True),
        ("disable_overlap_schedule", 1),
        ("disable_overlap_schedule", "1"),
        ("disable_overlap_schedule", "true"),
        ("disable_overlap_schedule", "True"),
        ("disable_overlap_schedule", " yes "),
        ("disable_overlap_schedule", "on"),
        ("disable_radix_cache", True),
        ("disable_radix_cache", 1),
        ("disable_radix_cache", " yes "),
    ],
)
def test_ming_tts_accepts_affirmative_unsupported_feature_flags(
    key: str, value: Any
) -> None:
    overrides = adjust_overrides(key, value)

    assert overrides[key] is True


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("disable_overlap_schedule", "does not currently support SGLang overlap"),
        ("disable_radix_cache", "requires disable_radix_cache=true"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [False, 0, "false", None],
)
def test_ming_tts_rejects_enabled_unsupported_feature_flags(
    key: str, message: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=message):
        adjust_overrides(key, value)


@pytest.mark.parametrize("value", [False, 0, "false", None])
def test_ming_tts_accepts_disabled_torch_compile(value: Any) -> None:
    overrides = adjust_overrides("enable_torch_compile", value)

    assert overrides["enable_torch_compile"] is value


@pytest.mark.parametrize("value", [True, 1, " yes "])
def test_ming_tts_rejects_enabled_torch_compile(value: Any) -> None:
    with pytest.raises(ValueError, match="torch.compile is not currently supported"):
        adjust_overrides("enable_torch_compile", value)
