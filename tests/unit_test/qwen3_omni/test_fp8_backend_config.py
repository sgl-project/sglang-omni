# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.model_runner.model_worker import (
    _apply_model_worker_backend_policy,
    _initialize_model_worker_backend_globals,
    _model_config_has_moe,
)


def _server_args(
    *,
    quantization: str | None = None,
    moe_runner_backend: str = "auto",
):
    return SimpleNamespace(
        quantization=quantization,
        moe_runner_backend=moe_runner_backend,
        fp8_gemm_runner_backend="auto",
        fp4_gemm_runner_backend="auto",
    )


def _model_config(
    *,
    quantization: str | None,
    has_moe: bool = True,
):
    attrs = {"num_experts_per_tok": 8} if has_moe else {}
    return SimpleNamespace(
        quantization=quantization,
        hf_text_config=SimpleNamespace(**attrs),
    )


def test_bf16_talker_preserves_existing_flashinfer_cutlass_default() -> None:
    server_args = _server_args()
    model_config = _model_config(quantization=None)

    effective = _apply_model_worker_backend_policy(
        server_args,
        model_config,
        "Qwen3OmniTalker",
    )

    assert effective is None
    assert server_args.quantization is None
    assert server_args.moe_runner_backend == "flashinfer_cutlass"


def test_native_fp8_talker_leaves_server_quantization_unset() -> None:
    server_args = _server_args()
    model_config = _model_config(quantization="fp8")

    effective = _apply_model_worker_backend_policy(
        server_args,
        model_config,
        "Qwen3OmniTalker",
    )

    assert effective == "fp8"
    assert server_args.quantization is None
    assert server_args.moe_runner_backend == "auto"


def test_native_fp8_talker_rejects_flashinfer_cutlass_backend() -> None:
    server_args = _server_args(moe_runner_backend="flashinfer_cutlass")
    model_config = _model_config(quantization="fp8")

    with pytest.raises(ValueError, match="native FP8.*flashinfer_cutlass"):
        _apply_model_worker_backend_policy(
            server_args,
            model_config,
            "Qwen3OmniTalker",
        )


def test_native_fp8_thinker_leaves_server_quantization_unset() -> None:
    server_args = _server_args()
    model_config = _model_config(quantization="fp8")

    effective = _apply_model_worker_backend_policy(
        server_args,
        model_config,
        "Qwen3OmniThinkerForCausalLM",
    )

    assert effective == "fp8"
    assert server_args.quantization is None
    assert server_args.moe_runner_backend == "auto"


def test_model_config_has_moe_prefers_effective_text_config() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=SimpleNamespace()),
        hf_text_config=SimpleNamespace(num_experts_per_tok=8),
    )

    assert _model_config_has_moe(model_config)


def test_backend_global_initialization_for_fp8_moe_model(monkeypatch) -> None:
    calls: list[str] = []

    _install_fake_backend_modules(monkeypatch, calls)

    _initialize_model_worker_backend_globals(
        _server_args(),
        _model_config(quantization="fp8"),
        "fp8",
    )

    assert calls == ["moe", "fp8"]


def test_backend_global_initialization_for_bf16_moe_omits_fp8(monkeypatch) -> None:
    calls: list[str] = []

    _install_fake_backend_modules(monkeypatch, calls)

    _initialize_model_worker_backend_globals(
        _server_args(),
        _model_config(quantization=None),
        None,
    )

    assert calls == ["moe"]


def _install_fake_backend_modules(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    _install_fake_module(monkeypatch, "sglang")
    _install_fake_module(monkeypatch, "sglang.srt")
    _install_fake_module(monkeypatch, "sglang.srt.layers")
    _install_fake_module(monkeypatch, "sglang.srt.layers.quantization")
    _install_fake_module(
        monkeypatch,
        "sglang.srt.layers.moe",
        initialize_moe_config=lambda server_args: calls.append("moe"),
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.layers.quantization.fp8_utils",
        initialize_fp8_gemm_config=lambda server_args: calls.append("fp8"),
    )


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attrs: object,
) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module
