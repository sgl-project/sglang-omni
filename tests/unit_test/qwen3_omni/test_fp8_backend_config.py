# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.model_runner import model_worker


@dataclass(frozen=True)
class BackendPolicyCase:
    name: str
    model_quantization: str | None
    server_quantization: str | None
    native_fp8_block_quant: bool
    model_arch_override: str | None
    has_moe: bool
    initial_moe_backend: str
    initial_fp8_gemm_backend: str | None
    ep_size: int
    cutlass_supported: bool
    expected_quantization: str | None
    expected_moe_backend: str | None = None
    expected_fp8_gemm_backend: str | None = None
    error_match: str | None = None


def _server_args(
    *,
    quantization: str | None = None,
    moe_runner_backend: str = "auto",
    fp8_gemm_runner_backend: str | None = "auto",
    ep_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        quantization=quantization,
        moe_runner_backend=moe_runner_backend,
        fp8_gemm_runner_backend=fp8_gemm_runner_backend,
        fp4_gemm_runner_backend="auto",
        ep_size=ep_size,
    )


def _model_config(
    *,
    quantization: str | None,
    native_fp8_block_quant: bool = False,
    has_moe: bool = True,
) -> SimpleNamespace:
    attrs = {"num_experts_per_tok": 8} if has_moe else {}
    quantization_config = (
        {"quant_method": "fp8", "weight_block_size": [128, 128]}
        if native_fp8_block_quant
        else None
    )
    return SimpleNamespace(
        quantization=quantization,
        hf_text_config=SimpleNamespace(**attrs),
        hf_config=SimpleNamespace(quantization_config=quantization_config),
    )


@pytest.mark.parametrize(
    "case",
    [
        BackendPolicyCase(
            name="bf16_talker_auto_uses_flashinfer_cutlass",
            model_quantization=None,
            server_quantization=None,
            native_fp8_block_quant=False,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization=None,
            expected_moe_backend="flashinfer_cutlass",
            expected_fp8_gemm_backend="auto",
        ),
        BackendPolicyCase(
            name="bf16_talker_explicit_triton_is_preserved",
            model_quantization=None,
            server_quantization=None,
            native_fp8_block_quant=False,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="triton",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization=None,
            expected_moe_backend="triton",
            expected_fp8_gemm_backend="auto",
        ),
        BackendPolicyCase(
            name="fp8_talker_auto_uses_cutlass_moe_and_triton_dense_gemm",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="cutlass",
            expected_fp8_gemm_backend="triton",
        ),
        BackendPolicyCase(
            name="fp8_thinker_auto_uses_cutlass_moe_and_preserves_dense_gemm_auto",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniThinkerForCausalLM",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="cutlass",
            expected_fp8_gemm_backend="auto",
        ),
        BackendPolicyCase(
            name="server_fp8_override_without_native_block_quant_stays_auto",
            model_quantization=None,
            server_quantization="fp8",
            native_fp8_block_quant=False,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="auto",
            expected_fp8_gemm_backend="auto",
        ),
        BackendPolicyCase(
            name="fp8_talker_dense_triton_is_independent_of_cutlass_moe_support",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=False,
            expected_quantization="fp8",
            expected_moe_backend="auto",
            expected_fp8_gemm_backend="triton",
        ),
        BackendPolicyCase(
            name="fp8_talker_dense_triton_does_not_require_moe",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=False,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="auto",
            expected_fp8_gemm_backend="triton",
        ),
        BackendPolicyCase(
            name="fp8_non_qwen_omni_arch_stays_auto",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="OtherMoEForCausalLM",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="auto",
            expected_fp8_gemm_backend="auto",
        ),
        BackendPolicyCase(
            name="fp8_non_qwen_omni_arch_preserves_flashinfer_cutlass",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="OtherMoEForCausalLM",
            has_moe=True,
            initial_moe_backend="flashinfer_cutlass",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="flashinfer_cutlass",
            expected_fp8_gemm_backend="auto",
        ),
        BackendPolicyCase(
            name="fp8_explicit_moe_triton_still_uses_talker_dense_triton_default",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="triton",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="triton",
            expected_fp8_gemm_backend="triton",
        ),
        BackendPolicyCase(
            name="fp8_explicit_moe_cutlass_still_uses_talker_dense_triton_default",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="cutlass",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="cutlass",
            expected_fp8_gemm_backend="triton",
        ),
        BackendPolicyCase(
            name="fp8_talker_explicit_dense_deep_gemm_is_preserved",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="deep_gemm",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="cutlass",
            expected_fp8_gemm_backend="deep_gemm",
        ),
        BackendPolicyCase(
            name="fp8_talker_explicit_dense_cutlass_is_preserved",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="cutlass",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="cutlass",
            expected_fp8_gemm_backend="cutlass",
        ),
        BackendPolicyCase(
            name="fp8_talker_unset_dense_backend_uses_triton",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend=None,
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_moe_backend="cutlass",
            expected_fp8_gemm_backend="triton",
        ),
        BackendPolicyCase(
            name="fp8_explicit_cutlass_requires_native_block_quant",
            model_quantization=None,
            server_quantization="fp8",
            native_fp8_block_quant=False,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="cutlass",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_fp8_gemm_backend="auto",
            error_match="requires a native serialized block-FP8 checkpoint",
        ),
        BackendPolicyCase(
            name="qwen3_omni_rejects_ep_size_above_one",
            model_quantization=None,
            server_quantization=None,
            native_fp8_block_quant=False,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="auto",
            initial_fp8_gemm_backend="auto",
            ep_size=2,
            cutlass_supported=True,
            expected_quantization=None,
            expected_fp8_gemm_backend="auto",
            error_match="does not support expert parallelism",
        ),
        BackendPolicyCase(
            name="fp8_rejects_flashinfer_cutlass",
            model_quantization="fp8",
            server_quantization=None,
            native_fp8_block_quant=True,
            model_arch_override="Qwen3OmniTalker",
            has_moe=True,
            initial_moe_backend="flashinfer_cutlass",
            initial_fp8_gemm_backend="auto",
            ep_size=1,
            cutlass_supported=True,
            expected_quantization="fp8",
            expected_fp8_gemm_backend="auto",
            error_match="native FP8.*flashinfer_cutlass",
        ),
    ],
    ids=lambda case: case.name,
)
def test_model_worker_backend_policy_precedence(
    monkeypatch: pytest.MonkeyPatch,
    case: BackendPolicyCase,
) -> None:
    """Covers quantization, architecture, MoE, hardware, and explicit override precedence."""
    monkeypatch.setattr(
        model_worker,
        "_is_fp8_cutlass_moe_supported",
        lambda: case.cutlass_supported,
    )
    monkeypatch.setattr(model_worker, "_is_h20_device", lambda: False)
    server_args = _server_args(
        quantization=case.server_quantization,
        moe_runner_backend=case.initial_moe_backend,
        fp8_gemm_runner_backend=case.initial_fp8_gemm_backend,
        ep_size=case.ep_size,
    )
    model_config = _model_config(
        quantization=case.model_quantization,
        native_fp8_block_quant=case.native_fp8_block_quant,
        has_moe=case.has_moe,
    )

    if case.error_match:
        with pytest.raises(ValueError, match=case.error_match):
            model_worker._apply_model_worker_backend_policy(
                server_args,
                model_config,
                case.model_arch_override,
            )
        return

    effective_quantization = model_worker._apply_model_worker_backend_policy(
        server_args,
        model_config,
        case.model_arch_override,
    )

    assert effective_quantization == case.expected_quantization
    assert server_args.quantization == case.server_quantization
    assert server_args.moe_runner_backend == case.expected_moe_backend
    assert server_args.fp8_gemm_runner_backend == case.expected_fp8_gemm_backend


def test_model_config_has_moe_prefers_effective_text_config() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=SimpleNamespace()),
        hf_text_config=SimpleNamespace(num_experts_per_tok=8),
    )

    assert model_worker._model_config_has_moe(model_config)


@pytest.mark.parametrize(
    (
        "cutlass_supported",
        "sm90_supported",
        "sm100_supported",
        "expected_supported",
    ),
    [
        pytest.param(True, True, False, True, id="h100_h200_h20_supported"),
        pytest.param(True, False, True, True, id="sm100_supported"),
        pytest.param(True, False, False, False, id="unsupported_gpu_rejected"),
        pytest.param(False, True, False, False, id="cutlass_runtime_rejected"),
    ],
)
def test_fp8_cutlass_moe_support_matches_sglang_0_5_12_post1_contract(
    monkeypatch: pytest.MonkeyPatch,
    cutlass_supported: bool,
    sm90_supported: bool,
    sm100_supported: bool,
    expected_supported: bool,
) -> None:
    """Mirrors the CUTLASS FP8 MoE assertions in pinned SGLang 0.5.12.post1."""
    _install_fake_cutlass_support_modules(
        monkeypatch,
        cutlass_supported=cutlass_supported,
        sm90_supported=sm90_supported,
        sm100_supported=sm100_supported,
    )

    assert model_worker._is_fp8_cutlass_moe_supported() is expected_supported


def test_backend_global_initialization_for_fp8_moe_model(monkeypatch) -> None:
    calls: list[str] = []

    _install_fake_backend_modules(monkeypatch, calls)

    model_worker._initialize_model_worker_backend_globals(
        _server_args(),
        _model_config(quantization="fp8", native_fp8_block_quant=True),
        "fp8",
    )

    assert calls == ["moe", "fp8"]


def test_backend_global_initialization_for_bf16_moe_omits_fp8(monkeypatch) -> None:
    calls: list[str] = []

    _install_fake_backend_modules(monkeypatch, calls)

    model_worker._initialize_model_worker_backend_globals(
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


def _install_fake_cutlass_support_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cutlass_supported: bool,
    sm90_supported: bool,
    sm100_supported: bool,
) -> None:
    _install_fake_module(monkeypatch, "sglang")
    _install_fake_module(monkeypatch, "sglang.srt")
    _install_fake_module(monkeypatch, "sglang.srt.layers")
    _install_fake_module(monkeypatch, "sglang.srt.layers.quantization")
    _install_fake_module(
        monkeypatch,
        "sglang.srt.layers.quantization.fp8_utils",
        cutlass_fp8_supported=lambda: cutlass_supported,
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.utils",
        is_sm90_supported=lambda: sm90_supported,
        is_sm100_supported=lambda: sm100_supported,
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
