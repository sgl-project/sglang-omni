# SPDX-License-Identifier: Apache-2.0
"""CPU policy tests for Whisper ASR (no accelerator required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.whisper_asr import engine_builder
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform


def whisper_builder() -> WhisperASREngineBuilder:
    return WhisperASREngineBuilder(
        encoder_attention_backend="auto",
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )


def build_attention_arguments(
    attention_backend: str,
    prefill_attention_backend: str | None = None,
    decode_attention_backend: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        quantization=None,
        attention_backend=attention_backend,
        prefill_attention_backend=prefill_attention_backend,
        decode_attention_backend=decode_attention_backend,
    )


def test_whisper_accelerator_keeps_compile_and_async_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_builder, "current_platform", CUDAOmniPlatform())
    builder = whisper_builder()

    assert builder.generation_defaults(dtype="float16")["enable_torch_compile"] is True
    assert builder.extra_scheduler_kwargs()["enable_async_decode"] is True


def test_whisper_cpu_disables_compile_and_async_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_builder, "current_platform", CPUOmniPlatform())
    builder = whisper_builder()

    defaults = builder.generation_defaults(dtype="float16")
    assert "attention_backend" not in defaults
    assert defaults["enable_torch_compile"] is False
    assert builder.extra_scheduler_kwargs()["enable_async_decode"] is False


@pytest.mark.parametrize(
    "attention_arguments",
    [
        build_attention_arguments("triton"),
        build_attention_arguments("intel_amx", prefill_attention_backend="triton"),
        build_attention_arguments("intel_amx", decode_attention_backend="triton"),
    ],
)
def test_whisper_cpu_rejects_unsupported_attention_backend(
    attention_arguments: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="supports intel_amx or torch_native"):
        CPUOmniPlatform().apply_model_worker_backend_policy(
            attention_arguments,
            SimpleNamespace(quantization=None),
            "WhisperForConditionalGeneration",
        )


@pytest.mark.parametrize(
    "attention_arguments",
    [
        build_attention_arguments("intel_amx"),
        build_attention_arguments("torch_native"),
        build_attention_arguments(
            "triton",
            prefill_attention_backend="intel_amx",
            decode_attention_backend="torch_native",
        ),
    ],
)
def test_whisper_cpu_accepts_supported_attention_backends(
    attention_arguments: SimpleNamespace,
) -> None:
    effective_quantization = CPUOmniPlatform().apply_model_worker_backend_policy(
        attention_arguments,
        SimpleNamespace(quantization=None),
        "WhisperForConditionalGeneration",
    )

    assert effective_quantization is None
