# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.dots_tts.engine_builder as engine_builder
import sglang_omni.models.dots_tts.stages as stages
from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder
from sglang_omni.models.dots_tts.stages import (
    create_sglang_latent_engine_executor,
    create_vocoder_executor,
)
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


def test_dots_engine_uses_shared_tts_builder() -> None:
    builder = DotsTTSEngineBuilder(optimize=True)

    assert isinstance(builder, TtsEngineBuilder)
    assert builder.optimize is True
    assert builder.generation_defaults(dtype="bfloat16")["max_running_requests"] == 16


def test_dots_engine_accepts_continuous_batching() -> None:
    DotsTTSEngineBuilder().adjust_overrides({"tp_size": 1, "max_running_requests": 16})


@pytest.mark.parametrize(
    ("is_npu", "optimize", "max_running_requests", "should_compile"),
    [
        (False, True, 1, True),
        (True, True, 1, False),
        (False, True, 16, False),
        (False, False, 1, False),
    ],
)
def test_single_request_compile_policy_matches_platform(
    monkeypatch: pytest.MonkeyPatch,
    is_npu: bool,
    optimize: bool,
    max_running_requests: int,
    should_compile: bool,
) -> None:
    configured: list[bool] = []
    monkeypatch.setattr(
        engine_builder,
        "current_platform",
        SimpleNamespace(is_npu=lambda: is_npu),
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.model_override_base.resolved_view",
        lambda _server_args: SimpleNamespace(max_running_requests=max_running_requests),
    )
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.stages.configure_optimized_kernels",
        lambda: configured.append(True),
    )

    DotsTTSEngineBuilder(optimize=optimize).customize_server_args(None)

    assert bool(configured) is should_compile


def test_accelerator_only_factories_reject_a_cpu_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sglang_omni.platforms.current_platform", CPUOmniPlatform())
    for factory in (create_sglang_latent_engine_executor, create_vocoder_executor):
        for device in (None, "cpu"):
            with pytest.raises(RuntimeError, match="requires an accelerator"):
                factory("stub", device=device)


@pytest.mark.parametrize(
    ("device_type", "precision", "expected_dtype", "error"),
    [
        ("npu", "float32", torch.float32, None),
        ("npu", "float16", torch.float16, None),
        ("cuda", "float32", None, None),
        ("cuda", "float16", None, "requires an NPU vocoder worker"),
        ("npu", "float64", None, "Unsupported NPU decoder precision"),
    ],
)
def test_vocoder_decoder_precision_validation(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    precision: str,
    expected_dtype: torch.dtype | None,
    error: str | None,
) -> None:
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda *_args: SimpleNamespace(type=device_type),
    )
    monkeypatch.setattr(
        stages,
        "current_platform",
        SimpleNamespace(is_npu=lambda: device_type == "npu", device_type=device_type),
    )

    class CodecReached(Exception):
        pass

    def load_codec(model_path: str, *, device: str, decoder_dtype: torch.dtype | None):
        assert model_path == "stub"
        assert decoder_dtype == expected_dtype
        raise CodecReached

    monkeypatch.setattr(stages, "load_dots_audio_codec", load_codec)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            create_vocoder_executor("stub", npu_decoder_precision=precision)
    else:
        with pytest.raises(CodecReached):
            create_vocoder_executor("stub", npu_decoder_precision=precision)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tp_size": 2, "max_running_requests": 16}, "does not implement TP"),
        (
            {
                "tp_size": 1,
                "max_running_requests": 16,
                "enable_torch_compile": True,
            },
            "backbone compile is disabled",
        ),
    ],
)
def test_dots_engine_rejects_unsupported_generation_modes(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTTSEngineBuilder().adjust_overrides(overrides)
