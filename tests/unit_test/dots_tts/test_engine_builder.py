# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang_omni.models.dots_tts.engine_builder as engine_builder
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
    ("is_cuda", "is_rocm", "optimize", "max_running_requests", "should_compile"),
    [
        (True, False, True, 1, True),
        (False, True, True, 1, True),
        (False, False, True, 1, False),
        (True, False, True, 16, False),
        (True, False, False, 1, False),
    ],
)
def test_single_request_compile_policy_matches_platform(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    is_rocm: bool,
    optimize: bool,
    max_running_requests: int,
    should_compile: bool,
) -> None:
    configured: list[bool] = []
    monkeypatch.setattr(
        engine_builder,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: is_cuda, is_rocm=lambda: is_rocm),
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
