# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS declares its attention backend through SGLang's override registry.

The provider runs inside ``ServerArgs.__post_init__``, before any pipeline stage
exists, so it is exercised against the registry rather than through the pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.arg_groups.overrides import collect_model_override_declarations

from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform


def _server_args(device: str, attention_backend: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        attention_backend=attention_backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )


@pytest.mark.parametrize(
    ("platform", "device", "attention_backend", "expected"),
    [
        (XPUOmniPlatform, "xpu", None, {"attention_backend": "intel_xpu"}),
        (XPUOmniPlatform, "xpu", "triton", {}),
        (XPUOmniPlatform, "cpu", None, {}),
        (CUDAOmniPlatform, "cuda", None, {}),
    ],
)
def test_the_platform_hook_decides_the_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
    platform: type[OmniPlatform],
    device: str,
    attention_backend: str | None,
    expected: dict[str, str],
) -> None:
    """An operator's explicit backend, a --device pinned off the platform, and a
    platform that declines all leave the backend alone."""
    monkeypatch.setattr(qwen3_stages, "current_platform", platform())

    assert (
        qwen3_stages._qwen3_tts_overrides(
            _server_args(device, attention_backend), object()
        )
        == expected
    )


def test_the_provider_is_registered_under_the_architecture_the_config_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards against the arch key in stages.py drifting from the pipeline config."""
    monkeypatch.setattr(qwen3_stages, "current_platform", XPUOmniPlatform())

    declarations = collect_model_override_declarations(
        Qwen3TTSPipelineConfig.architecture, _server_args("xpu", None), object()
    )

    assert {"attention_backend": "intel_xpu"} in [fields for _, fields in declarations]
