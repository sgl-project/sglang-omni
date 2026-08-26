# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS declares its XPU attention backend through SGLang's override registry.

The provider runs inside ``ServerArgs.__post_init__``, before any pipeline stage
exists, so it is exercised here against the registry rather than through the
pipeline.
"""

from __future__ import annotations

import pytest
from sglang.srt.arg_groups.overrides import collect_model_override_declarations

from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from tests.unit_test.fakes import FakeServerArgs


def _server_args(device: str, attention_backend: str | None) -> FakeServerArgs:
    return FakeServerArgs(
        device=device,
        attention_backend=attention_backend,
        is_attention_backend_not_set=lambda: attention_backend is None,
    )


@pytest.mark.parametrize(
    ("device", "attention_backend", "expected"),
    [
        ("xpu", None, {"attention_backend": "intel_xpu"}),
        ("xpu", "triton", {}),
        ("cuda", None, {}),
    ],
)
def test_only_an_unset_backend_on_xpu_resolves_to_intel_xpu(
    device: str,
    attention_backend: str | None,
    expected: dict[str, str],
) -> None:
    """An operator's explicit backend must survive, and CUDA must stay untouched."""
    assert (
        qwen3_stages._qwen3_tts_overrides(
            _server_args(device, attention_backend), object()
        )
        == expected
    )


def test_the_provider_is_registered_under_the_architecture_the_config_declares() -> (
    None
):
    """Guards against the arch key in stages.py drifting from the pipeline config."""
    declarations = collect_model_override_declarations(
        Qwen3TTSPipelineConfig.architecture, _server_args("xpu", None), object()
    )

    assert {"attention_backend": "intel_xpu"} in [fields for _, fields in declarations]
