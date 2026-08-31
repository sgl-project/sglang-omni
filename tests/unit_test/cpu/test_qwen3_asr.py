# SPDX-License-Identifier: Apache-2.0
"""CPU policy for the Qwen3-ASR pipeline."""

from __future__ import annotations

import pytest

from sglang_omni import platforms
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor


def _resolved_factory_kwargs(config: Qwen3ASRPipelineConfig) -> dict[str, object]:
    stage = config.stage_named("asr")
    return apply_typed_stage_kwargs(
        create_sglang_qwen3_asr_executor,
        resolve_stage_factory_kwargs(stage, config),
        resolve_stage_typed_kwargs(stage),
        stage_name=stage.name,
    )


def test_cpu_disables_qwen3_asr_encoder_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_cpu", lambda: True)
    config = Qwen3ASRPipelineConfig(model_path="unused")

    assert _resolved_factory_kwargs(config)["enable_encoder_cuda_graph"] is False


def test_explicit_encoder_cuda_graph_setting_overrides_cpu_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_cpu", lambda: True)
    config = Qwen3ASRPipelineConfig(model_path="unused")
    config.stage_named("asr").factory.enable_encoder_cuda_graph = True

    assert _resolved_factory_kwargs(config)["enable_encoder_cuda_graph"] is True
