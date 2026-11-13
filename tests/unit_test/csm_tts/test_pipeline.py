# SPDX-License-Identifier: Apache-2.0
"""Pipeline / registry contracts for CSM-1B."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.csm_tts.config import CsmTtsPipelineConfig
from sglang_omni.models.csm_tts.engine_builder import CsmTtsEngineBuilder
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_csm_pipeline_is_four_stage_and_streams_to_vocoder() -> None:
    config = CsmTtsPipelineConfig(model_path="fake-csm")
    stages = {stage.name: stage for stage in config.stages}

    assert list(stages) == [
        "preprocessing",
        "audio_encoder",
        "tts_engine",
        "vocoder",
    ]
    assert stages["tts_engine"].stream_to == ["vocoder"]
    assert stages["vocoder"].can_accept_stream_before_payload is True
    assert stages["vocoder"].terminal is True
    assert config.architecture == "CsmForConditionalGeneration"
    assert CsmTtsPipelineConfig.stage_config_cls("tts_engine").engine_stage
    assert config.supports_uploaded_voice_references() is True
    assert config.requires_uploaded_voice_for_named_voice() is False


def test_csm_is_registered_for_hf_and_alias_architectures() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("CsmForConditionalGeneration")
        is CsmTtsPipelineConfig
    )
    assert PIPELINE_CONFIG_REGISTRY.get_config("CsmTTS") is CsmTtsPipelineConfig


def test_csm_example_yaml_resolves() -> None:
    manager = ConfigManager.from_file("examples/configs/csm_1b.yaml")
    config = manager.config
    assert isinstance(config, CsmTtsPipelineConfig)
    stages = {stage.name: stage for stage in config.stages}
    assert stages["tts_engine"].factory.max_new_tokens == 125
    assert stages["tts_engine"].factory.enable_async_decode is False
    engine = stages["tts_engine"].engine
    assert engine is not None
    assert engine.max_running_requests == 4
    assert engine.disable_cuda_graph is True
    assert engine.model_dump().get("disable_radix_cache") is True
    assert stages["vocoder"].factory.dtype == "float32"


def test_csm_generation_defaults_keep_eager_baseline() -> None:
    defaults = _csm_builder().generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is True
    assert defaults["disable_radix_cache"] is True
    assert defaults["max_running_requests"] == 4


def test_csm_engine_lifecycle_callbacks_require_model() -> None:
    builder = _csm_builder()
    with pytest.raises(AssertionError):
        builder.make_abort_callback()
    with pytest.raises(AssertionError):
        builder.make_request_finished_callback()

    reset_calls: list[str] = []
    builder.model = SimpleNamespace(reset_request=reset_calls.append)
    builder.make_abort_callback()("req-1")
    builder.make_request_finished_callback()("req-2")
    assert reset_calls == ["req-1", "req-2"]


def _csm_builder() -> CsmTtsEngineBuilder:
    return CsmTtsEngineBuilder(
        max_new_tokens=125,
        max_running_requests=4,
        cuda_graph_max_bs=4,
        enable_async_decode=False,
        async_decode_min_batch_size=2,
    )
