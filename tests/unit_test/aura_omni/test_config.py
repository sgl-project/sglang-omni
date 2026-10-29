# SPDX-License-Identifier: Apache-2.0
"""AURA Omni pipeline config: the kwargs each stage's factory receives."""

from __future__ import annotations

from sglang_omni.models.aura_omni.config import AuraOmniPipelineConfig


def config() -> AuraOmniPipelineConfig:
    return AuraOmniPipelineConfig(model_path="aurateam/AURA")


def test_tts_preprocessing_shares_the_engine_frontend_by_default() -> None:
    assert config().stage_factory_kwargs("tts_preprocessing")["load_frontend"] is False


def test_tts_preprocessing_loads_its_own_frontend_in_its_own_process() -> None:
    pipeline = config()
    pipeline.stage_named("tts_preprocessing").process = "tts_frontend"

    assert pipeline.stage_factory_kwargs("tts_preprocessing")["load_frontend"] is True


def test_the_vocoder_reuses_the_engine_speech_tokenizer() -> None:
    kwargs = config().stage_factory_kwargs("vocoder")

    assert kwargs["shares_engine_speech_tokenizer"] is True


def test_a_vocoder_in_its_own_process_loads_no_shared_tokenizer() -> None:
    pipeline = config()
    pipeline.stage_named("vocoder").process = "vocoder_only"

    kwargs = pipeline.stage_factory_kwargs("vocoder")

    assert kwargs["shares_engine_speech_tokenizer"] is False


def test_a_vocoder_with_another_dtype_loads_no_shared_tokenizer() -> None:
    pipeline = config()
    pipeline.stage_named("vocoder").factory.dtype = "float32"

    kwargs = pipeline.stage_factory_kwargs("vocoder")

    assert kwargs["shares_engine_speech_tokenizer"] is False
