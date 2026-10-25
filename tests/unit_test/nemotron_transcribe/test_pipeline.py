# SPDX-License-Identifier: Apache-2.0
"""The Nemotron Transcribe pipeline registers and keeps whole-file audio as one request."""

from __future__ import annotations

import inspect

import pytest

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.nemotron_transcribe.config import (
    ARCHITECTURE,
    NemotronTranscribeFactoryArgs,
    NemotronTranscribePipelineConfig,
)
from sglang_omni.models.nemotron_transcribe.hf_config import num_audio_tokens
from sglang_omni.models.nemotron_transcribe.stages import (
    create_nemotron_transcribe_executor,
)
from sglang_omni.models.registry import import_pipeline_configs


def test_pipeline_is_one_whole_file_asr_stage() -> None:
    config = NemotronTranscribePipelineConfig(model_path="unused")

    assert config.allow_audio_chunking is False
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.stages[0].terminal is True


def test_architecture_registers_with_the_model_registry() -> None:
    configs = import_pipeline_configs("sglang_omni.models", "config")
    assert configs[ARCHITECTURE] is NemotronTranscribePipelineConfig


def test_audio_placeholder_counts_cover_one_hour_uploads() -> None:
    estimator = {
        "preprocessor": {"n_fft": 512, "hop_length": 160, "stft_pad_amount": 256},
        "subsampling": {"type": "feature_stacking", "factor": 8},
    }
    assert num_audio_tokens(16000, estimator) == 13
    assert num_audio_tokens(16000 * 3600, estimator) == 45000
    assert num_audio_tokens(1, estimator) == 1


def test_factory_signature_accepts_the_typed_pipeline_defaults() -> None:
    configuration = NemotronTranscribePipelineConfig(model_path="unused")
    typed_kwargs = resolve_stage_typed_kwargs(configuration.stages[0])
    inspect.signature(create_nemotron_transcribe_executor).bind(
        model_path="unused", **typed_kwargs
    )
    assert typed_kwargs["server_args_overrides"]["enable_torch_compile"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder_max_batch_size", 0),
        ("pre_lm_cache_max_entries", 0),
        ("pre_lm_max_batch_wait_ms", -1),
        ("chunked_prefill_size", 0),
        ("max_audio_seconds", -1),
        ("mm_embedding_cache_size_bytes", -1),
        ("perception_compile_mode", "reduce-overhead"),
    ],
)
def test_invalid_feature_options_are_rejected(field: str, value: int | str) -> None:
    with pytest.raises(ValueError):
        NemotronTranscribeFactoryArgs(**{field: value})
