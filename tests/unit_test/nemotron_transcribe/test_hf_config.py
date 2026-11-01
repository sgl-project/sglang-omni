# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration and placeholder geometry conversion."""

import math
from pathlib import Path

import pytest
from sglang.srt.configs.nemotron_h import NemotronHConfig
from transformers import AutoConfig

from sglang_omni.models.nemotron_transcribe.hf_config import (
    NemotronTranscribeConfig,
    num_audio_tokens,
    register_nemotron_transcribe_hf_config,
)


def raw_checkpoint_config() -> dict[str, object]:
    return {
        "llm_config": {
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
            "num_nextn_predict_layers": 2,
        },
        "eos_token_id": [11],
        "perception": {},
        "pe_encoder_config": {},
    }


def test_conversion_retains_hybrid_decoder_identity() -> None:
    configuration = NemotronTranscribeConfig.from_nemo_dict(raw_checkpoint_config())
    assert isinstance(configuration, NemotronHConfig)
    assert configuration.architectures == ["NemotronHForCausalLM"]
    assert configuration.num_nextn_predict_layers == 0
    assert configuration.eos_token_id == 11
    assert configuration.time_step_limit[0] == 0.0
    assert math.isinf(configuration.time_step_limit[1])


def test_hf_config_round_trip(tmp_path: Path) -> None:
    register_nemotron_transcribe_hf_config()
    configuration = NemotronTranscribeConfig.from_nemo_dict(raw_checkpoint_config())
    configuration.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(restored, NemotronTranscribeConfig)
    assert restored.architectures == configuration.architectures
    assert restored.audio_token_id == configuration.audio_token_id
    assert restored.eos_token_id == configuration.eos_token_id


@pytest.mark.parametrize(
    "configuration_field,value",
    [
        ("llm_config", None),
        ("perception", []),
        ("pe_encoder_config", None),
        ("audio_token_estimator", 1),
    ],
)
def test_malformed_checkpoint_sections_are_rejected(
    configuration_field: str, value: object
) -> None:
    configuration = raw_checkpoint_config()
    configuration[configuration_field] = value
    with pytest.raises(TypeError):
        NemotronTranscribeConfig.from_nemo_dict(configuration)


@pytest.mark.parametrize(
    "estimator",
    [
        {"preprocessor": {"hop_length": 0}},
        {"preprocessor": {"n_fft": -1}},
        {"preprocessor": {"stft_pad_amount": -1}},
        {"subsampling": {"factor": 0}},
    ],
)
def test_invalid_audio_geometry_is_rejected(estimator: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        num_audio_tokens(16000, estimator)


def test_nondefault_mamba_timestep_limits_are_preserved() -> None:
    configuration = raw_checkpoint_config()
    configuration["llm_config"]["time_step_limit"] = [0.02, 0.5]
    converted = NemotronTranscribeConfig.from_nemo_dict(configuration)
    assert converted.time_step_limit == (0.02, 0.5)
