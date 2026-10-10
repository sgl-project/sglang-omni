# SPDX-License-Identifier: Apache-2.0
"""Serving-layer validation for the precomputed speaker_embedding field."""

from __future__ import annotations

import pytest

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import (
    PreparedSpeechReferences,
    SpeechRequestValidator,
    build_tts_params,
)

DIM = 1024
EMBEDDING = [0.01] * DIM
BASE_REQUEST = {"model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "input": "hello"}


def make_validator(**overrides: object) -> SpeechRequestValidator:
    kwargs: dict[str, object] = {
        "default_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "speaker_embedding_dim": DIM,
    }
    kwargs.update(overrides)
    return SpeechRequestValidator(**kwargs)


def test_valid_speaker_embedding_lowers_into_tts_params() -> None:
    validator = make_validator()
    request = validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    tts_params = build_tts_params(request)
    assert tts_params["speaker_embedding"] == EMBEDDING


def test_speaker_embedding_requires_non_empty_list() -> None:
    validator = make_validator()
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request({**BASE_REQUEST, "speaker_embedding": []})
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "speaker_embedding"


def test_speaker_embedding_rejects_non_number_entries() -> None:
    validator = make_validator()
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request({**BASE_REQUEST, "speaker_embedding": [0.01, "x"]})
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "speaker_embedding"


def test_speaker_embedding_rejects_wrong_dimension() -> None:
    validator = make_validator()
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING[:-1]})
    assert exc_info.value.status_code == 400
    assert str(DIM) in exc_info.value.message


def test_speaker_embedding_rejects_explicit_x_vector_false() -> None:
    validator = make_validator()
    payload = {
        **BASE_REQUEST,
        "speaker_embedding": EMBEDDING,
        "x_vector_only_mode": False,
    }
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request(payload)
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "x_vector_only_mode"


def test_speaker_embedding_allows_explicit_x_vector_true() -> None:
    validator = make_validator()
    payload = {
        **BASE_REQUEST,
        "speaker_embedding": EMBEDDING,
        "x_vector_only_mode": True,
    }
    request = validator.parse_request(payload)
    assert build_tts_params(request)["x_vector_only_mode"] is True


def test_speaker_embedding_rejected_on_unsupported_model() -> None:
    validator = make_validator(speaker_embedding_dim=None)
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "speaker_embedding"


def test_speaker_embedding_rejects_non_base_task_type() -> None:
    validator = make_validator()
    payload = {
        **BASE_REQUEST,
        "speaker_embedding": EMBEDDING,
        "task_type": "CustomVoice",
    }
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_request(payload)
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "task_type"


def test_speaker_embedding_forces_base_task_in_tts_params() -> None:
    validator = make_validator()
    request = validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    tts_params = build_tts_params(request)
    assert tts_params["task_type"] == "Base"


def test_speaker_embedding_conflicts_with_ref_audio() -> None:
    validator = make_validator()
    request = validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    request = request.model_copy(update={"ref_audio": "/tmp/ref.wav"})
    prepared = PreparedSpeechReferences(request_updates={}, reference_descriptors=[])
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.validate_speech_references(request, prepared)
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "speaker_embedding"


def test_speaker_embedding_conflicts_with_references() -> None:
    validator = make_validator()
    request = validator.parse_request({**BASE_REQUEST, "speaker_embedding": EMBEDDING})
    request = request.model_copy(update={"references": [{"text": "transcript"}]})
    prepared = PreparedSpeechReferences(request_updates={}, reference_descriptors=[])
    with pytest.raises(SpeechAPIError) as exc_info:
        validator.validate_speech_references(request, prepared)
    assert exc_info.value.status_code == 400


def test_base_pipeline_config_rejects_speaker_embedding() -> None:
    config = PipelineConfig(
        model_path="/tmp/unused",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory_path="x:y",
                terminal=True,
            )
        ],
    )
    assert config.resolve_speaker_embedding_dim() is None


def test_qwen3_tts_resolves_dim_from_checkpoint_config(
    tmp_path: object,
) -> None:
    import json
    from pathlib import Path

    checkpoint = Path(tmp_path) / "qwen3-tts-base"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_text(
        json.dumps(
            {
                "tts_model_type": "base",
                "speaker_encoder_config": {"enc_dim": 1024},
            }
        ),
        encoding="utf-8",
    )
    config = Qwen3TTSPipelineConfig(model_path=str(checkpoint))
    assert config.resolve_speaker_embedding_dim() == 1024


def test_qwen3_tts_custom_voice_checkpoint_rejects_speaker_embedding(
    tmp_path: object,
) -> None:
    import json
    from pathlib import Path

    checkpoint = Path(tmp_path) / "qwen3-tts-custom-voice"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_text(
        json.dumps({"tts_model_type": "custom_voice"}),
        encoding="utf-8",
    )
    config = Qwen3TTSPipelineConfig(model_path=str(checkpoint))
    assert config.resolve_speaker_embedding_dim() is None
