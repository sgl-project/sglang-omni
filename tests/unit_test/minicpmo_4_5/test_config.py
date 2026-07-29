from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.models.minicpmo_4_5 import CAPABILITIES
from sglang_omni.models.minicpmo_4_5.config import (
    MINICPMO_DUPLEX_STAGE,
    MiniCPMO45PipelineConfig,
)


def test_pipeline_config_builds_one_native_sglang_stage() -> None:
    config = MiniCPMO45PipelineConfig(
        model_path="openbmb/MiniCPM-o-4_5",
        revision="checkpoint-sha",
        ref_audio_path="/voices/ref.wav",
        prompt_wav_path="/voices/tts-prompt.wav",
        context_length=32768,
        max_pending_units=3,
        max_pending_commands=9,
        session_ttl_s=45,
        server_args_overrides={"mem_fraction_static": 0.72},
    )

    assert config.architecture == "MiniCPMO"
    assert config.entry_stage == MINICPMO_DUPLEX_STAGE
    assert len(config.stages) == 1
    stage = config.stages[0]
    assert stage.process == MINICPMO_DUPLEX_STAGE
    assert stage.terminal is True
    assert stage.gpu == 0
    assert stage.tp_size == 1
    assert stage.factory_args == {
        "model_path": config.model_path,
        "revision": "checkpoint-sha",
        "ref_audio_path": "/voices/ref.wav",
        "prompt_wav_path": "/voices/tts-prompt.wav",
        "dtype": "bfloat16",
        "context_length": 32768,
        "max_pending_units": 3,
        "max_pending_commands": 9,
        "session_ttl_s": 45.0,
        "max_sessions": 1,
        "server_args_overrides": {"mem_fraction_static": 0.72},
        "duplex_sampling": config.duplex_sampling.model_dump(),
    }
    assert stage.factory_args["duplex_sampling"]["listen_top_k"] is None


def test_pipeline_config_has_no_demo_subprocess_settings() -> None:
    fields = MiniCPMO45PipelineConfig.model_fields
    assert "runtime_python" not in fields
    assert "demo_path" not in fields
    assert "pt_path" not in fields
    assert "runtime_backend" not in fields
    assert "runtime_timeout_s" not in fields


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sessions": 2},
        {"max_pending_units": 4, "max_pending_commands": 3},
        {"revision": "   "},
    ],
)
def test_pipeline_config_rejects_unsupported_session_topology(kwargs) -> None:
    with pytest.raises(ValidationError):
        MiniCPMO45PipelineConfig(model_path="model", **kwargs)


def test_pipeline_config_rejects_unimplemented_implicit_listen_speak_mode() -> None:
    with pytest.raises(ValidationError):
        MiniCPMO45PipelineConfig(
            model_path="model",
            duplex_sampling={"ls_mode": "implicit"},
        )


def test_pipeline_config_matches_demo_tts_sampling_defaults() -> None:
    config = MiniCPMO45PipelineConfig(model_path="model")

    assert config.duplex_sampling.tts_temperature == pytest.approx(0.8)
    assert config.duplex_sampling.tts_repetition_penalty == pytest.approx(1.05)


@pytest.mark.parametrize(
    "duplex_sampling",
    [
        {"tts_temperature": 0.0},
        {"tts_repetition_penalty": 0.0},
    ],
)
def test_pipeline_config_rejects_invalid_tts_sampling(duplex_sampling) -> None:
    with pytest.raises(ValidationError):
        MiniCPMO45PipelineConfig(
            model_path="model",
            duplex_sampling=duplex_sampling,
        )


def test_pipeline_config_rejects_empty_model_path() -> None:
    with pytest.raises(ValidationError):
        MiniCPMO45PipelineConfig(model_path="   ")


def test_pipeline_config_rejects_non_native_custom_stage() -> None:
    from sglang_omni.config import StageConfig

    with pytest.raises(ValidationError, match="must be named"):
        MiniCPMO45PipelineConfig(
            model_path="model",
            entry_stage="legacy_actor",
            stages=[
                StageConfig(
                    name="legacy_actor",
                    process="legacy_actor",
                    factory="somewhere.create_actor",
                    gpu=0,
                    terminal=True,
                )
            ],
        )


def test_pipeline_config_rejects_non_native_factory() -> None:
    from sglang_omni.config import StageConfig

    with pytest.raises(ValidationError, match="native SGLang factory"):
        MiniCPMO45PipelineConfig(
            model_path="model",
            stages=[
                StageConfig(
                    name=MINICPMO_DUPLEX_STAGE,
                    process=MINICPMO_DUPLEX_STAGE,
                    factory="legacy.create_actor",
                    gpu=0,
                    terminal=True,
                )
            ],
        )


def test_model_declares_native_realtime_capabilities() -> None:
    assert CAPABILITIES.supports_native_duplex is True
    assert CAPABILITIES.supports_realtime_audio_output is True
    assert CAPABILITIES.supports_realtime_video_input is True
    assert CAPABILITIES.supports_streaming_vocoder is True
