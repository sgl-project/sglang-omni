# SPDX-License-Identifier: Apache-2.0
"""Drama 3 pipeline configuration: topology, launch kwargs and the speech allowlist."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_CODEC_PATH,
    Drama3PipelineConfig,
)
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from tests.unit_test.pipeline.helpers import build_compiled_process_topology

VOCODER_WINDOWS = {
    "stream_stride": 4,
    "stream_followup_stride_ramp": (10, 12, 32),
    "stream_followup_stride": 64,
    "stream_overlap_tokens": 32,
    "stream_crossfade_samples": 0,
    "max_batch_size": 8,
    "max_batch_wait_ms": 2,
}


def launch_kwargs(config: Drama3PipelineConfig, stage_name: str) -> dict[str, object]:
    """What the launcher passes the stage's factory."""
    return resolve_stage_factory_args(config.stage_named(stage_name), config)


def test_every_process_on_the_shared_gpu_declares_its_memory() -> None:
    config = Drama3PipelineConfig(model_path="checkpoint")
    preprocessing, tts_engine, vocoder = config.stages

    topology = build_compiled_process_topology(config)

    assert topology.stage_to_process == {
        "preprocessing": "preprocessing",
        "tts_engine": "pipeline",
        "vocoder": "vocoder",
    }
    assert (preprocessing.total_reserve_bytes, preprocessing.enforce_total_reserve) == (
        6 * 1024**3,
        True,
    )
    assert (tts_engine.gpu_memory_fraction, vocoder.gpu_memory_fraction) == (0.70, 0.10)


def test_each_factory_gets_the_defaults_written_once() -> None:
    config = Drama3PipelineConfig(model_path="checkpoint")

    assert launch_kwargs(config, "preprocessing") == {
        "model_path": "checkpoint",
        "gpu_id": 0,
        "max_concurrency": 8,
        "reference_encode_dtype": "bfloat16",
        "max_reference_audio_seconds": 60.0,
        "max_prompt_rows": 8192,
        "codec_path": DEFAULT_CODEC_PATH,
    }
    assert launch_kwargs(config, "tts_engine") == {
        "model_path": "checkpoint",
        "gpu_id": 0,
        "total_gpu_memory_fraction": 0.70,
        "context_length": 16384,
        "max_running_requests": 16,
        "max_new_tokens": 2048,
        "max_prompt_rows": 8192,
        "first_frame_cuda_graphs": True,
        "first_frame_cuda_graph_max_requests": 16,
        "triton_sampling": True,
        "tower_fused_layers": True,
        "enable_async_decode": False,
        "async_decode_min_batch_size": 1,
        "stream_stride": 4,
        "stream_followup_stride_ramp": (10, 12, 32),
        "stream_followup_stride": 64,
    }
    assert launch_kwargs(config, "vocoder") == {
        "gpu_id": 0,
        "total_gpu_memory_fraction": 0.10,
        "codec_path": DEFAULT_CODEC_PATH,
        **VOCODER_WINDOWS,
        "window_cuda_graphs": True,
        "decode_dtype": "float16",
        "max_new_tokens": 2048,
    }


def test_preprocessing_follows_the_engine_cap_and_the_vocoder_codec() -> None:
    config = ConfigManager(Drama3PipelineConfig(model_path="checkpoint")).merge_config(
        [
            ("tts_engine.factory.max_prompt_rows", "4096"),
            ("vocoder.factory.codec_path", "local/codec.pth"),
        ]
    )
    assert isinstance(config, Drama3PipelineConfig)

    assert launch_kwargs(config, "preprocessing") == {
        "model_path": "checkpoint",
        "gpu_id": 0,
        "max_concurrency": 8,
        "reference_encode_dtype": "bfloat16",
        "max_reference_audio_seconds": 60.0,
        "max_prompt_rows": 4096,
        "codec_path": "local/codec.pth",
    }
    assert launch_kwargs(config, "tts_engine")["max_prompt_rows"] == 4096
    assert launch_kwargs(config, "vocoder")["codec_path"] == "local/codec.pth"


@pytest.mark.parametrize(
    ("field_name", "value", "owner"),
    [
        pytest.param("max_prompt_rows", "4096", "tts_engine", id="max-prompt-rows"),
        pytest.param("codec_path", "local/codec.pth", "vocoder", id="codec-path"),
    ],
)
def test_preprocessing_cannot_set_what_it_derives(
    field_name: str, value: str, owner: str
) -> None:
    manager = ConfigManager(Drama3PipelineConfig(model_path="checkpoint"))

    with pytest.raises(ValidationError) as rejection:
        manager.merge_config([(f"preprocessing.factory.{field_name}", value)])

    assert f"set stages.{owner}.factory.{field_name} instead" in str(rejection.value)


def test_the_engine_streams_by_the_vocoder_windows() -> None:
    config = ConfigManager(Drama3PipelineConfig(model_path="checkpoint")).merge_config(
        [
            ("vocoder.factory.stream_stride", "8"),
            ("vocoder.factory.stream_followup_stride_ramp", "[8, 16]"),
            ("vocoder.factory.stream_followup_stride", "48"),
        ]
    )
    assert isinstance(config, Drama3PipelineConfig)

    engine_kwargs = launch_kwargs(config, "tts_engine")
    vocoder_kwargs = launch_kwargs(config, "vocoder")
    for field_name, value in (
        ("stream_stride", 8),
        ("stream_followup_stride_ramp", (8, 16)),
        ("stream_followup_stride", 48),
    ):
        assert engine_kwargs[field_name] == vocoder_kwargs[field_name] == value


def test_the_vocoder_sizes_its_decodes_by_the_engine_cap_on_new_frames() -> None:
    config = ConfigManager(Drama3PipelineConfig(model_path="checkpoint")).merge_config(
        [("tts_engine.factory.max_new_tokens", "1024")]
    )
    assert isinstance(config, Drama3PipelineConfig)

    assert launch_kwargs(config, "vocoder")["max_new_tokens"] == 1024


def test_the_vocoder_cannot_set_the_engine_cap_on_new_frames() -> None:
    manager = ConfigManager(Drama3PipelineConfig(model_path="checkpoint"))

    with pytest.raises(ValidationError) as rejection:
        manager.merge_config([("vocoder.factory.max_new_tokens", "1024")])

    assert "set stages.tts_engine.factory.max_new_tokens instead" in str(
        rejection.value
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("stream_stride", "8", id="stride"),
        pytest.param("stream_followup_stride_ramp", "[8, 16]", id="ramp"),
        pytest.param("stream_followup_stride", "48", id="followup-stride"),
    ],
)
def test_the_engine_cannot_set_the_vocoder_windows(field_name: str, value: str) -> None:
    manager = ConfigManager(Drama3PipelineConfig(model_path="checkpoint"))

    with pytest.raises(ValidationError) as rejection:
        manager.merge_config([(f"tts_engine.factory.{field_name}", value)])

    assert f"set stages.vocoder.factory.{field_name} instead" in str(rejection.value)


def speech_validator() -> SpeechRequestValidator:
    return SpeechRequestValidator(
        default_model="drama3",
        supported_speech_fields=Drama3PipelineConfig.supported_speech_fields,
    )


def test_every_supported_speech_field_is_accepted() -> None:
    validator = speech_validator()

    validator.parse_raw_request(
        {
            "model": "drama3",
            "input": "hello",
            "voice": "default",
            "response_format": "pcm",
            "speed": 1.0,
            "stream": True,
            "stream_format": "sse",
            "references": [{"text": "take", "vq_codes": [[0]] * 10}],
            "ref_audio": "data:audio/wav;base64,AAAA",
            "ref_text": "take",
            "task_type": "Base",
            "max_new_tokens": 16,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 30,
            "repetition_penalty": 1.0,
        }
    )
    validator.parse_raw_request({"input": "hello", "speaker": "default"})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("instructions", "whisper"),
        ("language", "English"),
        ("seed", 0),
        ("stage_params", {}),
        ("x_vector_only_mode", False),
        ("stream_codec_output", False),
        ("suppress_bootstrap_silence", False),
        ("token_count", 1),
        ("duration_tokens", 1),
        ("initial_codec_chunk_frames", 0),
        ("stage_sampling", {}),
        ("infill", None),
    ],
)
def test_speech_fields_the_model_cannot_honor_are_refused(
    field_name: str, value: object
) -> None:
    with pytest.raises(SpeechAPIError) as rejection:
        speech_validator().parse_raw_request({"input": "hello", field_name: value})

    assert rejection.value.param == field_name
