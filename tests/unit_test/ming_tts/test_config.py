# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _audio_decode_stage(raw_config: dict[str, Any]) -> dict[str, Any]:
    return next(
        stage for stage in raw_config["stages"] if stage["name"] == AUDIO_DECODE_STAGE
    )


def test_ming_tts_pipeline_requires_audio_decode_stream_edge() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    tts_engine = next(
        stage for stage in raw["stages"] if stage["name"] == TTS_ENGINE_STAGE
    )
    assert tts_engine["stream_to"] == [AUDIO_DECODE_STAGE]

    tts_engine["stream_to"] = []
    with pytest.raises(
        ValueError,
        match="tts_engine stream_to must include 'audio_decode'",
    ):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_audio_decode_defaults_are_full_sequence_and_serial() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    factory_args = _audio_decode_stage(raw)["factory_args"]

    assert "decode_mode" not in factory_args
    assert (
        factory_args["initial_chunk_patches"] == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )
    assert factory_args["steady_chunk_patches"] == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    assert factory_args["max_batch_size"] == 1
    assert factory_args["max_batch_wait_ms"] == 0


def test_ming_tts_example_config_uses_supported_audio_decode_contract() -> None:
    config_path = _REPO_ROOT / "examples/configs/ming_omni_tts.yaml"
    with config_path.open() as config_file:
        raw = yaml.safe_load(config_file)
    assert raw.pop("config_cls") == "MingTTSPipelineConfig"
    assert (
        _audio_decode_stage(raw)["factory_args"]["initial_chunk_patches"]
        == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )
    config = MingTTSPipelineConfig.model_validate(raw)
    factory_args = next(
        stage.factory_args
        for stage in config.stages
        if stage.name == AUDIO_DECODE_STAGE
    )

    assert "decode_mode" not in factory_args
    assert (
        factory_args["initial_chunk_patches"] == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )
    assert factory_args["steady_chunk_patches"] == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    assert factory_args["max_batch_size"] == 1
    assert factory_args["max_batch_wait_ms"] == 0


def test_ming_tts_missing_initial_cadence_uses_default() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory_args"].pop("initial_chunk_patches")

    config = MingTTSPipelineConfig.model_validate(raw)
    factory_args = next(
        stage.factory_args
        for stage in config.stages
        if stage.name == AUDIO_DECODE_STAGE
    )

    assert (
        factory_args["initial_chunk_patches"] == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
def test_ming_tts_rejects_cadence_runtime_override(field: str) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    raw["runtime_overrides"] = {AUDIO_DECODE_STAGE: {field: 1}}

    with pytest.raises(ValueError, match=f"{field} is owned by"):
        MingTTSPipelineConfig.model_validate(raw)


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_ming_tts_rejects_invalid_cadence(field: str, value: Any) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory_args"][field] = value

    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_accepts_initial_cadence_larger_than_steady() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    factory_args = _audio_decode_stage(raw)["factory_args"]
    factory_args["initial_chunk_patches"] = 4
    factory_args["steady_chunk_patches"] = 2

    config = MingTTSPipelineConfig.model_validate(raw)
    validated_args = next(
        stage.factory_args
        for stage in config.stages
        if stage.name == AUDIO_DECODE_STAGE
    )

    assert validated_args["initial_chunk_patches"] == 4
    assert validated_args["steady_chunk_patches"] == 2


@pytest.mark.parametrize("source", ["factory_args", "runtime_overrides"])
def test_ming_tts_rejects_legacy_audio_decode_mode(source: str) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    if source == "factory_args":
        _audio_decode_stage(raw)["factory_args"]["decode_mode"] = "chunked"
    else:
        raw["runtime_overrides"] = {AUDIO_DECODE_STAGE: {"decode_mode": "chunked"}}

    with pytest.raises(ValueError, match="no longer supports 'decode_mode'"):
        MingTTSPipelineConfig.model_validate(raw)


@pytest.mark.parametrize("source", ["factory_args", "runtime_overrides"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("max_batch_size", 2, "max_batch_size=1 only"),
        ("max_batch_wait_ms", 1, "max_batch_wait_ms=0 only"),
    ],
)
def test_ming_tts_rejects_unsupported_audio_decode_batch_config(
    source: str,
    field: str,
    value: int,
    error: str,
) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    if source == "factory_args":
        _audio_decode_stage(raw)["factory_args"][field] = value
    else:
        raw["runtime_overrides"] = {AUDIO_DECODE_STAGE: {field: value}}

    with pytest.raises(ValueError, match=error):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_pipeline_requires_audio_decode_stream_capability() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode = _audio_decode_stage(raw)
    assert audio_decode["can_accept_stream_before_payload"] is True

    audio_decode["can_accept_stream_before_payload"] = False
    with pytest.raises(
        ValueError,
        match="audio_decode must set can_accept_stream_before_payload=true",
    ):
        MingTTSPipelineConfig.model_validate(raw)
