# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)


def audio_decode_stage(raw_config: dict[str, Any]) -> dict[str, Any]:
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


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("value", [True, 0])
def test_ming_tts_rejects_invalid_cadence(field: str, value: Any) -> None:
    """The positive-integer rule is a static declaration on the typed group."""
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode_stage(raw)["factory"][field] = value

    with pytest.raises(ValueError, match=field):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_accepts_initial_cadence_larger_than_steady() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    factory = audio_decode_stage(raw)["factory"]
    factory["initial_chunk_patches"] = 4
    factory["steady_chunk_patches"] = 2

    config = MingTTSPipelineConfig.model_validate(raw)
    factory = config.stage_named(AUDIO_DECODE_STAGE).factory

    assert factory.initial_chunk_patches == 4
    assert factory.steady_chunk_patches == 2


def test_ming_tts_rejects_legacy_audio_decode_mode() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode_stage(raw)["factory"]["decode_mode"] = "chunked"

    with pytest.raises(ValueError, match="no longer supports 'decode_mode'"):
        MingTTSPipelineConfig.model_validate(raw)


@pytest.mark.parametrize("value", [True, 0])
def test_ming_tts_rejects_invalid_audio_decode_stream_slots(value: Any) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode_stage(raw)["factory"]["stream_slots"] = value

    with pytest.raises(ValueError, match="stream_slots"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_rejects_cross_request_audio_decode_batching() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode_stage(raw)["factory"]["max_batch_size"] = 2

    with pytest.raises(ValueError, match="max_batch_size=1 only"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_rejects_nonzero_audio_decode_batch_wait() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode_stage(raw)["factory"]["max_batch_wait_ms"] = 1

    with pytest.raises(ValueError, match="max_batch_wait_ms=0 only"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_pipeline_requires_audio_decode_stream_capability() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode = audio_decode_stage(raw)
    assert audio_decode["can_accept_stream_before_payload"] is True

    audio_decode["can_accept_stream_before_payload"] = False
    with pytest.raises(
        ValueError,
        match="audio_decode must set can_accept_stream_before_payload=true",
    ):
        MingTTSPipelineConfig.model_validate(raw)
