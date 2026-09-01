# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.sources import sources_from_config_file
from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STREAM_SLOTS,
    MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH,
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
    factory = _audio_decode_stage(raw)["factory"]

    assert "decode_mode" not in factory
    assert factory["initial_chunk_patches"] == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    assert factory["steady_chunk_patches"] == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    assert factory["streaming_cuda_graph"] is MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH
    assert factory["stream_slots"] == MING_TTS_DEFAULT_STREAM_SLOTS
    assert factory["max_batch_size"] == 1
    assert factory["max_batch_wait_ms"] == 0


def test_ming_tts_example_config_uses_supported_audio_decode_contract() -> None:
    config_path = _REPO_ROOT / "examples/configs/ming_omni_tts.yaml"
    config, patches = sources_from_config_file(str(config_path))
    config = ConfigManager(config).merge_config([], extra_patches=patches)
    assert isinstance(config, MingTTSPipelineConfig)

    audio_decode = config.stage_named(AUDIO_DECODE_STAGE)
    factory = audio_decode.factory

    assert "decode_mode" not in (factory.model_extra or {})
    assert factory.initial_chunk_patches == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    assert factory.steady_chunk_patches == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    assert factory.streaming_cuda_graph is True
    assert factory.stream_slots == 8
    assert factory.max_batch_size == 1
    assert factory.max_batch_wait_ms == 0


def test_ming_tts_missing_initial_cadence_remains_unset() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"].pop("initial_chunk_patches")

    config = MingTTSPipelineConfig.model_validate(raw)
    factory = config.stage_named(AUDIO_DECODE_STAGE).factory

    assert factory.initial_chunk_patches is None


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_ming_tts_rejects_invalid_cadence(field: str, value: Any) -> None:
    """The positive-integer rule is a static declaration on the typed group."""
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"][field] = value

    with pytest.raises(ValueError, match=field):
        MingTTSPipelineConfig.model_validate(raw)


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("text", ["0", "true", "1.5"])
def test_ming_tts_rejects_invalid_cadence_via_dotted_override(
    field: str, text: str
) -> None:
    """The same rule holds when the write arrives through the merge; a
    boolean is refused by the lossless conversion rule on the way in."""
    config = MingTTSPipelineConfig(model_path="fake-model")
    with pytest.raises(ValueError, match=field):
        ConfigManager(config).merge_config(
            [(f"{AUDIO_DECODE_STAGE}.factory.{field}", text)]
        )


def test_ming_tts_accepts_initial_cadence_larger_than_steady() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    factory = _audio_decode_stage(raw)["factory"]
    factory["initial_chunk_patches"] = 4
    factory["steady_chunk_patches"] = 2

    config = MingTTSPipelineConfig.model_validate(raw)
    factory = config.stage_named(AUDIO_DECODE_STAGE).factory

    assert factory.initial_chunk_patches == 4
    assert factory.steady_chunk_patches == 2


def test_ming_tts_rejects_legacy_audio_decode_mode() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"]["decode_mode"] = "chunked"

    with pytest.raises(ValueError, match="no longer supports 'decode_mode'"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_accepts_positive_audio_decode_stream_slots() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"]["stream_slots"] = 2

    MingTTSPipelineConfig.model_validate(raw)


@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_ming_tts_rejects_invalid_audio_decode_stream_slots(value: Any) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"]["stream_slots"] = value

    with pytest.raises(ValueError, match="stream_slots"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_rejects_cross_request_audio_decode_batching() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"]["max_batch_size"] = 2

    with pytest.raises(ValueError, match="max_batch_size=1 only"):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_rejects_nonzero_audio_decode_batch_wait() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    _audio_decode_stage(raw)["factory"]["max_batch_wait_ms"] = 1

    with pytest.raises(ValueError, match="max_batch_wait_ms=0 only"):
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
