# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingOmniStreamingSpeechPipelineConfig wiring."""

from __future__ import annotations

import pytest

from sglang_omni.models.ming_omni.config import MingOmniStreamingSpeechPipelineConfig
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    DECODE_STAGE,
    SEGMENTER_STAGE,
    TALKER_STREAM_STAGE,
    THINKER_STAGE,
)


def _stage(config, name):
    return next(s for s in config.stages if s.name == name)


def _config_data_with_thinker_tp(*, talker_gpu: int) -> dict:
    data = MingOmniStreamingSpeechPipelineConfig(model_path="dummy").model_dump()
    for stage in data["stages"]:
        if stage["name"] == THINKER_STAGE:
            stage["gpu"] = [0, 1]
            stage["tp_size"] = 2
        elif stage["name"] == TALKER_STREAM_STAGE:
            stage["gpu"] = talker_gpu
    return data


def test_streaming_speech_topology_wires_segmenter_between_thinker_and_talker():
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    names = [s.name for s in config.stages]
    assert SEGMENTER_STAGE in names
    assert TALKER_STREAM_STAGE in names
    # Old non-streaming talker must NOT be present.
    assert "talker" not in names


def test_streaming_thinker_fans_out_to_decode_and_segmenter():
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, THINKER_STAGE)
    decode = _stage(config, DECODE_STAGE)
    assert thinker.next == [DECODE_STAGE, SEGMENTER_STAGE]
    assert thinker.stream_to == [DECODE_STAGE, SEGMENTER_STAGE]
    assert thinker.factory.enable_streaming_tts is True
    assert decode.can_accept_stream_before_payload is True


def test_segmenter_routes_to_talker_stream_and_accepts_pre_payload_streams():
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    seg = _stage(config, SEGMENTER_STAGE)
    assert seg.next == TALKER_STREAM_STAGE
    assert seg.stream_to == [TALKER_STREAM_STAGE]
    assert seg.can_accept_stream_before_payload is True


def test_talker_stream_is_terminal_and_accepts_pre_payload_streams():
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    talker = _stage(config, TALKER_STREAM_STAGE)
    assert talker.terminal is True
    assert talker.can_accept_stream_before_payload is True


def test_streaming_speech_rejects_talker_gpu_in_thinker_tp_range():
    data = _config_data_with_thinker_tp(talker_gpu=1)
    with pytest.raises(ValueError, match="collides with thinker TP range"):
        MingOmniStreamingSpeechPipelineConfig(**data)


def test_variants_dict_exposes_streaming_variant():
    from sglang_omni.models.ming_omni.config import Variants

    assert "streaming_speech" in Variants
    assert Variants["streaming_speech"] is MingOmniStreamingSpeechPipelineConfig


def test_replica_devices_override_the_declared_talker_gpu():
    data = _config_data_with_thinker_tp(talker_gpu=1)
    data["processes"] = {
        TALKER_STREAM_STAGE: {"num_replicas": 2, "replica_devices": [2, 3]}
    }

    MingOmniStreamingSpeechPipelineConfig(**data)


def test_colliding_replica_devices_are_rejected_at_config_entry():
    data = _config_data_with_thinker_tp(talker_gpu=5)
    data["processes"] = {
        TALKER_STREAM_STAGE: {"num_replicas": 2, "replica_devices": [1, 3]}
    }

    with pytest.raises(ValueError, match="collides with thinker TP range"):
        MingOmniStreamingSpeechPipelineConfig(**data)
