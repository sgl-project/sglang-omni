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


def test_streaming_speech_topology_wires_segmenter_between_thinker_and_talker():
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    names = [s.name for s in config.stages]
    assert SEGMENTER_STAGE in names
    assert TALKER_STREAM_STAGE in names
    # Old non-streaming talker must NOT be present.
    assert "talker" not in names


def test_streaming_thinker_fans_out_and_streams_to_segmenter():
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, THINKER_STAGE)
    assert sorted(thinker.next) == sorted([DECODE_STAGE, SEGMENTER_STAGE])
    assert thinker.stream_to == [SEGMENTER_STAGE]
    assert thinker.factory_args.get("enable_streaming_tts") is True


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
    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, THINKER_STAGE)
    talker = _stage(config, TALKER_STREAM_STAGE)
    thinker.gpu = [0, 1]
    thinker.tp_size = 2
    talker.gpu = 1  # collides
    with pytest.raises(ValueError, match="collides with thinker TP range"):
        config._validate_talker_stream_gpu_not_in_thinker_tp_range()


def test_variants_dict_exposes_streaming_variant():
    from sglang_omni.models.ming_omni.config import Variants

    assert "streaming_speech" in Variants
    assert Variants["streaming_speech"] is MingOmniStreamingSpeechPipelineConfig
