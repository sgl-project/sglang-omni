# SPDX-License-Identifier: Apache-2.0
"""Shared audio Session configuration without a cascade model dependency."""

import pytest

from sglang_omni.config.session_audio import AudioSessionPipelineConfig
from sglang_omni.models.vad.node import VADStageConfig
from sglang_omni.proto.streams import PCM16


def test_vad_graph_assembles_and_round_trips() -> None:
    config = AudioSessionPipelineConfig(
        model_path="unused",
        session_nodes={"vad": {"stages": ["vad"], "inputs": ["input:pcm16"]}},
        session_publish={"vad": ["session"]},
    )
    assert config.entry_stage == "vad"
    assert [stage.name for stage in config.stages] == ["vad"]
    assert isinstance(config.stages[0], VADStageConfig)
    rebuilt = AudioSessionPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.session_graph() == config.session_graph()
    assert rebuilt.session_ports().input[0].format == PCM16
    assert (
        rebuilt.session_limits().max_output_chunks == config.max_pending_output_chunks
    )


def test_unregistered_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="no stage group"):
        AudioSessionPipelineConfig(
            model_path="unused",
            session_nodes={"model": {"stages": ["missing"], "inputs": ["input:pcm16"]}},
            session_publish={"model": ["pcm16"]},
        )


def test_hosted_kind_is_not_a_common_audio_node() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AudioSessionPipelineConfig(
            model_path="unused",
            session_nodes={"model": {"kind": "missing", "inputs": ["input:pcm16"]}},
            session_publish={"model": ["pcm16"]},
        )
