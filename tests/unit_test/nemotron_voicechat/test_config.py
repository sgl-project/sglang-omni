# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.nemotron_voicechat.config import (
    CODE2WAV_STAGE,
    NemotronVoiceChatPipelineConfig,
)
from sglang_omni.models.nemotron_voicechat.payload_types import (
    FRAME_SAMPLES,
    VoiceChatFrameState,
)
from sglang_omni.serve.launcher import _warmup_frame_realtime_pipeline


def test_voicechat_pipeline_is_fixed_frame_and_audio_terminal() -> None:
    config = NemotronVoiceChatPipelineConfig(model_path="/checkpoint")

    assert config.realtime_audio.mode == "frame"
    assert config.realtime_audio.frame_samples == FRAME_SAMPLES
    assert type(config).code2wav_stage() == CODE2WAV_STAGE
    assert config.terminal_stages == [CODE2WAV_STAGE]
    assert [stage.name for stage in config.stages] == [
        "perception",
        "thinker",
        "talker",
        "code2wav",
    ]


def test_voicechat_pipeline_rejects_topology_changes() -> None:
    defaults = NemotronVoiceChatPipelineConfig(model_path="/checkpoint")
    stages = [stage.model_copy(deep=True) for stage in defaults.stages]
    stages[1].next = "code2wav"

    with pytest.raises(ValueError, match="Invalid Nemotron VoiceChat topology"):
        NemotronVoiceChatPipelineConfig(model_path="/checkpoint", stages=stages)


def test_voicechat_h100_example_uses_builder_session_capacity_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(repo_root / "examples/configs/nemotron_voicechat_h100.yaml")
    ).config
    stages = {stage.name: stage for stage in config.stages}

    assert isinstance(config, NemotronVoiceChatPipelineConfig)
    thinker_factory = stages["thinker"].factory.model_dump(exclude_none=True)
    talker_factory = stages["talker"].factory.model_dump(exclude_none=True)
    assert "server_args_overrides" not in thinker_factory
    assert "server_args_overrides" not in talker_factory
    assert "attention_backend" not in talker_factory
    assert stages["thinker"].engine is not None
    assert stages["talker"].engine is not None
    assert stages["thinker"].engine.mem_fraction_static == 0.45
    assert stages["talker"].engine.mem_fraction_static == 0.20


def test_voicechat_frame_state_requires_monotonic_frame_metadata() -> None:
    state = VoiceChatFrameState.from_data(
        {
            "event": "audio_frame",
            "session_id": "session",
            "frame_index": 0,
            "pcm16": "AA==",
        }
    )
    assert state.frame_index == 0

    wrapped = VoiceChatFrameState.from_data(
        {
            "raw_inputs": {
                "event": "audio_frame",
                "session_id": "session",
                "frame_index": 1,
                "pcm16": "AA==",
            }
        }
    )
    assert wrapped.frame_index == 1

    with pytest.raises(ValueError, match="frame_index"):
        VoiceChatFrameState.from_data(
            {
                "event": "audio_frame",
                "session_id": "session",
                "frame_index": -1,
                "pcm16": "AA==",
            }
        )


@pytest.mark.asyncio
async def test_voicechat_warmup_is_skipped_when_realtime_is_disabled() -> None:
    class UnexpectedClient:
        def generate(self, *args, **kwargs):
            raise AssertionError("warmup must not submit a request")

    config = NemotronVoiceChatPipelineConfig(model_path="/checkpoint")
    await _warmup_frame_realtime_pipeline(
        UnexpectedClient(),  # type: ignore[arg-type]
        config,
        enabled=False,
    )
