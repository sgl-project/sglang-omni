# SPDX-License-Identifier: Apache-2.0
import inspect

from sglang_omni.models.higgs_audio_asr.config import HiggsAudioASRPipelineConfig
from sglang_omni.models.higgs_audio_asr.configuration_higgs_audio_asr import (
    higgs_num_audio_tokens,
)
from sglang_omni.models.higgs_audio_asr.stages import (
    create_sglang_higgs_audio_asr_executor,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_higgs_audio_asr_config_registered():
    config = HiggsAudioASRPipelineConfig(model_path="bosonai/higgs-audio-v3-stt")

    assert config.entry_stage == "asr"
    assert config.stages[0].name == "asr"
    assert config.stages[0].terminal
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("HiggsAudio3Model")
        is HiggsAudioASRPipelineConfig
    )


def test_higgs_audio_asr_stage_defaults():
    signature = inspect.signature(create_sglang_higgs_audio_asr_executor)
    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["max_new_tokens"].default == 1024


def test_higgs_audio_token_lengths():
    # 4 s chunk: 400 mel -> conv2 200 -> pool 100 -> projector 50 (12.5/s)
    assert higgs_num_audio_tokens(400) == 50
    # partial last chunk (0.49 s = 49 mel frames)
    assert higgs_num_audio_tokens(49) == 6
