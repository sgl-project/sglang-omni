# SPDX-License-Identifier: Apache-2.0
import inspect

import torch

from sglang_omni.models.higgs_audio_asr.audio_tower import HiggsAudioTower
from sglang_omni.models.higgs_audio_asr.config import HiggsAudioASRPipelineConfig
from sglang_omni.models.higgs_audio_asr.configuration_higgs_audio_asr import (
    HiggsAudio3Config,
    HiggsAudioEncoderConfig,
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
    # Documented checkpoint behavior (transcribe.py / model card) is thinking on.
    assert signature.parameters["enable_thinking"].default is True


def test_higgs_audio_token_lengths():
    # 4 s chunk: 400 mel -> conv2 200 -> pool 100 -> projector 50 (12.5/s)
    assert higgs_num_audio_tokens(400) == 50
    # partial last chunk (0.49 s = 49 mel frames)
    assert higgs_num_audio_tokens(49) == 6


def test_tower_masking_isolates_partial_chunk():
    """A padded partial chunk batched with a longer chunk must produce the same
    embeddings as running it alone — i.e. it does not attend to the padding."""
    torch.manual_seed(0)
    enc = HiggsAudioEncoderConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=64,
    )
    cfg = HiggsAudio3Config(audio_encoder_config=enc, text_config={"hidden_size": 16})
    tower = HiggsAudioTower(cfg).eval()

    full_frames, partial_frames = 40, 20
    full = torch.randn(1, 8, full_frames)
    partial = torch.randn(1, 8, partial_frames)
    padded_partial = torch.nn.functional.pad(partial, (0, full_frames - partial_frames))

    with torch.no_grad():
        batched = tower(
            torch.cat([full, padded_partial], dim=0),
            mel_lengths=torch.tensor([full_frames, partial_frames]),
        )
        alone = tower(partial, mel_lengths=torch.tensor([partial_frames]))

    valid = (partial_frames - 1) // 2 // 2 + 1  # post conv2 + avg_pool
    torch.testing.assert_close(
        batched[1, :valid], alone[0, :valid], atol=1e-4, rtol=1e-4
    )
