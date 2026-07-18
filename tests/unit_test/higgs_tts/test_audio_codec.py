# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

from sglang_omni.models.higgs_tts import audio_codec


def test_higgs_codec_uses_upstream_transformers_architecture() -> None:
    assert audio_codec.HiggsAudioV2TokenizerConfig is HiggsAudioV2TokenizerConfig
    assert audio_codec.HiggsAudioV2TokenizerModel is HiggsAudioV2TokenizerModel

    config = HiggsAudioV2TokenizerConfig.from_json_file(
        audio_codec._BUNDLED_CODEC_CONFIG_PATH
    )
    with torch.device("meta"):
        model = audio_codec.HiggsAudioV2TokenizerModel(config)

    state = model.state_dict()
    assert len(state) == 527
    assert {
        key: tuple(state[key].shape)
        for key in (
            "acoustic_encoder.block.0.conv1.weight",
            "acoustic_decoder.block.0.conv_t1.weight",
            "quantizer.quantizers.0.codebook.embed",
            "semantic_model.encoder.layers.0.attention.q_proj.weight",
        )
    } == {
        "acoustic_encoder.block.0.conv1.weight": (128, 64, 16),
        "acoustic_decoder.block.0.conv_t1.weight": (1024, 512, 16),
        "quantizer.quantizers.0.codebook.embed": (1024, 64),
        "semantic_model.encoder.layers.0.attention.q_proj.weight": (768, 768),
    }
    assert config.frame_rate == 25
    assert config.num_quantizers == 8
