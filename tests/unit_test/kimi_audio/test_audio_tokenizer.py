# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.kimi_audio.audio_tokenizer import _encoder_weight_name


def test_encoder_weight_name_accepts_both_published_checkpoint_layouts() -> None:
    assert _encoder_weight_name("conv1.weight") == "conv1.weight"
    assert (
        _encoder_weight_name("model.encoder.layers.3.fc1.weight")
        == "layers.3.fc1.weight"
    )


def test_encoder_weight_name_rejects_training_buffers_and_decoder_weights() -> None:
    assert _encoder_weight_name("ema_count") is None
    assert _encoder_weight_name("model.decoder.layers.0.fc1.weight") is None
