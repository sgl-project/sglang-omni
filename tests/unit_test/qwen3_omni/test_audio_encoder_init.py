# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.qwen3_omni.components import audio_encoder


def test_audio_encoder_initializes_with_installed_transformers(monkeypatch):
    tower = torch.nn.Module()
    monkeypatch.setattr(audio_encoder, "load_thinker_config", lambda _: object())
    monkeypatch.setattr(
        audio_encoder, "build_audio_tower", lambda *args, **kwargs: tower
    )
    monkeypatch.setattr(audio_encoder, "share_segment_splits", lambda *args: None)

    encoder = audio_encoder.Qwen3OmniAudioEncoder("unused", device="cpu")

    assert encoder.audio_tower is tower
    lengths = encoder._downsample_lengths(torch.tensor([100, 200]))
    assert lengths.shape == (2,)
    assert 0 < lengths[0] < lengths[1]
