# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.mimo_audio.audio_tokenizer import MiMoAudioTokenizerEncoder


def _tiny_config() -> dict:
    return {
        "d_model": 16,
        "encoder_attention_heads": 4,
        "encoder_skip_layer_id": 1,
        "stride_size": 2,
        "avg_pooler": 2,
        "n_mels": 8,
        "kernel_size": 3,
        "rope_theta": 10_000,
        "encoder_attn_window_size": [-1, -1],
        "encoder_causal": False,
        "encoder_ffn_dim": 32,
        "encoder_layers": 1,
        "codebook_size": [8, 8, 4, 4, 4, 4, 4, 4],
    }


def test_input_only_tokenizer_has_no_decoder_or_vocoder() -> None:
    encoder = MiMoAudioTokenizerEncoder(_tiny_config())

    assert not hasattr(encoder, "decoder")
    assert not hasattr(encoder, "vocoder")
    assert all(
        forbidden not in name
        for name, _ in encoder.named_modules()
        for forbidden in ("decoder", "vocoder", "istft")
    )


def test_input_only_tokenizer_rejects_non_codes_mode() -> None:
    encoder = MiMoAudioTokenizerEncoder(_tiny_config())

    try:
        encoder.encode(
            input_features=torch.zeros((10, 8)),
            input_lens=torch.tensor([10]),
            return_codes_only=False,
        )
    except ValueError as exc:
        assert "codes-only" in str(exc)
    else:
        raise AssertionError("non-codes tokenizer mode was accepted")
