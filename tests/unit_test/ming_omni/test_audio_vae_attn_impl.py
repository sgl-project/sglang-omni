# SPDX-License-Identifier: Apache-2.0
"""Audio VAE backbones must not resolve to flash-attention.

transformers 5.6.0's flash-attention integration crashes on plain Qwen2
backbones (`s_aux.to(...)` with `s_aux=None`), which kills the first speech
request during prompt-wav registration. Pin the VAE backbones to sdpa.
"""

from __future__ import annotations

from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import (
    Decoder,
    Encoder,
)

_BACKBONE = {
    "hidden_size": 32,
    "intermediate_size": 64,
    # The encoder aggregator forces 4 layers; layer_types must cover them.
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
    "vocab_size": 16,
}


def test_encoder_backbones_pinned_to_sdpa():
    enc = Encoder(
        encoder_args=dict(_BACKBONE),
        input_dim=8,
        hop_size=4,
        latent_dim=8,
        patch_size=2,
    )
    assert enc.encoder.config._attn_implementation == "sdpa"
    assert enc.aggregator.config._attn_implementation == "sdpa"


def test_decoder_backbone_pinned_to_sdpa():
    dec = Decoder(
        decoder_args=dict(_BACKBONE),
        output_dim=8,
        latent_dim=8,
        patch_size=2,
    )
    assert dec.decoder.config._attn_implementation == "sdpa"
