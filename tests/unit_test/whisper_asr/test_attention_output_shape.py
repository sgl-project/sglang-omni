# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperSGLangCrossAttention,
    WhisperSGLangSelfAttention,
)

D_MODEL = 128
HEADS = 4
HEAD_DIM = D_MODEL // HEADS
TOKENS = 3


def _config() -> WhisperConfig:
    return WhisperConfig(
        d_model=D_MODEL,
        decoder_attention_heads=HEADS,
        decoder_layers=1,
        decoder_ffn_dim=256,
        encoder_attention_heads=HEADS,
        encoder_layers=1,
        encoder_ffn_dim=256,
        num_mel_bins=8,
        max_source_positions=16,
        max_target_positions=16,
        vocab_size=64,
    )


class _UnflattenedAttention(torch.nn.Module):
    """Stands in for the torch_native backend.

    It allocates its output with ``empty_like(q)``, so when the caller passes
    ``(tokens, heads, head_dim)`` the result keeps that rank instead of the
    ``(tokens, embed_dim)`` flashinfer returns.
    """

    def forward(self, q, k, v, forward_batch):
        del k, v, forward_batch
        return torch.zeros_like(q)


def test_self_attention_flattens_heads_before_out_proj() -> None:
    attn = WhisperSGLangSelfAttention(_config(), layer_id=0)
    attn.attn = _UnflattenedAttention()

    out = attn.forward(torch.randn(TOKENS, D_MODEL), forward_batch=None)

    assert out.shape == (TOKENS, D_MODEL)


def test_cross_attention_flattens_heads_before_out_proj() -> None:
    attn = WhisperSGLangCrossAttention(_config(), layer_id=0)
    attn.attn = _UnflattenedAttention()

    out = attn.forward(torch.randn(TOKENS, D_MODEL), forward_batch=None)

    assert out.shape == (TOKENS, D_MODEL)
