# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperSGLangCrossAttention,
    WhisperSGLangSelfAttention,
)


class _HeadShapedAttention(nn.Module):
    def forward(self, query, key, value, forward_batch):
        del key, value, forward_batch
        return query


def _attention_shell(cls):
    attention = cls.__new__(cls)
    nn.Module.__init__(attention)
    attention.embed_dim = 4
    attention.num_heads = 2
    attention.head_dim = 2
    attention.q_proj = nn.Identity()
    attention.k_proj = nn.Identity()
    attention.v_proj = nn.Identity()
    attention.out_proj = nn.Identity()
    attention.attn = _HeadShapedAttention()
    return attention


def test_whisper_self_attention_flattens_native_backend_heads() -> None:
    attention = _attention_shell(WhisperSGLangSelfAttention)

    output = attention(torch.ones(3, 4), object())

    assert output.shape == (3, 4)


def test_whisper_cross_attention_flattens_native_backend_heads() -> None:
    attention = _attention_shell(WhisperSGLangCrossAttention)

    output = attention(torch.ones(3, 4), torch.ones(5, 4), object())

    assert output.shape == (3, 4)
