# SPDX-License-Identifier: Apache-2.0
"""Strided-view Whisper encoder forward for dense single-rank CUDA graphs."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.whisper import WhisperEncoder, WhisperEncoderLayer


class PackedWhisperEncoder:
    """Drop-in forward for a loaded SGLang WhisperEncoder (tensor parallel 1).

    A plain callable rather than a module, so the encoder's parameters stay
    registered once, under the model's whisper_encoder.
    """

    def __init__(self, encoder: WhisperEncoder) -> None:
        self.encoder = encoder

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.encoder.parameters()

    def layer_forward(
        self, layer: WhisperEncoderLayer, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        attention = layer.self_attn
        normed = layer.self_attn_layer_norm(hidden_states)
        qkv = F.linear(normed, attention.qkv_proj.weight, attention.qkv_proj.bias)
        query, key, value = qkv.view(
            batch_size, sequence_length, 3, attention.num_heads, attention.head_dim
        ).unbind(2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        attended = attended.transpose(1, 2).reshape(-1, hidden_size)
        hidden_states = (
            torch.addmm(
                attention.out_proj.bias, attended, attention.out_proj.weight.t()
            )
            .view(batch_size, sequence_length, hidden_size)
            .add_(hidden_states)
        )
        normed = layer.final_layer_norm(hidden_states)
        intermediate = layer.activation_fn(
            F.linear(normed, layer.fc1.weight, layer.fc1.bias)
        )
        return (
            torch.addmm(
                layer.fc2.bias,
                intermediate.view(-1, intermediate.shape[-1]),
                layer.fc2.weight.t(),
            )
            .view(batch_size, sequence_length, hidden_size)
            .add_(hidden_states)
        )

    def __call__(
        self,
        input_features: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch | None,
    ) -> torch.Tensor:
        # forward_batch is part of the encoder call contract; the encoder's
        # self-attention runs SDPA directly and never reads it.
        encoder = self.encoder
        device = encoder.conv1.weight.device
        input_features = input_features.to(device=device)
        position_ids = position_ids.to(device=device)
        inputs_embeds = F.gelu(encoder.conv1(input_features))
        inputs_embeds = F.gelu(encoder.conv2(inputs_embeds))
        hidden_states = inputs_embeds.mT + encoder.embed_positions(position_ids)
        for layer in encoder.layers:
            hidden_states = self.layer_forward(layer, hidden_states)
        return encoder.layer_norm(hidden_states)
