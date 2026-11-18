# SPDX-License-Identifier: Apache-2.0
"""Zonos3 token mixers: global GQA attention and the gated delta net."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from sglang_omni.models.zonos3.hf_config import Zonos3Config

QUERY_KEY_NORM_EPSILON = 1e-6


class Zonos3Attention(nn.Module):
    """GQA with weightless QK norm, per-head temperature, interleaved RoPE and a head gate."""

    def __init__(self, config: Zonos3Config, layer_id: int) -> None:
        super().__init__()
        self.query_head_count = config.num_attention_heads
        self.key_value_head_count = config.num_key_value_heads
        self.head_dimension = config.head_dim
        hidden_size = config.hidden_size
        self.wq = nn.Linear(
            hidden_size, self.query_head_count * self.head_dimension, bias=False
        )
        self.wkv = nn.Linear(
            hidden_size, 2 * self.key_value_head_count * self.head_dimension, bias=False
        )
        self.wo = nn.Linear(
            self.query_head_count * self.head_dimension, hidden_size, bias=False
        )
        self.gater = nn.Linear(hidden_size, self.query_head_count, bias=False)
        self.temp = nn.Parameter(torch.empty(1, self.query_head_count, 1))
        self.rotary = get_rope(
            self.head_dimension,
            self.head_dimension,
            config.max_position_embeddings,
            int(config.rope_theta),
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.query_head_count,
            self.head_dimension,
            self.head_dimension**-0.5,
            self.key_value_head_count,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        gate = torch.sigmoid(self.gater(hidden_states))
        q = self.wq(hidden_states).view(
            token_count, self.query_head_count, self.head_dimension
        )
        k, v = self.wkv(hidden_states).chunk(2, dim=-1)
        k = k.reshape(token_count, self.key_value_head_count, self.head_dimension)
        q = F.rms_norm(
            q, (self.head_dimension,), None, QUERY_KEY_NORM_EPSILON
        ) * self.temp.abs().to(q.dtype)
        k = F.rms_norm(k, (self.head_dimension,), None, QUERY_KEY_NORM_EPSILON)
        q, k = self.rotary(
            positions,
            q.reshape(token_count, -1),
            k.reshape(token_count, -1),
        )
        attention_output = self.attn(q, k, v.contiguous(), forward_batch)
        attention_output = attention_output.view(
            token_count, self.query_head_count, self.head_dimension
        ) * gate.unsqueeze(-1)
        return self.wo(attention_output.reshape(token_count, -1))


class Zonos3GatedDeltaNet(nn.Module):
    """Gated delta rule mixer; the input projection packs q, k, v, output gate, decay and beta logits."""

    def __init__(self, config: Zonos3Config, layer_id: int) -> None:
        super().__init__()
        self.head_count = config.linear_num_key_heads
        self.head_dimension = config.linear_key_head_dim
        self.key_dimension = self.head_count * self.head_dimension
        convolution_dimension = 3 * self.key_dimension
        hidden_size = config.hidden_size
        self.in_proj = nn.Linear(
            hidden_size, 4 * self.key_dimension + 2 * self.head_count, bias=False
        )
        self.conv = nn.Conv1d(
            convolution_dimension,
            convolution_dimension,
            config.linear_conv_kernel_dim,
            groups=convolution_dimension,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.head_count, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.head_count))
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.head_count,
            num_k_heads=self.head_count,
            num_v_heads=self.head_count,
            head_q_dim=self.head_dimension,
            head_k_dim=self.head_dimension,
            head_v_dim=self.head_dimension,
            conv_weights=self.conv.weight.view(
                convolution_dimension, config.linear_conv_kernel_dim
            ),
            bias=None,
            activation="silu",
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        self.norm = RMSNormGated(
            self.head_dimension, eps=config.rms_norm_eps, norm_before_gate=True
        )
        self.wo = nn.Linear(self.key_dimension, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        mixed_qkv, output_gate, decay_logits, beta_logits = self.in_proj(
            hidden_states
        ).split(
            [
                3 * self.key_dimension,
                self.key_dimension,
                self.head_count,
                self.head_count,
            ],
            dim=-1,
        )
        # note (zhaochenyang20): the decode convolution and recurrence kernels read the projection's columns by row stride, which dropped 27 copy kernels from each decode step; prefill keeps its packed copies.
        if forward_batch.forward_mode.is_decode():
            core_output = self.attn(
                forward_batch, mixed_qkv=mixed_qkv, a=decay_logits, b=beta_logits
            )
        else:
            core_output = self.attn(
                forward_batch,
                mixed_qkv=mixed_qkv.contiguous(),
                a=decay_logits.contiguous(),
                b=beta_logits.contiguous(),
            )
        # The gated norm reads each token's head gates by row stride, so the gate stays a view of the projection.
        gated_output = self.norm(
            core_output.reshape(-1, self.head_dimension),
            output_gate.view(token_count, self.head_count, self.head_dimension),
        )
        return self.wo(gated_output.reshape(token_count, self.key_dimension))
