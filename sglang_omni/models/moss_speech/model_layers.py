# SPDX-License-Identifier: Apache-2.0
"""Graph-compatible decoder math preserving checkpoint rounding and request shapes."""

from __future__ import annotations

from functools import lru_cache

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.models.qwen3 import Qwen3Attention

from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig


class MossSpeechRMSNorm(RMSNorm):
    """Round residual addition and normalization before multiplying BF16 weights."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            pass
        else:
            x = x + residual
            residual = x
        original_dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        normalized = self.weight * normalized.to(original_dtype)
        return normalized if residual is None else (normalized, residual)


@lru_cache(maxsize=8)
def reference_rope_frequencies(head_dim: int, theta: float) -> torch.Tensor:
    """Compute frequencies on CPU to retain the checkpoint's rounding."""
    exponents = (
        torch.arange(0, head_dim, 2, device="cpu", dtype=torch.int64).float() / head_dim
    )
    return 1.0 / (theta**exponents)


def request_linear(
    hidden_states: torch.Tensor, weight: torch.Tensor, token_lengths: list[int]
) -> torch.Tensor:
    """Keep GEMM shapes independent of unrelated packed requests."""
    assert sum(token_lengths) == hidden_states.shape[0]
    if len(token_lengths) <= 1:
        return torch.nn.functional.linear(hidden_states, weight)
    else:
        return torch.cat(
            [
                torch.nn.functional.linear(request_hidden_states, weight)
                for request_hidden_states in hidden_states.split(token_lengths, dim=0)
            ],
            dim=0,
        )


def request_norm(
    norm: MossSpeechRMSNorm,
    hidden_states: torch.Tensor,
    token_lengths: list[int],
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if len(token_lengths) <= 1:
        return norm(hidden_states, residual)
    elif residual is None:
        return torch.cat(
            [
                norm(request_hidden_states)
                for request_hidden_states in hidden_states.split(token_lengths, dim=0)
            ]
        )
    else:
        normalized = [
            norm(request_hidden_states, request_residual)
            for request_hidden_states, request_residual in zip(
                hidden_states.split(token_lengths, dim=0),
                residual.split(token_lengths, dim=0),
            )
        ]
        return (
            torch.cat([request_result[0] for request_result in normalized]),
            torch.cat([request_result[1] for request_result in normalized]),
        )


class MossSpeechDecoderLayer(torch.nn.Module):
    """Reuse native weights, attention, and KV storage with dual-tail math."""

    inverse_frequencies: torch.Tensor

    def __init__(
        self,
        config: MossSpeechConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn: Qwen3Attention = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=0,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.register_buffer(
            "inverse_frequencies",
            reference_rope_frequencies(config.head_dim, config.rope_theta).to(
                self.self_attn.qkv_proj.weight.device
            ),
            persistent=False,
        )
        self.mlp: Qwen2MLP = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            reduce_results=False,
        )
        self.input_layernorm: MossSpeechRMSNorm = MossSpeechRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm: MossSpeechRMSNorm = MossSpeechRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn.q_norm = MossSpeechRMSNorm(
            config.head_dim, eps=config.rms_norm_eps
        )
        self.self_attn.k_norm = MossSpeechRMSNorm(
            config.head_dim, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        token_lengths: list[int],
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = request_norm(
                self.input_layernorm, hidden_states, token_lengths
            )
        else:
            hidden_states, residual = request_norm(
                self.input_layernorm, hidden_states, token_lengths, residual
            )
        attention = self.self_attn
        query_weight, key_weight, value_weight = attention.qkv_proj.weight.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=0
        )
        q = request_linear(hidden_states, query_weight, token_lengths)
        k = request_linear(hidden_states, key_weight, token_lengths)
        v = request_linear(hidden_states, value_weight, token_lengths)
        q = request_norm(
            attention.q_norm,
            q.reshape(-1, attention.num_heads, attention.head_dim),
            token_lengths,
        )
        k = request_norm(
            attention.k_norm,
            k.reshape(-1, attention.num_kv_heads, attention.head_dim),
            token_lengths,
        )
        with torch.autocast(device_type=positions.device.type, enabled=False):
            frequencies = (
                self.inverse_frequencies[None, :, None]
                @ positions[None, None, :].float()
            ).transpose(1, 2)
            angles = torch.cat((frequencies, frequencies), dim=-1)
            cosine = angles.cos().to(q.dtype)[0, :, None, :]
            sine = angles.sin().to(q.dtype)[0, :, None, :]
        query_left, query_right = q.chunk(2, dim=-1)
        key_left, key_right = k.chunk(2, dim=-1)
        q = (q * cosine + torch.cat((-query_right, query_left), dim=-1) * sine).reshape(
            -1, attention.q_size
        )
        k = (k * cosine + torch.cat((-key_right, key_left), dim=-1) * sine).reshape(
            -1, attention.kv_size
        )
        hidden_states = attention.attn(q, k, v, forward_batch)
        hidden_states = request_linear(
            hidden_states, attention.o_proj.weight, token_lengths
        )
        hidden_states, residual = request_norm(
            self.post_attention_layernorm, hidden_states, token_lengths, residual
        )
        gate_weight, up_weight = self.mlp.gate_up_proj.weight.chunk(2, dim=0)
        gate = request_linear(hidden_states, gate_weight, token_lengths)
        up = request_linear(hidden_states, up_weight, token_lengths)
        hidden_states = request_linear(
            torch.nn.functional.silu(gate) * up,
            self.mlp.down_proj.weight,
            token_lengths,
        )
        return hidden_states, residual
