# SPDX-License-Identifier: Apache-2.0
"""LFM3 text backbone on the SGLang runtime.

Hybrid layers: conv (gated short convolution, width 7), full_attention (GQA, partial zero-tail RoPE) and sliding_attention (GQA, full RoPE, 256-token window). Every layer has sandwich norms and every attention layer a per-head sigmoid output gate.

Conv layers (short_conv.py) keep their state in SGLang's MambaPool, which the runtime allocates for the LFM2 config.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers
from torch import nn

from sglang_omni.models.lfm3_audio.hf_config import (
    ATTENTION_LAYER_TYPES,
    Lfm3SGLangConfig,
    RopeParameters,
)
from sglang_omni.models.lfm3_audio.short_conv import Lfm3ShortConv

# The backbone's weight prefix in LFM3-Audio's export layout.
CHECKPOINT_PREFIX = "lfm.model."


def rope_kwargs(rope_parameters: RopeParameters) -> tuple[RopeParameters, float, float]:
    """(rope_scaling, base, partial_rotary_factor) for get_rope.

    LFM3's partial RoPE rotates the first factor * head_dim / 2 frequencies of each half and leaves the tail unrotated (zero inverse frequency), which is SGLang's proportional (Gemma 4) rotary embedding.
    """
    base = float(rope_parameters["rope_theta"])
    factor = float(rope_parameters.get("partial_rotary_factor", 1.0))
    rope_type = rope_parameters.get("rope_type", "default")
    if factor < 1.0 and rope_type == "default":
        rope_type = "proportional"
    else:
        pass
    scaling = {"rope_type": rope_type}
    scaling.update(
        {
            key: value
            for key, value in rope_parameters.items()
            if key not in ("rope_theta", "rope_type", "partial_rotary_factor")
        }
    )
    return scaling, base, factor


class Lfm3MLP(nn.Module):
    def __init__(self, config: Lfm3SGLangConfig, prefix: str = "") -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            prefix=add_prefix("w1", prefix),
        )
        self.w3 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            prefix=add_prefix("w3", prefix),
        )
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.w1(x)
        up, _ = self.w3(x)
        out, _ = self.w2(F.silu(gate) * up)
        return out


class Lfm3Attention(nn.Module):
    """GQA with Q/K RMSNorm, per-layer-type RoPE and a sigmoid gate per head."""

    def __init__(
        self,
        config: Lfm3SGLangConfig,
        layer_id: int,
        sliding_window: int | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        layer_type = (
            "sliding_attention" if sliding_window is not None else "full_attention"
        )
        scaling, base, factor = rope_kwargs(config.rope_for(layer_type))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_scaling=scaling,
            base=base,
            is_neox_style=True,
            partial_rotary_factor=factor,
            dtype=torch.get_default_dtype(),
        )
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            config.num_attention_heads,
            bias=False,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        # note (ruilin): the checkpoint's window counts tokens; RadixAttention takes the largest q-kv distance.
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(
                sliding_window - 1 if sliding_window is not None else -1
            ),
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(num_tokens, q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(num_tokens, kv_size)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, forward_batch)
        gate_logits, _ = self.gate_proj(hidden_states)
        gate = torch.sigmoid(gate_logits).view(num_tokens, self.num_heads, 1)
        attn_out = (
            attn_out.view(num_tokens, self.num_heads, self.head_dim) * gate
        ).reshape(num_tokens, q_size)
        out, _ = self.o_proj(attn_out)
        return out


class Lfm3DecoderLayer(nn.Module):
    def __init__(
        self, config: Lfm3SGLangConfig, layer_id: int, prefix: str = ""
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type in ATTENTION_LAYER_TYPES
        self.use_sandwich_norm = bool(config.use_sandwich_norm)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            pass
        if self.layer_type == "full_attention":
            self.self_attn = Lfm3Attention(
                config, layer_id, None, prefix=add_prefix("self_attn", prefix)
            )
        elif self.layer_type == "sliding_attention":
            if config.sliding_window is None:
                raise ValueError("sliding_attention layers need config.sliding_window")
            else:
                pass
            self.self_attn = Lfm3Attention(
                config,
                layer_id,
                int(config.sliding_window),
                prefix=add_prefix("self_attn", prefix),
            )
        elif self.layer_type == "conv":
            self.conv = Lfm3ShortConv(
                config, layer_id, prefix=add_prefix("conv", prefix)
            )
        else:
            raise ValueError(f"Unsupported LFM3 layer type {self.layer_type!r}")
        self.feed_forward = Lfm3MLP(config, prefix=add_prefix("feed_forward", prefix))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states
        else:
            pass
        normed = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            operator_out = self.self_attn(positions, normed, forward_batch)
        else:
            operator_out = self.conv(normed, forward_batch)
        if self.use_sandwich_norm:
            operator_out = self.post_operator_norm(operator_out)
        else:
            pass
        hidden_states = hidden_states + operator_out
        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        else:
            pass
        return hidden_states + ffn_out


class Lfm3TextModel(nn.Module):
    def __init__(self, config: Lfm3SGLangConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        def get_layer(idx: int, prefix: str, **_kwargs: object) -> Lfm3DecoderLayer:
            return Lfm3DecoderLayer(config, idx, prefix=prefix)

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=add_prefix("layers", prefix)
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return self.embedding_norm(hidden_states)

    def load_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        parameters: dict[str, nn.Parameter],
    ) -> str | None:
        """Load one backbone weight, name relative to the model; returns the parameter name, or None for an unknown weight."""
        if ".conv.conv.weight" in name:
            name = name.replace(".conv.conv.weight", ".conv.conv_weight")
            loaded_weight = loaded_weight.squeeze(1)
        elif ".conv.conv.bias" in name:
            name = name.replace(".conv.conv.bias", ".conv.conv_bias")
        else:
            pass
        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        )
        for stacked_name, weight_name, shard_id in stacked:
            if f".{weight_name}." in name:
                parameter_name = name.replace(weight_name, stacked_name)
                parameter = parameters[parameter_name]
                parameter.weight_loader(parameter, loaded_weight, shard_id)
                return parameter_name
            else:
                pass
        parameter = parameters.get(name)
        if parameter is None:
            return None
        else:
            pass
        # note (ruilin): SGLang attaches weight_loader only to the parameters it shards; the norms take the plain copy.
        weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
        weight_loader(parameter, loaded_weight)
        return name
