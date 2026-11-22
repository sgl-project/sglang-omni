# SPDX-License-Identifier: Apache-2.0
"""LFM3 text backbone on the SGLang runtime.

Hybrid layers: ``conv`` (gated short convolution, width 7), ``full_attention``
(GQA, partial zero-tail RoPE) and ``sliding_attention`` (GQA, full RoPE,
256-token window). Every layer has sandwich norms and every attention layer a
per-head sigmoid output gate, as in the checkpoint's ``modeling_lfm3.py``.

Conv layers keep their state in SGLang's MambaPool (the config is an LFM2
config, so the runtime allocates it). The width-7 convolution runs in torch:
SGLang 0.5.23's causal-conv1d kernels serve widths 2-5 only.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs
from torch import nn

from sglang_omni.models.lfm3_audio.hf_config import (
    ATTENTION_LAYER_TYPES,
    Lfm3SGLangConfig,
)

# Checkpoint prefix of the backbone in the LFM3-Audio export.
CHECKPOINT_PREFIX = "lfm.model."


def rope_kwargs(rope_parameters: dict) -> tuple[dict, float, float]:
    """(rope_scaling, base, partial_rotary_factor) for ``get_rope``.

    LFM3's partial RoPE rotates the first ``factor * head_dim / 2`` frequencies
    of each half and leaves the tail unrotated (zero inverse frequency), which
    is SGLang's ``proportional`` (Gemma 4) rotary embedding.
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
        # The checkpoint's window counts tokens; RadixAttention takes the largest q-kv distance.
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
        tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(tokens, q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(tokens, kv_size)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, forward_batch)
        gate_logits, _ = self.gate_proj(hidden_states)
        gate = torch.sigmoid(gate_logits).view(tokens, self.num_heads, 1)
        attn_out = (
            attn_out.view(tokens, self.num_heads, self.head_dim) * gate
        ).reshape(tokens, q_size)
        out, _ = self.o_proj(attn_out)
        return out


class Lfm3ShortConv(nn.Module):
    """Gated short convolution: in_proj -> (B, C, x) -> conv(B*x) -> C * out -> out_proj.

    The MambaPool conv state of a request holds the last ``K - 1`` values of
    ``B * x`` per channel, shaped ``(hidden, K - 1)``.
    """

    def __init__(
        self, config: Lfm3SGLangConfig, layer_idx: int, prefix: str = ""
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel = int(config.conv_L_cache)
        self.hidden_size = config.hidden_size
        tp_size = get_parallel().tp_size
        self.hidden_size_per_partition = self.hidden_size // tp_size
        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,
            bias=bool(config.conv_bias),
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=bool(config.conv_bias),
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )
        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(
            self.conv_weight,
            {"weight_loader": sharded_weight_loader(0, parallel_group="tp")},
        )
        if config.conv_bias:
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition))
            set_weight_attrs(
                self.conv_bias,
                {"weight_loader": sharded_weight_loader(0, parallel_group="tp")},
            )
        else:
            self.register_parameter("conv_bias", None)

    def conv_decode(
        self,
        bx: torch.Tensor,
        conv_state: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """One token per request: dot the rolling window, then advance the state in place."""
        state = conv_state[cache_indices].to(bx.dtype)
        window = torch.cat([state, bx.unsqueeze(-1)], dim=-1)
        out = (window * self.conv_weight.to(bx.dtype).unsqueeze(0)).sum(dim=-1)
        if self.conv_bias is not None:
            out = out + self.conv_bias.to(out.dtype)
        else:
            pass
        conv_state[cache_indices] = window[:, :, 1:].to(conv_state.dtype)
        return out

    def conv_prefill(
        self,
        bx: torch.Tensor,
        conv_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-request causal depthwise conv over ``state || B*x`` with ``F.conv1d``."""
        width = self.conv_kernel
        channels = self.hidden_size_per_partition
        weight = self.conv_weight.to(bx.dtype).unsqueeze(1)
        bias = self.conv_bias.to(bx.dtype) if self.conv_bias is not None else None
        starts = query_start_loc.tolist()
        slots = cache_indices.tolist()
        initial = (
            has_initial_state.tolist()
            if has_initial_state is not None
            else [True] * len(slots)
        )
        bx_t = bx.transpose(0, 1)
        outputs: list[torch.Tensor] = []
        for row, slot in enumerate(slots):
            start, end = starts[row], starts[row + 1]
            if start == end:
                continue
            else:
                pass
            if initial[row]:
                state = conv_state[slot].to(bx.dtype)
            else:
                state = bx.new_zeros(channels, width - 1)
            window = torch.cat([state, bx_t[:, start:end]], dim=-1)
            outputs.append(
                F.conv1d(window.unsqueeze(0), weight, bias=bias, groups=channels)[0]
            )
            conv_state[slot] = window[:, -(width - 1) :].to(conv_state.dtype)
        if not outputs:
            return bx.new_zeros(0, channels)
        else:
            pass
        return torch.cat(outputs, dim=-1).transpose(0, 1)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states
        else:
            pass
        meta = get_attn_backend().conv_state_metadata(self.layer_idx, forward_batch)
        conv_state = meta.layer_cache.conv[0]
        proj, _ = self.in_proj(hidden_states)
        b_gate, c_gate, x = proj.chunk(3, dim=-1)
        bx = b_gate * x
        if forward_batch.forward_mode.is_decode():
            conv_out = self.conv_decode(bx, conv_state, meta.cache_indices)
        else:
            conv_out = self.conv_prefill(
                bx,
                conv_state,
                meta.cache_indices,
                meta.query_start_loc,
                getattr(meta, "has_initial_state", None),
            )
        output, _ = self.out_proj(c_gate * conv_out)
        return output


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
        self.num_attention_layers = sum(
            1
            for layer_type in config.layer_types
            if layer_type in ATTENTION_LAYER_TYPES
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
        params: dict[str, nn.Parameter],
    ) -> str | None:
        """Load one backbone weight, ``name`` relative to the model; returns the parameter name."""
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
        for param_name, weight_name, shard_id in stacked:
            if f".{weight_name}." in name:
                mapped = name.replace(weight_name, param_name)
                param = params[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                return mapped
            else:
                pass
        param = params.get(name)
        if param is None:
            return None
        else:
            pass
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, loaded_weight)
        return name


def backbone_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Strip the export prefix from backbone weights; others pass through unchanged."""
    for name, tensor in weights:
        if name.startswith(CHECKPOINT_PREFIX):
            yield "language_model." + name[len(CHECKPOINT_PREFIX) :], tensor
        else:
            yield name, tensor
