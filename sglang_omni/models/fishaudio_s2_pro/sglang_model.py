# SPDX-License-Identifier: Apache-2.0
"""SGLang-native S2-Pro text model with paged KV cache via RadixAttention.

Loads fish_speech FishQwen3OmniForCausalLM checkpoint weights but replaces the
static KVCache with SGLang's RadixAttention + paged token_to_kv_pool.

The audio decoder (FishQwen3AudioDecoder) is kept unchanged with its own
static KVCache — it only handles 11 tokens per step, no OOM risk.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import Tensor, nn

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


class S2ProAttention(nn.Module):
    """S2-Pro attention using RadixAttention for paged KV cache.

    Loads fused wqkv weights from fish_speech checkpoint, splits into q/k/v
    for QKVParallelLinear. Applies QK-norm and RoPE via SGLang's get_rope().
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_base,
            is_neox_style=False,  # fish_speech uses interleaved (GPT-J) RoPE
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_norm:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class S2ProDecoderLayer(nn.Module):
    """S2-Pro decoder layer: attention + SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = S2ProAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            qk_norm=qk_norm,
        )
        # SwiGLU: gate_proj (w1) and up_proj (w3) merged, down_proj (w2)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Pre-norm attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)

        # Pre-norm FFN
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # SwiGLU
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        del gate, up
        hidden_states, _ = self.down_proj(hidden_states)

        return hidden_states, residual


class S2ProSGLangTextModel(nn.Module):
    """SGLang-native S2-Pro text model (embedding → layers → norm → lm_head).

    Input shape: (num_tokens, dim) — flattened for continuous batching.
    """

    def __init__(
        self,
        config: Any = None,
        quant_config: Any = None,
        vocab_size: int = 155776,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        tie_word_embeddings: bool = True,
    ) -> None:
        super().__init__()

        # When called by SGLang's model loader, config is a FishQwen3OmniConfig
        if config is not None:
            tc = config.text_config
            vocab_size = tc.vocab_size
            hidden_size = tc.dim
            intermediate_size = tc.intermediate_size
            num_layers = tc.n_layer
            num_heads = tc.n_head
            num_kv_heads = tc.n_local_heads
            head_dim = tc.head_dim
            rope_base = tc.rope_base
            max_position_embeddings = tc.max_seq_len
            rms_norm_eps = tc.norm_eps
            qk_norm = tc.attention_qk_norm
            tie_word_embeddings = tc.tie_word_embeddings

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings

        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = make_layers(
            num_layers,
            lambda idx, prefix: S2ProDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_id=idx,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                qk_norm=qk_norm,
            ),
            prefix="layers",
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        if not tie_word_embeddings:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            self.lm_head = ParallelLMHead(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[Tensor] = None,
    ) -> LogitsProcessorOutput:
        """Forward pass. Returns LogitsProcessorOutput with hidden_states."""
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds

        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        # Prune to last-token positions for extend (prefill) mode
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]

        # Compute logits (tied weights: use embedding weight as lm_head)
        if self.tie_word_embeddings:
            logits = torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def get_embed_tokens(self):
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        """Load weights from fish_speech FishQwen3OmniForCausalLM checkpoint.

        Checkpoint keys (text_model only):
            text_model.model.embeddings.weight
            text_model.model.layers.N.attention.wqkv.weight   → split into q/k/v
            text_model.model.layers.N.attention.wo.weight      → o_proj
            text_model.model.layers.N.attention.q_norm.weight  → q_norm
            text_model.model.layers.N.attention.k_norm.weight  → k_norm
            text_model.model.layers.N.attention_norm.weight    → input_layernorm
            text_model.model.layers.N.ffn_norm.weight          → post_attention_layernorm
            text_model.model.layers.N.feed_forward.w1.weight   → gate_up_proj (shard 0)
            text_model.model.layers.N.feed_forward.w3.weight   → gate_up_proj (shard 1)
            text_model.model.layers.N.feed_forward.w2.weight   → down_proj
            text_model.model.norm.weight
        """
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Strip text_model.model. prefix
            if name.startswith("text_model.model."):
                name = name[len("text_model.model.") :]
            else:
                # Skip non-text-model weights (audio_decoder, etc.)
                continue

            # Remap checkpoint names to SGLang model names
            if self._load_remapped_weight(name, loaded_weight, params_dict):
                continue

            # Direct match
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug("Skipping weight: %s", name)

    def _load_remapped_weight(
        self,
        name: str,
        loaded_weight: Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        # Remap table: checkpoint suffix → (target suffix, shard_id or None)
        remap = {
            "attention.wqkv.weight": None,  # handled specially
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "down_proj.weight",
            "embeddings.weight": "embed_tokens.weight",
            "norm.weight": "norm.weight",
        }

        for ckpt_suffix, target in remap.items():
            if not name.endswith(ckpt_suffix):
                continue

            prefix = name[: -len(ckpt_suffix)]

            # Special case: fused wqkv → split into q, k, v shards
            if target is None:
                return self._load_fused_qkv(prefix, loaded_weight, params_dict)

            if isinstance(target, tuple):
                target_suffix, shard_id = target
            else:
                target_suffix = target
                shard_id = None

            target_name = prefix + target_suffix
            param = params_dict[target_name]
            if shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            return True

        return False

    def _load_fused_qkv(
        self,
        prefix: str,
        wqkv: Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        target_name = prefix + "self_attn.qkv_proj.weight"
        if target_name not in params_dict:
            return True

        param = params_dict[target_name]
        # Fish_speech wqkv layout: [q || k || v]
        layer = self.layers[int(prefix.split(".")[1])]
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size

        q, k, v = wqkv.split([q_size, kv_size, kv_size], dim=0)
        for shard_id, weight in [("q", q), ("k", k), ("v", v)]:
            param.weight_loader(param, weight, shard_id)
        return True


def _default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    param.data.copy_(loaded_weight)


EntryClass = S2ProSGLangTextModel
