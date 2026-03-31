# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MiniCPM-V LLM model with paged attention.

This module provides a SGLang-compatible LLM backbone for MiniCPM-V,
supporting both 2.6 and 4.5 architectures:

MiniCPM-V 2.6 (MiniCPM-3.0 backbone):
- QKVParallelLinear + RadixAttention (with optional QK norm)
- MergedColumnParallelLinear (gate_up) + RowParallelLinear (down) + SiLU
- RMSNorm for layer normalization
- Default: hidden_size=2560, num_layers=62, num_kv_heads=8

MiniCPM-V 4.5 (Qwen3-8B backbone):
- Same architecture but with different dimensions
- Default: hidden_size=4096, num_layers=36, num_kv_heads=8
- vocab_size=151748, max_position_embeddings=40960

Image embedding injection is handled via forward_batch.input_embeds,
which is set by the SGLang model runner when omni_model_inputs is present.

The weight loading strips "llm." prefix from MiniCPM-V checkpoints.
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
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


class MiniCPMAttention(nn.Module):
    """MiniCPM attention layer with QK norm support.

    MiniCPM 3.0 uses standard LLaMA-style attention with optional QK norm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        qk_norm: bool = False,
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
            is_neox_style=True,
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
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(-1, self.q_size)
            k = k.view(-1, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMDecoderLayer(nn.Module):
    """MiniCPM decoder layer with LLaMA-style architecture."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = MiniCPMAttention(
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        del gate, up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


class MiniCPMVSGLangLLM(nn.Module):
    """SGLang-native MiniCPM-V LLM with paged attention.

    This model implements the LLaMA-style LLM backbone optimized for
    SGLang's continuous batching and CUDA graph capture.

    Supports both versions:
    - MiniCPM-V 2.6: MiniCPM-3.0 backbone (62 layers, hidden_size=2560)
    - MiniCPM-V 4.5: Qwen3-8B backbone (36 layers, hidden_size=4096)

    Image embedding injection is handled via forward_batch.input_embeds,
    which is set by the SGLang model runner when omni_model_inputs is present.
    """

    def __init__(
        self,
        config: Any = None,
        quant_config: Any = None,
        # Dynamic defaults based on config - these are fallbacks
        # 2.6: vocab_size=122880, hidden_size=2560, num_layers=62
        # 4.5: vocab_size=151748, hidden_size=4096, num_layers=36
        vocab_size: int = 151748,  # Default to 4.5 (Qwen3)
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_base: float = 1000000.0,  # Qwen3 default
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = False,
        tie_word_embeddings: bool = False,
    ) -> None:
        super().__init__()

        # Extract config from HF config if provided
        if config is not None:
            # MiniCPM-V may have llm_config nested or direct attributes
            llm_cfg = getattr(config, "llm_config", config)
            vocab_size = getattr(llm_cfg, "vocab_size", vocab_size)
            hidden_size = getattr(llm_cfg, "hidden_size", hidden_size)
            intermediate_size = getattr(llm_cfg, "intermediate_size", intermediate_size)
            num_layers = getattr(llm_cfg, "num_hidden_layers", num_layers)
            num_heads = getattr(llm_cfg, "num_attention_heads", num_heads)
            num_kv_heads = getattr(
                llm_cfg, "num_key_value_heads", num_kv_heads
            )
            head_dim = getattr(
                llm_cfg, "head_dim", hidden_size // num_heads
            )
            rope_base = getattr(llm_cfg, "rope_theta", rope_base)
            max_position_embeddings = getattr(
                llm_cfg, "max_position_embeddings", max_position_embeddings
            )
            rms_norm_eps = getattr(llm_cfg, "rms_norm_eps", rms_norm_eps)
            qk_norm = getattr(llm_cfg, "qk_norm", qk_norm)
            tie_word_embeddings = getattr(
                llm_cfg, "tie_word_embeddings", tie_word_embeddings
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings

        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = make_layers(
            num_layers,
            lambda idx, prefix: MiniCPMDecoderLayer(
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
        # Check for pre-computed image embeddings from model runner
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds

        if input_embeds is not None:
            # Prefill with image embeddings already injected
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Extend mode: extract last token hidden states
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]

        # Compute logits
        if self.tie_word_embeddings:
            logits = torch.nn.functional.linear(
                hidden_states, self.embed_tokens.weight
            )
        else:
            logits = self.lm_head(hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def get_embed_tokens(self):
        """Return the embedding layer for external access."""
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        """Load weights from MiniCPM-V checkpoint.

        Strips "llm." or "model.llm." prefix and handles Q/K/V → qkv_proj fusion.
        """
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Strip LLM prefix
            if name.startswith("llm.model."):
                name = name[len("llm.model."):]
            elif name.startswith("model.llm.model."):
                name = name[len("model.llm.model."):]
            elif name.startswith("llm."):
                name = name[len("llm."):]
            elif name.startswith("model.llm."):
                name = name[len("model.llm."):]
            else:
                # Skip non-LLM weights
                continue

            # Handle fused QKV projection
            if self._load_remapped_weight(name, loaded_weight, params_dict):
                continue

            # Direct parameter loading
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
        """Handle weight remapping for fused projections.

        MiniCPM checkpoint format:
        - layers.{i}.self_attn.q_proj.weight
        - layers.{i}.self_attn.k_proj.weight
        - layers.{i}.self_attn.v_proj.weight
        - layers.{i}.mlp.gate_proj.weight (gate)
        - layers.{i}.mlp.up_proj.weight (up)
        - layers.{i}.mlp.down_proj.weight

        SGLang target format:
        - layers.{i}.self_attn.qkv_proj.weight (fused q,k,v)
        - layers.{i}.gate_up_proj.weight (fused gate,up)
        - layers.{i}.down_proj.weight
        """
        remap = {
            # Attention projections
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", "q"),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", "k"),
            "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", "v"),
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            # QK norm (if present)
            "self_attn.q_norm.weight": "self_attn.q_norm.weight",
            "self_attn.k_norm.weight": "self_attn.k_norm.weight",
            # Layer norms
            "input_layernorm.weight": "input_layernorm.weight",
            "post_attention_layernorm.weight": "post_attention_layernorm.weight",
            # MLP projections
            "mlp.gate_proj.weight": ("gate_up_proj.weight", 0),
            "mlp.up_proj.weight": ("gate_up_proj.weight", 1),
            "mlp.down_proj.weight": "down_proj.weight",
        }

        for ckpt_suffix, target in remap.items():
            if not name.endswith(ckpt_suffix):
                continue
            prefix = name[: -len(ckpt_suffix)]

            if isinstance(target, tuple):
                target_suffix, shard_id = target
            else:
                target_suffix, shard_id = target, None

            target_name = prefix + target_suffix
            if target_name not in params_dict:
                logger.debug("Target param not found: %s", target_name)
                return True

            param = params_dict[target_name]
            if shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            return True

        # Handle embed_tokens and norm separately
        if name == "embed_tokens.weight":
            if "embed_tokens.weight" in params_dict:
                param = params_dict["embed_tokens.weight"]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
                return True
        elif name == "norm.weight":
            if "norm.weight" in params_dict:
                param = params_dict["norm.weight"]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
                return True

        return False


def _default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    """Default weight loader that copies weights directly."""
    param.data.copy_(loaded_weight)


EntryClass = MiniCPMVSGLangLLM
