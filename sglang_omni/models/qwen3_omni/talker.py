"""SGLang-native Talker model for Qwen3-Omni.

Simplified implementation:
- Reuse Thinker's components where possible
- Only define Talker-specific parts (Shared Expert MoE)
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.config.qwen3_omni import (
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
)

# Reuse Thinker's components
from sglang_omni.models.qwen3_omni.thinker import (
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import tensor_model_parallel_all_reduce
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QuantizationConfig,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads."""
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class ResizeMLP(nn.Module):
    """Simple Linear-SiLU-Linear projection (used for text/hidden projection).

    Field names match HF checkpoint: linear_fc1, linear_fc2.
    """

    def __init__(
        self,
        in_size: int,
        intermediate_size: int,
        out_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_fc1 = ReplicatedLinear(
            in_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.act = nn.SiLU()
        self.linear_fc2 = ReplicatedLinear(
            intermediate_size,
            out_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.linear_fc1(x)
        out = self.act(out)
        out, _ = self.linear_fc2(out)
        return out


# ---------------------------------------------------------------------------
# Talker-specific MLP (Shared Expert MoE)
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerDenseMLP(nn.Module):
    """Standard SwiGLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3OmniMoeTalkerSharedExpertMLP(nn.Module):
    """Shared expert MLP with reduce_results=False for unified all-reduce."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,  # Don't all-reduce here; unified with routed experts
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3OmniMoeTalkerSparseMoeBlock(Qwen3OmniMoeThinkerTextSparseMoeBlock):
    """MoE block with Shared Expert (Talker-specific).

    Inherits from Thinker's MoE for routed experts (topk, experts, gate).
    Adds shared expert with gated output.

    All-reduce is unified: both routed and shared expert outputs stay as
    per-rank partial sums until combined, then a single all-reduce is applied.
    """

    def __init__(
        self,
        layer_id: int,
        config: Qwen3OmniMoeTalkerTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # Initialize parent (Thinker's MoE: topk, experts, gate)
        super().__init__(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Shared expert (reduce_results=False to avoid double all-reduce)
        self.shared_expert = Qwen3OmniMoeTalkerSharedExpertMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=add_prefix("shared_expert_gate", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Shared branch must consume the original MLP input before routed experts.
        # The fused MoE implementation mutates `hidden_states` in-place.
        shared_output = self.shared_expert(hidden_states)
        shared_gate, _ = self.shared_expert_gate(hidden_states)
        shared_output = shared_output * torch.sigmoid(shared_gate)

        # Routed experts (fused triton kernel, CUDA-graph safe)
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_output = self.experts(hidden_states, topk_output)

        final_hidden_states = routed_output + shared_output
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# Talker DecoderLayer (minimal override of Thinker's)
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerDecoderLayer(Qwen3OmniMoeThinkerTextDecoderLayer):
    """Talker decoder layer: inherit from Thinker, only replace MLP with Shared Expert MoE."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # Call parent's __init__ (Thinker's DecoderLayer)
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )

        # Replace MLP with Talker's Shared Expert MoE
        self.mlp = Qwen3OmniMoeTalkerSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )


# ---------------------------------------------------------------------------
# Talker Text Model
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerTextModel(nn.Module):
    """Talker's MoE text backbone (20-layer, with shared expert).

    Uses codec_embedding instead of embed_tokens.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Codec embedding (standard nn.Embedding, not VocabParallel - vocab is small)
        self.codec_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Decoder layers
        alt_stream = torch.cuda.Stream()
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Qwen3OmniMoeTalkerDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def get_input_embeddings(self):
        return self.codec_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        if input_embeds is None:
            hidden_states = self.codec_embedding(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        capture_layers = set(self.layers_to_capture or [])
        aux_hidden_states = []

        # Match the hidden-capture contract used by the compare tooling:
        # layer 0 is the embedding output (input to the first transformer layer).
        if 0 in capture_layers or "embed" in capture_layers:
            aux_hidden_states.append(hidden_states.clone())

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                captured_last_layer_outputs=(
                    aux_hidden_states if i in capture_layers and i != 0 else None
                ),
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Code Predictor (single class, matches HF checkpoint structure)
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """Code predictor for generating RVQ codes (layers 1 to N-1, N=num_code_groups).

    Matches HF checkpoint structure:
    - code_predictor.model.codec_embedding: ModuleList[N-1]  (15 embeddings)
    - code_predictor.model.layers: ModuleList[num_layers]     (5 dense decoder layers)
    - code_predictor.model.norm: RMSNorm
    - code_predictor.lm_head: ModuleList[N-1]                 (15 output heads)
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        cp_config = config.code_predictor_config

        # Wrapper to match HF checkpoint path (code_predictor.model.*)
        self.model = nn.Module()

        # Codec embeddings: 15 embeddings for layers 1-15 (layer 0 uses TextModel's codec_head)
        self.model.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(cp_config.vocab_size, cp_config.hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # 5 dense decoder layers
        alt_stream = torch.cuda.Stream()
        self.model.layers = nn.ModuleList()
        for idx in range(cp_config.num_hidden_layers):
            # Create a decoder layer similar to Thinker but with dense MLP
            layer = nn.Module()
            layer.self_attn = Qwen3OmniMoeThinkerTextAttention(
                hidden_size=cp_config.hidden_size,
                num_heads=cp_config.num_attention_heads,
                num_kv_heads=cp_config.num_key_value_heads,
                layer_id=idx,
                rope_theta=getattr(cp_config, "rope_theta", 1000000.0),
                rope_scaling=getattr(cp_config, "rope_scaling", None),
                max_position_embeddings=getattr(
                    cp_config, "max_position_embeddings", 32768
                ),
                head_dim=getattr(
                    cp_config,
                    "head_dim",
                    cp_config.hidden_size // cp_config.num_attention_heads,
                ),
                rms_norm_eps=cp_config.rms_norm_eps,
                attention_bias=cp_config.attention_bias,
                config=cp_config,
                quant_config=quant_config,
                prefix=add_prefix(f"model.layers.{idx}.self_attn", prefix),
                dual_chunk_attention_config=None,
                alt_stream=alt_stream,
            )
            layer.mlp = Qwen3OmniMoeTalkerDenseMLP(
                cp_config.hidden_size,
                cp_config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix(f"model.layers.{idx}.mlp", prefix),
            )
            layer.input_layernorm = RMSNorm(
                cp_config.hidden_size, eps=cp_config.rms_norm_eps
            )
            layer.post_attention_layernorm = RMSNorm(
                cp_config.hidden_size, eps=cp_config.rms_norm_eps
            )
            self.model.layers.append(layer)

        self.model.norm = RMSNorm(cp_config.hidden_size, eps=cp_config.rms_norm_eps)

        # 15 LM heads for predicting layers 1-15
        self.lm_head = nn.ModuleList(
            [
                ReplicatedLinear(
                    cp_config.hidden_size,
                    cp_config.vocab_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix(f"lm_head.{i}", prefix),
                )
                for i in range(config.num_code_groups - 1)
            ]
        )

        # KV cache dimensions (read from first layer after construction)
        first_attn = self.model.layers[0].self_attn
        self._cache_num_kv_heads = first_attn.num_kv_heads
        self._cache_num_heads = first_attn.num_heads
        self._cache_head_dim = first_attn.head_dim
        self._cache_num_layers = cp_config.num_hidden_layers
        self._cache_max_seq = config.num_code_groups + 1  # 17
        self._kv_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # KV-cached forward (prefill + decode)
    # ------------------------------------------------------------------

    def _ensure_kv_cache(
        self, device: torch.device, dtype: torch.dtype, max_batch_size: int = 1
    ) -> None:
        if (
            self._kv_cache is not None
            and self._kv_cache.device == device
            and self._kv_cache.dtype == dtype
            and self._kv_cache.shape[2] >= max_batch_size
        ):
            return
        # [num_layers, 2(K/V), max_batch, num_kv_heads, max_seq, head_dim]
        self._kv_cache = torch.zeros(
            self._cache_num_layers,
            2,
            max_batch_size,
            self._cache_num_kv_heads,
            self._cache_max_seq,
            self._cache_head_dim,
            device=device,
            dtype=dtype,
        )

    def _forward_cached(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        """Forward through code predictor using KV cache.

        Args:
            inputs_embeds: [1, seq_len, hidden_size]
            positions: [seq_len] position indices
            cache_pos: starting slot in the KV cache to write new K,V
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.model.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states.reshape(-1, hidden_size))
            normed = normed.reshape(batch_size, seq_len, hidden_size)
            attn_out = self._cached_self_attention(
                attn=layer.self_attn,
                hidden_states=normed,
                positions=positions,
                layer_idx=layer_idx,
                cache_pos=cache_pos,
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(
                hidden_states.reshape(-1, hidden_size)
            )
            mlp_out = layer.mlp(normed).reshape(batch_size, seq_len, hidden_size)
            hidden_states = residual + mlp_out

        hidden_states = self.model.norm(hidden_states.reshape(-1, hidden_size))
        return hidden_states.reshape(batch_size, seq_len, hidden_size)

    def _cached_self_attention(
        self,
        *,
        attn: Qwen3OmniMoeThinkerTextAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
        cache_pos: int,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size)

        qkv, _ = attn.qkv_proj(flat_hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            head_dim=attn.head_dim,
            alt_stream=attn.alt_stream,
        )
        q, k = attn.rotary_emb(
            positions.reshape(-1), q, k, fused_set_kv_buffer_arg=None
        )

        q = q.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(
            1, 2
        )
        k = k.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(
            1, 2
        )
        v = v.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(
            1, 2
        )

        # Write new K,V into cache
        kv_end = cache_pos + seq_len
        assert self._kv_cache is not None
        self._kv_cache[layer_idx, 0, :batch_size, :, cache_pos:kv_end, :] = k
        self._kv_cache[layer_idx, 1, :batch_size, :, cache_pos:kv_end, :] = v

        # Read full cached K,V
        k_full = self._kv_cache[layer_idx, 0, :batch_size, :, :kv_end, :]
        v_full = self._kv_cache[layer_idx, 1, :batch_size, :, :kv_end, :]

        # GQA expansion
        num_kv_groups = attn.num_heads // attn.num_kv_heads
        k_full = _repeat_kv(k_full, num_kv_groups)
        v_full = _repeat_kv(v_full, num_kv_groups)

        attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * attn.scaling

        # Causal mask only needed for prefill (seq_len > 1)
        if seq_len > 1:
            q_pos = torch.arange(cache_pos, kv_end, device=q.device)
            k_pos = torch.arange(kv_end, device=q.device)
            causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.matmul(attn_weights, v_full)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size * seq_len, attn.num_heads * attn.head_dim
        )
        attn_output, _ = attn.o_proj(attn_output)
        return attn_output.reshape(batch_size, seq_len, hidden_size)

    # ------------------------------------------------------------------

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward through the code predictor (matches vLLM-Omni's mtp_block pattern).

        Args:
            inputs_embeds: [batch, seq_len, hidden_size] or [total_tokens, hidden_size]
            positions: [total_tokens] position indices
            forward_batch: SGLang's forward batch info (None for direct call)

        Returns:
            hidden_states: same shape as inputs_embeds
        """
        if forward_batch is None:
            return self._forward_direct(
                inputs_embeds=inputs_embeds, positions=positions
            )

        # SGLang layers expect 2D [total_tokens, hidden]; reshape if 3D
        needs_reshape = inputs_embeds.ndim == 3
        if needs_reshape:
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            hidden_states = inputs_embeds.reshape(-1, hidden_size)
        else:
            hidden_states = inputs_embeds

        for layer in self.model.layers:
            # Pre-norm self-attention with residual
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states = residual + hidden_states

            # Pre-norm MLP with residual
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # Final norm
        hidden_states = self.model.norm(hidden_states)

        if needs_reshape:
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        return hidden_states

    def _forward_direct(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run the code predictor without SGLang runtime state.

        The streaming code predictor executor invokes this model directly, so there is
        no `ForwardBatch` and no KV-cache backend. Fall back to a small eager causal
        attention path that reuses the loaded SGLang weights.
        """
        needs_reshape = inputs_embeds.ndim == 3
        if needs_reshape:
            hidden_states = inputs_embeds
        else:
            hidden_states = inputs_embeds.unsqueeze(0)

        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_positions = self._flatten_positions(
            positions=positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
        )

        for layer in self.model.layers:
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states.reshape(-1, hidden_size))
            normed = normed.reshape(batch_size, seq_len, hidden_size)
            attn_out = self._direct_self_attention(
                attn=layer.self_attn,
                hidden_states=normed,
                positions=flat_positions,
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(
                hidden_states.reshape(-1, hidden_size)
            )
            mlp_out = layer.mlp(normed).reshape(batch_size, seq_len, hidden_size)
            hidden_states = residual + mlp_out

        hidden_states = self.model.norm(hidden_states.reshape(-1, hidden_size))
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)
        if needs_reshape:
            return hidden_states
        return hidden_states.squeeze(0)

    def _flatten_positions(
        self,
        *,
        positions: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if positions.ndim == 1:
            if positions.numel() == seq_len and batch_size > 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)
            elif positions.numel() != batch_size * seq_len:
                raise ValueError(
                    f"Unexpected positions shape {tuple(positions.shape)} for "
                    f"batch_size={batch_size}, seq_len={seq_len}"
                )
        elif positions.ndim == 2:
            if tuple(positions.shape) != (batch_size, seq_len):
                raise ValueError(
                    f"Unexpected positions shape {tuple(positions.shape)} for "
                    f"batch_size={batch_size}, seq_len={seq_len}"
                )
        else:
            raise ValueError(f"Unsupported positions rank: {positions.ndim}")
        return positions.to(device=device, dtype=torch.long).reshape(-1)

    def _direct_self_attention(
        self,
        *,
        attn: Qwen3OmniMoeThinkerTextAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size)

        qkv, _ = attn.qkv_proj(flat_hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            head_dim=attn.head_dim,
            alt_stream=attn.alt_stream,
        )
        q, k = attn.rotary_emb(positions, q, k, fused_set_kv_buffer_arg=None)

        q = q.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(
            1, 2
        )
        k = k.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(
            1, 2
        )
        v = v.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(
            1, 2
        )

        num_kv_groups = attn.num_heads // attn.num_kv_heads
        k = _repeat_kv(k, num_kv_groups)
        v = _repeat_kv(v, num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_weights.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size * seq_len, attn.num_heads * attn.head_dim
        )
        attn_output, _ = attn.o_proj(attn_output)
        return attn_output.reshape(batch_size, seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Top-level Talker Model
# ---------------------------------------------------------------------------


class Qwen3OmniTalker(nn.Module):
    """Talker: Text-to-Audio generation model."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # SGLang passes the top-level HF config; extract the talker sub-config
        if not isinstance(config, Qwen3OmniMoeTalkerConfig):
            if hasattr(config, "talker_config"):
                config = config.talker_config
            else:
                config = Qwen3OmniMoeTalkerConfig()
        self.config = config

        # Projection MLPs (thinker hidden -> talker hidden)
        self.text_projection = ResizeMLP(
            config.thinker_hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("text_projection", prefix),
        )
        self.hidden_projection = ResizeMLP(
            config.thinker_hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("hidden_projection", prefix),
        )

        # Main components
        self.model = Qwen3OmniMoeTalkerTextModel(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.codec_head = ReplicatedLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("codec_head", prefix),
        )
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictor(
            config,
            quant_config=quant_config,
            prefix=add_prefix("code_predictor", prefix),
        )

        # LogitsProcessor for SGLang pipeline integration
        from sglang.srt.layers.logits_processor import (
            LogitsProcessor as SGLangLogitsProcessor,
        )

        self.logits_processor = SGLangLogitsProcessor(config.text_config)

        # Inline code predictor state (set by setup_code_predictor_decode)
        self._cp_enabled = False
        self._output_codes: Optional[torch.Tensor] = None
        self._output_embeds: Optional[torch.Tensor] = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def prepare_input_embeds(
        self,
        thinker_embeds: Optional[torch.Tensor] = None,
        thinker_hidden_states: Optional[torch.Tensor] = None,
        is_multimodal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project thinker outputs to talker's hidden dimension.

        - Text positions:       text_projection(thinker_embeds)
        - Multimodal positions:  hidden_projection(thinker_hidden_states)

        If no mask is provided, all positions use text_projection.
        """
        if thinker_hidden_states is None or is_multimodal_mask is None:
            return self.text_projection(thinker_embeds)
        if thinker_embeds is None:
            return self.hidden_projection(thinker_hidden_states)

        # Mixed: use mask to select projection
        output = torch.empty(
            (*thinker_embeds.shape[:-1], self.config.text_config.hidden_size),
            device=thinker_embeds.device,
            dtype=thinker_embeds.dtype,
        )
        if is_multimodal_mask.any():
            output[is_multimodal_mask] = self.hidden_projection(
                thinker_hidden_states[is_multimodal_mask]
            )
        text_mask = ~is_multimodal_mask
        if text_mask.any():
            output[text_mask] = self.text_projection(thinker_embeds[text_mask])
        return output

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
        input_deepstack_mask: Optional[torch.Tensor] = None,
        input_embeds_are_projected: bool = False,
    ):
        """Forward pass through the talker MoE backbone.

        When input_embeds is provided (prefill with thinker hidden states),
        project them via prepare_input_embeds before running the backbone.

        Args:
            input_ids: codec token ids (used during decode)
            positions: position indices
            forward_batch: SGLang's forward batch info
            input_embeds: thinker hidden states [total_tokens, thinker_hidden_size]
                          (provided by SGLang when Req.input_embeds is set)
            input_deepstack_embeds: optional layer-N thinker hidden states
            input_deepstack_mask: positions that should use hidden_projection
            input_embeds_are_projected: whether `input_embeds` is already in talker space

        Returns:
            LogitsProcessorOutput with codec logits
        """
        if input_embeds is not None and not input_embeds_are_projected:
            # Prefill: project thinker hidden states → talker dimension
            deepstack_hidden = input_deepstack_embeds
            deepstack_mask = input_deepstack_mask
            if deepstack_hidden is not None and deepstack_mask is not None:
                input_embeds = self.prepare_input_embeds(
                    thinker_embeds=input_embeds,
                    thinker_hidden_states=deepstack_hidden,
                    is_multimodal_mask=deepstack_mask,
                )
            else:
                input_embeds = self.prepare_input_embeds(thinker_embeds=input_embeds)

        # Decode: apply feedback from pre-allocated buffers (CUDA-graph safe)
        if input_embeds is None and self._cp_enabled:
            bs = input_ids.shape[0]
            base_embeds = self.model.codec_embedding(input_ids)
            mask = self._feedback_mask[:bs].unsqueeze(-1)
            input_embeds = torch.where(mask, self._feedback_buffer[:bs], base_embeds)
            self._feedback_mask.zero_()

        torch.cuda.nvtx.range_push("talker_transformer")
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        torch.cuda.nvtx.range_pop()
        if forward_batch.forward_mode.is_extend() and input_embeds is not None:
            return self._manual_extend_logits(hidden_states, forward_batch)
        logits_output = self.logits_processor(
            input_ids, hidden_states, self.codec_head, forward_batch
        )
        # Inline code prediction during decode (unified talker+MTP forward)
        if self._cp_enabled and not forward_batch.forward_mode.is_extend():
            torch.cuda.nvtx.range_push("code_predictor_inline")
            self._decode_codebooks(logits_output.next_token_logits, hidden_states)
            torch.cuda.nvtx.range_pop()
        return logits_output

    def _manual_extend_logits(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Compute next-token logits for talker prefill without LogitsProcessor.

        The projected-prompt prefill path only needs next-token logits. Using a
        tiny local implementation avoids the generic SGLang logits processor path
        that currently fails on this extend batch.
        """
        last_index = self._extend_last_index(forward_batch, hidden_states.device)
        pruned_states = hidden_states[last_index]
        next_token_logits, _ = self.codec_head(pruned_states)
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=pruned_states,
        )

    def _extend_last_index(
        self,
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)

        if (
            forward_batch.padded_static_len is not None
            and forward_batch.padded_static_len >= 0
        ):
            idx = torch.arange(
                len(extend_seq_lens), device=device, dtype=extend_seq_lens.dtype
            )
            return (
                idx * forward_batch.padded_static_len
                + extend_seq_lens.to(device=device)
                - 1
            )

        seq_lens = extend_seq_lens.to(device=device)
        return torch.cumsum(seq_lens, dim=0) - 1

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute layer-0 codec logits."""
        logits, _ = self.codec_head(hidden_states)
        return logits

    # ------------------------------------------------------------------
    # Inline code predictor (unified with talker forward, following #153)
    # ------------------------------------------------------------------

    def setup_code_predictor_decode(self, max_batch_size: int = 1) -> None:
        """Pre-allocate output buffers for inline code prediction.

        Call after model load, before CUDA graph capture.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        num_groups = self.config.num_code_groups
        hidden_size = self.config.text_config.hidden_size

        # Code predictor output buffers
        self._output_codes = torch.zeros(
            max_batch_size, num_groups, dtype=torch.long, device=device
        )
        self._output_embeds = torch.zeros(
            max_batch_size, hidden_size, dtype=dtype, device=device
        )
        # Feedback input buffers (written by model runner before forward)
        self._feedback_buffer = torch.zeros(
            max_batch_size, hidden_size, dtype=dtype, device=device
        )
        self._feedback_mask = torch.zeros(
            max_batch_size, dtype=torch.bool, device=device
        )
        self.code_predictor._ensure_kv_cache(device, dtype, max_batch_size)
        # Pre-allocate position tensors (no tensor creation during CUDA graph capture)
        self._cp_prefill_pos = torch.arange(2, device=device)
        self._cp_decode_pos = [
            torch.tensor([i + 2], device=device) for i in range(num_groups - 1)
        ]
        self._cp_enabled = True

    def _decode_codebooks(
        self,
        next_token_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Run batched code predictor inline after talker forward.

        Args:
            next_token_logits: [bs, vocab_size]
            hidden_states: [bs, hidden_size] last-token hidden from talker
        """
        bs = next_token_logits.shape[0]
        device = next_token_logits.device
        num_groups = self.config.num_code_groups

        # Greedy sample layer0 code
        layer0_codes = next_token_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]
        layer0_embed = self.model.codec_embedding(layer0_codes)  # [bs, 1, hidden]

        # Prefill: [hidden_state, layer0_embed] at positions [0, 1]
        prefill_input = torch.cat(
            [
                hidden_states.unsqueeze(1),
                layer0_embed,
            ],
            dim=1,
        )  # [bs, 2, hidden]
        prefill_pos = self._cp_prefill_pos
        hidden_out = self.code_predictor._forward_cached(
            prefill_input, prefill_pos, cache_pos=0
        )

        self._output_codes[:bs, 0] = layer0_codes.squeeze(-1)
        embeds_sum = layer0_embed[:, 0, :]  # [bs, hidden]

        # Decode: predict layers 1 to N-1 (greedy, batched)
        for layer_idx in range(num_groups - 1):
            logits, _ = self.code_predictor.lm_head[layer_idx](hidden_out[:, -1:, :])
            code = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [bs, 1]
            self._output_codes[:bs, layer_idx + 1] = code.squeeze(-1)

            new_embed = self.code_predictor.model.codec_embedding[layer_idx](
                code
            )  # [bs, 1, hidden]
            embeds_sum = embeds_sum + new_embed[:, 0, :]

            decode_pos = self._cp_decode_pos[layer_idx]
            hidden_out = self.code_predictor._forward_cached(
                new_embed, decode_pos, cache_pos=layer_idx + 2
            )

        self._output_embeds[:bs] = embeds_sum

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load weights from HuggingFace checkpoint."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict

        # Stacked parameters mapping
        stacked_params = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MoE expert parameters mapping
        expert_params = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.text_config.num_experts,
        )

        for name, loaded_weight in weights:
            # Support both monolithic (talker.xxx) and split (xxx) checkpoints
            if name.startswith("talker."):
                name = name[len("talker.") :]
            elif "." in name and name.split(".")[0] in ("thinker", "code2wav"):
                continue

            # 1. Handle stacked parameters (qkv_proj, gate_up_proj)
            handled = False
            for param_name, weight_name, shard_id in stacked_params:
                if weight_name in name and "mlp.experts" not in name:
                    param = params_dict.get(name.replace(weight_name, param_name))
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, shard_id)
                        handled = True
                        break
            if handled:
                continue

            # 2. Handle MoE expert parameters
            for param_name, weight_name, expert_id, shard_id in expert_params:
                if weight_name in name:
                    mapped = name.replace(weight_name, param_name)
                    param = params_dict.get(mapped)
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        handled = True
                        break
            if handled:
                continue

            # 3. Direct parameter loading
            param = params_dict.get(name)
            if param is not None:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
