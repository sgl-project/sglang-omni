"""
SGLang-native Talker model for Qwen3-Omni compatiable with hf formatting.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni_v1.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)
from sglang_omni_v1.models.qwen3_omni.hf_config import (
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
)
from sglang_omni_v1.vendor.sglang.core import ForwardBatch
from sglang_omni_v1.vendor.sglang.distributed import tensor_model_parallel_all_reduce
from sglang_omni_v1.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QuantizationConfig,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang_omni_v1.vendor.sglang.models import apply_qk_norm
from sglang_omni_v1.vendor.sglang.server_args import get_global_server_args
from sglang_omni_v1.vendor.sglang.utils import make_layers


def _bind_default_weight_loaders(module: nn.Module) -> None:
    for param in module.parameters():
        if "weight_loader" not in param.__dict__:
            param.weight_loader = default_weight_loader


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
        hidden_states = hidden_states.view(-1, hidden_dim)
        linear_hidden_states = hidden_states
        linear_dtype = self.shared_expert_gate.weight.dtype
        if linear_hidden_states.dtype != linear_dtype:
            linear_hidden_states = linear_hidden_states.to(dtype=linear_dtype)

        # Shared branch must consume the original MLP input before routed experts.
        # The fused MoE implementation mutates `hidden_states` in-place.
        shared_output = self.shared_expert(linear_hidden_states)
        shared_gate, _ = self.shared_expert_gate(linear_hidden_states)
        shared_output = shared_output * torch.sigmoid(shared_gate)

        # --- Routed experts (no all-reduce yet) ---
        router_logits, _ = self.gate(linear_hidden_states)
        topk_output = self.topk(linear_hidden_states, router_logits)
        routed_output = self.experts(linear_hidden_states, topk_output)

        # --- Combine then unified all-reduce ---
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                **kwargs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


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
        max_batch_size = get_global_server_args().max_running_requests
        self._cp_enabled = True
        self._feedback_buffer = torch.zeros(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self._feedback_mask = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=self.codec_embedding.weight.device,
        )

        # Disable fused_qk_norm_rope so the separate QK-norm + RoPE path is
        # used.  The fp32 weight promotion + cast_x_before_out_mul is applied
        # in load_weights() (after weights are loaded from checkpoint).
        for idx in range(self.start_layer, self.end_layer):
            self.layers[idx].self_attn.use_fused_qk_norm_rope = False
            self.layers[idx].self_attn.compatible_with_fused_qk_norm_rope = False

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
            if self._cp_enabled:
                bs = hidden_states.shape[0]
                feedback_mask = self._feedback_mask[:bs]
                hidden_states = torch.where(
                    feedback_mask.unsqueeze(-1),
                    self._feedback_buffer[:bs].to(hidden_states.dtype),
                    hidden_states,
                )
                self._feedback_mask[:bs] = False
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
                rope_theta=cp_config.rope_theta,
                rope_scaling=cp_config.rope_scaling,
                max_position_embeddings=cp_config.max_position_embeddings,
                head_dim=cp_config.head_dim,
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

        # Use SDPA to match HF's attention computation
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
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
        if not isinstance(config, Qwen3OmniMoeTalkerConfig):
            config = Qwen3OmniMoeTalkerConfig(**config.talker_config.to_dict())
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

        device = self.model.codec_embedding.weight.device
        hidden_size = config.text_config.hidden_size
        predictor_len = config.num_code_groups + 1
        max_batch_size = get_global_server_args().max_running_requests
        self._cp_enabled = self.model._cp_enabled
        self._feedback_buffer = self.model._feedback_buffer
        self._feedback_mask = self.model._feedback_mask
        self._predictor_input_buffer = torch.zeros(
            max_batch_size,
            predictor_len,
            hidden_size,
            device=device,
            dtype=self.model.codec_embedding.weight.dtype,
        )
        self._predictor_positions = torch.arange(
            predictor_len,
            device=device,
            dtype=torch.long,
        )
        predictor_num_layers = len(self.code_predictor.model.layers)
        predictor_num_kv_heads = self.code_predictor.model.layers[
            0
        ].self_attn.num_kv_heads
        predictor_head_dim = self.code_predictor.model.layers[0].self_attn.head_dim
        self._predictor_k_cache = torch.zeros(
            predictor_num_layers,
            max_batch_size,
            predictor_num_kv_heads,
            predictor_len,
            predictor_head_dim,
            device=device,
            dtype=self.model.codec_embedding.weight.dtype,
        )
        self._predictor_v_cache = torch.zeros_like(self._predictor_k_cache)
        self._sampled_token_ids = torch.zeros(
            max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self._repetition_mask = torch.zeros(
            max_batch_size,
            config.text_config.vocab_size,
            dtype=torch.bool,
            device=device,
        )
        self._repetition_penalties = torch.ones(
            max_batch_size,
            1,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._suppress_mask = torch.zeros(
            max_batch_size,
            config.text_config.vocab_size,
            dtype=torch.bool,
            device=device,
        )
        self._sampling_temperatures = torch.ones(
            max_batch_size,
            1,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._sampling_top_ps = torch.ones(
            max_batch_size,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._sampling_top_ks = torch.full(
            (max_batch_size,),
            1,
            dtype=torch.int32,
            device=device,
        )
        self._sampling_min_ps = torch.zeros(
            max_batch_size,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._output_codes = torch.zeros(
            max_batch_size,
            config.num_code_groups,
            dtype=torch.long,
            device=device,
        )
        self._output_embeds = torch.zeros(
            max_batch_size,
            hidden_size,
            device=device,
            dtype=self.model.codec_embedding.weight.dtype,
        )
        _bind_default_weight_loaders(self)
        self._cached_params_dict = dict(self.named_parameters())
        self._sampler = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    @staticmethod
    def _sample_code_predictor_token(logits: torch.Tensor) -> torch.Tensor:
        # Match HF generate(do_sample=False, temperature=0.0) behavior for the
        # residual code predictor by taking the highest-logit token directly.
        next_code = torch.argmax(logits[:, -1, :], dim=-1)
        if next_code.ndim == 1:
            next_code = next_code.unsqueeze(-1)
        return next_code

    def prepare_decode_buffers(self, requests: list) -> None:
        batch_size = len(requests)
        self._repetition_mask[:batch_size] = False
        self._repetition_penalties[:batch_size].fill_(1.0)
        self._suppress_mask[:batch_size] = False
        self._sampling_temperatures[:batch_size].fill_(1.0)
        self._sampling_top_ps[:batch_size].fill_(1.0)
        self._sampling_top_ks[:batch_size].fill_(1)
        self._sampling_min_ps[:batch_size].zero_()

        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            sampling_params = req.sampling_params

            penalty = float(sampling_params.repetition_penalty)
            self._repetition_penalties[row_idx, 0] = penalty
            self._sampling_temperatures[row_idx, 0] = float(sampling_params.temperature)
            self._sampling_top_ps[row_idx] = float(sampling_params.top_p)
            self._sampling_top_ks[row_idx] = int(sampling_params.top_k)
            self._sampling_min_ps[row_idx] = float(sampling_params.min_p)
            if penalty != 1.0 and req.output_ids:
                token_ids = torch.as_tensor(
                    list({int(token_id) for token_id in req.output_ids}),
                    dtype=torch.long,
                    device=self._repetition_mask.device,
                )
                valid = token_ids[
                    (token_ids >= 0) & (token_ids < self._repetition_mask.shape[1])
                ]
                if valid.numel() > 0:
                    self._repetition_mask[row_idx, valid] = True

            suppress_tokens = data.suppress_tokens or req._codec_suppress_tokens
            if suppress_tokens:
                token_ids = torch.as_tensor(
                    [int(token_id) for token_id in suppress_tokens],
                    dtype=torch.long,
                    device=self._suppress_mask.device,
                )
                valid = token_ids[
                    (token_ids >= 0) & (token_ids < self._suppress_mask.shape[1])
                ]
                if valid.numel() > 0:
                    self._suppress_mask[row_idx, valid] = True

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

        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend() and input_embeds is not None:
            return self._manual_extend_logits(hidden_states, forward_batch)
        logits_output = self._manual_decode_logits(hidden_states)
        if forward_batch.forward_mode.is_decode():
            sampled_token_ids = self._sample_decode_tokens(
                logits_output.next_token_logits,
                forward_batch,
            )
            batch_size = sampled_token_ids.shape[0]
            self._sampled_token_ids[:batch_size].copy_(sampled_token_ids)
            self.code_predictor_forward(
                sampled_token_ids.unsqueeze(1),
                hidden_states.unsqueeze(1),
            )
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

    def _manual_decode_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> LogitsProcessorOutput:
        """Compute codec logits directly for decode/CUDA-graph capture."""
        next_token_logits, _ = self.codec_head(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=hidden_states,
        )

    def _sample_decode_tokens(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        logits = logits.clone()

        penalties = self._repetition_penalties[:batch_size].to(dtype=logits.dtype)
        penalized_logits = torch.where(
            logits > 0, logits / penalties, logits * penalties
        )
        logits = torch.where(
            self._repetition_mask[:batch_size], penalized_logits, logits
        )
        logits = logits.masked_fill(self._suppress_mask[:batch_size], float("-inf"))

        logits_output = LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=None,
        )
        if self._sampler is None:
            return torch.argmax(logits, dim=-1)
        sampling_info = self._build_static_sampling_info(batch_size)
        sampled = self._sampler(
            logits_output,
            sampling_info,
            False,
            [0] * batch_size,
            [[] for _ in range(batch_size)],
            forward_batch.positions,
        )
        if sampled.ndim > 1:
            sampled = sampled.squeeze(-1)
        return sampled

    def _build_static_sampling_info(self, batch_size: int) -> SamplingBatchInfo:
        return SamplingBatchInfo(
            temperatures=self._sampling_temperatures[:batch_size],
            top_ps=self._sampling_top_ps[:batch_size],
            top_ks=self._sampling_top_ks[:batch_size],
            min_ps=self._sampling_min_ps[:batch_size],
            # Keep sampler control flow static during graph capture while
            # preserving SGLang's actual sampling kernel semantics.
            is_all_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=self.config.text_config.vocab_size,
            grammars=[],
            vocab_mask=None,
            apply_mask_func=None,
            penalizer_orchestrator=None,
            acc_linear_penalties=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            sampling_seed=None,
            device="cuda",
            logit_bias=None,
        )

    def _extend_last_index(
        self,
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = forward_batch.extend_seq_lens
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

    def _code_predictor_forward_incremental(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the predictor one token at a time using a per-layer single-turn KV cache."""
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)

        batch_size, seq_len = layer0_codes.shape
        if talker_hidden.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "talker_hidden shape must align with layer0_codes: "
                f"{tuple(talker_hidden.shape)} vs {tuple(layer0_codes.shape)}"
            )

        predictor_input = self._predictor_input_buffer[:batch_size]
        predictor_input.zero_()
        num_groups = self.config.num_code_groups
        runtime_single_token = seq_len == 1
        if runtime_single_token:
            self._output_codes[:batch_size].zero_()
            self._output_embeds[:batch_size].zero_()
            result_codes = self._output_codes[:batch_size].unsqueeze(-1)
            summed_embeddings = self._output_embeds[:batch_size].unsqueeze(1)
        else:
            result_codes = torch.empty(
                (batch_size, num_groups, seq_len),
                dtype=torch.long,
                device=layer0_codes.device,
            )
            summed_embeddings = torch.empty(
                (batch_size, seq_len, predictor_input.shape[-1]),
                dtype=predictor_input.dtype,
                device=predictor_input.device,
            )

        for pos in range(seq_len):
            layer0_code = layer0_codes[:, pos : pos + 1]
            layer0_embed = self.get_input_embeddings()(layer0_code).to(
                dtype=predictor_input.dtype
            )
            pos_codes = result_codes[:, :, pos]
            pos_summed = summed_embeddings[:, pos, :]
            pos_summed.zero_()
            predictor_input[:, 0, :] = talker_hidden[:, pos, :].to(
                dtype=predictor_input.dtype
            )
            predictor_input[:, 1, :] = layer0_embed[:, 0, :]
            pos_codes[:, 0].copy_(layer0_code[:, 0])
            pos_summed.add_(layer0_embed[:, 0, :])

            cache_len = 0
            self._predictor_forward_one_token(
                token_embeds=predictor_input[:, 0:1, :],
                batch_size=batch_size,
                cache_len=cache_len,
            )
            cache_len += 1
            last_hidden = self._predictor_forward_one_token(
                token_embeds=predictor_input[:, 1:2, :],
                batch_size=batch_size,
                cache_len=cache_len,
            )
            cache_len += 1

            for layer_idx in range(num_groups - 1):
                logits, _ = self.code_predictor.lm_head[layer_idx](last_hidden)
                next_code = self._sample_code_predictor_token(logits)
                pos_codes[:, layer_idx + 1].copy_(next_code[:, 0])

                new_embed = self.code_predictor.model.codec_embedding[layer_idx](
                    next_code
                ).to(dtype=predictor_input.dtype)
                predictor_input[:, layer_idx + 2, :] = new_embed[:, 0, :]
                pos_summed.add_(new_embed[:, 0, :])
                if layer_idx < num_groups - 2:
                    last_hidden = self._predictor_forward_one_token(
                        token_embeds=new_embed,
                        batch_size=batch_size,
                        cache_len=cache_len,
                    )
                    cache_len += 1

        return result_codes, summed_embeddings

    def _predictor_forward_one_token(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        """Process one predictor token against the cached prefix."""
        hidden_states = token_embeds
        hidden_size = hidden_states.shape[-1]
        positions = self._predictor_positions[cache_len : cache_len + 1].repeat(
            batch_size
        )

        for layer_idx, layer in enumerate(self.code_predictor.model.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states.reshape(-1, hidden_size))
            normed = normed.reshape(batch_size, 1, hidden_size)
            attn_out = self._predictor_cached_self_attention(
                layer_idx=layer_idx,
                attn=layer.self_attn,
                hidden_states=normed,
                positions=positions,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(
                hidden_states.reshape(-1, hidden_size)
            )
            mlp_out = layer.mlp(normed).reshape(batch_size, 1, hidden_size)
            hidden_states = residual + mlp_out

        hidden_states = self.code_predictor.model.norm(
            hidden_states.reshape(-1, hidden_size)
        )
        return hidden_states.reshape(batch_size, 1, hidden_size)

    def _predictor_cached_self_attention(
        self,
        *,
        layer_idx: int,
        attn: Qwen3OmniMoeThinkerTextAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        """Append one token to the per-layer KV cache and attend over the cached prefix."""
        _, seq_len, hidden_size = hidden_states.shape
        if seq_len != 1:
            raise ValueError(
                f"Cached predictor attention expects exactly one token, got seq_len={seq_len}"
            )

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
            positions.to(device=flat_hidden.device, dtype=torch.long),
            q,
            k,
            fused_set_kv_buffer_arg=None,
        )

        q = q.reshape(batch_size, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

        layer_k_cache = self._predictor_k_cache[layer_idx, :batch_size]
        layer_v_cache = self._predictor_v_cache[layer_idx, :batch_size]
        layer_k_cache[:, :, cache_len : cache_len + 1, :].copy_(k)
        layer_v_cache[:, :, cache_len : cache_len + 1, :].copy_(v)

        cached_k = layer_k_cache[:, :, : cache_len + 1, :]
        cached_v = layer_v_cache[:, :, : cache_len + 1, :]
        num_kv_groups = attn.num_heads // attn.num_kv_heads
        cached_k = _repeat_kv(cached_k, num_kv_groups)
        cached_v = _repeat_kv(cached_v, num_kv_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            cached_k,
            cached_v,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, attn.num_heads * attn.head_dim
        )
        attn_output, _ = attn.o_proj(attn_output)
        return attn_output.reshape(batch_size, 1, hidden_size)

    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate residual RVQ codes in batch and mirror outputs into static buffers."""
        result_codes, summed_embeddings = self._code_predictor_forward_incremental(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
        )
        return result_codes, summed_embeddings

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load weights from HuggingFace checkpoint."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

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
                        param.weight_loader(param, loaded_weight, shard_id)
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
                        param.weight_loader(
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
                param.weight_loader(param, loaded_weight)
