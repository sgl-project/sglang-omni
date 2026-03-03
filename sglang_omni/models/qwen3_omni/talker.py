"""SGLang-native Talker model for Qwen3-Omni.

Simplified implementation:
- Reuse Thinker's components where possible
- Only define Talker-specific parts (Shared Expert MoE)
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
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
    top_k_top_p_sampling_from_probs,
)
from sglang_omni.vendor.sglang.utils import make_layers

# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------


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

        # --- Routed experts (no all-reduce yet) ---
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_output = self.experts(hidden_states, topk_output)

        # --- Shared expert (no all-reduce, reduce_results=False) ---
        shared_output = self.shared_expert(hidden_states)
        shared_gate, _ = self.shared_expert_gate(hidden_states)
        shared_output = shared_output * torch.sigmoid(shared_gate)

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
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

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

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward through the code predictor (matches vLLM-Omni's mtp_block pattern).

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            positions: [total_tokens] position indices
            forward_batch: SGLang's forward batch info

        Returns:
            hidden_states: [batch, seq_len, hidden_size] - final hidden states
        """
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
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level Talker Model
# ---------------------------------------------------------------------------


class Qwen3OmniTalker(nn.Module):
    """Talker: Text-to-Audio generation model."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the talker MoE backbone.

        Args:
            input_ids: codec token ids (used when inputs_embeds is None)
            positions: position indices
            forward_batch: SGLang's forward batch info
            inputs_embeds: pre-computed input embeddings (from prepare_input_embeds)

        Returns:
            hidden_states from the talker backbone
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute layer-0 codec logits."""
        logits, _ = self.codec_head(hidden_states)
        return logits

    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate residual RVQ codes (layers 1 to N-1) for each position.

        Matches vLLM-Omni's code_predictor_forward:
        - Per-position autoregressive loop
        - Growing sequence: [talker_hidden, layer0_embed, layer1_embed, ...]
        - Last-token hidden state → lm_head → predicted code

        Args:
            layer0_codes: [batch, seq_len] - layer-0 codec codes from argmax/sample
            talker_hidden: [batch, seq_len, hidden] - hidden states from talker backbone

        Returns:
            result_codes: [batch, num_code_groups, seq_len] - all codes (layer 0 + predicted)
            summed_embeddings: [batch, seq_len, hidden] - sum of all layer embeddings
        """
        batch_size, seq_len = layer0_codes.shape
        num_groups = self.config.num_code_groups
        all_codes_per_pos = []
        all_summed_per_pos = []

        for pos in range(seq_len):
            layer0_code = layer0_codes[:, pos : pos + 1]  # [batch, 1]
            layer0_embed = self.model.codec_embedding(layer0_code)  # [batch, 1, hidden]
            last_hidden = talker_hidden[:, pos : pos + 1, :]  # [batch, 1, hidden]

            # Initial input: [talker_hidden_at_pos, layer0_embed]
            current_input = torch.cat(
                [last_hidden, layer0_embed], dim=1
            )  # [batch, 2, hidden]
            pos_codes = [layer0_code]

            # Predict layers 1 to N-1 autoregressively
            for layer_idx in range(num_groups - 1):
                # Forward through code predictor transformer
                predictor_hidden = self.code_predictor(
                    inputs_embeds=current_input,
                    positions=torch.arange(
                        current_input.shape[1], device=current_input.device
                    ),
                    forward_batch=None,
                )

                # Predict from last token's hidden state (top_k=50, top_p=0.8 matching HF/vLLM-Omni)
                logits, _ = self.code_predictor.lm_head[layer_idx](
                    predictor_hidden[:, -1:, :]
                )
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                code = top_k_top_p_sampling_from_probs(
                    probs, top_k=50, top_p=0.8
                )  # [batch, 1]
                pos_codes.append(code)

                # Append new embedding to growing sequence
                new_embed = self.code_predictor.model.codec_embedding[layer_idx](code)
                current_input = torch.cat([current_input, new_embed], dim=1)

            # Stack all layers for this position: [batch, num_code_groups, 1]
            all_codes_per_pos.append(torch.stack(pos_codes, dim=1))

            # Build summed_embeddings for this position (for Code2Wav):
            # current_input = [talker_hidden, l0_embed, l1_embed, ..., lN-1_embed]
            # We want sum of all codec embeddings: l0 + l1 + ... + lN-1
            # That's current_input[:, 1:, :] (skip the talker_hidden at index 0)
            codec_embeds = current_input[:, 1:, :]  # [batch, num_code_groups, hidden]
            pos_summed = codec_embeds.sum(dim=1, keepdim=True)  # [batch, 1, hidden]
            all_summed_per_pos.append(pos_summed)

        # [batch, num_code_groups, seq_len]
        result_codes = torch.cat(all_codes_per_pos, dim=2)
        # [batch, seq_len, hidden]
        summed_embeddings = torch.cat(all_summed_per_pos, dim=1)

        return result_codes, summed_embeddings

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
            # Strip "talker." prefix if present
            if not name.startswith("talker."):
                continue
            name = name[len("talker.") :]

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
