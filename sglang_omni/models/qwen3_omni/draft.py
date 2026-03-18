# SPDX-License-Identifier: Apache-2.0
"""Eagle3 Draft Model for Qwen3-Omni."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeTextConfig
from sglang_omni.models.qwen3_omni.thinker import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    LayerCommunicator,
    LayerScatterModes,
    MergedColumnParallelLinear,
    ParallelLMHead,
    QuantizationConfig,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
)
from sglang_omni.vendor.sglang.utils import add_prefix


class Qwen3OmniEagle3DenseMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        should_allreduce_fusion: bool,
        use_reduce_scatter: bool,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3OmniEagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", False)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        self.self_attn = Qwen3OmniMoeThinkerTextAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            dual_chunk_attention_config=dual_chunk_attention_config,
            alt_stream=alt_stream,
        )

        # Use Dense MLP for Draft Model
        # Assuming intermediate_size is provided in config or derived
        intermediate_size = getattr(
            config, "intermediate_size", config.hidden_size * 4
        )
        self.mlp = Qwen3OmniEagle3DenseMLP(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1,  # Draft model typically has 1 layer
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
            hidden_states,
            residual,
            forward_batch,
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected
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


class Qwen3OmniEagle3DraftModel(nn.Module):
    """
    Eagle3 Draft Model for Qwen3-Omni.
    It uses a dense Transformer structure and a projection layer (fc) to match
    the target model's hidden states.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Eagle3 specific projection layer
        # It projects target model's hidden state (typically same size, but could differ)
        self.fc = ReplicatedLinear(
            config.hidden_size * 2, # Example: target + embed or just hidden_size
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        # Single Decoder Layer
        alt_stream = torch.cuda.Stream()
        self.layer = Qwen3OmniEagle3DecoderLayer(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            prefix=add_prefix("layer", prefix),
            alt_stream=alt_stream,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Will be set by set_embed_and_head
        self.embed_tokens = None
        self.lm_head = None
        
        from sglang.srt.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config)

    def set_embed_and_head(self, embed: nn.Module, head: nn.Module):
        self.embed_tokens = embed
        self.lm_head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        hidden_states: torch.Tensor = None, # Target model's hidden states
    ) -> torch.Tensor:
        if input_embeds is None and self.embed_tokens is not None:
            input_embeds = self.embed_tokens(input_ids)
            
        # In Eagle3, draft model usually combines its own embedding with target's hidden state
        if hidden_states is not None:
            # Concatenate or process based on specific Eagle3 implementation
            # Assuming concatenation of target hidden and current token embed
            combined_states = torch.cat([hidden_states, input_embeds], dim=-1)
            draft_hidden, _ = self.fc(combined_states)
        else:
            # Fallback or initial step
            draft_hidden = input_embeds

        residual = None
        
        draft_hidden, residual = self.layer(
            positions, draft_hidden, forward_batch, residual
        )

        if draft_hidden.shape[0] != 0:
            if residual is None:
                draft_hidden = self.norm(draft_hidden)
            else:
                draft_hidden, _ = self.norm(draft_hidden, residual)

        if forward_batch.return_logprob and self.lm_head is not None:
            return self.logits_processor(
                input_ids,
                draft_hidden,
                self.lm_head,
                forward_batch,
            )
        
        return draft_hidden

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Implement specific weight loading for Draft Model if necessary
        # Often it just loads its own small checkpoint
        pass

EntryClass = Qwen3OmniEagle3DraftModel
