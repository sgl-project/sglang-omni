# SPDX-License-Identifier: Apache-2.0
"""Zonos3 backbone on SGLang: gated delta nets, two global attention layers, latent MoE."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_schedule
from torch import nn

from sglang_omni.models.zonos3.backbone_kernels import (
    select_experts,
    store_and_mix_values,
)
from sglang_omni.models.zonos3.hf_config import Zonos3Config
from sglang_omni.models.zonos3.mixers import Zonos3Attention, Zonos3GatedDeltaNet

# Checkpoint tensors stored with a leading split axis that the model fuses into one linear.
FUSED_CHECKPOINT_SUFFIXES = (".attention.wkv.weight", ".feed_forward.w_in.weight")
CHECKPOINT_RENAMES = (
    ("multi_embedder.embedders.", "embedders."),
    (".attention.out_weight", ".attention.norm.weight"),
    (".experts.gate_up_proj", ".experts.w13_weight"),
    (".experts.down_proj", ".experts.w2_weight"),
)
# The speaker embedding path is not used by the reference-prefix voice cloning layout.
UNUSED_CHECKPOINT_PREFIXES = ("speaker_projection.",)
# The runner fills this table every decode step, so the checkpoint has no tensor for it.
RUNTIME_PARAMETER_NAMES = frozenset({"decode_input_embedding.weight"})
# Checkpoint tensors of the decoder stack, which the model nests under its backbone module.
BACKBONE_CHECKPOINT_PREFIXES = ("layers.", "block_attnres_queries", "out_norm.")
BACKBONE_PARAMETER_PREFIX = "model."
# note (zhaochenyang20): searched on one H100 for these experts (64 of them, 256-wide latent, 1536-wide) at the 1 to 64 token decode batches; the generic table was tuned for wider inputs, and the ten MoE layers took 284 us against 319 us at 12 tokens and 170 us against 239 us at 4, with outputs bitwise equal (16-row tiles accumulate the same products in the same order).
SMALL_BATCH_EXPERT_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4,
}
SMALL_BATCH_EXPERT_MAX_TOKENS = 64


class FloatRMSNorm(nn.Module):
    """RMS norm computed in float32; returns float32."""

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        # note (zhaochenyang20): the checkpoint stores this weight in bfloat16, so float32 holds it exactly and no step casts it.
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.weight,
            self.epsilon,
        )


class Zonos3DenseFeedForward(nn.Module):
    def __init__(self, config: Zonos3Config) -> None:
        super().__init__()
        self.w_in = nn.Linear(
            config.hidden_size, 2 * config.dense_intermediate_size, bias=False
        )
        self.w_out = nn.Linear(
            config.dense_intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up, gate = self.w_in(hidden_states).chunk(2, dim=-1)
        return self.w_out(up * F.silu(gate))


class Zonos3Router(nn.Module):
    """Top-k on bias-corrected logits; weights are the renormalized softmax of the picks."""

    def __init__(self, config: Zonos3Config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.rmsnorm_eda = FloatRMSNorm(config.hidden_size, config.rms_norm_eps)
        # note (zhaochenyang20): the checkpoint stores this weight in bfloat16, so float32 holds it exactly and no step casts it.
        self.router_linear = nn.Linear(
            config.hidden_size, config.num_experts, bias=False, dtype=torch.float32
        )
        self.balancing_biases = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor) -> StandardTopKOutput:
        router_logits = self.router_linear(self.rmsnorm_eda(hidden_states))
        expert_weights, expert_ids = select_experts(
            router_logits, self.balancing_biases, top_k=self.top_k
        )
        return StandardTopKOutput(
            topk_weights=expert_weights,
            topk_ids=expert_ids,
            router_logits=router_logits,
        )


class Zonos3LatentMoE(nn.Module):
    """Experts run in a down-projected latent space."""

    def __init__(
        self,
        config: Zonos3Config,
        layer_id: int,
        quantization_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.router = Zonos3Router(config)
        self.latent_down = nn.Linear(
            config.hidden_size, config.expert_latent_size, bias=False
        )
        self.latent_up = nn.Linear(
            config.expert_latent_size, config.hidden_size, bias=False
        )
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            hidden_size=config.expert_latent_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            top_k=config.num_experts_per_tok,
            reduce_results=False,
            quant_config=quantization_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # A larger batch keeps the generic table's tiles.
        with override_config(
            SMALL_BATCH_EXPERT_KERNEL_CONFIG
            if hidden_states.shape[0] <= SMALL_BATCH_EXPERT_MAX_TOKENS
            else None
        ):
            expert_output = self.experts(
                self.latent_down(hidden_states), self.router(hidden_states)
            )
        return self.latent_up(expert_output)


class Zonos3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Zonos3Config,
        layer_id: int,
        quantization_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.is_full_attention = layer_id in config.full_attention_layer_ids
        self.attention_norm = FloatRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = FloatRMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.is_full_attention:
            self.attention: Zonos3Attention | Zonos3GatedDeltaNet = Zonos3Attention(
                config, layer_id
            )
        else:
            self.attention = Zonos3GatedDeltaNet(config, layer_id)
        if layer_id == 0:
            self.feed_forward: Zonos3DenseFeedForward | Zonos3LatentMoE = (
                Zonos3DenseFeedForward(config)
            )
        else:
            self.feed_forward = Zonos3LatentMoE(config, layer_id, quantization_config)


class Zonos3Backbone(nn.Module):
    """Each sublayer reads a softmax mix of all earlier sublayer outputs instead of a running residual."""

    def __init__(
        self, config: Zonos3Config, quantization_config: QuantizationConfig | None
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.layers = nn.ModuleList(
            Zonos3DecoderLayer(config, layer_id, quantization_config)
            for layer_id in range(config.num_hidden_layers)
        )
        # One residual query per sublayer input plus the final output.
        self.query_count = 2 * config.num_hidden_layers + 1
        # note (zhaochenyang20): the checkpoint stores the queries in bfloat16, so float32 holds them exactly and no step casts them.
        self.block_attnres_queries = nn.Parameter(
            torch.empty(self.query_count, hidden_size, dtype=torch.float32)
        )
        self.out_norm = FloatRMSNorm(hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Model-dtype hidden states [tokens, hidden] of input_embeds [tokens, hidden].

        input_ids is unused; the prefill CUDA graph runner calls the backbone with it.
        """
        token_count, hidden_size = input_embeds.shape
        epsilon = self.config.rms_norm_eps
        model_dtype = input_embeds.dtype
        # Model-dtype sublayer outputs held in float32, and each one's logits against every query.
        values = torch.empty(
            (token_count, self.query_count, hidden_size),
            dtype=torch.float32,
            device=input_embeds.device,
        )
        value_logits = torch.empty(
            (token_count, self.query_count, self.query_count),
            dtype=torch.float32,
            device=input_embeds.device,
        )
        mixer_input = store_and_mix_values(
            F.rms_norm(input_embeds.float(), (hidden_size,), None, epsilon).to(
                model_dtype
            ),
            values,
            self.block_attnres_queries,
            value_logits,
            self.layers[0].attention_norm.weight,
            value_index=0,
            store_epsilon=epsilon,
            mix_epsilon=self.layers[0].attention_norm.epsilon,
            output_dtype=model_dtype,
        )
        for layer_index, layer in enumerate(self.layers):
            sublayer_index = 2 * layer_index
            if layer.is_full_attention:
                mixer_output = layer.attention(mixer_input, positions, forward_batch)
            else:
                mixer_output = layer.attention(mixer_input, forward_batch)
            feed_forward_input = store_and_mix_values(
                mixer_output,
                values,
                self.block_attnres_queries,
                value_logits,
                layer.ffn_norm.weight,
                value_index=sublayer_index + 1,
                store_epsilon=epsilon,
                mix_epsilon=layer.ffn_norm.epsilon,
                output_dtype=model_dtype,
            )
            # The last layer's output is mixed for the final query under the output norm.
            if layer_index + 1 < len(self.layers):
                next_norm = self.layers[layer_index + 1].attention_norm
            else:
                next_norm = self.out_norm
            mixer_input = store_and_mix_values(
                layer.feed_forward(feed_forward_input),
                values,
                self.block_attnres_queries,
                value_logits,
                next_norm.weight,
                value_index=sublayer_index + 2,
                store_epsilon=epsilon,
                mix_epsilon=next_norm.epsilon,
                output_dtype=model_dtype,
            )
        return mixer_input


class Zonos3SGLangModel(nn.Module):
    """Frame embeddings in, backbone hidden states out; the runner computes the logits.

    Prefill reads summed frame embeddings from input_embeds; decode reads the leading rows of decode_input_embedding, which the runner fills with each request's last frame in batch order.
    """

    def __init__(
        self,
        config: Zonos3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """The signature is the SGLang model loader contract; prefix is unused."""
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embedders = nn.ModuleList(
            [
                nn.Embedding(config.audio_vocab_size, hidden_size)
                for _ in range(config.codebook_count)
            ]
            + [nn.Embedding(config.text_pad_id + 1, hidden_size)]
        )
        self.condition_embeddings = nn.ModuleList(
            nn.Embedding(bucket_count + 1, hidden_size)
            for bucket_count in config.condition_bucket_counts
        )
        # The prefill CUDA graph runner captures the module named model, which owns the layers.
        self.model = Zonos3Backbone(config, quant_config)
        self.multi_output = nn.Linear(
            hidden_size, config.codebook_count * config.audio_vocab_size, bias=False
        )
        self.decode_input_embedding = nn.Embedding(
            get_schedule().max_running_requests, hidden_size
        )
        self.requires_grad_(False)

    def embed_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Float32 sum of the column embeddings of int64 frame rows [tokens, frame_width]."""
        embeddings = self.embedders[0](rows[:, 0]).float()
        for column_index in range(1, rows.shape[1]):
            embeddings = (
                embeddings + self.embedders[column_index](rows[:, column_index]).float()
            )
        return embeddings

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Model-dtype logits [tokens, codebooks, audio vocab] before the soft cap, which the sampler applies."""
        return self.multi_output(hidden_states).view(
            -1, self.config.codebook_count, self.config.audio_vocab_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput:
        """omni_prefill_rids comes with the prefill embeddings; the hidden states do not depend on it."""
        if input_embeds is not None:
            pass
        elif forward_batch.forward_mode.is_decode():
            # note (zhaochenyang20): the runner's row ids count up from zero, so the rows are a view and the decode graph skips a gather kernel.
            input_embeds = self.decode_input_embedding.weight[: input_ids.shape[0]]
        else:
            raise RuntimeError("Zonos3 prefill needs the runner's frame embeddings")
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((hidden_states.shape[0], 1)),
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        parameters = dict(self.named_parameters())
        loaded_names: set[str] = set()
        for checkpoint_name, tensor in weights:
            if checkpoint_name.startswith(UNUSED_CHECKPOINT_PREFIXES):
                continue
            else:
                pass
            parameter_name = checkpoint_name
            for checkpoint_fragment, parameter_fragment in CHECKPOINT_RENAMES:
                parameter_name = parameter_name.replace(
                    checkpoint_fragment, parameter_fragment
                )
            if checkpoint_name.startswith(BACKBONE_CHECKPOINT_PREFIXES):
                parameter_name = f"{BACKBONE_PARAMETER_PREFIX}{parameter_name}"
            else:
                pass
            parameter = parameters[parameter_name]
            if checkpoint_name.endswith(FUSED_CHECKPOINT_SUFFIXES):
                tensor = tensor.reshape(parameter.shape)
            elif tensor.shape != parameter.shape:
                raise ValueError(
                    f"{checkpoint_name} has shape {tuple(tensor.shape)}; the model "
                    f"expects {tuple(parameter.shape)}"
                )
            else:
                pass
            parameter.data.copy_(tensor.to(parameter.dtype))
            loaded_names.add(parameter_name)
        missing_names = set(parameters) - loaded_names - RUNTIME_PARAMETER_NAMES
        if missing_names:
            raise ValueError(f"checkpoint is missing {sorted(missing_names)}")
        else:
            pass


EntryClass = Zonos3SGLangModel
