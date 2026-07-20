# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Ming-Omni-TTS 16.8B AR model wrapper."""

from __future__ import annotations

import logging
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import nn

from sglang_omni.models.ming_omni.talker.talker_module.aggregator import Aggregator
from sglang_omni.models.ming_tts.flow_matching import (
    FlowLoss,
    build_cfm_sde_random,
    build_cfm_timesteps,
)
from sglang_omni.models.ming_tts.hf_config import MING_TTS_TAIL_ATTN_BACKEND
from sglang_omni.models.ming_tts.weight_loading import (
    MING_TTS_LM_HEAD_SKIP_REASON,
    MING_TTS_ROTARY_BUFFER_SKIP_REASON,
    OWNER_AR_MODEL,
    OWNER_AUDIO_VAE,
    OWNER_INTENTIONAL_SKIP,
    OWNER_TTS_HEADS,
    OWNER_UNKNOWN,
    MingTTSWeightReport,
    assert_ming_tts_weight_coverage,
    classify_ming_tts_weight,
)
from sglang_omni.models.weight_loader import default_weight_loader
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang_omni.vendor.sglang.layers import (
    LayerCommunicator,
    LayerScatterModes,
    MergedColumnParallelLinear,
    MRotaryEmbedding,
    QKVParallelLinear,
    QuantizationConfig,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    TopK,
    VocabParallelEmbedding,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_moe_impl_class,
    get_rope,
    should_skip_post_experts_all_reduce,
)
from sglang_omni.vendor.sglang.models import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang_omni.vendor.sglang.server_args import get_global_server_args
from sglang_omni.vendor.sglang.utils import add_prefix

logger = logging.getLogger(__name__)

_MING_TTS_CFM_STEPS = 10


@dataclass
class MingTTSTailInputs:
    hidden_states: torch.Tensor
    latent_history: torch.Tensor
    cfg: torch.Tensor
    sigma: torch.Tensor
    temperature: torch.Tensor


@dataclass
class MingTTSTailOutputs:
    sampled: torch.Tensor
    feedback_embeddings: torch.Tensor
    stop_prob: torch.Tensor


class _MingTTSTailGraph:
    def __init__(self, model: Any, batch_size: int) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: MingTTSTailInputs | None = None
        self.noise: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sde_random: torch.Tensor | None = None
        self.outputs: MingTTSTailOutputs | None = None

    def capture(self) -> None:
        weight = self.model._decode_input_embedding.weight
        device = weight.device
        hidden_dtype = weight.dtype
        float_dtype = torch.float32
        batch_size = self.batch_size
        self.inputs = MingTTSTailInputs(
            hidden_states=torch.zeros(
                batch_size,
                1,
                int(self.model.hidden_size),
                device=device,
                dtype=hidden_dtype,
            ),
            latent_history=torch.zeros(
                batch_size,
                int(self.model.history_patch_size),
                int(self.model.latent_dim),
                device=device,
                dtype=float_dtype,
            ),
            cfg=torch.full((batch_size,), 2.0, device=device, dtype=float_dtype),
            sigma=torch.full((batch_size,), 0.25, device=device, dtype=float_dtype),
            temperature=torch.zeros(batch_size, device=device, dtype=float_dtype),
        )
        self.noise, self.timesteps, self.sde_random = (
            self.model._make_tail_sampling_inputs(
                batch_size=batch_size,
                device=device,
            )
        )

        context = (
            torch.autocast(device_type="cuda", dtype=hidden_dtype)
            if device.type == "cuda" and hidden_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    self.model._compute_tail_step(
                        self.inputs,
                        noise=self.noise,
                        timesteps=self.timesteps,
                        sde_random=self.sde_random,
                    )
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            torch.cuda.synchronize(device=device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.outputs = self.model._compute_tail_step(
                    self.inputs,
                    noise=self.noise,
                    timesteps=self.timesteps,
                    sde_random=self.sde_random,
                )
        self.graph = graph

    def replay(
        self,
        inputs: MingTTSTailInputs,
        *,
        noise: torch.Tensor,
        sde_random: torch.Tensor,
    ) -> MingTTSTailOutputs:
        batch_size = int(inputs.hidden_states.shape[0])
        self.inputs.hidden_states[:batch_size].copy_(inputs.hidden_states)
        self.inputs.hidden_states[batch_size:].zero_()
        self.inputs.latent_history[:batch_size].copy_(inputs.latent_history)
        self.inputs.latent_history[batch_size:].zero_()
        self.inputs.cfg[:batch_size].copy_(inputs.cfg)
        self.inputs.cfg[batch_size:].zero_()
        self.inputs.sigma[:batch_size].copy_(inputs.sigma)
        self.inputs.sigma[batch_size:].zero_()
        self.inputs.temperature[:batch_size].copy_(inputs.temperature)
        self.inputs.temperature[batch_size:].zero_()
        self.noise[:batch_size].copy_(noise)
        self.noise[batch_size:].zero_()
        self.sde_random[:, :batch_size].copy_(sde_random)
        self.sde_random[:, batch_size:].zero_()
        self.graph.replay()
        return MingTTSTailOutputs(
            sampled=self.outputs.sampled[:batch_size].clone(),
            feedback_embeddings=self.outputs.feedback_embeddings[:batch_size].clone(),
            stop_prob=self.outputs.stop_prob[:batch_size].clone(),
        )


class _MingTTSTailGraphCache:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.graphs: dict[int, _MingTTSTailGraph] = {}
        self.buckets: tuple[int, ...] = ()

    def capture(self, batch_sizes: list[int]) -> None:
        self.buckets = tuple(sorted({int(batch_size) for batch_size in batch_sizes}))
        for batch_size in reversed(self.buckets):
            graph = _MingTTSTailGraph(self.model, batch_size)
            graph.capture()
            self.graphs[batch_size] = graph

    def replay(
        self,
        inputs: MingTTSTailInputs,
        *,
        noise: torch.Tensor,
        sde_random: torch.Tensor,
    ) -> MingTTSTailOutputs:
        batch_size = int(inputs.hidden_states.shape[0])
        for bucket in self.buckets:
            if bucket >= batch_size:
                graph = self.graphs[bucket]
                return graph.replay(inputs, noise=noise, sde_random=sde_random)
        raise RuntimeError(
            "Ming TTS tail CUDA graph bucket does not cover active batch "
            f"{batch_size}; captured={list(self.buckets)}"
        )


class MingBailingMoeAttention(nn.Module):
    """BailingMoe attention with SGLang-managed paged KV cache."""

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        if self.num_heads % attn_tp_size != 0:
            raise ValueError(
                "Ming BailingMoe attention heads must be divisible by "
                f"attention TP size, got heads={self.num_heads}, "
                f"tp_size={attn_tp_size}"
            )
        if self.num_kv_heads >= attn_tp_size:
            if self.num_kv_heads % attn_tp_size != 0:
                raise ValueError(
                    "Ming BailingMoe KV heads must be divisible by attention "
                    f"TP size, got kv_heads={self.num_kv_heads}, "
                    f"tp_size={attn_tp_size}"
                )
        elif attn_tp_size % self.num_kv_heads != 0:
            raise ValueError(
                "Ming BailingMoe KV heads must either divide or be divided by "
                f"attention TP size, got kv_heads={self.num_kv_heads}, "
                f"tp_size={attn_tp_size}"
            )

        self.num_heads_per_tp = self.num_heads // attn_tp_size
        self.num_kv_heads_per_tp = max(1, self.num_kv_heads // attn_tp_size)
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=bool(getattr(config, "use_qkv_bias", False)),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("query_key_value", prefix),
        )
        self.dense = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "use_bias", False)),
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("dense", prefix),
        )

        rope_scaling = getattr(config, "runtime_rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=float(config.rope_theta),
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads_per_tp,
            self.head_dim,
            1.0 / math.sqrt(self.head_dim),
            num_kv_heads=self.num_kv_heads_per_tp,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def _prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        if isinstance(self.rotary_emb, MRotaryEmbedding):
            if positions.dim() == 1:
                return positions.unsqueeze(0).expand(3, -1)
            return positions
        if positions.dim() == 2:
            return positions[0]
        return positions

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        positions = self._prepare_positions(positions)
        rotary_dim = int(getattr(self.rotary_emb, "rotary_dim", self.head_dim))
        can_fuse_set_kv = (
            not isinstance(self.rotary_emb, MRotaryEmbedding)
            and self.head_dim == rotary_dim
            and enable_fused_set_kv_buffer(forward_batch)
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if can_fuse_set_kv
                else None
            ),
        )
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not can_fuse_set_kv,
        )
        output, _ = self.dense(attn_output)
        return output


class MingBailingMoeMLP(nn.Module):
    """BailingMoe SwiGLU MLP using SGLang tensor-parallel layers."""

    def __init__(
        self,
        config: Any,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.gate_up_proj = MergedColumnParallelLinear(
            int(config.hidden_size),
            [int(intermediate_size), int(intermediate_size)],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            int(intermediate_size),
            int(config.hidden_size),
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if self.tp_size == 1 and hidden_states.shape[0] == 0:
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return hidden_states


class MingBailingMoeGate(nn.Module):
    """Replicated BailingMoe router weight with official softmax top-k semantics."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(int(config.num_experts), int(config.hidden_size))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )


class MingBailingMoeSparseMoeBlock(nn.Module):
    """BailingMoe sparse block: official routing, SGLang FusedMoE execution."""

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_experts = int(config.num_experts)
        self.num_experts_per_tok = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.multi_gate = bool(getattr(config, "multi_gate", False))
        self.tp_size = get_tensor_model_parallel_world_size()

        self.gate = MingBailingMoeGate(config)
        if self.multi_gate:
            # Note (yzxiao): These modality gates are loaded for checkpoint
            # coverage even though TTS serving does not use modality masks.
            self.image_gate = MingBailingMoeGate(config)
            self.audio_gate = MingBailingMoeGate(config)
        self.topk = TopK(
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        FusedMoE = get_moe_impl_class(quant_config)
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=False,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        num_shared_experts = int(getattr(config, "num_shared_experts", 0) or 0)
        if num_shared_experts > 0:
            self.shared_experts = MingBailingMoeMLP(
                config,
                int(config.moe_intermediate_size) * num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, original_shape[-1])
        shared_input = (
            hidden_states.clone() if self.shared_experts is not None else hidden_states
        )

        router_logits = self.gate(hidden_states).float()
        topk_output = self.topk(hidden_states, router_logits)
        hidden_states = self.experts(hidden_states, topk_output)
        if self.shared_experts is not None:
            hidden_states = hidden_states + self.shared_experts(shared_input)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states.view(original_shape)


class MingBailingMoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attention = MingBailingMoeAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        self.is_layer_sparse = self._is_layer_sparse(config, layer_id)
        if self.is_layer_sparse:
            self.mlp = MingBailingMoeSparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = MingBailingMoeMLP(
                config=config,
                intermediate_size=int(config.intermediate_size),
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )
        self.post_attention_layernorm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=int(config.num_hidden_layers),
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=self._is_layer_sparse(config, layer_id - 1),
            is_next_layer_sparse=self._is_layer_sparse(config, layer_id + 1),
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=layer_id == int(config.num_hidden_layers) - 1,
        )

    @staticmethod
    def _is_layer_sparse(config: Any, layer_id: int) -> bool:
        return getattr(config, "num_experts", None) is not None and layer_id >= int(
            getattr(config, "first_k_dense_replace", 0) or 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
            )
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.attention(positions, hidden_states, forward_batch)
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
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
            hidden_states,
            forward_batch=forward_batch,
            should_allreduce_fusion=should_allreduce_fusion,
            use_reduce_scatter=use_reduce_scatter,
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states,
                residual,
                forward_batch,
            )
        return hidden_states, residual


class MingBailingMoeTextModel(nn.Module):
    """BailingMoe decoder body with SGLang KV-cache ownership."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self._check_supported_bailing_moe_config(config)
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.word_embeddings = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("word_embeddings", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MingBailingMoeDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.start_layer = 0
        self.end_layer = int(config.num_hidden_layers)
        self.norm = RMSNorm(self.hidden_size, eps=float(config.rms_norm_eps))

    @staticmethod
    def _check_supported_bailing_moe_config(config: Any) -> None:
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                "Ming-Omni-TTS BailingMoE currently supports only "
                f"hidden_act='silu'; got {hidden_act!r}"
            )
        if getattr(config, "use_qk_norm", False):
            raise ValueError(
                "Ming-Omni-TTS BailingMoE does not yet support use_qk_norm=True"
            )
        if getattr(config, "use_sliding_window", False):
            raise ValueError(
                "Ming-Omni-TTS BailingMoE does not yet support sliding-window "
                "attention"
            )
        if getattr(config, "moe_router_enable_expert_bias", False):
            raise ValueError(
                "Ming-Omni-TTS BailingMoE does not yet support router expert bias"
            )

        score_function = getattr(config, "score_function", None)
        if score_function not in (None, "softmax"):
            raise ValueError(
                "Ming-Omni-TTS BailingMoE currently supports only softmax "
                f"routing; got score_function={score_function!r}"
            )
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is not None:
            raise ValueError(
                "Ming-Omni-TTS BailingMoE does not yet support router_dtype; "
                f"got {router_dtype!r}"
            )

        for field in ("n_group", "num_expert_group", "topk_group"):
            value = getattr(config, field, None)
            if value not in (None, 0):
                raise ValueError(
                    "Ming-Omni-TTS BailingMoE does not yet support grouped "
                    f"top-k routing; got {field}={value!r}"
                )

        shared_intermediate = getattr(
            config,
            "moe_shared_expert_intermediate_size",
            None,
        )
        if shared_intermediate is not None and int(shared_intermediate) != int(
            config.moe_intermediate_size
        ):
            raise ValueError(
                "Ming-Omni-TTS BailingMoE currently expects shared expert "
                "intermediate size to match moe_intermediate_size; got "
                f"moe_shared_expert_intermediate_size={shared_intermediate!r}, "
                f"moe_intermediate_size={config.moe_intermediate_size!r}"
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.word_embeddings(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        layers = self.layers
        for layer_id in range(self.start_layer, self.end_layer):
            hidden_states, residual = layers[layer_id](
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MingTTSSGLangModel(nn.Module):
    """SGLang wrapper for Ming-TTS AR backbone and weighted TTS heads.

    forward covers the hidden-state backbone path used by SGLang scheduling,
    CUDA graph, and TP.  FlowLoss sampling and latent feedback transitions run
    in the model runner's AR tail after this forward returns.
    """

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".query_key_value.",
        ".dense.",
    ]

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.llm_config = getattr(config, "llm_config", config)
        self.quant_config = quant_config
        self.model = MingBailingMoeTextModel(
            self.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.hidden_size = int(self.llm_config.hidden_size)
        self.vocab_size = int(self.llm_config.vocab_size)

        max_batch_size = 1
        try:
            server_args = get_global_server_args()
        except (AssertionError, RuntimeError):
            server_args = None
        if server_args is not None:
            max_batch_size = int(server_args.max_running_requests)
            if not bool(getattr(server_args, "disable_cuda_graph", True)):
                max_batch_size = max(
                    max_batch_size,
                    int(getattr(server_args, "cuda_graph_max_bs", 1) or 1),
                )
        tail_attn_backend = MING_TTS_TAIL_ATTN_BACKEND

        weight = self.model.word_embeddings.weight
        self._decode_input_embedding = nn.Embedding(
            max_batch_size,
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._decode_input_embedding.weight.requires_grad_(False)
        self.register_buffer(
            "_decode_input_row_ids",
            torch.arange(max_batch_size, dtype=torch.long, device=weight.device),
            persistent=False,
        )

        if not hasattr(self.config, "audio_tokenizer_config"):
            raise ValueError(
                "MingTTSSGLangModel requires the top-level bailingmm config, "
                "not only llm_config, because FlowLoss/Aggregator shapes live "
                "in audio_tokenizer_config, ditar_config, and aggregator_config."
            )

        audio_config = self.config.audio_tokenizer_config
        self.latent_dim = int(audio_config.enc_kwargs["latent_dim"])
        self.patch_size = int(self.config.ditar_config["patch_size"])
        self.history_patch_size = int(
            self.config.ditar_config.get("history_patch_size", self.patch_size)
        )
        self.tail_attn_backend = tail_attn_backend
        aggregator_config = dict(self.config.aggregator_config)
        aggregator_config["attn_backend"] = tail_attn_backend
        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim,
            llm_input_dim=self.hidden_size,
            **aggregator_config,
        )
        ditar_config = dict(self.config.ditar_config)
        ditar_config["attn_backend"] = tail_attn_backend
        self.flowloss = FlowLoss(
            z_channels=self.latent_dim,
            llm_cond_dim=self.hidden_size,
            **ditar_config,
        )
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.hidden_size, bias=True)
        self._tail_graphs = None
        self._cfm_timesteps: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def stage_decode_feedback(
        self,
        feedback_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(feedback_embeddings.shape[0])
        weight = self._decode_input_embedding.weight
        weight[:batch_size].copy_(
            feedback_embeddings.to(device=weight.device, dtype=weight.dtype)
        )
        return self._decode_input_row_ids[:batch_size]

    @torch.no_grad()
    def init_tail_graphs(self, batch_sizes: list[int]) -> None:
        graphs = _MingTTSTailGraphCache(self)
        graphs.capture(batch_sizes)
        self._tail_graphs = graphs
        logger.info(
            "Ming TTS tail CUDA graphs captured for bs=%s",
            list(graphs.buckets),
        )

    @torch.no_grad()
    def run_tail_step(self, inputs: MingTTSTailInputs) -> MingTTSTailOutputs:
        noise, timesteps, sde_random = self._make_tail_sampling_inputs(
            batch_size=int(inputs.hidden_states.shape[0]),
            device=inputs.hidden_states.device,
        )
        tail_graphs = self._tail_graphs
        if tail_graphs is not None:
            return tail_graphs.replay(
                inputs,
                noise=noise,
                sde_random=sde_random,
            )
        return self._compute_tail_step(
            inputs,
            noise=noise,
            timesteps=timesteps,
            sde_random=sde_random,
        )

    def _compute_tail_step(
        self,
        inputs: MingTTSTailInputs,
        *,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
    ) -> MingTTSTailOutputs:
        sampled = self.flowloss.sample(
            z=inputs.hidden_states,
            latent_history=inputs.latent_history,
            noise=noise,
            cfg=inputs.cfg,
            sigma=inputs.sigma,
            temperature=inputs.temperature,
            timesteps=timesteps,
            sde_random=sde_random,
        )
        feedback = self.linear_proj_audio(sampled).reshape(
            int(inputs.hidden_states.shape[0]),
            -1,
        )
        stop_prob = self.stop_head(inputs.hidden_states).softmax(dim=-1)[:, 0, 1]
        return MingTTSTailOutputs(
            sampled=sampled,
            feedback_embeddings=feedback,
            stop_prob=stop_prob,
        )

    def _make_tail_sampling_inputs(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        noise = torch.randn(
            batch_size,
            int(self.latent_dim),
            int(self.patch_size),
            device=device,
        )
        timesteps = self._cfm_timesteps
        if timesteps is None or timesteps.device != device:
            timesteps = build_cfm_timesteps(
                steps=_MING_TTS_CFM_STEPS,
                device=device,
                dtype=noise.dtype,
            )
            self._cfm_timesteps = timesteps
        sde_random = build_cfm_sde_random(
            steps=_MING_TTS_CFM_STEPS,
            device=device,
            dtype=noise.dtype,
            batch_size=batch_size,
            patch_size=int(self.patch_size),
            latent_dim=int(self.latent_dim),
        )
        return noise, timesteps, sde_random

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Any = None,
    ) -> LogitsProcessorOutput:
        del pp_proxy_tensors

        if input_embeds is None:
            input_embeds = getattr(forward_batch, "input_embeds", None)

        forward_mode = forward_batch.forward_mode
        is_decode = bool(forward_mode.is_decode())
        is_extend = bool(forward_mode.is_extend())
        if input_embeds is None and is_decode:
            input_embeds = self._decode_input_embedding(input_ids)
            input_ids = None

        mrope_positions = getattr(forward_batch, "mrope_positions", None)
        if mrope_positions is not None:
            positions = mrope_positions

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if is_extend:
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            if extend_seq_lens is None:
                sample_hidden_states = hidden_states[-1:].contiguous()
            else:
                last_index = (
                    torch.cumsum(
                        extend_seq_lens.to(
                            device=hidden_states.device,
                            dtype=torch.long,
                        ),
                        dim=0,
                    )
                    - 1
                )
                sample_hidden_states = hidden_states[last_index]
        else:
            sample_hidden_states = hidden_states
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        report = MingTTSWeightReport(
            loaded={OWNER_AR_MODEL: 0, OWNER_TTS_HEADS: 0},
            skipped={MING_TTS_LM_HEAD_SKIP_REASON: []},
            deferred={OWNER_AUDIO_VAE: []},
        )
        loaded_param_names: set[str] = set()
        num_experts = int(self.llm_config.num_experts)

        def load_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        def load_gate_up_weight(
            name: str,
            loaded_weight: torch.Tensor,
        ) -> tuple[str, str] | None:
            if ".experts." in name:
                return None
            for weight_name, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, ".gate_up_proj.")
                param = params_dict.get(mapped_name)
                if param is None:
                    return None
                param.weight_loader(param, loaded_weight, shard_id)
                return mapped_name, str(shard_id)
            return None

        def load_fused_expert_weight(
            name: str,
            loaded_weight: torch.Tensor,
        ) -> tuple[str, str] | None:
            if ".experts." not in name:
                return None
            match = re.search(r"experts\.(\d+)\.(gate_proj|down_proj|up_proj)", name)
            if match is None:
                return None

            expert_id = int(match.group(1))
            if expert_id >= num_experts:
                return None

            weight_type = match.group(2)
            param_name = "experts.w2_weight"
            shard_id = "w2"
            if weight_type == "gate_proj":
                param_name = "experts.w13_weight"
                shard_id = "w1"
            elif weight_type == "up_proj":
                param_name = "experts.w13_weight"
                shard_id = "w3"

            weight_name = f"experts.{expert_id}.{weight_type}.weight"
            if weight_name not in name:
                return None
            mapped_name = name.replace(weight_name, param_name)
            param = params_dict.get(mapped_name)
            if param is None:
                return None
            param.weight_loader(
                param,
                loaded_weight,
                mapped_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return mapped_name, f"{shard_id}:{expert_id}"

        for name in params_dict:
            if name.endswith("gate_up_proj.weight"):
                report.add_required_shards(name, ("0", "1"))
            elif name.endswith("experts.w13_weight"):
                shards = []
                for expert_id in range(num_experts):
                    shards.append(f"w1:{expert_id}")
                    shards.append(f"w3:{expert_id}")
                report.add_required_shards(name, shards)
            elif name.endswith("experts.w2_weight"):
                report.add_required_shards(
                    name,
                    [f"w2:{expert_id}" for expert_id in range(num_experts)],
                )

        for original_name, loaded_weight in weights:
            owner = classify_ming_tts_weight(original_name)
            if owner == OWNER_INTENTIONAL_SKIP:
                report.skipped.setdefault(
                    MING_TTS_LM_HEAD_SKIP_REASON,
                    [],
                ).append(original_name)
                continue
            if owner == OWNER_AUDIO_VAE:
                report.deferred.setdefault(OWNER_AUDIO_VAE, []).append(original_name)
                continue
            if owner == OWNER_UNKNOWN:
                report.leftovers.append(original_name)
                continue
            if "rotary_emb." in original_name or original_name.endswith(
                ".rotary_embed.inv_freq"
            ):
                report.skipped.setdefault(
                    MING_TTS_ROTARY_BUFFER_SKIP_REASON,
                    [],
                ).append(original_name)
                continue

            name = original_name
            if name.startswith("model.model."):
                name = "model." + name[len("model.model.") :]
            elif name.startswith(("word_embeddings.", "layers.", "norm.")):
                name = "model." + name

            packed = load_fused_expert_weight(name, loaded_weight)
            if packed is not None:
                target_param, shard_id = packed
                loaded_param_names.add(target_param)
                report.add_loaded(owner, original_name, target_param=target_param)
                report.add_loaded_shard(target_param, shard_id)
                continue

            packed = load_gate_up_weight(name, loaded_weight)
            if packed is not None:
                target_param, shard_id = packed
                loaded_param_names.add(target_param)
                report.add_loaded(owner, original_name, target_param=target_param)
                report.add_loaded_shard(target_param, shard_id)
                continue

            param = params_dict.get(name)
            if param is not None:
                load_param(param, loaded_weight)
                loaded_param_names.add(name)
                report.add_loaded(owner, original_name, target_param=name)
            else:
                report.leftovers.append(original_name)

        runtime_params = {"_decode_input_embedding.weight"}
        missing_params = sorted(set(params_dict) - loaded_param_names - runtime_params)
        if missing_params:
            report.missing["model_params"] = missing_params

        assert_ming_tts_weight_coverage(report)
        self._weight_load_report = report
        logger.info("%s", report.summary())


EntryClass = MingTTSSGLangModel
