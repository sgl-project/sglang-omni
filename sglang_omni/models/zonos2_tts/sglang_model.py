# SPDX-License-Identifier: Apache-2.0
"""SGLang-native ZONOS2 TTS model with custom MoE backbone.

ZONOS2 is an MoE-based TTS model that generates multi-codebook DAC audio codes.
Key architecture features:
- Input is 2D: (seq_len, frame_width) where frame_width = n_codebooks + 1
- Multi-codebook embedding: sum of per-codebook embeddings
- Mixed dense/MoE backbone: first/last layers are dense, middle layers are MoE
- EDA (Exponential Decay Attention) routing with MLP router
- Per-layer top-k overrides via special_topk_layers
- QK-norm with learnable temperature + sigmoid gating on attention output
- Single MultiOutputHead predicts all n_codebooks codes per frame
- Logit soft-capping (loss_softcap=15.0)
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang_omni.models.zonos2_tts.payload_types import (
    ZONOS2_AUDIO_PAD_ID,
    ZONOS2_CODEBOOK_SIZE,
    ZONOS2_EOA_ID,
    ZONOS2_LOSS_SOFTCAP,
    ZONOS2_N_CODEBOOKS,
    ZONOS2_TEXT_VOCAB,
)
from sglang_omni.models.zonos2_tts.state_pool import Zonos2TTSDecodeStatePool
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    StandardTopKOutput,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_moe_impl_class,
    get_rope,
)
from sglang_omni.vendor.sglang.utils import add_prefix

logger = logging.getLogger(__name__)


def _unwrap_norm_output(output: Any) -> torch.Tensor:
    """Handle RMSNorm variants that return either a tensor or (tensor, residual)."""
    if isinstance(output, tuple):
        return output[0]
    return output


# ============================================================================
# Configuration helpers
# ============================================================================


def _resolve_layer_topk(
    default_topk: int,
    special_topk_layers: Optional[dict],
    layer_idx: int,
) -> int:
    """Resolve per-layer top-k from special_topk_layers config."""
    if special_topk_layers and layer_idx in special_topk_layers:
        return special_topk_layers[layer_idx]
    return default_topk


def _is_moe_layer(
    layer_id: int,
    num_layers: int,
    moe_n_experts: int,
    moe_start_from_layer: int,
    moe_end_from_layer: int,
) -> bool:
    """Check if a layer should be MoE based on config."""
    if moe_n_experts <= 1:
        return False
    if layer_id < moe_start_from_layer:
        return False
    if (num_layers - layer_id) <= moe_end_from_layer:
        return False
    return True


# ============================================================================
# ZONOS2 Attention
# ============================================================================


class Zonos2Attention(nn.Module):
    """ZONOS2 attention with QK-norm, learnable temperature, and sigmoid gating.

    Key differences from standard attention:
    - QK normalization with per-head learnable temperature scaling
    - Sigmoid gating on attention output (element-wise per head)
    - Interleaved RoPE format (is_neox=False)
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        attn_tp_size = get_attention_tp_size()
        attn_tp_rank = get_attention_tp_rank()
        self.num_heads_per_tp = self.num_heads // attn_tp_size
        self.num_kv_heads_per_tp = max(1, self.num_kv_heads // attn_tp_size)
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        # QKV projection: ZONOS2 uses separate wq and wkv (chunked K+V)
        # We map to SGLang's QKVParallelLinear for TP support
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        # Learnable temperature for QK-norm: [1, num_heads_per_tp, 1]
        # Checkpoint: layers.{N}.attention.temp
        self.temp = nn.Parameter(
            torch.ones(1, self.num_heads_per_tp, 1), requires_grad=False
        )

        # Sigmoid gater: Linear(hidden_size, num_heads) -> sigmoid
        # Checkpoint: layers.{N}.attention.gater.weight
        self.gater = ReplicatedLinear(
            self.hidden_size, self.num_heads, bias=False,
            prefix=add_prefix("gater", prefix),
        )

        # RoPE - ZONOS2 uses interleaved format
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=False,  # ZONOS2 uses interleaved RoPE
        )

        # RadixAttention for paged KV cache
        self.attn = RadixAttention(
            self.num_heads_per_tp,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads_per_tp,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Compute sigmoid gate
        gate_logits, _ = self.gater(hidden_states)
        # Gate is per-head: [tokens, num_heads] -> need to shard for TP
        attn_tp_size = get_attention_tp_size()
        attn_tp_rank = get_attention_tp_rank()
        if attn_tp_size > 1:
            heads_per_tp = self.num_heads // attn_tp_size
            gate_logits = gate_logits[:, attn_tp_rank * heads_per_tp:(attn_tp_rank + 1) * heads_per_tp]
        gate = torch.sigmoid(gate_logits)  # [tokens, num_heads_per_tp]

        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for multi-head
        q = q.view(-1, self.num_heads_per_tp, self.head_dim)
        k = k.view(-1, self.num_kv_heads_per_tp, self.head_dim)

        # QK normalization with learnable temperature
        q = F.rms_norm(q, (self.head_dim,), eps=1e-6) * self.temp.abs().to(q.dtype)
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)

        # ZONOS2 applies interleaved RoPE on flattened head dimensions.
        q, k = self.rotary_emb(
            forward_batch.positions,
            q.flatten(-2),
            k.flatten(-2),
        )
        q = q.view(-1, self.num_heads_per_tp, self.head_dim)

        # Attention with paged KV cache
        attn_output = self.attn(q, k, v, forward_batch)

        # Apply sigmoid gating per head
        attn_output = attn_output.view(-1, self.num_heads_per_tp, self.head_dim)
        attn_output = attn_output * gate.unsqueeze(-1)
        attn_output = attn_output.view(-1, self.num_heads_per_tp * self.head_dim)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


# ============================================================================
# Dense FFN
# ============================================================================


class Zonos2DenseMLP(nn.Module):
    """Dense SwiGLU MLP for non-MoE layers.

    Checkpoint naming:
    - layers.{N}.feed_forward.w_in.weight (3D: [2, intermediate, hidden])
    - layers.{N}.feed_forward.w_out.weight
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # gate_up_proj maps to w_in. In the reference implementation the first
        # half is the up projection and the second half is the SiLU gate.
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        # down_proj maps to w_out
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        up, gate = gate_up.chunk(2, dim=-1)
        x = up * F.silu(gate)
        x, _ = self.down_proj(x)
        return x


# ============================================================================
# EDA Router
# ============================================================================


class Zonos2Router(nn.Module):
    """ZONOS2 EDA (Exponential Decay Attention) Router.

    Architecture:
    - down_proj: Linear(hidden_size, router_dim, bias=True)
    - EDA blending with previous layer's router states
    - RMSNorm on blended states
    - router_mlp: 3-layer MLP (router_dim -> router_dim -> num_experts)
    - Balanced top-k selection with learnable biases

    Checkpoint naming:
    - layers.{N}.feed_forward.router.down_proj.weight/bias
    - layers.{N}.feed_forward.router.router_mlp.{0,2,4}.weight/bias
    - layers.{N}.feed_forward.router.rmsnorm_eda.weight
    - layers.{N}.feed_forward.router.router_states_scale
    - layers.{N}.feed_forward.router.balancing_biases
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.router_dim = config.moe_router_dim
        self.num_experts = config.moe_n_experts
        self.top_k = _resolve_layer_topk(
            config.num_experts_per_tok, config.special_topk_layers, layer_id
        )
        self.use_legacy_balancing = config.moe_balancing_strategy == "legacy"

        # Down projection to router dimension
        self.down_proj = nn.Linear(self.hidden_size, self.router_dim, bias=True)

        # Router MLP: router_dim -> GELU -> router_dim -> GELU -> num_experts
        self.router_mlp_0 = nn.Linear(self.router_dim, self.router_dim, bias=True)
        self.router_mlp_2 = nn.Linear(self.router_dim, self.router_dim, bias=True)
        self.router_mlp_4 = nn.Linear(self.router_dim, self.num_experts, bias=False)

        # RMSNorm for EDA
        self.rmsnorm_eda = RMSNorm(self.router_dim, eps=config.rms_norm_eps)

        # EDA: use_eda is True for all layers except the first MoE layer
        self.use_eda = layer_id != config.moe_start_from_layer
        if self.use_eda:
            self.router_states_scale = nn.Parameter(
                torch.ones(self.router_dim), requires_grad=False
            )

        # Balancing biases for load balancing
        self.balancing_biases = nn.Parameter(
            torch.zeros(self.num_experts), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert indices.

        Returns:
            topk_weights: [tokens, top_k] routing weights
            topk_ids: [tokens, top_k] expert indices
            router_states_next: [tokens, router_dim] for next layer's EDA
        """
        # Down-project to router dimension
        h = self.down_proj(hidden_states)

        # EDA: blend with previous router states
        if self.use_eda and router_states is not None:
            h = h + router_states * self.router_states_scale

        # Save for next layer's EDA
        router_states_next = h.clone()

        # Normalize
        h = _unwrap_norm_output(self.rmsnorm_eda(h))

        # Router MLP
        h = F.gelu(self.router_mlp_0(h))
        h = F.gelu(self.router_mlp_2(h))
        router_logits = self.router_mlp_4(h)

        # Softmax routing probabilities
        expert_prob = torch.softmax(router_logits.float(), dim=-1)

        # Balanced top-k selection
        with torch.no_grad():
            bias = self.balancing_biases.detach().float()
            if self.use_legacy_balancing:
                routing_scores = expert_prob.float() + bias
            else:
                routing_scores = expert_prob.float() - bias
            _, expert_choice = torch.topk(routing_scores, self.top_k, dim=-1)

        # Gather actual probabilities for selected experts
        topk_weights = torch.gather(expert_prob, dim=-1, index=expert_choice)
        topk_ids = expert_choice.to(dtype=torch.int32)

        return topk_weights, topk_ids, router_states_next


# ============================================================================
# MoE FFN
# ============================================================================


class Zonos2MoEBlock(nn.Module):
    """ZONOS2 MoE feedforward block with EDA router.

    Checkpoint naming:
    - layers.{N}.feed_forward.router.*
    - layers.{N}.feed_forward.experts.* (w1, w2, w3 or w13)
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.top_k = _resolve_layer_topk(
            config.num_experts_per_tok, config.special_topk_layers, layer_id
        )

        # EDA Router
        self.router = Zonos2Router(
            config, layer_id, prefix=add_prefix("router", prefix)
        )

        # FusedMoE experts
        FusedMoECls = get_moe_impl_class(quant_config)
        self.experts = FusedMoECls(
            num_experts=config.moe_n_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=True,
            prefix=add_prefix("experts", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoE layer.

        Returns:
            output: [tokens, hidden_dim]
            router_states_next: [tokens, router_dim] for next MoE layer
        """
        # Compute routing
        topk_weights, topk_ids, router_states_next = self.router(
            hidden_states, router_states
        )

        # FusedMoE forward with pre-computed routing
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=None,
        )
        output = self.experts(hidden_states, topk_output)

        return output, router_states_next


# ============================================================================
# Decoder Layer
# ============================================================================


class Zonos2DecoderLayer(nn.Module):
    """ZONOS2 transformer decoder layer with attention + dense/MoE FFN.

    Checkpoint naming:
    - layers.{N}.attention_norm.weight
    - layers.{N}.attention.*
    - layers.{N}.ffn_norm.weight
    - layers.{N}.feed_forward.*
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # Determine if this layer is MoE or dense
        self.is_moe = _is_moe_layer(
            layer_id,
            config.num_layers,
            config.moe_n_experts,
            config.moe_start_from_layer,
            config.moe_end_from_layer,
        )

        # Pre-attention norm
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.attention = Zonos2Attention(
            config, layer_id, quant_config,
            prefix=add_prefix("attention", prefix),
        )

        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN: MoE or dense
        if self.is_moe:
            self.feed_forward = Zonos2MoEBlock(
                config, layer_id, quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:
            self.feed_forward = Zonos2DenseMLP(
                config, quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        router_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Pre-attention norm with residual
        if residual is None:
            residual = hidden_states
            hidden_states = _unwrap_norm_output(self.attention_norm(hidden_states))
        else:
            hidden_states, residual = self.attention_norm(hidden_states, residual)

        # Attention
        hidden_states = self.attention(hidden_states, forward_batch)

        # Pre-FFN norm with residual
        hidden_states, residual = self.ffn_norm(hidden_states, residual)

        # FFN (MoE or dense)
        if self.is_moe:
            hidden_states, router_states = self.feed_forward(
                hidden_states, router_states
            )
        else:
            hidden_states = self.feed_forward(hidden_states)
            # Dense layers don't produce router states; pass through None
            router_states = None

        return hidden_states, residual, router_states


# ============================================================================
# Full Model
# ============================================================================


class Zonos2SGLangModel(nn.Module):
    """ZONOS2 TTS model with custom MoE backbone for SGLang inference.

    Architecture:
    - MultiEmbedding: n_codebooks audio embeddings + 1 text embedding, summed
    - Embedding norm (RMSNorm, no learnable params)
    - Optional speaker projection (for voice cloning)
    - Mixed dense/MoE transformer backbone with EDA routing
    - Output norm (RMSNorm)
    - MultiOutputHead: single linear projecting to n_codebooks * audio_vocab
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = self._normalize_config(config)
        self.quant_config = quant_config
        self.hidden_size = int(self.config.hidden_size)
        self.n_codebooks = int(self.config.n_codebooks)
        self.codebook_size = int(self.config.codebook_size)
        self.audio_vocab = self.codebook_size + 2  # +2 for eoa_id and audio_pad_id
        self.text_vocab = getattr(self.config, "text_vocab", None)
        self.loss_softcap = float(getattr(self.config, "loss_softcap", ZONOS2_LOSS_SOFTCAP))

        # Frame width: n_codebooks audio + 1 text column
        self.frame_width = self.n_codebooks + (1 if self.text_vocab is not None else 0)

        # Multi-codebook embeddings
        self.embedding_list = nn.ModuleList()
        for idx in range(self.n_codebooks):
            self.embedding_list.append(
                VocabParallelEmbedding(
                    self.audio_vocab,
                    self.hidden_size,
                    prefix=add_prefix(f"multi_embedder.embedders.{idx}", prefix),
                )
            )
        if self.text_vocab is not None:
            self.embedding_list.append(
                VocabParallelEmbedding(
                    int(self.text_vocab) + 1,
                    self.hidden_size,
                    prefix=add_prefix(
                        f"multi_embedder.embedders.{self.n_codebooks}", prefix
                    ),
                )
            )

        # Embedding norm is non-affine in the reference implementation.
        self.emb_norm = RMSNorm(
            self.hidden_size,
            eps=self.config.rms_norm_eps,
            has_weight=False,
        )

        # Speaker projection (optional, for voice cloning)
        speaker_enabled = bool(getattr(self.config, "speaker_enabled", False))
        speaker_dim = int(getattr(self.config, "speaker_embedding_dim", 128))
        speaker_lda_dim = getattr(self.config, "speaker_lda_dim", None)
        if speaker_enabled:
            if speaker_lda_dim is not None:
                self.speaker_lda_weight = nn.Parameter(
                    torch.empty(int(speaker_lda_dim), speaker_dim)
                )
                self.speaker_lda_bias = nn.Parameter(
                    torch.empty(int(speaker_lda_dim))
                )
                proj_input_dim = int(speaker_lda_dim)
            else:
                self.speaker_lda_weight = None
                self.speaker_lda_bias = None
                proj_input_dim = speaker_dim
            self.speaker_projection_weight = nn.Parameter(
                torch.empty(self.hidden_size, proj_input_dim)
            )
            self.speaker_projection_bias = nn.Parameter(
                torch.empty(self.hidden_size)
            )
        else:
            self.speaker_lda_weight = None
            self.speaker_lda_bias = None
            self.speaker_projection_weight = None
            self.speaker_projection_bias = None

        # Transformer layers (mixed dense/MoE)
        self.layers = nn.ModuleList([
            Zonos2DecoderLayer(
                self.config, layer_id, quant_config,
                prefix=add_prefix(f"layers.{layer_id}", prefix),
            )
            for layer_id in range(self.config.num_layers)
        ])

        # Output norm
        self.out_norm = RMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)

        # Multi-output head: exact checkpoint projection, not a padded LM vocab head.
        self.multi_output_head = nn.Linear(
            self.hidden_size,
            self.n_codebooks * self.audio_vocab,
            bias=False,
        )

        # Decode-time input embedding (rewritten each step by model_runner)
        max_batch_size = 16
        try:
            from sglang.srt.server_args import get_global_server_args
            max_batch_size = get_global_server_args().max_running_requests
        except Exception:
            pass
        self._decode_input_embedding = nn.Embedding(max_batch_size, self.hidden_size)
        self._decode_input_embedding.weight.requires_grad_(False)
        self._state_pool = (
            Zonos2TTSDecodeStatePool(self)
            if bool(getattr(self.config, "enable_decode_state_pool", False))
            else None
        )

    def reset_request(self, rid: str) -> None:
        """Release decode-state pool state for a finished or aborted request."""
        if self._state_pool is not None:
            self._state_pool.release_row(rid)

    @staticmethod
    def _normalize_config(config: Any) -> Any:
        """Normalize ZONOS2 config to have all required fields."""
        # Core dimensions
        hidden_size = int(getattr(config, "hidden_size", getattr(config, "dim", 2048)))
        num_layers = int(
            getattr(
                config,
                "num_layers",
                getattr(config, "num_hidden_layers", getattr(config, "n_layers", 24)),
            )
        )
        num_qo_heads = int(
            getattr(
                config,
                "num_qo_heads",
                getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "n_heads", hidden_size // 128),
                ),
            )
        )
        num_kv_heads = int(
            getattr(
                config,
                "num_kv_heads",
                getattr(
                    config,
                    "num_key_value_heads",
                    getattr(config, "n_kv_heads", num_qo_heads),
                ),
            )
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // num_qo_heads))
        raw_intermediate = getattr(config, "intermediate_size", None)
        if raw_intermediate is None:
            multiplier = float(getattr(config, "ffn_dim_multiplier", 1.5))
            multiple_of = int(getattr(config, "multiple_of", 256))
            raw_size = int(multiplier * hidden_size)
            raw_intermediate = multiple_of * (
                (raw_size + multiple_of - 1) // multiple_of
            )
        intermediate_size = int(raw_intermediate)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", getattr(config, "norm_eps", 1e-5)))
        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        max_position = int(
            getattr(config, "max_position_embeddings", getattr(config, "max_seqlen", 6144))
        )

        # TTS-specific
        n_codebooks = int(getattr(config, "n_codebooks", ZONOS2_N_CODEBOOKS))
        codebook_size = int(getattr(config, "codebook_size", ZONOS2_CODEBOOK_SIZE))
        eoa_id = int(getattr(config, "eoa_id", ZONOS2_EOA_ID))
        audio_pad_id = int(getattr(config, "audio_pad_id", ZONOS2_AUDIO_PAD_ID))
        loss_softcap = float(getattr(config, "loss_softcap", ZONOS2_LOSS_SOFTCAP))
        text_vocab = getattr(config, "text_vocab", ZONOS2_TEXT_VOCAB)

        # MoE-specific
        moe_n_experts = int(getattr(config, "moe_n_experts", getattr(config, "num_experts", 16)))
        num_experts_per_tok = int(getattr(config, "moe_router_topk",
                                          getattr(config, "num_experts_per_tok", 1)))
        moe_start_from_layer = int(getattr(config, "moe_start_from_layer", 3))
        moe_end_from_layer = int(getattr(config, "moe_end_from_layer", 1))
        moe_router_dim = int(getattr(config, "moe_router_dim", 128))
        moe_intermediate_size = int(getattr(config, "moe_intermediate_size", intermediate_size))
        moe_balancing_strategy = str(getattr(config, "moe_balancing_strategy", "legacy"))
        special_topk_layers = getattr(config, "special_topk_layers", None)
        if isinstance(special_topk_layers, Mapping):
            special_topk_layers = {
                int(layer): int(topk) for layer, topk in special_topk_layers.items()
            }

        # Speaker
        speaker_enabled = bool(getattr(config, "speaker_enabled", True))
        speaker_embedding_dim = int(getattr(config, "speaker_embedding_dim", 2048))
        speaker_lda_dim = getattr(config, "speaker_lda_dim", 1024)

        # Set all fields on config
        config.hidden_size = hidden_size
        config.num_layers = num_layers
        config.num_qo_heads = num_qo_heads
        config.num_kv_heads = num_kv_heads
        config.head_dim = head_dim
        config.intermediate_size = intermediate_size
        config.rms_norm_eps = rms_norm_eps
        config.rope_theta = rope_theta
        config.max_position_embeddings = max_position
        config.n_codebooks = n_codebooks
        config.codebook_size = codebook_size
        config.eoa_id = eoa_id
        config.audio_pad_id = audio_pad_id
        config.loss_softcap = loss_softcap
        config.text_vocab = text_vocab
        config.moe_n_experts = moe_n_experts
        config.num_experts_per_tok = num_experts_per_tok
        config.moe_start_from_layer = moe_start_from_layer
        config.moe_end_from_layer = moe_end_from_layer
        config.moe_router_dim = moe_router_dim
        config.moe_intermediate_size = moe_intermediate_size
        config.moe_balancing_strategy = moe_balancing_strategy
        config.special_topk_layers = special_topk_layers
        config.speaker_enabled = speaker_enabled
        config.speaker_embedding_dim = speaker_embedding_dim
        config.speaker_lda_dim = speaker_lda_dim

        return config

    @property
    def device(self) -> torch.device:
        return self.embedding_list[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding_list[0].weight.dtype

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute summed multi-codebook embeddings from 2D input."""
        return self._prepare_multi_modal_inputs(input_ids)

    def _apply_emb_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply embedding RMSNorm. Used by model_runner for prefill/decode."""
        return _unwrap_norm_output(self.emb_norm(x))

    @torch.no_grad()
    def project_speaker_embedding(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """Project a raw speaker embedding into the model hidden space."""

        if self.speaker_projection_weight is None:
            raise RuntimeError("Current ZONOS2 checkpoint has no speaker projection")
        x = torch.as_tensor(
            speaker_embedding,
            device=self.speaker_projection_weight.device,
            dtype=self.speaker_projection_weight.dtype,
        ).squeeze()
        if x.dim() == 2:
            x = x[0] if x.shape[0] == 1 else x.mean(dim=0)
        if x.dim() != 1:
            raise ValueError(
                f"ZONOS2 speaker embedding must be 1D or 2D, got {tuple(x.shape)}"
            )
        if self.speaker_lda_weight is not None:
            x = F.linear(
                x,
                self.speaker_lda_weight.to(dtype=x.dtype),
                self.speaker_lda_bias.to(dtype=x.dtype),
            )
        projected = F.linear(
            x,
            self.speaker_projection_weight.to(dtype=x.dtype),
            self.speaker_projection_bias.to(dtype=x.dtype),
        )
        return projected.to(device=self.device, dtype=self.dtype)

    def _prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Sum embeddings from all codebook columns."""
        if input_ids.dim() == 1:
            total = input_ids.shape[0]
            if total % self.frame_width == 0:
                input_ids_2d = input_ids.view(total // self.frame_width, self.frame_width)
            else:
                input_ids_2d = torch.full(
                    (total, self.frame_width),
                    self.config.audio_pad_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids_2d[:, -1] = input_ids
        elif input_ids.dim() == 2:
            input_ids_2d = input_ids
        else:
            raise ValueError(
                f"ZONOS2 input_ids must be 1D or 2D, got shape {tuple(input_ids.shape)}"
            )

        # Sum embeddings from all columns
        embeds = torch.zeros(
            input_ids_2d.shape[0],
            self.hidden_size,
            device=input_ids_2d.device,
            dtype=self.dtype,
        )
        for idx, embed_layer in enumerate(self.embedding_list):
            col_ids = input_ids_2d[:, idx].contiguous()
            embeds.add_(embed_layer(col_ids))

        return embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected, pp_proxy_tensors

        if input_embeds is None:
            forward_mode = getattr(forward_batch, "forward_mode", None)
            is_decode = (
                forward_mode is not None
                and hasattr(forward_mode, "is_decode")
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                input_embeds = self._decode_input_embedding(input_ids)
            else:
                input_embeds = self._prepare_multi_modal_inputs(input_ids)

        # Apply embedding norm (non-affine)
        hidden_states = _unwrap_norm_output(self.emb_norm(input_embeds))

        # Run transformer layers
        residual = None
        router_states = None
        for layer in self.layers:
            hidden_states, residual, router_states = layer(
                hidden_states, forward_batch, residual, router_states
            )

        # Final norm
        hidden_states, _ = self.out_norm(hidden_states, residual)

        # Select last-token hidden states for sampling
        sample_hidden_states = self._select_sample_hidden_states(
            hidden_states, forward_batch
        )

        # Return hidden states for the model_runner to compute multi-codebook logits
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
        )

    def compute_multi_codebook_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-codebook logits from hidden states.

        Args:
            hidden_states: [batch_size, hidden_size]

        Returns:
            logits: [batch_size, n_codebooks, audio_vocab] with soft-capping applied
        """
        logits = self.multi_output_head(hidden_states)
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.n_codebooks, self.audio_vocab)

        # Apply soft capping
        if self.loss_softcap > 0:
            logits = self.loss_softcap * torch.tanh(logits / self.loss_softcap)

        return logits

    @staticmethod
    def _select_sample_hidden_states(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Select the last token's hidden state per sequence for sampling."""
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_extend = (
            forward_mode is not None
            and hasattr(forward_mode, "is_extend")
            and bool(forward_mode.is_extend())
        )
        if not is_extend:
            return hidden_states
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return hidden_states[-1:].contiguous()
        last_index = (
            torch.cumsum(
                extend_seq_lens.to(device=hidden_states.device, dtype=torch.long), dim=0
            )
            - 1
        )
        return hidden_states[last_index]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load weights from ZONOS2 checkpoint format.

        ZONOS2 checkpoint naming:
        - multi_embedder.embedders.{N}.weight -> embedding_list.{N}.weight
        - layers.{N}.attention.wq.weight -> layers.{N}.attention.qkv_proj (q shard)
        - layers.{N}.attention.wkv.weight -> layers.{N}.attention.qkv_proj (k,v shards)
        - layers.{N}.attention.wo.weight -> layers.{N}.attention.o_proj.weight
        - layers.{N}.attention.temp -> layers.{N}.attention.temp
        - layers.{N}.attention.gater.weight -> layers.{N}.attention.gater.weight
        - layers.{N}.attention_norm.weight -> layers.{N}.attention_norm.weight
        - layers.{N}.ffn_norm.weight -> layers.{N}.ffn_norm.weight
        - layers.{N}.feed_forward.w_in.weight -> layers.{N}.feed_forward.gate_up_proj (dense)
        - layers.{N}.feed_forward.w_out.weight -> layers.{N}.feed_forward.down_proj (dense)
        - layers.{N}.feed_forward.router.* -> layers.{N}.feed_forward.router.* (MoE)
        - layers.{N}.feed_forward.experts.* -> layers.{N}.feed_forward.experts.* (MoE)
        - out_norm.weight -> out_norm.weight
        - multi_output.weight -> multi_output_head.weight
        - speaker_lda_projection.weight/bias
        - speaker_projection.weight/bias
        """
        params_dict = dict(self.named_parameters())
        loaded_count = 0
        skipped_count = 0
        unmatched_names: list[str] = []

        for original_name, loaded_weight in self._iter_normalized_weights(weights):
            name = original_name
            if not isinstance(loaded_weight, torch.Tensor):
                skipped_count += 1
                unmatched_names.append(name)
                continue

            if (
                ".router.ent_denom" in name
                or ".router.ent_numer" in name
                or ".router.normalized_entropy" in name
            ):
                skipped_count += 1
                continue

            # === Multi-embedder ===
            if name.startswith("multi_embedder.embedders."):
                mapped = name.replace("multi_embedder.embedders.", "embedding_list.")
                if mapped in params_dict:
                    self._load_param(params_dict[mapped], loaded_weight)
                    loaded_count += 1
                continue

            # === Multi-output head ===
            if name == "multi_output.weight":
                mapped = "multi_output_head.weight"
                if mapped in params_dict:
                    self._load_param(params_dict[mapped], loaded_weight)
                    loaded_count += 1
                continue

            # === Output norm ===
            if name == "out_norm.weight":
                if "out_norm.weight" in params_dict:
                    self._load_param(params_dict["out_norm.weight"], loaded_weight)
                    loaded_count += 1
                continue

            # === Embedding norm (non-affine, skip) ===
            if name == "emb_norm.weight":
                skipped_count += 1
                continue

            # === Speaker projection ===
            if name == "speaker_lda_projection.weight":
                if self.speaker_lda_weight is not None:
                    self.speaker_lda_weight.data.copy_(loaded_weight)
                    loaded_count += 1
                continue
            if name == "speaker_lda_projection.bias":
                if self.speaker_lda_bias is not None:
                    self.speaker_lda_bias.data.copy_(loaded_weight)
                    loaded_count += 1
                continue
            if name == "speaker_projection.weight":
                if self.speaker_projection_weight is not None:
                    self.speaker_projection_weight.data.copy_(loaded_weight)
                    loaded_count += 1
                continue
            if name == "speaker_projection.bias":
                if self.speaker_projection_bias is not None:
                    self.speaker_projection_bias.data.copy_(loaded_weight)
                    loaded_count += 1
                continue

            # === Skip rotary embeddings ===
            if "rotary_emb" in name:
                skipped_count += 1
                continue

            # === Transformer layers ===
            if name.startswith("layers."):
                loaded = self._load_layer_weight(name, loaded_weight, params_dict)
                if loaded:
                    loaded_count += 1
                else:
                    unmatched_names.append(name)
                    skipped_count += 1
                continue

            # === Fallback: try direct match ===
            if name in params_dict:
                self._load_param(params_dict[name], loaded_weight)
                loaded_count += 1
            else:
                unmatched_names.append(name)
                skipped_count += 1

        if unmatched_names:
            logger.warning(
                "ZONOS2 loader skipped %d unmatched weights (first 10): %s",
                len(unmatched_names),
                unmatched_names[:10],
            )
        logger.info(
            "ZONOS2 weight loading: loaded=%d skipped=%d", loaded_count, skipped_count
        )

    @classmethod
    def _iter_normalized_weights(
        cls,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, Any]]:
        for original_name, loaded_weight in weights:
            if isinstance(loaded_weight, Mapping) and original_name in (
                "model",
                "state_dict",
                "module",
            ):
                for inner_name, inner_weight in loaded_weight.items():
                    yield cls._normalize_weight_name(str(inner_name)), inner_weight
                continue
            yield cls._normalize_weight_name(str(original_name)), loaded_weight

    @staticmethod
    def _normalize_weight_name(name: str) -> str:
        if name.startswith("model."):
            name = name[len("model.") :]
        name = name.replace(".parametrizations.", ".")
        if name.endswith(".original"):
            name = name[: -len(".original")]
        return name

    def _load_layer_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
    ) -> bool:
        """Load a single transformer layer weight with proper mapping."""

        # === Attention: wq -> qkv_proj (q shard) ===
        if ".attention.wq.weight" in name:
            mapped = name.replace(".attention.wq.weight", ".attention.qkv_proj.weight")
            if mapped in params_dict:
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, "q")
                return True
            return False

        # === Attention: wkv -> qkv_proj (k, v shards) ===
        # wkv.weight is [2, kv_dim, hidden] or [2*kv_dim, hidden]
        if ".attention.wkv.weight" in name:
            mapped = name.replace(".attention.wkv.weight", ".attention.qkv_proj.weight")
            if mapped in params_dict:
                param = params_dict[mapped]
                # Split wkv into k and v
                if loaded_weight.dim() == 3:
                    k_weight = loaded_weight[0]  # [kv_dim, hidden]
                    v_weight = loaded_weight[1]  # [kv_dim, hidden]
                else:
                    half = loaded_weight.shape[0] // 2
                    k_weight = loaded_weight[:half]
                    v_weight = loaded_weight[half:]
                param.weight_loader(param, k_weight, "k")
                param.weight_loader(param, v_weight, "v")
                return True
            return False

        # === Attention: wo -> o_proj ===
        if ".attention.wo.weight" in name:
            mapped = name.replace(".attention.wo.weight", ".attention.o_proj.weight")
            if mapped in params_dict:
                self._load_param(params_dict[mapped], loaded_weight)
                return True
            return False

        # === Attention: temp ===
        if ".attention.temp" in name and not name.endswith(".weight"):
            mapped = name  # layers.{N}.attention.temp
            if mapped in params_dict:
                param = params_dict[mapped]
                if param.shape != loaded_weight.shape and loaded_weight.dim() == 3:
                    tp_rank = get_attention_tp_rank()
                    per_rank = param.shape[1]
                    start = tp_rank * per_rank
                    loaded_weight = loaded_weight[:, start : start + per_rank, :]
                param.data.copy_(loaded_weight)
                return True
            return False

        # === Attention: gater ===
        if ".attention.gater.weight" in name:
            mapped = name  # layers.{N}.attention.gater.weight
            if mapped in params_dict:
                self._load_param(params_dict[mapped], loaded_weight)
                return True
            return False

        # === Dense FFN: w_in -> gate_up_proj ===
        # w_in.weight is [2, intermediate, hidden] (chunked gate + up)
        if ".feed_forward.w_in.weight" in name:
            mapped = name.replace(
                ".feed_forward.w_in.weight", ".feed_forward.gate_up_proj.weight"
            )
            if mapped in params_dict:
                param = params_dict[mapped]
                if loaded_weight.dim() == 3:
                    # Reference dense FFN w_in is [up, gate].
                    up_weight = loaded_weight[0]
                    gate_weight = loaded_weight[1]
                else:
                    half = loaded_weight.shape[0] // 2
                    up_weight = loaded_weight[:half]
                    gate_weight = loaded_weight[half:]
                param.weight_loader(param, up_weight, 0)
                param.weight_loader(param, gate_weight, 1)
                return True
            return False

        # === Dense FFN: w_out -> down_proj ===
        if ".feed_forward.w_out.weight" in name:
            mapped = name.replace(
                ".feed_forward.w_out.weight", ".feed_forward.down_proj.weight"
            )
            if mapped in params_dict:
                self._load_param(params_dict[mapped], loaded_weight)
                return True
            return False

        # === MoE Router weights ===
        if ".feed_forward.router." in name:
            # Map router_mlp.{0,2,4} to router_mlp_{0,2,4}
            mapped = name
            mapped = mapped.replace(".router_mlp.0.", ".router_mlp_0.")
            mapped = mapped.replace(".router_mlp.2.", ".router_mlp_2.")
            mapped = mapped.replace(".router_mlp.4.", ".router_mlp_4.")
            if mapped in params_dict:
                params_dict[mapped].data.copy_(loaded_weight)
                return True
            # Try direct match
            if name in params_dict:
                params_dict[name].data.copy_(loaded_weight)
                return True
            return False

        # === MoE Expert weights ===
        if ".feed_forward.experts." in name:
            return self._load_moe_expert_weight(name, loaded_weight, params_dict)

        # === Norms and other direct matches ===
        if name in params_dict:
            self._load_param(params_dict[name], loaded_weight)
            return True

        return False

    def _load_moe_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
    ) -> bool:
        """Load MoE expert weights, handling fusion of w1+w3 -> gate_up_proj.

        ZONOS2 checkpoint formats:
        - experts.w1.weight: [num_experts, intermediate, hidden] (gate)
        - experts.w2.weight: [num_experts, hidden, intermediate] (down)
        - experts.w3.weight: [num_experts, intermediate, hidden] (up)
        - experts.w13: [num_experts, 2*intermediate, hidden] (SonicMoE interleaved)
        """
        base = name.split(".feed_forward.experts.")[0]
        w13_key = f"{base}.feed_forward.experts.w13_weight"
        w2_key = f"{base}.feed_forward.experts.w2_weight"

        if name.endswith(".feed_forward.experts.w1.weight") and w13_key in params_dict:
            param = params_dict[w13_key]
            return self._load_packed_moe_experts(param, loaded_weight, name, "w1")

        if name.endswith(".feed_forward.experts.w3.weight") and w13_key in params_dict:
            param = params_dict[w13_key]
            return self._load_packed_moe_experts(param, loaded_weight, name, "w3")

        if name.endswith(".feed_forward.experts.w2.weight") and w2_key in params_dict:
            param = params_dict[w2_key]
            return self._load_packed_moe_experts(param, loaded_weight, name, "w2")

        # SonicMoE stores w13 interleaved as gate/up alternating rows.
        if (name.endswith(".feed_forward.experts.w13") or ".experts.w13" in name) and w13_key in params_dict:
            if loaded_weight.dim() != 3:
                return False
            gate = loaded_weight[:, 0::2, :].contiguous()
            up = loaded_weight[:, 1::2, :].contiguous()
            param = params_dict[w13_key]
            return (
                self._load_packed_moe_experts(param, gate, name, "w1")
                and self._load_packed_moe_experts(param, up, name, "w3")
            )

        if (name.endswith(".feed_forward.experts.w2") or ".experts.w2" in name) and w2_key in params_dict:
            param = params_dict[w2_key]
            return self._load_packed_moe_experts(param, loaded_weight, name, "w2")

        from sglang_omni.models.qwen3_omni.components.thinker_model import (
            extract_fused_experts,
        )

        # Try to use SGLang's FusedMoE weight loading
        # Map w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj
        mapped_name = name
        mapped_name = mapped_name.replace(".experts.w1.weight", ".experts.gate_proj.weight")
        mapped_name = mapped_name.replace(".experts.w3.weight", ".experts.up_proj.weight")
        mapped_name = mapped_name.replace(".experts.w2.weight", ".experts.down_proj.weight")

        # For per-expert weights (w1, w2, w3), use extract_fused_experts
        res = extract_fused_experts(
            name=mapped_name,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.moe_n_experts,
        )
        if res:
            param_name, weight_name, expert_id, shard_id = res
            fused_key = mapped_name.replace(
                weight_name + ".weight", param_name + "weight"
            )
            if fused_key in params_dict:
                param = params_dict[fused_key]
                param.weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                return True

        # Fallback: direct match
        if name in params_dict:
            self._load_param(params_dict[name], loaded_weight)
            return True

        return False

    def _load_packed_moe_experts(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        name: str,
        shard_id: str,
    ) -> bool:
        num_experts = int(self.config.moe_n_experts)
        if loaded_weight.dim() < 3 or int(loaded_weight.shape[0]) != num_experts:
            return False
        weight_name = name if "weight" in name else f"{name}.weight"
        for expert_id, expert_weight in enumerate(loaded_weight.unbind(dim=0)):
            param.weight_loader(
                param,
                expert_weight.contiguous(),
                weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
        return True

    @staticmethod
    def _load_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        pass


EntryClass = Zonos2SGLangModel
