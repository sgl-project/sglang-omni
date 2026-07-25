# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 config adapter.

ZONOS2 ships params.json instead of an HF config.json, so AutoConfig cannot load
it. load_zonos2_pretrained_config reads params.json and builds a Zonos2Config.
"""

from __future__ import annotations

import json
import os
from typing import Any

from transformers import PretrainedConfig

ARCHITECTURE = "Zonos2ForCausalLM"


def _round_ffn(dim: int, multiplier: float, multiple_of: int) -> int:
    """Intermediate size: dim * multiplier rounded up to multiple_of."""
    hidden = int(dim * multiplier)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class Zonos2Config(PretrainedConfig):
    """HF-style config for the ZONOS2 MoE TTS backbone."""

    model_type = "zonos2"

    def __init__(
        self,
        *,
        # ---- backbone ----
        n_layers: int = 28,
        dim: int = 2048,
        head_dim: int = 128,
        n_heads: int | None = None,
        n_kv_heads: int = 4,
        ffn_dim_multiplier: float = 1.5,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seqlen: int = 6144,
        dtype: str = "bfloat16",
        # ---- audio / text vocab ----
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        eoa_id: int = 1024,
        audio_pad_id: int = 1025,
        text_vocab: int = 519,
        loss_softcap: float = 15.0,
        # ---- speaker cloning ----
        speaker_enabled: bool = True,
        speaker_embedding_dim: int = 2048,
        speaker_lda_dim: int = 1024,
        speaker_background_token_enabled: bool = True,
        accurate_mode_token_enabled: bool = True,
        # ---- conditioning ----
        speaking_rate_num_buckets: int = 8,
        speaking_rate_buckets: list[str] | None = None,
        quality_num_buckets: int = 60,
        quality_features: list[str] | None = None,
        quality_buckets: dict[str, list[str]] | None = None,
        # ---- MoE (sonic) ----
        moe_impl: str = "sonic",
        moe_n_experts: int = 16,
        moe_router_topk: int = 1,
        special_topk_layers: dict[str, int] | None = None,
        moe_router_dim: int = 128,
        moe_start_from_layer: int = 3,
        moe_end_from_layer: int = 1,
        moe_balancing_strategy: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Native field names are kept verbatim for the weight loader.
        self.n_layers = n_layers
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads if n_heads else dim // head_dim
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seqlen = max_seqlen
        self.zonos_dtype = dtype

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.eoa_id = eoa_id
        self.audio_pad_id = audio_pad_id
        # codes + eoa + pad
        self.audio_vocab = codebook_size + 2
        self.text_vocab = text_vocab
        self.loss_softcap = loss_softcap

        self.speaker_enabled = speaker_enabled
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_lda_dim = speaker_lda_dim
        self.speaker_background_token_enabled = speaker_background_token_enabled
        self.accurate_mode_token_enabled = accurate_mode_token_enabled

        self.speaking_rate_num_buckets = speaking_rate_num_buckets
        self.speaking_rate_buckets = speaking_rate_buckets or []
        self.quality_num_buckets = quality_num_buckets
        self.quality_features = quality_features or []
        self.quality_buckets = quality_buckets or {}

        self.moe_impl = moe_impl
        self.moe_n_experts = moe_n_experts
        self.moe_router_topk = moe_router_topk
        self.special_topk_layers = {
            int(k): int(v) for k, v in (special_topk_layers or {}).items()
        }
        self.moe_router_dim = moe_router_dim
        self.moe_start_from_layer = moe_start_from_layer
        self.moe_end_from_layer = moe_end_from_layer
        self.moe_balancing_strategy = moe_balancing_strategy

        self.intermediate_size = _round_ffn(dim, ffn_dim_multiplier, multiple_of)

        # HF aliases so generic sglang/transformers code works.
        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = n_kv_heads
        self.max_position_embeddings = max_seqlen
        self.rms_norm_eps = norm_eps
        self.vocab_size = self.audio_vocab

        kwargs.setdefault("architectures", [ARCHITECTURE])
        super().__init__(**kwargs)

    def is_moe_layer(self, layer_id: int) -> bool:
        """MoE band is [moe_start_from_layer, n_layers-1-moe_end_from_layer]."""
        last_moe = self.n_layers - 1 - self.moe_end_from_layer
        return self.moe_start_from_layer <= layer_id <= last_moe

    def topk_for_layer(self, layer_id: int) -> int:
        return self.special_topk_layers.get(layer_id, self.moe_router_topk)

    def uses_eda(self, layer_id: int) -> bool:
        # First MoE layer has no EDA carry (router_states_scale absent there).
        return self.is_moe_layer(layer_id) and layer_id != self.moe_start_from_layer


def _resolve_params_json(model_path: str) -> str:
    """Return a local path to params.json for a dir, a json file, or an HF repo id."""
    local = os.path.join(model_path, "params.json")
    if os.path.isfile(local):
        return local
    if os.path.isfile(model_path) and model_path.endswith(".json"):
        return model_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_path, filename="params.json")


def load_zonos2_pretrained_config(model_path: str) -> Zonos2Config:
    """Build a Zonos2Config from a model dir / json file / HF repo id."""
    with open(_resolve_params_json(model_path), "r") as f:
        params = json.load(f)
    return Zonos2Config(**params)
