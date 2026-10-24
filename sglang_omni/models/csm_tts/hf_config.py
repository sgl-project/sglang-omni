# SPDX-License-Identifier: Apache-2.0
"""HF config wrapper + synthetic Llama / depth sub-configs for CSM-1B.

Native CsmConfig.get_text_config() returns self with vocab_size=2051.
SGLang needs the 2048-hidden / 16-layer / 128256-vocab backbone view.
Must stay sglang/CUDA-free (registry import).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import transformers
from transformers import CsmConfig, LlamaConfig

# Pinned CSM-1B rope scaling (config.json:105-111 backbone, :80-86 depth);
# also the fallback when a (test) config omits the fields.
_BACKBONE_ROPE_SCALING: dict[str, Any] = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.125,
    "high_freq_factor": 0.5,
    "original_max_position_embeddings": 1024,
}
_DEPTH_ROPE_SCALING: dict[str, Any] = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.001953125,
    "high_freq_factor": 0.0078125,
    "original_max_position_embeddings": 16,
}
_DEFAULT_ROPE_THETA = 500000.0


def cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        value = None
    elif isinstance(cfg, dict):
        value = cfg.get(name)
    else:
        value = getattr(cfg, name, None)
    return default if value is None else value


def as_rope_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if v is not None}
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
        if isinstance(raw, dict):
            return {k: v for k, v in raw.items() if v is not None}
    out: dict[str, Any] = {}
    for key in (
        "rope_type",
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
        "rope_theta",
    ):
        value = getattr(obj, key, None)
        if value is not None:
            out[key] = value
    return out


def rope_fields(
    cfg: Any, default_scaling: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """(rope_theta, rope_scaling) from rope_scaling or v5 rope_parameters."""
    theta = cfg_get(cfg, "rope_theta", None)
    scaling = as_rope_dict(cfg_get(cfg, "rope_scaling", None))
    params = as_rope_dict(cfg_get(cfg, "rope_parameters", None))
    if params:
        theta = params.get("rope_theta", theta)
        for key, value in params.items():
            if key != "rope_theta" and key not in scaling:
                scaling[key] = value
    merged = dict(default_scaling)
    merged.update(scaling)
    merged.setdefault("rope_type", "llama3")
    return float(theta if theta is not None else _DEFAULT_ROPE_THETA), merged


def build_backbone_llama_config(csm_cfg: Any) -> LlamaConfig:
    """SGLang-loadable LlamaConfig. tie_word_embeddings avoids an unused text lm_head."""
    rope_theta, rope_scaling = rope_fields(csm_cfg, _BACKBONE_ROPE_SCALING)
    return LlamaConfig(
        vocab_size=cfg_get(csm_cfg, "text_vocab_size", 128256),
        hidden_size=cfg_get(csm_cfg, "hidden_size", 2048),
        intermediate_size=cfg_get(csm_cfg, "intermediate_size", 8192),
        num_hidden_layers=cfg_get(csm_cfg, "num_hidden_layers", 16),
        num_attention_heads=cfg_get(csm_cfg, "num_attention_heads", 32),
        num_key_value_heads=cfg_get(csm_cfg, "num_key_value_heads", 8),
        head_dim=cfg_get(csm_cfg, "head_dim", 64),
        hidden_act=cfg_get(csm_cfg, "hidden_act", "silu"),
        max_position_embeddings=cfg_get(csm_cfg, "max_position_embeddings", 2048),
        rms_norm_eps=cfg_get(csm_cfg, "rms_norm_eps", 1e-5),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        attention_bias=cfg_get(csm_cfg, "attention_bias", False),
        attention_dropout=cfg_get(csm_cfg, "attention_dropout", 0.0),
        mlp_bias=cfg_get(csm_cfg, "mlp_bias", False),
        tie_word_embeddings=True,
        bos_token_id=cfg_get(csm_cfg, "bos_token_id", 128000),
        eos_token_id=128001,
        pad_token_id=cfg_get(csm_cfg, "pad_token_id", 128002),
    )


def build_depth_decoder_config(csm_cfg: Any) -> SimpleNamespace:
    """Depth-decoder knobs. head_dim=128 is explicit, not hidden/heads."""
    raw = cfg_get(csm_cfg, "depth_decoder_config", None)
    rope_theta, rope_scaling = rope_fields(raw, _DEPTH_ROPE_SCALING)
    return SimpleNamespace(
        hidden_size=cfg_get(raw, "hidden_size", 1024),
        num_hidden_layers=cfg_get(raw, "num_hidden_layers", 4),
        num_attention_heads=cfg_get(raw, "num_attention_heads", 8),
        head_dim=cfg_get(raw, "head_dim", 128),
        num_key_value_heads=cfg_get(raw, "num_key_value_heads", 2),
        intermediate_size=cfg_get(raw, "intermediate_size", 8192),
        max_position_embeddings=cfg_get(raw, "max_position_embeddings", 33),
        backbone_hidden_size=cfg_get(raw, "backbone_hidden_size", 2048),
        vocab_size=cfg_get(raw, "vocab_size", 2051),
        num_codebooks=cfg_get(raw, "num_codebooks", 32),
        rms_norm_eps=cfg_get(raw, "rms_norm_eps", 1e-5),
        hidden_act=cfg_get(raw, "hidden_act", "silu"),
        attention_bias=cfg_get(raw, "attention_bias", False),
        mlp_bias=cfg_get(raw, "mlp_bias", False),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )


class CsmTtsHfConfig(CsmConfig):
    """CsmConfig whose get_text_config() returns the backbone Llama config."""

    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        del decoder
        cached = self.__dict__.get("_csm_backbone_llama_config")
        if cached is None:
            cached = build_backbone_llama_config(self)
            self.__dict__["_csm_backbone_llama_config"] = cached
        return cached

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output.pop("_csm_backbone_llama_config", None)
        return output


__all__ = [
    "CsmTtsHfConfig",
    "build_backbone_llama_config",
    "build_depth_decoder_config",
]
