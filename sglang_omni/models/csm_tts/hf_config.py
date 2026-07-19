# SPDX-License-Identifier: Apache-2.0
# Builds on HuggingFace Transformers' CSM config (`transformers/models/csm`, Apache-2.0).
"""HF config wrapper + synthetic sub-configs for CSM-1B (``sesame/csm-1b``).

CSM's native ``transformers.CsmConfig`` stores the backbone fields FLAT (not as
a nested ``text_config``) and its ``get_text_config()`` returns ``self`` with
``vocab_size=2051`` — useless for SGLang's ``ModelConfig.from_server_args``,
which reads head counts / hidden size / layers / vocab through
``get_text_config()``. This module synthesizes a real ``LlamaConfig`` for the
backbone and a small namespace config for the depth decoder.

Must stay sglang/CUDA-free: imported by ``csm_tts/__init__.py`` during the
registry pkgutil scan (contract).


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


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    """Field read off a PretrainedConfig / SimpleNamespace / dict / None."""
    if cfg is None:
        value = None
    elif isinstance(cfg, dict):
        value = cfg.get(name)
    else:
        value = getattr(cfg, name, None)
    return default if value is None else value


def _rope_fields(
    cfg: Any, default_scaling: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """``(rope_theta, rope_scaling)`` from either layout: transformers<5
    ``rope_theta`` + ``rope_scaling`` attrs (the checkpoint's serialized form)
    or the v5 merged ``rope_parameters`` dict."""
    theta = _cfg_get(cfg, "rope_theta", None)
    scaling = _cfg_get(cfg, "rope_scaling", None)
    if scaling is None:
        params = _cfg_get(cfg, "rope_parameters", None)
        if isinstance(params, dict):
            theta = params.get("rope_theta", theta)
            scaling = {k: v for k, v in params.items() if k != "rope_theta"}
    scaling = dict(scaling) if scaling else dict(default_scaling)
    scaling.setdefault("rope_type", "llama3")
    return float(theta if theta is not None else _DEFAULT_ROPE_THETA), scaling


def build_backbone_llama_config(csm_cfg: Any) -> LlamaConfig:
    """Synthesize the SGLang-loadable backbone ``LlamaConfig``.

    Reads the FLAT backbone fields off the HF ``CsmConfig`` and realizes:

    - ``hidden_size=2048``, ``num_hidden_layers=16``, ``num_attention_heads=32``,
      ``num_key_value_heads=8``, ``head_dim=64``, ``intermediate_size=8192``
    - ``max_position_embeddings=2048`` (hard model ceiling)
    - ``rope_theta=500000``,
      ``rope_scaling={"rope_type": "llama3", "factor": 32.0,
      "low_freq_factor": 0.125, "high_freq_factor": 0.5,
      "original_max_position_embeddings": 1024}``
    - ``vocab_size=128256`` (text vocab IS the backbone embedding)
    - ``tie_word_embeddings=True`` — sglang ``LlamaForCausalLM``'s lm_head then
      aliases ``embed_tokens``, avoiding a dead 525 MB bf16 text head that we
      never use (cb0 comes from ``codebook0_head``, not the backbone lm_head).

    Args:
        csm_cfg: the (possibly wrapped) HF ``CsmConfig`` for ``sesame/csm-1b``.

    Returns:
        A concrete ``transformers.LlamaConfig``.
    """
    rope_theta, rope_scaling = _rope_fields(csm_cfg, _BACKBONE_ROPE_SCALING)
    return LlamaConfig(
        vocab_size=_cfg_get(csm_cfg, "text_vocab_size", 128256),
        hidden_size=_cfg_get(csm_cfg, "hidden_size", 2048),
        intermediate_size=_cfg_get(csm_cfg, "intermediate_size", 8192),
        num_hidden_layers=_cfg_get(csm_cfg, "num_hidden_layers", 16),
        num_attention_heads=_cfg_get(csm_cfg, "num_attention_heads", 32),
        num_key_value_heads=_cfg_get(csm_cfg, "num_key_value_heads", 8),
        head_dim=_cfg_get(csm_cfg, "head_dim", 64),
        hidden_act=_cfg_get(csm_cfg, "hidden_act", "silu"),
        max_position_embeddings=_cfg_get(csm_cfg, "max_position_embeddings", 2048),
        rms_norm_eps=_cfg_get(csm_cfg, "rms_norm_eps", 1e-5),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        attention_bias=_cfg_get(csm_cfg, "attention_bias", False),
        attention_dropout=_cfg_get(csm_cfg, "attention_dropout", 0.0),
        mlp_bias=_cfg_get(csm_cfg, "mlp_bias", False),
        tie_word_embeddings=True,
        bos_token_id=_cfg_get(csm_cfg, "bos_token_id", 128000),
        eos_token_id=128001,  # <|end_of_text|>; cb0 ids (≤2050) can never collide
        pad_token_id=_cfg_get(csm_cfg, "pad_token_id", 128002),
    )


def build_depth_decoder_config(csm_cfg: Any) -> SimpleNamespace:
    """Synthesize the depth-decoder config namespace.

    Depth decoder: 4 layers, ``hidden_size=1024``, 8 heads with EXPLICIT
    ``head_dim=128`` (do not derive head_dim from hidden), 2 KV heads (GQA),
    ``intermediate_size=8192`` (SwiGLU), ``max_position_embeddings=33`` (1
    backbone hidden + 32 codebook positions), depth rope ``theta=500000``
    llama3-scaled with ``factor=32.0``, ``low_freq_factor=0.001953125``,
    ``high_freq_factor=0.0078125``, ``original_max_position_embeddings=16``.

    Args:
        csm_cfg: the HF ``CsmConfig``.

    Returns:
        ``SimpleNamespace`` (or ``PretrainedConfig``) consumed by
        :class:`sglang_omni.models.csm_tts.modeling.CsmDepthDecoder`.
    """
    raw = _cfg_get(csm_cfg, "depth_decoder_config", None)
    rope_theta, rope_scaling = _rope_fields(raw, _DEPTH_ROPE_SCALING)
    return SimpleNamespace(
        hidden_size=_cfg_get(raw, "hidden_size", 1024),
        num_hidden_layers=_cfg_get(raw, "num_hidden_layers", 4),
        num_attention_heads=_cfg_get(raw, "num_attention_heads", 8),
        head_dim=_cfg_get(raw, "head_dim", 128),
        num_key_value_heads=_cfg_get(raw, "num_key_value_heads", 2),
        intermediate_size=_cfg_get(raw, "intermediate_size", 8192),
        max_position_embeddings=_cfg_get(raw, "max_position_embeddings", 33),
        backbone_hidden_size=_cfg_get(raw, "backbone_hidden_size", 2048),
        vocab_size=_cfg_get(raw, "vocab_size", 2051),
        num_codebooks=_cfg_get(raw, "num_codebooks", 32),
        rms_norm_eps=_cfg_get(raw, "rms_norm_eps", 1e-5),
        hidden_act=_cfg_get(raw, "hidden_act", "silu"),
        attention_bias=_cfg_get(raw, "attention_bias", False),
        mlp_bias=_cfg_get(raw, "mlp_bias", False),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )


class CsmTtsHfConfig(CsmConfig):
    """``CsmConfig`` subclass whose ``get_text_config()`` returns the realized
    synthetic backbone ``LlamaConfig``.

    Subclassing the native class keeps ``isinstance`` and field compatibility
    for HF-side tooling, and inherits ``model_type="csm"`` so
    ``AutoConfig.register("csm", ..., exist_ok=True)`` passes the model-type
    consistency check (override semantics: ``_LazyConfigMapping._extra_content``
    is consulted BEFORE the native mapping — the override wins in-process;
    confirm in the container via ``AutoConfig.for_model("csm")``).

    """

    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        """Return the synthetic backbone ``LlamaConfig`` (NOT ``self``).

        Native ``CsmConfig.get_text_config()`` would return ``self`` with
        ``vocab_size=2051``; SGLang reads head counts / hidden / layers / vocab
        through this hook, so it must see the 2048-hidden / 16-layer / 128256-
        vocab backbone view.
        """
        del decoder
        cached = self.__dict__.get("_csm_backbone_llama_config")
        if cached is None:
            cached = build_backbone_llama_config(self)
            # Cached for object stability (callers may hold + mutate the
            # returned reference, higgs pattern); stripped from serialization
            # in to_dict() below.
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
