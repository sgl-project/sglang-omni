# SPDX-License-Identifier: Apache-2.0
"""HF config wrapper + synthetic Llama / depth sub-configs for CSM-1B.

Native CsmConfig.get_text_config() returns self with vocab_size=2051.
SGLang needs the 2048-hidden / 16-layer / 128256-vocab backbone view.
Must stay sglang/CUDA-free (registry import).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import transformers
from transformers import CsmConfig, LlamaConfig

RopeScaling = dict[str, str | float | int]
ConfigValue = (
    str
    | float
    | int
    | bool
    | RopeScaling
    | transformers.PretrainedConfig
    | SimpleNamespace
    | None
)
ConfigSource = (
    transformers.PretrainedConfig | SimpleNamespace | Mapping[str, ConfigValue] | None
)

# pinned CSM-1B rope scaling from config.json; also the test fallback.
BACKBONE_ROPE_SCALING: RopeScaling = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.125,
    "high_freq_factor": 0.5,
    "original_max_position_embeddings": 1024,
}
DEPTH_ROPE_SCALING: RopeScaling = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.001953125,
    "high_freq_factor": 0.0078125,
    "original_max_position_embeddings": 16,
}
DEFAULT_ROPE_THETA = 500000.0
ROPE_FIELD_NAMES = (
    "rope_type",
    "factor",
    "low_freq_factor",
    "high_freq_factor",
    "original_max_position_embeddings",
    "rope_theta",
)
BACKBONE_LLAMA_CONFIG_KEY = "csm_backbone_llama_config"


def config_value(config: ConfigSource, name: str, default: ConfigValue) -> ConfigValue:
    """Read one optional field from a config of any supported shape."""
    if config is None:
        value: ConfigValue = None
    elif isinstance(config, Mapping):
        value = config.get(name)
    else:
        # HF renames and drops fields across versions; every read is optional.
        value = getattr(config, name, None)
    return default if value is None else value


def as_rope_dict(rope_config: ConfigValue) -> RopeScaling:
    """Rope knobs as a plain dict, dropping unset entries."""
    if rope_config is None:
        fields: Mapping[str, ConfigValue] = {}
    elif isinstance(rope_config, Mapping):
        fields = rope_config
    elif isinstance(rope_config, transformers.PretrainedConfig):
        fields = rope_config.to_dict()
    elif isinstance(rope_config, SimpleNamespace):
        fields = {
            key: value
            for key, value in vars(rope_config).items()
            if key in ROPE_FIELD_NAMES
        }
    else:
        raise TypeError(
            f"rope config must be a mapping or a config object, got "
            f"{type(rope_config).__name__}"
        )
    return {
        key: value
        for key, value in fields.items()
        if value is not None and isinstance(value, (str, float, int))
    }


def rope_fields(
    config: ConfigSource, default_scaling: RopeScaling
) -> tuple[float, RopeScaling]:
    """(rope_theta, rope_scaling) from rope_scaling or v5 rope_parameters."""
    theta = config_value(config, "rope_theta", None)
    scaling = as_rope_dict(config_value(config, "rope_scaling", None))
    parameters = as_rope_dict(config_value(config, "rope_parameters", None))
    if parameters:
        theta = parameters.get("rope_theta", theta)
        for key, value in parameters.items():
            if key != "rope_theta" and key not in scaling:
                scaling[key] = value
            else:
                pass
    else:
        pass
    merged = dict(default_scaling)
    merged.update(scaling)
    merged.setdefault("rope_type", "llama3")
    return float(theta if theta is not None else DEFAULT_ROPE_THETA), merged


def build_backbone_llama_config(csm_config: ConfigSource) -> LlamaConfig:
    """SGLang-loadable LlamaConfig. tie_word_embeddings avoids an unused text lm_head."""
    rope_theta, rope_scaling = rope_fields(csm_config, BACKBONE_ROPE_SCALING)
    return LlamaConfig(
        vocab_size=config_value(csm_config, "text_vocab_size", 128256),
        hidden_size=config_value(csm_config, "hidden_size", 2048),
        intermediate_size=config_value(csm_config, "intermediate_size", 8192),
        num_hidden_layers=config_value(csm_config, "num_hidden_layers", 16),
        num_attention_heads=config_value(csm_config, "num_attention_heads", 32),
        num_key_value_heads=config_value(csm_config, "num_key_value_heads", 8),
        head_dim=config_value(csm_config, "head_dim", 64),
        hidden_act=config_value(csm_config, "hidden_act", "silu"),
        max_position_embeddings=config_value(
            csm_config, "max_position_embeddings", 2048
        ),
        rms_norm_eps=config_value(csm_config, "rms_norm_eps", 1e-5),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        attention_bias=config_value(csm_config, "attention_bias", False),
        attention_dropout=config_value(csm_config, "attention_dropout", 0.0),
        mlp_bias=config_value(csm_config, "mlp_bias", False),
        tie_word_embeddings=True,
        bos_token_id=config_value(csm_config, "bos_token_id", 128000),
        eos_token_id=128001,
        pad_token_id=config_value(csm_config, "pad_token_id", 128002),
    )


def build_depth_decoder_config(csm_config: ConfigSource) -> SimpleNamespace:
    """Depth-decoder knobs. head_dim=128 is explicit, not hidden/heads."""
    depth_config = config_value(csm_config, "depth_decoder_config", None)
    rope_theta, rope_scaling = rope_fields(depth_config, DEPTH_ROPE_SCALING)
    return SimpleNamespace(
        hidden_size=config_value(depth_config, "hidden_size", 1024),
        num_hidden_layers=config_value(depth_config, "num_hidden_layers", 4),
        num_attention_heads=config_value(depth_config, "num_attention_heads", 8),
        head_dim=config_value(depth_config, "head_dim", 128),
        num_key_value_heads=config_value(depth_config, "num_key_value_heads", 2),
        intermediate_size=config_value(depth_config, "intermediate_size", 8192),
        max_position_embeddings=config_value(
            depth_config, "max_position_embeddings", 33
        ),
        backbone_hidden_size=config_value(depth_config, "backbone_hidden_size", 2048),
        vocab_size=config_value(depth_config, "vocab_size", 2051),
        num_codebooks=config_value(depth_config, "num_codebooks", 32),
        rms_norm_eps=config_value(depth_config, "rms_norm_eps", 1e-5),
        hidden_act=config_value(depth_config, "hidden_act", "silu"),
        attention_bias=config_value(depth_config, "attention_bias", False),
        mlp_bias=config_value(depth_config, "mlp_bias", False),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )


class CsmTtsHfConfig(CsmConfig):
    """CsmConfig whose get_text_config() returns the backbone Llama config."""

    # PretrainedConfig.get_text_config passes decoder; CSM has one text config.
    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        cached = self.__dict__.get(BACKBONE_LLAMA_CONFIG_KEY)
        if cached is None:
            cached = build_backbone_llama_config(self)
            self.__dict__[BACKBONE_LLAMA_CONFIG_KEY] = cached
        else:
            pass
        return cached

    def to_dict(self) -> dict[str, ConfigValue]:
        output = super().to_dict()
        output.pop(BACKBONE_LLAMA_CONFIG_KEY, None)
        return output


__all__ = [
    "BACKBONE_ROPE_SCALING",
    "ConfigSource",
    "ConfigValue",
    "CsmTtsHfConfig",
    "DEPTH_ROPE_SCALING",
    "RopeScaling",
    "build_backbone_llama_config",
    "build_depth_decoder_config",
]
