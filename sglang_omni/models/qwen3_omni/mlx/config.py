# SPDX-License-Identifier: Apache-2.0
"""Native MLX configuration parser for Qwen3-Omni.

The parser mirrors the nested Transformers 5.12.1 ``Qwen3OmniMoeConfig`` layout
(``thinker_config``/``talker_config`` each holding a ``text_config`` and, for the
talker, a ``code_predictor_config``). Missing required fields raise ``KeyError``
with the full dotted path so misconfigured checkpoints fail loudly rather than
silently defaulting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _require(raw: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in raw:
        raise KeyError(f"missing required Qwen3-Omni config field: {path}")
    return raw[key]


def _require_expert_count(raw: Mapping[str, Any], path: str) -> int:
    """Resolve the MoE expert count under either canonical or alias name.

    The thinker text config stores ``num_experts`` directly, while the talker
    text config serializes it under the ``num_local_experts`` alias (Transformers
    ``attribute_map``). Both normalize to ``num_experts`` here.
    """

    if "num_experts" in raw:
        return int(raw["num_experts"])
    if "num_local_experts" in raw:
        return int(raw["num_local_experts"])
    raise KeyError(f"missing required Qwen3-Omni config field: {path}.num_experts")


def _rope_container(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the rope sub-dict under either the normalized or legacy key.

    Transformers 5.12.1 normalizes rope into a single ``rope_parameters`` dict
    (holding ``rope_type``/``rope_theta``/``mrope_section``). Published
    checkpoints instead store a ``rope_scaling`` dict (``rope_type`` +
    ``mrope_section``) alongside a *sibling* ``rope_theta``. A native parser
    reading ``config.json`` directly must accept both.
    """

    normalized = raw.get("rope_parameters")
    if normalized is not None:
        return normalized
    legacy = raw.get("rope_scaling")
    if legacy is not None:
        return legacy
    return {}


def _require_rope_theta(raw: Mapping[str, Any], path: str) -> float:
    """Read ``rope_theta`` from the normalized rope dict or the legacy sibling."""

    rope = _rope_container(raw)
    if "rope_theta" in rope:
        return float(rope["rope_theta"])
    if "rope_theta" in raw:
        return float(raw["rope_theta"])
    raise KeyError(f"missing required Qwen3-Omni config field: {path}.rope_theta")


def _require_mrope_section(raw: Mapping[str, Any], path: str) -> tuple[int, ...]:
    """Read the required ``mrope_section`` from either rope layout.

    ``mrope_section`` is mandatory for the M-RoPE text stacks; it must never
    silently default to ``None``. A config missing it under both rope layouts
    raises naming the full dotted path.
    """

    rope = _rope_container(raw)
    mrope = rope.get("mrope_section")
    if mrope is None:
        mrope = raw.get("mrope_section")
    if mrope is None:
        raise KeyError(
            f"missing required Qwen3-Omni config field: {path}.mrope_section"
        )
    return tuple(int(v) for v in mrope)


@dataclass(frozen=True)
class MoeTextConfig:
    """Text-decoder config for a thinker/talker MoE stack."""

    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    mrope_section: tuple[int, ...]
    tie_word_embeddings: bool
    shared_expert_intermediate_size: int | None
    # Published Qwen3-Omni thinker/talker configs set this false, but it must be
    # parsed rather than assumed: a true value adds q/k/v/o projection biases,
    # which a strict weight load would otherwise reject with a key error.
    attention_bias: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "MoeTextConfig":
        shared = raw.get("shared_expert_intermediate_size")
        return cls(
            hidden_size=int(_require(raw, "hidden_size", f"{path}.hidden_size")),
            head_dim=int(_require(raw, "head_dim", f"{path}.head_dim")),
            num_attention_heads=int(
                _require(raw, "num_attention_heads", f"{path}.num_attention_heads")
            ),
            num_key_value_heads=int(
                _require(raw, "num_key_value_heads", f"{path}.num_key_value_heads")
            ),
            num_hidden_layers=int(
                _require(raw, "num_hidden_layers", f"{path}.num_hidden_layers")
            ),
            intermediate_size=int(
                _require(raw, "intermediate_size", f"{path}.intermediate_size")
            ),
            moe_intermediate_size=int(
                _require(raw, "moe_intermediate_size", f"{path}.moe_intermediate_size")
            ),
            num_experts=_require_expert_count(raw, path),
            num_experts_per_tok=int(
                _require(raw, "num_experts_per_tok", f"{path}.num_experts_per_tok")
            ),
            norm_topk_prob=bool(raw.get("norm_topk_prob", False)),
            decoder_sparse_step=int(raw.get("decoder_sparse_step", 1)),
            mlp_only_layers=tuple(raw.get("mlp_only_layers") or ()),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            vocab_size=int(_require(raw, "vocab_size", f"{path}.vocab_size")),
            max_position_embeddings=int(
                _require(
                    raw, "max_position_embeddings", f"{path}.max_position_embeddings"
                )
            ),
            rope_theta=_require_rope_theta(raw, path),
            mrope_section=_require_mrope_section(raw, path),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            shared_expert_intermediate_size=(
                int(shared) if shared is not None else None
            ),
            attention_bias=bool(raw.get("attention_bias", False)),
        )


@dataclass(frozen=True)
class CodePredictorConfig:
    """Config for the talker code predictor stack."""

    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    num_code_groups: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "CodePredictorConfig":
        return cls(
            hidden_size=int(_require(raw, "hidden_size", f"{path}.hidden_size")),
            head_dim=int(_require(raw, "head_dim", f"{path}.head_dim")),
            num_attention_heads=int(
                _require(raw, "num_attention_heads", f"{path}.num_attention_heads")
            ),
            num_key_value_heads=int(
                _require(raw, "num_key_value_heads", f"{path}.num_key_value_heads")
            ),
            num_hidden_layers=int(
                _require(raw, "num_hidden_layers", f"{path}.num_hidden_layers")
            ),
            intermediate_size=int(
                _require(raw, "intermediate_size", f"{path}.intermediate_size")
            ),
            num_code_groups=int(
                _require(raw, "num_code_groups", f"{path}.num_code_groups")
            ),
            vocab_size=int(_require(raw, "vocab_size", f"{path}.vocab_size")),
            max_position_embeddings=int(
                _require(
                    raw, "max_position_embeddings", f"{path}.max_position_embeddings"
                )
            ),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=_require_rope_theta(raw, path),
        )


@dataclass(frozen=True)
class ThinkerConfig:
    """Thinker section of the Omni config."""

    text_config: MoeTextConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "ThinkerConfig":
        text_raw = _require(raw, "text_config", f"{path}.text_config")
        return cls(
            text_config=MoeTextConfig.from_dict(text_raw, path=f"{path}.text_config")
        )


@dataclass(frozen=True)
class TalkerConfig:
    """Talker section of the Omni config."""

    text_config: MoeTextConfig
    code_predictor_config: CodePredictorConfig
    num_code_groups: int
    # The thinker hidden size the talker's resize MLPs (``text_projection`` /
    # ``hidden_projection``) consume. Published Qwen3-Omni configs serialize it
    # on ``talker_config`` itself; it stays optional so a config that predates
    # the field still parses, and the caller then falls back to the parsed
    # thinker text hidden size.
    thinker_hidden_size: int | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "TalkerConfig":
        text_raw = _require(raw, "text_config", f"{path}.text_config")
        predictor_raw = _require(
            raw, "code_predictor_config", f"{path}.code_predictor_config"
        )
        predictor = CodePredictorConfig.from_dict(
            predictor_raw, path=f"{path}.code_predictor_config"
        )
        num_code_groups = int(
            _require(raw, "num_code_groups", f"{path}.num_code_groups")
        )
        # The talker embeds group 0 with its own codec table and delegates the
        # remaining groups to the predictor's per-group tables and heads, so a
        # disagreement here would silently build a talker that emits a
        # different number of codes than code2wav's quantizer count expects.
        if num_code_groups != predictor.num_code_groups:
            raise ValueError(
                f"{path}.num_code_groups={num_code_groups} disagrees with "
                f"{path}.code_predictor_config.num_code_groups="
                f"{predictor.num_code_groups}"
            )
        thinker_hidden_size = raw.get("thinker_hidden_size")
        return cls(
            text_config=MoeTextConfig.from_dict(text_raw, path=f"{path}.text_config"),
            code_predictor_config=predictor,
            num_code_groups=num_code_groups,
            thinker_hidden_size=(
                int(thinker_hidden_size) if thinker_hidden_size is not None else None
            ),
        )


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization metadata carried by a converted MLX checkpoint.

    ``bits`` and ``group_size`` are parsed from the checkpoint's ``quantization``
    block so the pre-load quantization path is driven by real configuration
    rather than hard-coded literals. ``mode`` defaults to affine (the only mode
    the converter currently emits).
    """

    bits: int
    group_size: int
    mode: str = "affine"

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, path: str = "quantization"
    ) -> "QuantizationConfig":
        return cls(
            bits=int(_require(raw, "bits", f"{path}.bits")),
            group_size=int(_require(raw, "group_size", f"{path}.group_size")),
            mode=str(raw.get("mode", "affine")),
        )


@dataclass(frozen=True)
class Qwen3OmniMlxConfig:
    """Native MLX view of the nested Qwen3-Omni configuration."""

    thinker: ThinkerConfig
    talker: TalkerConfig
    quantization: QuantizationConfig | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Qwen3OmniMlxConfig":
        thinker_raw = _require(raw, "thinker_config", "thinker_config")
        talker_raw = _require(raw, "talker_config", "talker_config")
        quant_raw = raw.get("quantization")
        return cls(
            thinker=ThinkerConfig.from_dict(thinker_raw, path="thinker"),
            talker=TalkerConfig.from_dict(talker_raw, path="talker"),
            quantization=(
                QuantizationConfig.from_dict(quant_raw)
                if quant_raw is not None
                else None
            ),
        )
