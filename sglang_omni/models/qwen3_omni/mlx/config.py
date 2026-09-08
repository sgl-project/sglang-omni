# SPDX-License-Identifier: Apache-2.0
"""Native MLX configuration parser for Qwen3-Omni.

The parser mirrors the nested Transformers 5.12.1 ``Qwen3OmniMoeConfig`` layout
(``thinker_config``/``talker_config`` each holding a ``text_config`` and, for the
talker, a ``code_predictor_config``). Missing required fields raise ``KeyError``
with the full dotted path so misconfigured checkpoints fail loudly. The public
MLX checkpoint omits the standard 48-by-48 vision position-table size, so that
single Transformers default is restored explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_DEFAULT_VISION_POSITION_EMBEDDINGS = 48 * 48


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


def _require_int_tuple(raw: Mapping[str, Any], key: str, path: str) -> tuple[int, ...]:
    values = _require(raw, key, path)
    return tuple(int(v) for v in values)


def _vision_deepstack_indexes(raw: Mapping[str, Any], path: str) -> tuple[int, ...]:
    return _require_int_tuple(raw, "deepstack_visual_indexes", path)


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
class VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    hidden_act: str
    num_position_embeddings: int
    deepstack_visual_indexes: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "VisionConfig":
        return cls(
            depth=int(_require(raw, "depth", f"{path}.depth")),
            hidden_size=int(_require(raw, "hidden_size", f"{path}.hidden_size")),
            intermediate_size=int(
                _require(raw, "intermediate_size", f"{path}.intermediate_size")
            ),
            num_heads=int(_require(raw, "num_heads", f"{path}.num_heads")),
            in_channels=int(_require(raw, "in_channels", f"{path}.in_channels")),
            patch_size=int(_require(raw, "patch_size", f"{path}.patch_size")),
            temporal_patch_size=int(
                _require(raw, "temporal_patch_size", f"{path}.temporal_patch_size")
            ),
            spatial_merge_size=int(
                _require(raw, "spatial_merge_size", f"{path}.spatial_merge_size")
            ),
            out_hidden_size=int(
                _require(raw, "out_hidden_size", f"{path}.out_hidden_size")
            ),
            hidden_act=str(_require(raw, "hidden_act", f"{path}.hidden_act")),
            num_position_embeddings=int(
                raw.get(
                    "num_position_embeddings",
                    _DEFAULT_VISION_POSITION_EMBEDDINGS,
                )
            ),
            deepstack_visual_indexes=_vision_deepstack_indexes(
                raw, f"{path}.deepstack_visual_indexes"
            ),
        )


@dataclass(frozen=True)
class AudioConfig:
    d_model: int
    encoder_attention_heads: int
    encoder_layers: int
    encoder_ffn_dim: int
    num_mel_bins: int
    downsample_hidden_size: int
    max_source_positions: int
    output_dim: int
    activation_function: str
    scale_embedding: bool
    n_window: int
    n_window_infer: int
    conv_chunksize: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "AudioConfig":
        return cls(
            d_model=int(_require(raw, "d_model", f"{path}.d_model")),
            encoder_attention_heads=int(
                _require(
                    raw, "encoder_attention_heads", f"{path}.encoder_attention_heads"
                )
            ),
            encoder_layers=int(
                _require(raw, "encoder_layers", f"{path}.encoder_layers")
            ),
            encoder_ffn_dim=int(
                _require(raw, "encoder_ffn_dim", f"{path}.encoder_ffn_dim")
            ),
            num_mel_bins=int(_require(raw, "num_mel_bins", f"{path}.num_mel_bins")),
            downsample_hidden_size=int(
                _require(
                    raw, "downsample_hidden_size", f"{path}.downsample_hidden_size"
                )
            ),
            max_source_positions=int(
                _require(raw, "max_source_positions", f"{path}.max_source_positions")
            ),
            output_dim=int(_require(raw, "output_dim", f"{path}.output_dim")),
            activation_function=str(
                _require(raw, "activation_function", f"{path}.activation_function")
            ),
            scale_embedding=bool(
                _require(raw, "scale_embedding", f"{path}.scale_embedding")
            ),
            n_window=int(_require(raw, "n_window", f"{path}.n_window")),
            n_window_infer=int(
                _require(raw, "n_window_infer", f"{path}.n_window_infer")
            ),
            conv_chunksize=int(
                _require(raw, "conv_chunksize", f"{path}.conv_chunksize")
            ),
        )


@dataclass(frozen=True)
class Code2WavConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    attention_dropout: float
    hidden_act: str
    max_position_embeddings: int
    sliding_window: int | None
    codebook_size: int
    num_quantizers: int
    upsampling_ratios: tuple[int, ...]
    decoder_dim: int
    upsample_rates: tuple[int, ...]
    layer_scale_initial_scale: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str) -> "Code2WavConfig":
        sliding_window = _require(raw, "sliding_window", f"{path}.sliding_window")
        return cls(
            hidden_size=int(_require(raw, "hidden_size", f"{path}.hidden_size")),
            intermediate_size=int(
                _require(raw, "intermediate_size", f"{path}.intermediate_size")
            ),
            num_hidden_layers=int(
                _require(raw, "num_hidden_layers", f"{path}.num_hidden_layers")
            ),
            num_attention_heads=int(
                _require(raw, "num_attention_heads", f"{path}.num_attention_heads")
            ),
            num_key_value_heads=int(
                _require(raw, "num_key_value_heads", f"{path}.num_key_value_heads")
            ),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=_require_rope_theta(raw, path),
            attention_bias=bool(raw.get("attention_bias", False)),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            hidden_act=str(raw.get("hidden_act", "silu")),
            max_position_embeddings=int(
                _require(
                    raw, "max_position_embeddings", f"{path}.max_position_embeddings"
                )
            ),
            sliding_window=(
                int(sliding_window) if sliding_window is not None else None
            ),
            codebook_size=int(_require(raw, "codebook_size", f"{path}.codebook_size")),
            num_quantizers=int(
                _require(raw, "num_quantizers", f"{path}.num_quantizers")
            ),
            upsampling_ratios=_require_int_tuple(
                raw, "upsampling_ratios", f"{path}.upsampling_ratios"
            ),
            decoder_dim=int(_require(raw, "decoder_dim", f"{path}.decoder_dim")),
            upsample_rates=_require_int_tuple(
                raw, "upsample_rates", f"{path}.upsample_rates"
            ),
            layer_scale_initial_scale=float(raw.get("layer_scale_initial_scale", 0.01)),
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
    vision: VisionConfig
    audio: AudioConfig
    code2wav: Code2WavConfig
    quantization: QuantizationConfig | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Qwen3OmniMlxConfig":
        thinker_raw = _require(raw, "thinker_config", "thinker_config")
        talker_raw = _require(raw, "talker_config", "talker_config")
        code2wav_raw = _require(raw, "code2wav_config", "code2wav_config")
        quant_raw = raw.get("quantization")
        thinker = ThinkerConfig.from_dict(thinker_raw, path="thinker")
        talker = TalkerConfig.from_dict(talker_raw, path="talker")
        vision_raw = _require(
            thinker_raw, "vision_config", "thinker_config.vision_config"
        )
        audio_raw = _require(thinker_raw, "audio_config", "thinker_config.audio_config")
        vision = VisionConfig.from_dict(vision_raw, path="thinker.vision_config")
        audio = AudioConfig.from_dict(audio_raw, path="thinker.audio_config")
        code2wav = Code2WavConfig.from_dict(code2wav_raw, path="code2wav_config")
        if code2wav.num_quantizers != talker.num_code_groups:
            raise ValueError(
                "code2wav_config.num_quantizers="
                f"{code2wav.num_quantizers} disagrees with "
                f"talker_config.num_code_groups={talker.num_code_groups}"
            )
        return cls(
            thinker=thinker,
            talker=talker,
            vision=vision,
            audio=audio,
            code2wav=code2wav,
            quantization=(
                QuantizationConfig.from_dict(quant_raw)
                if quant_raw is not None
                else None
            ),
        )
