# SPDX-License-Identifier: Apache-2.0
"""HF config for Drama 3 checkpoints (model_type fish_omni).

A fish_omni_text text_config converts to Drama3TextConfig, the form a saved config
stores and reloads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from transformers import PretrainedConfig

from sglang_omni.models.fishaudio_drama3.text_config import (
    FISH_TEXT_MODEL_TYPE,
    Drama3TextConfig,
    checked_int,
    checked_number,
    checked_text_config,
    fish_text_kwargs,
)

FISH_SCHEMA_VERSION = 1
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"

__all__ = [
    "FishOmniConfig",
    "FishOmniAudioDecoderConfig",
    "Drama3TextConfig",
    "checked_int",
    "checked_number",
    "checked_text_config",
    "fish_text_kwargs",
]


class FishOmniAudioDecoderConfig(PretrainedConfig):
    """The codebook decoder tower config, under the checkpoint's own field names.

    rotary_interleaved and attention_qk_norm are stored but not read: the
    tower always uses interleaved rope and per-head q/k norm, whatever they say.
    """

    model_type = "fish_qwen3_audio_decoder"
    base_config_key = "audio_decoder_config"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        decoder_type: str | None = None,
        dim: int | None = None,
        n_head: int | None = None,
        n_local_heads: int | None = None,
        head_dim: int | None = None,
        n_layer: int | None = None,
        intermediate_size: int | None = None,
        norm_eps: float | None = None,
        num_codebooks: int | None = None,
        vocab_size: int | None = None,
        text_dim: int | None = None,
        max_seq_len: int | None = None,
        shared_enrichment_layers: int | None = None,
        rope_base: float | None = None,
        rotary_percent: float | None = None,
        rotary_interleaved: bool | None = None,
        attention_output_gate: bool | None = None,
        attention_qk_norm: bool | None = None,
        attention_qkv_bias: bool = False,
        attention_o_bias: bool = False,
        layernorm_zero_centered_gamma: bool = False,
        vq_feature_scope: str = "all_vq",
        vq_feature_placement: str | None = None,
        vq_backbone_codebooks: int = 0,
        mtp_spec: Mapping[str, object] | None = None,
        use_moe: bool = False,
        mtp_use_swa: bool = False,
        window_size: int | None = None,
        window_attn_skip_freq: int | None = None,
        window_rotary_base: float | None = None,
        dropout: float = 0.0,
        **kwargs: object,
    ) -> None:
        declared_model_type = kwargs.pop("model_type", self.model_type)
        if declared_model_type != self.model_type:
            raise ValueError(
                f"audio_decoder_config.model_type={declared_model_type!r}; expected "
                f"{self.model_type!r}"
            )
        else:
            pass
        self.dim = checked_int(dim, "audio_decoder_config.dim", 1)
        self.n_head = checked_int(n_head, "audio_decoder_config.n_head", 1)
        self.n_local_heads = checked_int(
            n_local_heads, "audio_decoder_config.n_local_heads", 1
        )
        self.head_dim = checked_int(head_dim, "audio_decoder_config.head_dim", 1)
        self.n_layer = checked_int(n_layer, "audio_decoder_config.n_layer", 1)
        self.intermediate_size = checked_int(
            intermediate_size, "audio_decoder_config.intermediate_size", 1
        )
        self.num_codebooks = checked_int(
            num_codebooks, "audio_decoder_config.num_codebooks", 2
        )
        self.vocab_size = checked_int(vocab_size, "audio_decoder_config.vocab_size", 1)
        self.text_dim = checked_int(text_dim, "audio_decoder_config.text_dim", 1)
        self.max_seq_len = checked_int(
            max_seq_len, "audio_decoder_config.max_seq_len", 1
        )
        self.shared_enrichment_layers = checked_int(
            shared_enrichment_layers, "audio_decoder_config.shared_enrichment_layers", 1
        )
        self.vq_backbone_codebooks = checked_int(
            vq_backbone_codebooks, "audio_decoder_config.vq_backbone_codebooks", 0
        )
        self.norm_eps = checked_number(
            norm_eps, "audio_decoder_config.norm_eps", math.inf
        )
        self.rope_base = checked_number(
            rope_base, "audio_decoder_config.rope_base", math.inf
        )
        self.rotary_percent = checked_number(
            rotary_percent, "audio_decoder_config.rotary_percent", 1.0
        )
        if mtp_spec is None:
            self.mtp_spec: dict[str, object] = {}
        elif isinstance(mtp_spec, Mapping):
            self.mtp_spec = dict(mtp_spec)
        else:
            raise ValueError(
                f"audio_decoder_config.mtp_spec must be a dict or null; got "
                f"{mtp_spec!r}"
            )
        if decoder_type != "codebook_mtp_shared":
            raise NotImplementedError(
                f"audio_decoder_config.decoder_type={decoder_type!r} is not supported"
            )
        elif vq_feature_scope != "all_vq":
            raise NotImplementedError(
                f"audio_decoder_config.vq_feature_scope={vq_feature_scope!r} is not "
                f"supported"
            )
        elif vq_feature_placement == "backbone_partial" and not (
            1 <= self.vq_backbone_codebooks <= self.num_codebooks - 1
        ):
            raise ValueError(
                f"backbone_partial needs 1 <= vq_backbone_codebooks <= "
                f"{self.num_codebooks - 1}; got {self.vq_backbone_codebooks}"
            )
        elif vq_feature_placement == "mtp_shared_pre" and self.vq_backbone_codebooks:
            raise ValueError("mtp_shared_pre needs vq_backbone_codebooks = 0")
        elif vq_feature_placement not in ("backbone_partial", "mtp_shared_pre"):
            raise NotImplementedError(
                f"audio_decoder_config.vq_feature_placement={vq_feature_placement!r} "
                f"is not supported"
            )
        elif self.n_layer != self.shared_enrichment_layers:
            raise ValueError(
                f"audio_decoder_config.n_layer={self.n_layer} differs from "
                f"shared_enrichment_layers={self.shared_enrichment_layers}"
            )
        elif self.max_seq_len != self.num_codebooks + 1:
            raise ValueError(
                f"audio_decoder_config.max_seq_len={self.max_seq_len} is not "
                f"num_codebooks + 1 = {self.num_codebooks + 1}"
            )
        elif self.dim != self.text_dim:
            raise NotImplementedError(
                f"audio_decoder_config.dim={self.dim} differs from text_dim="
                f"{self.text_dim}"
            )
        elif self.n_head % self.n_local_heads != 0:
            raise ValueError(
                f"audio_decoder_config.n_head={self.n_head} is not a multiple of "
                f"n_local_heads={self.n_local_heads}"
            )
        elif self.rotary_dim % 2 != 0 or self.rotary_dim == 0:
            raise ValueError(
                f"head_dim * rotary_percent must give a positive even rotary width; "
                f"got {self.rotary_dim}"
            )
        elif attention_output_gate is not True:
            raise NotImplementedError(
                "the decoder tower is implemented with the attention output gate only"
            )
        elif layernorm_zero_centered_gamma is not True:
            raise NotImplementedError(
                "the decoder tower is implemented with zero-centered norm gammas only"
            )
        elif self.mtp_spec.get("attention_variant", "full_attention") != (
            "full_attention"
        ):
            raise NotImplementedError("the decoder tower is full attention only")
        elif (
            use_moe
            or any(
                key in ("num_experts", "num_experts_per_tok") or key.startswith("moe_")
                for key in self.mtp_spec
            )
            or mtp_use_swa
            or attention_qkv_bias
            or attention_o_bias
            or window_size is not None
            or window_attn_skip_freq is not None
            or window_rotary_base is not None
            or dropout != 0.0
        ):
            raise NotImplementedError(
                "MoE, sliding windows, attention biases and dropout are not part of "
                "the decoder tower"
            )
        else:
            pass
        self.decoder_type = decoder_type
        self.rotary_interleaved = rotary_interleaved
        self.attention_output_gate = True
        self.attention_qk_norm = attention_qk_norm
        self.layernorm_zero_centered_gamma = True
        self.vq_feature_scope = vq_feature_scope
        self.vq_feature_placement = vq_feature_placement
        super().__init__(**kwargs)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)


class FishOmniConfig(PretrainedConfig):
    """Drama 3 checkpoint: a Qwen3.5 MoE backbone plus the codebook decoder tower."""

    model_type = "fish_omni"
    sub_configs = {
        "text_config": Drama3TextConfig,
        "audio_decoder_config": FishOmniAudioDecoderConfig,
    }
    has_no_defaults_at_init = True

    def __init__(
        self,
        text_config: Mapping[str, object] | None = None,
        audio_decoder_config: Mapping[str, object] | None = None,
        audio_encoder_config: Mapping[str, object] | None = None,
        semantic_start_token_id: int | None = None,
        semantic_end_token_id: int | None = None,
        speech_template_version: str | None = None,
        extra_samplable_token_ids: list[int] | None = None,
        fish_schema_version: int = FISH_SCHEMA_VERSION,
        **kwargs: object,
    ) -> None:
        if fish_schema_version != FISH_SCHEMA_VERSION:
            raise ValueError(
                f"fish_schema_version={fish_schema_version!r} is not supported "
                f"(expected {FISH_SCHEMA_VERSION})"
            )
        elif speech_template_version != SPEECH_TEMPLATE_VERSION:
            raise ValueError(
                f"speech_template_version={speech_template_version!r}; this model "
                f"only builds {SPEECH_TEMPLATE_VERSION} prompts"
            )
        elif audio_encoder_config is not None:
            raise NotImplementedError("audio_encoder_config is not supported")
        elif extra_samplable_token_ids:
            raise NotImplementedError("extra_samplable_token_ids is not supported")
        elif not isinstance(text_config, Mapping):
            raise ValueError(f"text_config must be a dict; got {text_config!r}")
        elif not isinstance(audio_decoder_config, Mapping):
            raise ValueError(
                f"audio_decoder_config must be a dict; got {audio_decoder_config!r}"
            )
        elif text_config.get("model_type") == FISH_TEXT_MODEL_TYPE:
            text_kwargs = fish_text_kwargs(text_config)
        elif text_config.get("model_type") == Qwen3_5MoeTextConfig.model_type:
            text_kwargs = {
                key: value for key, value in text_config.items() if key != "model_type"
            }
        else:
            raise ValueError(
                f"text_config.model_type={text_config.get('model_type')!r}; expected "
                f"{FISH_TEXT_MODEL_TYPE!r} or {Qwen3_5MoeTextConfig.model_type!r}"
            )
        self.audio_decoder_config = FishOmniAudioDecoderConfig(**audio_decoder_config)
        tower_layer_count = (
            self.audio_decoder_config.shared_enrichment_layers
            + self.audio_decoder_config.num_codebooks
            - 1
        )
        if text_kwargs.get("tower_layer_count", tower_layer_count) != tower_layer_count:
            raise ValueError(
                f"text_config.tower_layer_count={text_kwargs['tower_layer_count']!r} "
                f"disagrees with the decoder's {tower_layer_count} tower layers"
            )
        else:
            text_kwargs["tower_layer_count"] = tower_layer_count
        self.text_config = checked_text_config(text_kwargs)
        self.fish_schema_version = fish_schema_version
        self.speech_template_version = speech_template_version
        self.audio_encoder_config = None
        super().__init__(**kwargs)
        if not self.architectures:
            self.architectures = ["FishOmniModel"]
        else:
            pass

        text_vocab_size = self.text_config.vocab_size
        self.semantic_start_token_id = checked_int(
            semantic_start_token_id, "semantic_start_token_id", 0
        )
        self.semantic_end_token_id = checked_int(
            semantic_end_token_id, "semantic_end_token_id", self.semantic_start_token_id
        )
        eos_token_id = checked_int(self.eos_token_id, "eos_token_id", 0)
        if self.semantic_end_token_id >= text_vocab_size:
            raise ValueError(
                f"semantic_end_token_id={self.semantic_end_token_id} is outside the "
                f"text vocabulary of {text_vocab_size}"
            )
        elif (
            self.semantic_end_token_id - self.semantic_start_token_id + 1
            != self.audio_decoder_config.vocab_size
        ):
            raise ValueError(
                f"the semantic band [{self.semantic_start_token_id}, "
                f"{self.semantic_end_token_id}] does not match the codebook size "
                f"{self.audio_decoder_config.vocab_size}"
            )
        elif eos_token_id >= text_vocab_size:
            raise ValueError(
                f"eos_token_id={eos_token_id} is outside the text vocabulary of "
                f"{text_vocab_size}"
            )
        elif self.semantic_start_token_id <= eos_token_id <= self.semantic_end_token_id:
            raise ValueError(f"eos_token_id={eos_token_id} is inside the semantic band")
        elif self.audio_decoder_config.dim != self.text_config.hidden_size:
            raise ValueError(
                f"audio_decoder_config.dim={self.audio_decoder_config.dim} differs "
                f"from text hidden_size={self.text_config.hidden_size}"
            )
        else:
            pass
