# SPDX-License-Identifier: Apache-2.0
"""Drama 3 HF config: backbone config conversion, tower layer ids, rejections."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Protocol

import pytest
from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from sglang.srt.utils.hf_transformers_utils import get_rope_config

from sglang_omni.models.fishaudio_drama3.hf_config import (
    Drama3TextConfig,
    FishOmniConfig,
)

FISH_ONLY_TEXT_KEYS = (
    "layer_types",
    "mlp_layer_types",
    "rotary_interleaved",
    "use_qk_norm",
    "layernorm_zero_centered_gamma",
    "shared_expert_gate",
)
# Qwen3_5MoeTextConfig names the full-attention block type "attention".
BACKBONE_BLOCK_TYPES = {
    "linear_attention": "linear_attention",
    "full_attention": "attention",
}

ConfigDict = dict[str, object]


def fish_text_config_dict() -> ConfigDict:
    return {
        "model_type": "fish_omni_text",
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 96,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "mlp_layer_types": ["moe"] * 4,
        "rope_parameters": {
            "rope_theta": 1000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 8,
        "mamba_ssm_dtype": "float32",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "shared_expert_gate": True,
        "intermediate_size": 16,
        "rotary_interleaved": False,
        "use_qk_norm": True,
        "attn_output_gate": True,
        "layernorm_zero_centered_gamma": True,
    }


def audio_decoder_config_dict() -> ConfigDict:
    # head_dim * n_head != dim, grouped kv heads and a partial rotary width on
    # purpose; the two fields the tower must not trust are set to False.
    return {
        "model_type": "fish_qwen3_audio_decoder",
        "decoder_type": "codebook_mtp_shared",
        "dim": 32,
        "text_dim": 32,
        "n_head": 4,
        "n_local_heads": 2,
        "head_dim": 12,
        "n_layer": 2,
        "shared_enrichment_layers": 2,
        "intermediate_size": 24,
        "norm_eps": 1e-6,
        "num_codebooks": 4,
        "vocab_size": 8,
        "max_seq_len": 5,
        "rope_base": 1000.0,
        "rotary_percent": 0.5,
        "rotary_interleaved": False,
        "attention_qk_norm": False,
        "attention_output_gate": True,
        "layernorm_zero_centered_gamma": True,
        "vq_feature_scope": "all_vq",
        "vq_feature_placement": "backbone_partial",
        "vq_backbone_codebooks": 1,
        "mtp_spec": {"attention_variant": "full_attention"},
        "use_moe": False,
    }


def fish_config_dict() -> ConfigDict:
    """A small fish_omni config dict: text vocab 96, semantic band [80, 87], eos 3."""
    return {
        "model_type": "fish_omni",
        "architectures": ["FishOmniModel"],
        "text_config": fish_text_config_dict(),
        "audio_decoder_config": audio_decoder_config_dict(),
        "audio_encoder_config": None,
        "semantic_start_token_id": 80,
        "semantic_end_token_id": 87,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "speech_template_version": "reference-bracket-v1",
        "fish_schema_version": 1,
        "quantization_config": {
            "quant_method": "bsgemm-moe",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
        },
    }


def test_fish_text_config_becomes_a_qwen3_5_moe_text_config() -> None:
    config = FishOmniConfig(**fish_config_dict())
    text_config = config.text_config

    assert isinstance(text_config, Drama3TextConfig)
    assert isinstance(text_config, Qwen3_5MoeTextConfig)
    assert text_config.model_type == "qwen3_5_moe_text"
    assert text_config.layers_block_type == [
        BACKBONE_BLOCK_TYPES[layer_type]
        for layer_type in fish_text_config_dict()["layer_types"]
    ]
    # Four backbone layers, then the decoder's 2 shared + 3 residual tower layers.
    assert text_config.tower_layer_count == 5
    assert text_config.full_attention_layer_ids == [1, 3, 4, 5, 6, 7, 8]
    assert text_config.linear_layer_ids == [0, 2]
    assert get_rope_config(text_config)[0] == 1000.0
    assert text_config.rope_theta == 1000.0
    assert text_config.partial_rotary_factor == 0.5
    assert text_config.attn_output_gate is True
    assert text_config.num_experts == 4
    assert text_config.shared_expert_intermediate_size == 16
    assert not set(FISH_ONLY_TEXT_KEYS) & set(text_config.to_dict())
    assert config.architectures == ["FishOmniModel"]
    assert config.quantization_config["quant_method"] == "bsgemm-moe"
    assert config.audio_decoder_config.rotary_dim == 6
    assert config.audio_decoder_config.rotary_interleaved is False
    assert config.audio_decoder_config.attention_qk_norm is False


def test_saved_config_reloads_to_the_same_config(tmp_path: Path) -> None:
    """The converted text config is the saved form, and it reloads unchanged."""
    config = FishOmniConfig(**fish_config_dict())

    config.save_pretrained(tmp_path)
    reloaded = FishOmniConfig.from_pretrained(tmp_path)
    rebuilt = FishOmniConfig.from_dict(config.to_dict())

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["text_config"]["model_type"] == "qwen3_5_moe_text"
    assert saved["text_config"]["tower_layer_count"] == 5
    assert "layer_types" not in saved["text_config"]
    assert reloaded.to_dict() == config.to_dict()
    assert rebuilt.to_dict() == config.to_dict()
    assert (
        reloaded.text_config.layers_block_type == config.text_config.layers_block_type
    )


class ConfigChange(Protocol):
    def __call__(self, config: ConfigDict) -> None: ...


def changed_text(key: str, value: object) -> ConfigChange:
    def change(config: ConfigDict) -> None:
        text_config = config["text_config"]
        assert isinstance(text_config, dict)
        text_config[key] = value

    return change


def removed_text(key: str) -> ConfigChange:
    def change(config: ConfigDict) -> None:
        text_config = config["text_config"]
        assert isinstance(text_config, dict)
        text_config.pop(key)

    return change


def changed_saved_text(key: str, value: object) -> ConfigChange:
    """Swap in the saved (converted) text config, with key set to value."""

    def change(config: ConfigDict) -> None:
        text_config = FishOmniConfig(**fish_config_dict()).text_config.to_dict()
        text_config[key] = value
        config["text_config"] = text_config

    return change


def removed_saved_text(key: str) -> ConfigChange:
    def change(config: ConfigDict) -> None:
        text_config = FishOmniConfig(**fish_config_dict()).text_config.to_dict()
        text_config.pop(key)
        config["text_config"] = text_config

    return change


def changed_decoder(key: str, value: object) -> ConfigChange:
    def change(config: ConfigDict) -> None:
        decoder_config = config["audio_decoder_config"]
        assert isinstance(decoder_config, dict)
        decoder_config[key] = value

    return change


def changed_top(key: str, value: object) -> ConfigChange:
    def change(config: ConfigDict) -> None:
        config[key] = value

    return change


def without_template_version(config: ConfigDict) -> None:
    config.pop("speech_template_version")


def band_past_vocabulary(config: ConfigDict) -> None:
    changed_top("semantic_start_token_id", 90)(config)
    changed_top("semantic_end_token_id", 97)(config)


def decoder_wider_than_backbone(config: ConfigDict) -> None:
    changed_decoder("dim", 48)(config)
    changed_decoder("text_dim", 48)(config)


REJECTED_CONFIGS = [
    pytest.param(
        changed_text("layer_types", ["full_attention"] * 3),
        "layer_types",
        id="layer-count",
    ),
    pytest.param(
        changed_text("layer_types", ["linear_attention", "sliding"] * 2),
        "layer_types",
        id="unknown-layer-type",
    ),
    pytest.param(
        changed_text("layer_types", ["linear_attention"] * 4),
        "layer_types",
        id="no-full-layer",
    ),
    pytest.param(
        changed_text(
            "layer_types",
            [
                "linear_attention",
                "full_attention",
                "full_attention",
                "linear_attention",
            ],
        ),
        "not periodic",
        id="non-periodic",
    ),
    pytest.param(
        changed_text("mlp_layer_types", ["moe", "dense"] * 2), "moe", id="dense-layer"
    ),
    pytest.param(
        changed_text("rotary_interleaved", True),
        "rotary_interleaved",
        id="text-gptj-rope",
    ),
    pytest.param(changed_text("use_qk_norm", False), "use_qk_norm", id="no-qk-norm"),
    pytest.param(
        changed_text("attn_output_gate", False), "attn_output_gate", id="no-gate"
    ),
    pytest.param(
        changed_text("layernorm_zero_centered_gamma", False),
        "layernorm_zero_centered_gamma",
        id="plain-gamma",
    ),
    pytest.param(
        removed_text("layernorm_zero_centered_gamma"),
        "layernorm_zero_centered_gamma",
        id="gamma-flag-missing",
    ),
    pytest.param(
        changed_text("shared_expert_gate", False),
        "shared_expert_gate",
        id="ungated-shared",
    ),
    pytest.param(
        changed_text("rope_parameters", {"rope_theta": 1000.0, "rope_type": "yarn"}),
        "rope_type",
        id="scaled-rope",
    ),
    pytest.param(
        changed_text(
            "rope_parameters", {"rope_theta": 1000.0, "mrope_section": [2, 1, 1]}
        ),
        "mrope_section",
        id="mrope",
    ),
    pytest.param(
        changed_text(
            "rope_parameters",
            {"rope_theta": 1000.0, "type": "linear", "factor": 4.0},
        ),
        "rope_type",
        id="scaled-rope-legacy-key",
    ),
    pytest.param(
        changed_text(
            "rope_parameters", {"rope_theta": 1000.0, "partial_rotary_factor": 1.5}
        ),
        "partial_rotary_factor",
        id="rotary-factor",
    ),
    pytest.param(
        changed_text(
            "rope_parameters", {"rope_theta": math.inf, "partial_rotary_factor": 0.5}
        ),
        "rope_theta",
        id="infinite-rope-theta",
    ),
    pytest.param(
        changed_text("partial_rotary_factor", 0.25),
        "disagrees with rope_parameters",
        id="second-rotary-factor",
    ),
    pytest.param(
        changed_text("rope_scaling", {"rope_type": "linear", "factor": 4.0}),
        "disagrees with rope_parameters",
        id="second-rope-dict",
    ),
    pytest.param(changed_text("head_dim", 6), "rotary width", id="odd-rotary-width"),
    pytest.param(
        changed_text("shared_expert_intermediate_size", -1),
        "shared_expert_intermediate_size",
        id="negative-shared-expert",
    ),
    pytest.param(
        changed_text("rms_norm_eps", -1), "rms_norm_eps", id="negative-norm-eps"
    ),
    pytest.param(
        changed_text("rms_norm_eps", math.inf), "rms_norm_eps", id="infinite-norm-eps"
    ),
    pytest.param(
        changed_text("num_attention_heads", 0), "num_attention_heads", id="zero-heads"
    ),
    pytest.param(
        changed_text("num_key_value_heads", 3), "num_key_value_heads", id="kv-heads"
    ),
    pytest.param(
        changed_text("num_experts_per_tok", 5),
        "num_experts_per_tok",
        id="top-k-experts",
    ),
    pytest.param(
        changed_text("model_type", "qwen3_next"), "model_type", id="text-model-type"
    ),
    pytest.param(
        changed_saved_text("attn_output_gate", False),
        "attn_output_gate",
        id="saved-form-no-gate",
    ),
    pytest.param(
        changed_saved_text("partial_rotary_factor", 0.0),
        "disagrees with rope_parameters",
        id="saved-form-second-rotary-factor",
    ),
    pytest.param(
        changed_saved_text("rope_theta", 5.0),
        "disagrees with rope_parameters",
        id="saved-form-second-rope-theta",
    ),
    pytest.param(
        changed_decoder("decoder_type", "codebook_mtp"),
        "decoder_type",
        id="decoder-type",
    ),
    pytest.param(
        changed_decoder("vq_feature_placement", "backbone"),
        "vq_feature_placement",
        id="placement",
    ),
    pytest.param(
        changed_decoder("vq_backbone_codebooks", 4),
        "vq_backbone_codebooks",
        id="backbone-codebooks",
    ),
    pytest.param(changed_decoder("max_seq_len", 6), "max_seq_len", id="frame-width"),
    pytest.param(changed_decoder("text_dim", 48), "text_dim", id="text-dim"),
    pytest.param(
        changed_decoder("n_local_heads", 3), "n_local_heads", id="decoder-kv-heads"
    ),
    pytest.param(
        changed_decoder("rotary_percent", 0.25),
        "rotary width",
        id="decoder-odd-rotary-width",
    ),
    pytest.param(
        changed_decoder("rotary_percent", 0.01),
        "rotary width",
        id="decoder-zero-rotary-width",
    ),
    pytest.param(
        changed_decoder("attention_output_gate", False),
        "output gate",
        id="decoder-no-gate",
    ),
    pytest.param(
        changed_decoder("layernorm_zero_centered_gamma", False),
        "zero-centered",
        id="decoder-plain-gamma",
    ),
    pytest.param(changed_decoder("use_moe", True), "MoE", id="decoder-moe"),
    pytest.param(
        changed_decoder(
            "mtp_spec", {"attention_variant": "full_attention", "num_experts": 4}
        ),
        "MoE",
        id="decoder-moe-experts",
    ),
    pytest.param(
        changed_decoder("mtp_spec", {"moe_intermediate_size": 16}),
        "MoE",
        id="decoder-moe-width",
    ),
    pytest.param(
        changed_decoder("mtp_spec", ["full_attention"]),
        "mtp_spec",
        id="decoder-spec-list",
    ),
    pytest.param(changed_decoder("norm_eps", 0.0), "norm_eps", id="decoder-eps"),
    pytest.param(
        changed_decoder("rope_base", math.inf), "rope_base", id="decoder-infinite-base"
    ),
    pytest.param(
        changed_decoder("model_type", "fish_qwen3_audio_encoder"),
        "model_type",
        id="decoder-model-type",
    ),
    pytest.param(decoder_wider_than_backbone, "hidden_size", id="decoder-dim"),
    pytest.param(
        changed_top("semantic_end_token_id", 88), "codebook size", id="band-width"
    ),
    pytest.param(
        changed_top("semantic_start_token_id", -1),
        "semantic_start_token_id",
        id="negative-start",
    ),
    pytest.param(
        changed_top("semantic_end_token_id", 79),
        "semantic_end_token_id",
        id="end-before-start",
    ),
    pytest.param(
        band_past_vocabulary, "outside the text vocabulary", id="band-past-vocab"
    ),
    pytest.param(
        changed_top("eos_token_id", 84), "inside the semantic band", id="eos-in-band"
    ),
    pytest.param(changed_top("eos_token_id", 96), "eos_token_id", id="eos-past-vocab"),
    pytest.param(changed_top("eos_token_id", True), "eos_token_id", id="eos-bool"),
    pytest.param(
        changed_top("speech_template_version", "legacy-s2"),
        "speech_template_version",
        id="template",
    ),
    pytest.param(
        without_template_version, "speech_template_version", id="template-absent"
    ),
    pytest.param(
        changed_top("extra_samplable_token_ids", [5]),
        "extra_samplable_token_ids",
        id="extra-samplable",
    ),
    pytest.param(
        changed_top("audio_encoder_config", {"d_model": 8}),
        "audio_encoder_config",
        id="audio-in",
    ),
    pytest.param(
        changed_top("fish_schema_version", 2),
        "fish_schema_version",
        id="schema-version",
    ),
    pytest.param(
        changed_saved_text("tower_layer_count", 4),
        "tower_layer_count",
        id="saved-tower-layer-count",
    ),
]


@pytest.mark.parametrize(("change", "error"), REJECTED_CONFIGS)
def test_unsupported_configs_are_rejected(change: ConfigChange, error: str) -> None:
    config = fish_config_dict()
    change(config)

    with pytest.raises((ValueError, NotImplementedError), match=error):
        FishOmniConfig(**config)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(removed_text("attn_output_gate"), id="fish-form"),
        pytest.param(removed_saved_text("attn_output_gate"), id="saved-form"),
    ],
)
def test_a_text_config_without_attn_output_gate_loads_with_the_gate(
    change: ConfigChange, tmp_path: Path
) -> None:
    config_dict = fish_config_dict()
    change(config_dict)

    config = FishOmniConfig(**config_dict)
    config.save_pretrained(tmp_path)
    reloaded = FishOmniConfig.from_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert config.text_config.attn_output_gate is True
    assert saved["text_config"]["attn_output_gate"] is True
    assert config.to_dict() == FishOmniConfig(**fish_config_dict()).to_dict()
    assert reloaded.to_dict() == config.to_dict()


def test_saved_form_without_rope_copies_reads_rope_parameters() -> None:
    text_config = FishOmniConfig(**fish_config_dict()).text_config.to_dict()
    text_config.pop("rope_theta")
    text_config.pop("partial_rotary_factor")
    config = fish_config_dict()
    config["text_config"] = text_config

    reloaded = FishOmniConfig(**config).text_config

    assert reloaded.rope_theta == 1000.0
    assert reloaded.partial_rotary_factor == 0.5


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(removed_text("shared_expert_intermediate_size"), id="missing"),
        pytest.param(changed_text("shared_expert_intermediate_size", None), id="null"),
    ],
)
def test_null_shared_expert_size_means_no_shared_expert(change: ConfigChange) -> None:
    config = fish_config_dict()
    change(config)

    assert FishOmniConfig(**config).text_config.shared_expert_intermediate_size == 0


@pytest.fixture(scope="module")
def checkpoint_config_path() -> Path:
    """config.json in FISH_DRAMA3_CHECKPOINT_DIR; skipped when absent."""
    configured_directory = os.environ.get("FISH_DRAMA3_CHECKPOINT_DIR")
    if configured_directory is None:
        pytest.skip("FISH_DRAMA3_CHECKPOINT_DIR is not set")
    else:
        pass
    config_path = Path(configured_directory).expanduser() / "config.json"
    if not config_path.is_file():
        pytest.skip(f"no checkpoint config at {config_path}")
    else:
        pass
    return config_path


def test_checkpoint_config_loads_and_round_trips(
    checkpoint_config_path: Path, tmp_path: Path
) -> None:
    raw_config = json.loads(checkpoint_config_path.read_text())

    config = FishOmniConfig.from_pretrained(checkpoint_config_path.parent)
    config.save_pretrained(tmp_path)
    reloaded = FishOmniConfig.from_pretrained(tmp_path)

    decoder_config = config.audio_decoder_config
    assert config.text_config.layers_block_type == [
        BACKBONE_BLOCK_TYPES[layer_type]
        for layer_type in raw_config["text_config"]["layer_types"]
    ]
    assert (
        config.semantic_end_token_id - config.semantic_start_token_id + 1
        == decoder_config.vocab_size
    )
    assert not (
        config.semantic_start_token_id
        <= config.eos_token_id
        <= config.semantic_end_token_id
    )
    assert decoder_config.dim == config.text_config.hidden_size
    assert reloaded.to_dict() == config.to_dict()
