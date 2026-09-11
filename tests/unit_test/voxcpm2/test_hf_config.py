# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 checkpoint config: runtime view and the flattened SGLang view."""

from __future__ import annotations

import json

from sglang_omni.models.voxcpm2.hf_config import VoxCPM2Config, load_voxcpm2_config

_LM_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 32768,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "vocab_size": 73448,
    "use_mup": False,
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4,
    "kv_channels": 128,
}

_CONFIG = {
    "architecture": "voxcpm2",
    "lm_config": _LM_CONFIG,
    "patch_size": 4,
    "feat_dim": 64,
    "scalar_quantization_latent_dim": 512,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 8,
    "residual_lm_no_rope": True,
    "encoder_config": {"hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16},
    "dit_config": {
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "mean_mode": False,
        "cfm_config": {"solver": "euler", "inference_cfg_rate": 2.0},
    },
    "audio_vae_config": {
        "latent_dim": 64,
        "sample_rate": 16000,
        "out_sample_rate": 48000,
    },
    "max_length": 8192,
    "dtype": "bfloat16",
}


def _write_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    return str(tmp_path)


def test_runtime_config_reads_the_nested_sections(tmp_path):
    config = load_voxcpm2_config(_write_checkpoint(tmp_path))
    assert config.patch_size == 4
    assert config.feat_dim == 64
    assert config.residual_lm_num_layers == 8
    assert config.residual_lm_no_rope is True
    assert config.sample_rate == 16000
    assert config.out_sample_rate == 48000
    assert config.latent_dim == 64
    assert config.cfm["solver"] == "euler"
    assert config.dit_mean_mode is False


def test_sglang_view_reports_both_stacks_as_one_depth():
    """The KV pool is sized from this number, so it covers both stacks."""
    config = VoxCPM2Config(lm_config=_LM_CONFIG, voxcpm2_config=_CONFIG)
    assert config.num_hidden_layers == 28 + 8
    assert config.lm_config.num_hidden_layers == 28


def test_sglang_view_lifts_the_fields_sglang_reads():
    config = VoxCPM2Config(lm_config=_LM_CONFIG, voxcpm2_config=_CONFIG)
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 2
    assert config.vocab_size == 73448


def test_sub_config_name_stays_off_sglangs_text_config_lookup():
    """Renaming lm_config to one of these hands SGLang the 28-layer depth."""
    config = VoxCPM2Config(lm_config=_LM_CONFIG, voxcpm2_config=_CONFIG)
    for claimed in ("text_config", "llm_config", "language_config", "thinker_config"):
        assert not hasattr(config, claimed)
