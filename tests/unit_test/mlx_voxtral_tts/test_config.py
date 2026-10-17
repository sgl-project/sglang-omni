# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sglang_omni_mlx.voxtral_tts.config import AudioTokenizerConfig, read_voxtral_config

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

PARAMS: dict[str, JsonValue] = {
    "dim": 3072,
    "n_layers": 26,
    "head_dim": 128,
    "hidden_dim": 9216,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 131072,
    "rope_theta": 1_000_000.0,
    "norm_eps": 1e-5,
    "multimodal": {
        "audio_model_args": {
            "semantic_codebook_size": 8192,
            "acoustic_codebook_size": 21,
            "n_acoustic_codebook": 36,
            "audio_token_id": 24,
            "acoustic_transformer_args": {
                "input_dim": 3072,
                "dim": 768,
                "n_layers": 3,
                "head_dim": 128,
                "hidden_dim": 2048,
                "n_heads": 6,
                "n_kv_heads": 2,
                "use_biases": False,
            },
        },
        "audio_tokenizer_args": {
            "voice": {"cheerful_female": 4},
            "sampling_rate": 24000,
            "pretransform_patch_size": 240,
            "patch_proj_kernel_size": 7,
            "semantic_codebook_size": 8192,
            "semantic_dim": 256,
            "acoustic_codebook_size": 21,
            "acoustic_dim": 36,
            "causal": True,
            "attn_sliding_window_size": 16,
            "half_attn_window_upon_downsampling": True,
            "dim": 1024,
            "hidden_dim": 4096,
            "head_dim": 128,
            "n_heads": 8,
            "n_kv_heads": 8,
            "qk_norm_eps": 1e-6,
            "qk_norm": True,
            "use_biases": False,
            "norm_eps": 1e-2,
            "layer_scale": True,
            "layer_scale_init": None,
            "decoder_transformer_lengths_str": "2,2,2,2",
            "decoder_convs_kernels_str": "3,4,4,4",
            "decoder_convs_strides_str": "1,2,2,2",
        },
    },
}


def remove_config_value(
    params: dict[str, JsonValue], config_key_path: tuple[str, ...]
) -> None:
    config_section = params
    for config_key in config_key_path[:-1]:
        nested_value = config_section[config_key]
        assert isinstance(nested_value, dict)
        config_section = nested_value
    else:
        pass
    config_section.pop(config_key_path[-1])


def test_read_voxtral_config_parses_required_fields_and_shipped_defaults(
    tmp_path: Path,
) -> None:
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(PARAMS))

    config = read_voxtral_config(tmp_path)
    tokenizer_config = AudioTokenizerConfig.from_voxtral_config(config)

    assert config.text.dim == 3072
    assert config.audio.audio_token_id == 24
    assert config.audio.acoustic_transformer.norm_eps == 1e-5
    assert config.voices == {"cheerful_female": 4}
    assert tokenizer_config.sampling_rate == 24000
    assert tokenizer_config.encoder_convolution_strides == (2, 2, 2, 1)
    assert tokenizer_config.layer_scale_initial is None
    assert tokenizer_config.transformer_lengths == (2, 2, 2, 2)


@pytest.mark.parametrize(
    "config_key_path",
    [
        ("dim",),
        ("multimodal", "audio_model_args", "audio_token_id"),
        (
            "multimodal",
            "audio_model_args",
            "acoustic_transformer_args",
            "input_dim",
        ),
    ],
)
def test_read_voxtral_config_rejects_missing_required_fields(
    tmp_path: Path, config_key_path: tuple[str, ...]
) -> None:
    params = copy.deepcopy(PARAMS)
    remove_config_value(params, config_key_path)
    (tmp_path / "params.json").write_text(json.dumps(params))

    with pytest.raises(KeyError, match=config_key_path[-1]):
        read_voxtral_config(tmp_path)


@pytest.mark.parametrize(
    "config_key_path",
    [
        ("multimodal", "audio_tokenizer_args", "sampling_rate"),
        ("multimodal", "audio_tokenizer_args", "decoder_convs_strides_str"),
    ],
)
def test_audio_tokenizer_config_rejects_missing_required_fields(
    tmp_path: Path, config_key_path: tuple[str, ...]
) -> None:
    params = copy.deepcopy(PARAMS)
    remove_config_value(params, config_key_path)
    (tmp_path / "params.json").write_text(json.dumps(params))
    config = read_voxtral_config(tmp_path)

    with pytest.raises(KeyError, match=config_key_path[-1]):
        AudioTokenizerConfig.from_voxtral_config(config)


def test_audio_tokenizer_config_rejects_mismatched_decoder_stage_counts(
    tmp_path: Path,
) -> None:
    (tmp_path / "params.json").write_text(json.dumps(PARAMS))
    config = read_voxtral_config(tmp_path)
    config.audio_tokenizer["decoder_convs_strides_str"] = "1,2"

    with pytest.raises(ValueError, match="configuration lengths differ"):
        AudioTokenizerConfig.from_voxtral_config(config)
