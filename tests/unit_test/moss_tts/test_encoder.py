# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.moss_tts.audio_tokenizer import (
    MossAudioEncoder,
    MossAudioTokenizerEncoder,
    MossAudioTokenizerProjectedTransformer,
    MossAudioTokenizerVocoder,
    MossAudioTokenizerVocoderDecoder,
    MossAudioVocoder,
    _normalize_moss_audio_tokenizer_v1_transformer_state_dict,
    _PatchedPretransform,
    _ResidualLFQ,
    load_moss_audio_encoder,
    load_moss_audio_vocoder,
    resolve_moss_audio_attention_backend,
    resolve_moss_audio_dtype,
    resolve_moss_audio_sample_rate,
)


def _tiny_config() -> dict:
    return {
        "architectures": ["RemoteCodeMustNotBeImported"],
        "auto_map": {"AutoModel": "missing_remote_module.Model"},
        "model_type": "moss-audio-tokenizer",
        "sample_rate": 8,
        "sampling_rate": 8,
        "downsample_rate": 2,
        "number_channels": 1,
        "enable_channel_interleave": False,
        "compute_dtype": "fp32",
        "causal_transformer_context_duration": 1.0,
        "encoder_kwargs": [
            {"module_type": "PatchedPretransform", "patch_size": 2},
            {
                "module_type": "Transformer",
                "input_dimension": 2,
                "output_dimension": 4,
                "d_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 8,
                "causal": True,
                "norm": "layer_norm",
                "positional_embedding": "rope",
                "max_period": 10_000,
                "gating": "none",
                "layer_scale": 0.01,
                "context_duration": 1.0,
            },
        ],
        "decoder_kwargs": [
            {
                "module_type": "Transformer",
                "input_dimension": 4,
                "output_dimension": 4,
                "d_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 8,
                "causal": True,
                "norm": "layer_norm",
                "positional_embedding": "rope",
                "max_period": 10_000,
                "gating": "none",
                "layer_scale": 0.01,
                "context_duration": 1.0,
            },
            {"module_type": "PatchedPretransform", "patch_size": 2},
        ],
        "quantizer_type": "rlfq",
        "quantizer_kwargs": {
            "input_dim": 4,
            "rvq_dim": 4,
            "output_dim": 4,
            "num_quantizers": 2,
            "codebook_size": 4,
            "codebook_dim": 2,
            "quantizer_type": "rlfq",
        },
    }


def _tiny_moss_audio_tokenizer_v1_config() -> dict:
    config = _tiny_config()
    config.pop("number_channels")
    config.pop("enable_channel_interleave")
    config.pop("compute_dtype")
    return config


def test_repository_encoder_cpu_fallback_preserves_batch_lengths() -> None:
    model = MossAudioTokenizerEncoder(
        _tiny_config(),
        parameter_device="cpu",
    ).eval()

    output = model.batch_encode(
        [torch.randn(1, 8), torch.randn(1, 6)],
        num_quantizers=2,
    )

    assert output.audio_codes.shape == (2, 2, 4)
    assert output.audio_codes_lengths.tolist() == [4, 3]
    assert output.encoder_hidden_states.shape == (2, 4, 4)
    assert not model.supports_packed_attention()
    resolution = model.resolve_attention_backend("cpu")
    assert resolution.backend == "sdpa"
    assert resolution.fallback_reason is not None


def test_repository_encoder_defaults_missing_compute_dtype_to_bfloat16() -> None:
    config = _tiny_config()
    config.pop("compute_dtype")

    model = MossAudioTokenizerEncoder(config, parameter_device="cpu")

    assert model.encoder_dtype is torch.bfloat16
    assert model.compute_dtype is torch.bfloat16


def test_repository_encoder_uses_shared_packed_attention_wrapper() -> None:
    model = MossAudioTokenizerEncoder(
        _tiny_config(),
        parameter_device="cpu",
    )
    stage = model.encoder[1]

    assert isinstance(stage, MossAudioTokenizerProjectedTransformer)
    attention = stage.transformer.layers[0].self_attn
    assert attention.attention_backend == "auto"


def test_repository_encoder_uses_configured_attention_implementation() -> None:
    config = _tiny_config()
    config["attention_implementation"] = "sdpa"
    model = MossAudioTokenizerEncoder(
        config,
        parameter_device="cpu",
    )

    attention = model.encoder[1].transformer.layers[0].self_attn
    assert attention.attention_backend == "sdpa"


def test_repository_vocoder_uses_configured_attention_implementation() -> None:
    config = _tiny_config()
    config["attention_implementation"] = "sdpa"
    model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
    )

    attention = model.decoder[0].transformer.layers[0].self_attn
    assert attention.attention_backend == "sdpa"


@pytest.mark.parametrize(
    ("attention_backend", "attention_implementation", "expected"),
    [
        ("auto", None, "auto"),
        ("auto", "flash_attention_2", "auto"),
        ("auto", "sdpa", "sdpa"),
        ("packed_flash_attention", "sdpa", "packed_flash_attention"),
    ],
)
def test_repository_attention_backend_selection(
    attention_backend: str,
    attention_implementation: str | None,
    expected: str,
) -> None:
    assert (
        resolve_moss_audio_attention_backend(
            attention_backend,
            attention_implementation,
        )
        == expected
    )


def test_repository_attention_backend_selection_rejects_unknown_implementation() -> (
    None
):
    with pytest.raises(ValueError, match="attention_implementation"):
        resolve_moss_audio_attention_backend("auto", "eager")


def test_repository_encoder_loads_local_weights_without_remote_code(tmp_path) -> None:
    config = _tiny_config()
    expected_model = MossAudioTokenizerEncoder(
        config,
        parameter_device="cpu",
    ).eval()
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )

    loaded = load_moss_audio_encoder(
        str(tmp_path),
        device="cpu",
    ).model

    expected = expected_model.state_dict()
    actual = loaded.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_repository_encoder_strict_flash_fails_before_loading_weights(
    tmp_path,
) -> None:
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(_tiny_config(), config_file)

    with pytest.raises(
        RuntimeError,
        match="attention_backend='packed_flash_attention'.*is unavailable",
    ):
        load_moss_audio_encoder(
            str(tmp_path),
            device="cpu",
            compute_dtype=torch.bfloat16,
            attention_backend="packed_flash_attention",
        )


def test_repository_encoder_materializes_compute_dtype_at_load(tmp_path) -> None:
    config = _tiny_config()
    config["compute_dtype"] = "bfloat16"
    config["encoder_kwargs"][1]["norm"] = "rms_norm_f32"
    expected_model = MossAudioTokenizerEncoder(
        config,
        parameter_device="cpu",
        # note (Zhang Yiyang): Simulate the FP32 checkpoint representation. The
        # loader below must materialize the encoder in the requested BF16
        # compute dtype.
        compute_dtype=torch.float32,
    ).eval()
    assert {parameter.dtype for parameter in expected_model.encoder.parameters()} == {
        torch.float32
    }
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )

    loaded = load_moss_audio_encoder(
        str(tmp_path),
        device="cpu",
        compute_dtype=torch.bfloat16,
    ).model

    assert loaded.encoder_dtype is torch.bfloat16
    assert loaded.encoder[1].input_proj.weight.dtype is torch.bfloat16
    assert loaded.encoder[1].transformer.layers[0].norm1.alpha.dtype is torch.float32
    assert {parameter.dtype for parameter in loaded.encoder.parameters()} == {
        torch.bfloat16,
        torch.float32,
    }
    assert {parameter.dtype for parameter in loaded.quantizer.parameters()} == {
        torch.float32
    }


def test_repository_encoder_materialized_bfloat16_does_not_use_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    config["compute_dtype"] = "bfloat16"
    model = MossAudioTokenizerEncoder(
        config,
        parameter_device="cpu",
        compute_dtype=torch.bfloat16,
        attention_backend="sdpa",
    ).eval()

    original_autocast = torch.autocast

    def reject_enabled_autocast(*args, **kwargs):
        if kwargs.get("enabled", True):
            raise AssertionError("materialized BF16 inference must not use autocast")
        return original_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", reject_enabled_autocast)
    output = model.batch_encode(
        [torch.randn(1, 8), torch.randn(1, 6)],
        num_quantizers=2,
    )

    assert output.audio_codes_lengths.tolist() == [4, 3]


def test_repository_encoder_skips_missing_decoder_shard(tmp_path) -> None:
    config = _tiny_config()
    expected_model = MossAudioTokenizerEncoder(
        config,
        parameter_device="cpu",
        compute_dtype=torch.float32,
    ).eval()
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    selected_weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in expected_model.state_dict().items()
    }
    selected_shard = "model-00001-of-00002.safetensors"
    missing_decoder_shard = "model-00002-of-00002.safetensors"
    save_file(selected_weights, tmp_path / selected_shard)
    weight_map = {name: selected_shard for name in selected_weights}
    weight_map["decoder.0.weight"] = missing_decoder_shard
    with (tmp_path / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as index_file:
        json.dump({"metadata": {}, "weight_map": weight_map}, index_file)

    loaded = load_moss_audio_encoder(
        str(tmp_path),
        device="cpu",
        compute_dtype=torch.float32,
    ).model

    assert isinstance(loaded.encoder, torch.nn.ModuleList)
    assert not (tmp_path / missing_decoder_shard).exists()


def test_repository_vocoder_loads_only_local_quantizer_and_decoder(
    tmp_path,
) -> None:
    config = _tiny_config()
    expected_model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    ).eval()
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    save_file(
        {
            f"quantizer.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.quantizer.state_dict().items()
        }
        | {
            f"decoder.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.decoder.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )

    loaded = load_moss_audio_vocoder(
        str(tmp_path),
        device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    ).model

    assert isinstance(loaded.decoder, MossAudioTokenizerVocoderDecoder)
    expected_quantizer = expected_model.quantizer.state_dict()
    actual_quantizer = loaded.quantizer.state_dict()
    assert actual_quantizer.keys() == expected_quantizer.keys()
    for name in expected_quantizer:
        torch.testing.assert_close(actual_quantizer[name], expected_quantizer[name])
    expected_decoder = expected_model.decoder.state_dict()
    actual_decoder = loaded.decoder.state_dict()
    assert actual_decoder.keys() == expected_decoder.keys()
    for name in expected_decoder:
        torch.testing.assert_close(actual_decoder[name], expected_decoder[name])


def test_repository_vocoder_materializes_compute_dtype_at_load(tmp_path) -> None:
    config = _tiny_config()
    config["decoder_kwargs"][0]["norm"] = "rms_norm_f32"
    expected_model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        # note (Zhang Yiyang): Simulate the FP32 checkpoint representation. The
        # loader below must materialize the decoder in the requested BF16
        # compute dtype.
        compute_dtype=torch.float32,
    ).eval()
    assert {parameter.dtype for parameter in expected_model.decoder.parameters()} == {
        torch.float32
    }
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    save_file(
        {
            f"quantizer.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.quantizer.state_dict().items()
        }
        | {
            f"decoder.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.decoder.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )

    loaded = load_moss_audio_vocoder(
        str(tmp_path),
        device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    ).model

    assert loaded.decoder_dtype is torch.bfloat16
    assert {parameter.dtype for parameter in loaded.decoder.parameters()} == {
        torch.bfloat16,
        torch.float32,
    }
    assert loaded.decoder[0].transformer.layers[0].norm1.alpha.dtype is torch.float32
    assert {parameter.dtype for parameter in loaded.quantizer.parameters()} == {
        torch.float32
    }


def test_repository_vocoder_skips_missing_encoder_shard(tmp_path) -> None:
    config = _tiny_config()
    expected_model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    ).eval()
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    selected_weights = {
        f"quantizer.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in expected_model.quantizer.state_dict().items()
    } | {
        f"decoder.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in expected_model.decoder.state_dict().items()
    }
    selected_shard = "model-00001-of-00002.safetensors"
    missing_encoder_shard = "model-00002-of-00002.safetensors"
    save_file(selected_weights, tmp_path / selected_shard)
    weight_map = {name: selected_shard for name in selected_weights}
    weight_map["encoder.0.weight"] = missing_encoder_shard
    with (tmp_path / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as index_file:
        json.dump({"metadata": {}, "weight_map": weight_map}, index_file)

    loaded = load_moss_audio_vocoder(
        str(tmp_path),
        device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    ).model

    assert isinstance(loaded.decoder, MossAudioTokenizerVocoderDecoder)
    assert not (tmp_path / missing_encoder_shard).exists()


def test_repository_vocoder_constructs_decoder_frame_rate_and_upsampling() -> None:
    model = MossAudioTokenizerVocoder(
        _tiny_config(),
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    transformer = model.decoder[0]
    patch = model.decoder[1]

    assert isinstance(transformer, MossAudioTokenizerProjectedTransformer)
    assert transformer.transformer.layers[0].self_attn.context == 4
    assert isinstance(patch, _PatchedPretransform)
    assert not patch.is_downsample
    values = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    output, lengths = patch(values, torch.tensor([2]))
    assert output.shape == (1, 2, 4)
    assert lengths.tolist() == [4]


def test_repository_vocoder_materialized_bfloat16_does_not_use_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    config["decoder_kwargs"] = [{"module_type": "PatchedPretransform", "patch_size": 2}]
    model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    ).eval()
    vocoder = MossAudioVocoder(model, device="cpu")

    original_autocast = torch.autocast

    def reject_enabled_autocast(*args, **kwargs):
        if kwargs.get("enabled", True):
            raise AssertionError("materialized BF16 inference must not use autocast")
        return original_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", reject_enabled_autocast)
    [audio] = vocoder.decode_codes(torch.zeros((2, 2), dtype=torch.long))

    assert model.decoder_dtype is torch.bfloat16
    assert model.compute_dtype is torch.bfloat16
    assert audio.dtype is torch.float32


def test_repository_quantizer_decode_matches_codebook_sum() -> None:
    torch.manual_seed(0)
    quantizer = _ResidualLFQ(_tiny_config()["quantizer_kwargs"], device="cpu")
    codes = torch.randint(0, 4, (2, 3, 5))

    expected = quantizer.output_proj(
        quantizer.quantizers[0].decode_code(codes[0])
        + quantizer.quantizers[1].decode_code(codes[1])
    )

    torch.testing.assert_close(quantizer.decode_codes(codes), expected)


def test_repository_moss_audio_tokenizer_v1_vocoder_normalizes_checkpoint_fields(
    tmp_path,
) -> None:
    config = _tiny_moss_audio_tokenizer_v1_config()
    expected_model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    ).eval()
    with (tmp_path / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file)
    decoder_state = {
        name.replace(".ffn.linear1.", ".linear1.")
        .replace(".ffn.linear2.", ".linear2.")
        .replace(".self_attn.in_proj.", ".self_attn.in_projs.0.")
        .replace(".self_attn.out_proj.", ".self_attn.out_projs.0."): tensor
        for name, tensor in expected_model.decoder.state_dict().items()
    }
    save_file(
        {
            f"quantizer.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in expected_model.quantizer.state_dict().items()
        }
        | {
            f"decoder.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in decoder_state.items()
        },
        tmp_path / "model.safetensors",
    )

    loaded = load_moss_audio_vocoder(
        str(tmp_path),
        device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
    ).model

    expected = expected_model.decoder.state_dict()
    actual = loaded.decoder.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_repository_encoder_normalizes_moss_audio_tokenizer_v1_checkpoint_fields() -> (
    None
):
    model = MossAudioTokenizerEncoder(
        _tiny_moss_audio_tokenizer_v1_config(),
        parameter_device="cpu",
        compute_dtype=torch.bfloat16,
    )
    stage = model.encoder[1]
    state_dict = stage.state_dict()

    assert model._uses_moss_audio_tokenizer_v1_weights
    assert model.compute_dtype is torch.bfloat16
    assert list(state_dict) == [
        "input_proj.weight",
        "transformer.layers.0.norm1.weight",
        "transformer.layers.0.norm1.bias",
        "transformer.layers.0.self_attn.in_proj.weight",
        "transformer.layers.0.self_attn.out_proj.weight",
        "transformer.layers.0.layer_scale_1.scale",
        "transformer.layers.0.norm2.weight",
        "transformer.layers.0.norm2.bias",
        "transformer.layers.0.ffn.linear1.weight",
        "transformer.layers.0.ffn.linear2.weight",
        "transformer.layers.0.layer_scale_2.scale",
    ]
    assert "transformer.layers.0.ffn.linear1.weight" in state_dict
    assert "transformer.layers.0.self_attn.in_proj.weight" in state_dict

    moss_audio_tokenizer_v1_state_dict = {
        name.replace(".ffn.linear1.", ".linear1.")
        .replace(".ffn.linear2.", ".linear2.")
        .replace(".self_attn.in_proj.", ".self_attn.in_projs.0.")
        .replace(".self_attn.out_proj.", ".self_attn.out_projs.0."): tensor
        for name, tensor in state_dict.items()
    }
    normalized = _normalize_moss_audio_tokenizer_v1_transformer_state_dict(
        moss_audio_tokenizer_v1_state_dict
    )

    assert normalized.keys() == state_dict.keys()


@pytest.mark.parametrize(
    "compute_dtype",
    [torch.float16],
)
def test_repository_encoder_rejects_float16(
    compute_dtype: torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="dtype"):
        MossAudioTokenizerEncoder(
            _tiny_config(),
            parameter_device="cpu",
            compute_dtype=compute_dtype,
        )


@pytest.mark.parametrize(
    ("decoder_dtype", "compute_dtype"),
    [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
def test_repository_vocoder_rejects_float16(
    decoder_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="dtype"):
        MossAudioTokenizerVocoder(
            _tiny_config(),
            parameter_device="cpu",
            decoder_dtype=decoder_dtype,
            compute_dtype=compute_dtype,
        )


@pytest.mark.parametrize(
    ("value", "allow_none", "expected"),
    [
        ("float32", False, torch.float32),
        ("bfloat16", False, torch.bfloat16),
        (torch.float32, False, torch.float32),
        (torch.bfloat16, False, torch.bfloat16),
        (None, True, None),
    ],
)
def test_repository_encoder_resolves_configured_dtype(
    value: str | torch.dtype | None,
    allow_none: bool,
    expected: torch.dtype | None,
) -> None:
    assert (
        resolve_moss_audio_dtype(
            value,
            name="dtype",
            allow_none=allow_none,
        )
        is expected
    )


@pytest.mark.parametrize("value", ["float16", "fp16", torch.float16])
def test_repository_encoder_dtype_resolver_rejects_float16(value) -> None:
    with pytest.raises(ValueError, match="dtype"):
        resolve_moss_audio_dtype(
            value,
            name="dtype",
            allow_none=False,
        )


def test_shared_audio_encoder_uses_model_channel_contract() -> None:
    class FakeModel:
        config = type("Config", (), {"sampling_rate": 48000, "number_channels": 2})()

        def __init__(self) -> None:
            self.prepared = None

        def batch_encode(self, waveforms, *, num_quantizers):
            self.prepared = waveforms
            return type(
                "Output",
                (),
                {
                    "audio_codes": torch.zeros(
                        num_quantizers, len(waveforms), 4, dtype=torch.long
                    ),
                    "audio_codes_lengths": torch.full(
                        (len(waveforms),), 4, dtype=torch.long
                    ),
                },
            )()

    model = FakeModel()
    encoder = MossAudioEncoder(model, device="cpu")
    encoder.encode_wavs([torch.ones(1, 4)], 48000, num_quantizers=2)

    assert model.prepared is not None
    assert model.prepared[0].shape == (2, 4)


@pytest.mark.parametrize(
    ("model_attrs", "config_attrs", "expected"),
    [
        ({"sampling_rate": 48_000}, {"sampling_rate": 24_000}, 48_000),
        ({}, {"sampling_rate": 24_000}, 24_000),
        ({}, {"sample_rate": 16_000}, 16_000),
    ],
)
def test_shared_audio_sample_rate_resolution(
    model_attrs: dict[str, int],
    config_attrs: dict[str, int],
    expected: int,
) -> None:
    model = type("Model", (), model_attrs)()
    config = type("Config", (), config_attrs)()

    assert resolve_moss_audio_sample_rate(model, config) == expected


def test_shared_audio_sample_rate_resolution_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="sampling_rate or sample_rate"):
        resolve_moss_audio_sample_rate(object(), object())
