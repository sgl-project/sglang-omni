# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder as TorchWhisper

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni_mlx.moss_transcribe_diarize.model import (  # noqa: E402
    AudioEncoderConfig,
    MossTranscribeDiarize,
    checkpoint_weights,
    load_moss_transcribe_diarize,
)
from sglang_omni_mlx.text_decoder import TextDecoderConfig  # noqa: E402

AUDIO_CONFIG = {
    "num_mel_bins": 4,
    "d_model": 8,
    "encoder_layers": 1,
    "encoder_attention_heads": 2,
    "encoder_ffn_dim": 16,
    "max_source_positions": 8,
}
TEXT_CONFIG = {
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
}


@pytest.fixture(params=["cpu", "gpu"])
def mlx_device(request: pytest.FixtureRequest) -> Iterator[float]:
    previous = mx.default_device()
    device = mx.cpu if request.param == "cpu" else mx.gpu
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    else:
        pass
    mx.set_default_device(device)
    mx.random.seed(42)
    try:
        yield 3e-4 if device == mx.cpu else 5e-3
    finally:
        mx.set_default_device(previous)


def tiny_model() -> MossTranscribeDiarize:
    return MossTranscribeDiarize(
        AudioEncoderConfig(**AUDIO_CONFIG),
        TextDecoderConfig(**TEXT_CONFIG),
        audio_merge_size=2,
        adaptor_input_size=16,
    )


def test_audio_encoder_matches_torch(mlx_device: float) -> None:
    torch.manual_seed(42)
    torch_encoder = TorchWhisper(WhisperConfig(**AUDIO_CONFIG)).eval()
    torch_adaptor = torch.nn.Sequential(
        torch.nn.Linear(16, 8),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
    ).eval()
    model = tiny_model()
    weights = {
        f"model.whisper_encoder.{name}": mx.array(value.detach().numpy())
        for name, value in torch_encoder.state_dict().items()
    }
    weights.update(
        {
            f"model.vq_adaptor.layers.{name}": mx.array(value.detach().numpy())
            for name, value in torch_adaptor.state_dict().items()
        }
    )
    model.load_weights(list(checkpoint_weights(weights).items()), strict=False)
    features = torch.randn(3, 4, 16)
    token_lengths = np.array([3, 2, 3])
    with torch.inference_mode():
        encoded = torch_encoder(features, return_dict=True).last_hidden_state
        joined = torch.cat(
            [encoded[0:1, :6], encoded[1:2, :4], encoded[2:3, :6]], dim=1
        )
        expected = torch_adaptor(joined.reshape(1, 8, 16))[0]
    actual = model.encode_audio(mx.array(features.numpy()), token_lengths)
    np.testing.assert_allclose(
        np.array(actual), expected.numpy(), atol=mlx_device, rtol=mlx_device
    )


def test_checkpoint_mapping_is_idempotent() -> None:
    weights = {
        "model.whisper_encoder.conv1.weight": mx.arange(96).reshape(8, 4, 3),
        "model.whisper_encoder.conv2.weight": mx.arange(192).reshape(8, 8, 3),
        "model.vq_adaptor.layers.0.weight": mx.ones((8, 16)),
        "model.language_model.embed_tokens.weight": mx.ones((32, 8)),
    }
    mapped = checkpoint_weights(weights)
    remapped = checkpoint_weights(mapped)
    assert mapped["whisper_encoder.conv1.weight"].shape == (8, 3, 4)
    for name, value in mapped.items():
        np.testing.assert_array_equal(np.array(remapped[name]), np.array(value))


def test_loads_tiny_official_weight_layout(tmp_path: Path) -> None:
    model = tiny_model()
    mapped = dict(tree_flatten(model.parameters()))
    official: dict[str, mx.array] = {}
    for name, value in mapped.items():
        checkpoint_name = f"model.{name}"
        checkpoint_name = checkpoint_name.replace(
            "model.model.", "model.language_model."
        )
        checkpoint_name = checkpoint_name.replace(
            "vq_adaptor.linear1.", "vq_adaptor.layers.0."
        )
        checkpoint_name = checkpoint_name.replace(
            "vq_adaptor.linear2.", "vq_adaptor.layers.2."
        )
        checkpoint_name = checkpoint_name.replace(
            "vq_adaptor.layer_norm.", "vq_adaptor.layers.3."
        )
        if checkpoint_name in {
            "model.whisper_encoder.conv1.weight",
            "model.whisper_encoder.conv2.weight",
        }:
            value = value.transpose(0, 2, 1)
        else:
            pass
        official[checkpoint_name] = value
    mx.save_safetensors(str(tmp_path / "model.safetensors"), official)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "audio_config": AUDIO_CONFIG,
                "text_config": {**TEXT_CONFIG, "tie_word_embeddings": True},
                "audio_merge_size": 2,
                "adaptor_input_dim": 16,
            }
        )
    )
    loaded = load_moss_transcribe_diarize(tmp_path)
    expected_parameters = tree_flatten(model.parameters())
    actual_parameters = tree_flatten(loaded.parameters())
    for (name, expected), (_, actual) in zip(expected_parameters, actual_parameters):
        np.testing.assert_array_equal(
            np.array(actual), np.array(expected), err_msg=name
        )
