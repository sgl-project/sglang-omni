# SPDX-License-Identifier: Apache-2.0
"""Hardware-backed parity checks for the Fun-ASR Apple implementation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.fun_asr.mlx.config import ModelConfig
from sglang_omni.models.fun_asr.mlx.model import FunASRModel
from sglang_omni.models.fun_asr.mlx.runner import FunASRMlxModelRunner
from sglang_omni.models.fun_asr.sglang_model import (
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
)


@pytest.fixture(params=["cpu", "gpu"])
def mlx_device(request):
    previous = mx.default_device()
    device = mx.cpu if request.param == "cpu" else mx.gpu
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    mx.set_default_device(device)
    mx.random.seed(42)
    # Metal's fast SDPA has different accumulation precision from CPU SDPA.
    yield 3e-5 if device == mx.cpu else 5e-3
    mx.set_default_device(previous)


def tiny_config():
    return ModelConfig.from_dict(
        {
            "audio_config": {
                "num_mel_bins": 4,
                "num_stacked_frames": 2,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "num_hidden_layers": 3,
                "num_timestamp_prediction_layers": 1,
                "fsmn_kernel_size": 3,
            },
            "adaptor_config": {
                "hidden_size": 8,
                "intermediate_size": 2,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "projector_hidden_size": 16,
            },
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 256,
                "tie_word_embeddings": True,
                "rope_parameters": {"rope_theta": 1000000.0},
            },
        }
    )


@pytest.mark.parametrize("length", [1, 7, 8, 9, 17])
def test_mlx_audio_matches_torch_with_padding(length, mlx_device):
    torch.manual_seed(42)
    encoder = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
    ).eval()
    adaptor = FunAsrNanoAdaptor(
        encoder_dim=8, llm_dim=8, ffn_dim=16, num_layers=1, attention_heads=2
    ).eval()
    model = FunASRModel(tiny_config())
    weights = {
        f"model.audio_tower.{k}": mx.array(v.numpy())
        for k, v in encoder.state_dict().items()
    }
    weights.update(
        {
            f"model.multi_modal_projector.{k}": mx.array(v.numpy())
            for k, v in adaptor.state_dict().items()
        }
    )
    # Load each audio submodule strictly, so a Torch-side rename fails here as a
    # missing weight instead of leaving MLX layers randomly initialized.
    sanitized = model.sanitize(weights)
    for name in ("audio_tower", "multi_modal_projector"):
        getattr(model, name).load_weights(
            [
                (key.removeprefix(f"{name}."), value)
                for key, value in sanitized.items()
                if key.startswith(f"{name}.")
            ],
            strict=True,
        )
    features = torch.randn(1, 8, length + 5)
    mask = torch.arange(length + 5)[None] < length
    with torch.inference_mode():
        expected = adaptor(encoder(features[:, :, :length].transpose(1, 2)))[
            0, : (length + 7) // 8
        ]
    actual = model.get_audio_features(
        mx.array(features.numpy()), mx.array(mask.numpy())
    )
    np.testing.assert_allclose(
        np.array(actual), expected.numpy(), atol=mlx_device, rtol=mlx_device
    )


def test_mlx_prefill_replaces_hashed_audio_and_decode_uses_cache(mlx_device):
    runner = object.__new__(FunASRMlxModelRunner)
    runner.model = FunASRModel(tiny_config())
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(
            audio_token_id=10,
            mm_items=[
                SimpleNamespace(
                    feature=torch.zeros(1, 8, 16),
                    feature_attention_mask=torch.ones(1, 16),
                    pad_value=999,
                )
            ],
        )
    )
    ids, embeddings = runner.audio_prefill_inputs(req, [1, 999, 999, 2])
    assert ids.tolist() == [[1, 10, 10, 2]]
    cache = runner.model.make_cache()
    first = runner.model.forward_last_logits(embeddings, cache=cache)
    token = mx.argmax(first[:, -1], axis=-1)[:, None]
    decode = runner.model(token, cache=cache)
    full = mx.concatenate([embeddings, runner.model.model.embed_tokens(token)], axis=1)
    expected = runner.model.forward_last_logits(full)
    np.testing.assert_allclose(
        np.array(decode), np.array(expected), atol=mlx_device, rtol=mlx_device
    )
    assert cache[0].offset == 5


def test_mlx_rejects_audio_placeholder_mismatch():
    model = FunASRModel(tiny_config())
    with pytest.raises(ValueError, match="placeholder span"):
        model.build_inputs_embeds(
            mx.array([[1, 10, 2]]), mx.zeros((2, 8)), audio_start=1, num_audio_tokens=1
        )


def test_mlx_weight_mapping_transposes_conv_once():
    model = FunASRModel(tiny_config())
    value = mx.ones((8, 1, 3))
    key = "audio_tower.layers.0.self_attn.fsmn.conv.weight"
    result = model.sanitize({f"model.{key}": value})
    assert result[key].shape == (8, 3, 1)
    # Already-native weights must not be transposed a second time.
    assert model.sanitize(result)[key].shape == (8, 3, 1)


def test_mlx_config_requires_flat_checkpoint_blocks():
    # A pre-flat checkpoint has no audio_config block at all.
    with pytest.raises(KeyError, match="audio_config"):
        ModelConfig.from_dict({"text_config": {}, "encoder_config": {}})
    # Shape-determining fields have no defaults to fall back on.
    with pytest.raises(TypeError, match="hidden_size"):
        ModelConfig.from_dict({"text_config": {}, "audio_config": {}})
