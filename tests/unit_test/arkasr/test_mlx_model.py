# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.arkasr.mlx.config import (
    AudioEncoderConfig,
    ModelConfig,
    TextConfig,
)
from sglang_omni.models.arkasr.mlx.model import ArkasrModel
from sglang_omni.models.arkasr.mlx.runner import make_arkasr_mlx_runner_class


def _tiny_model(*, tie_word_embeddings: bool = True) -> ArkasrModel:
    mx.random.seed(0)
    return ArkasrModel(
        ModelConfig(
            audio_config=AudioEncoderConfig(
                num_mel_bins=8,
                encoder_layers=2,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                max_source_positions=64,
            ),
            text_config=TextConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                tie_word_embeddings=tie_word_embeddings,
            ),
            audio_token_id=10,
            merge_factor=2,
        )
    )


def test_model_config_from_dict_flat_layout() -> None:
    config = ModelConfig.from_dict(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "audio_token_id": 151663,
            "merge_factor": 4,
            "use_rope": True,
            "whisper_config": {
                "d_model": 1280,
                "encoder_layers": 32,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "num_mel_bins": 128,
            },
        }
    )

    assert config.text_config.hidden_size == 2048
    assert config.text_config.num_hidden_layers == 36
    assert config.text_config.num_key_value_heads == 2
    assert config.audio_config.d_model == 1280
    assert config.audio_config.encoder_layers == 32
    assert config.audio_config.head_dim == 64
    assert config.audio_token_id == 151663
    assert config.merge_factor == 4
    assert config.use_rope is True


def test_native_mlx_audio_prefill_forward() -> None:
    model = _tiny_model()
    audio_features = model.get_audio_features(mx.zeros((1, 8, 20)), None)
    input_ids = mx.array([[1, *([10] * audio_features.shape[0]), 2]], dtype=mx.int32)
    embeddings = model.build_inputs_embeds(
        input_ids,
        audio_features,
        audio_start=1,
        num_audio_tokens=audio_features.shape[0],
    )
    logits = model.forward_last_logits(embeddings, cache=model.make_cache())

    mx.eval(logits)
    assert logits.shape == (1, 1, 64)
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_native_mlx_prefill_only_projects_last_position() -> None:
    model = _tiny_model()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    embeddings = model.embed_tokens(input_ids)

    full_logits = model(input_ids, input_embeddings=embeddings)
    last_logits = model.forward_last_logits(embeddings)

    mx.eval(full_logits, last_logits)
    assert last_logits.shape == (1, 1, 64)
    assert mx.allclose(
        last_logits,
        full_logits[:, -1:, :],
        rtol=1e-2,
        atol=1e-2,
    ).item()


def test_native_mlx_rope_safe_matches_per_sequence_rope() -> None:
    from mlx import nn

    from sglang_omni.models.arkasr.mlx.model import rope_safe

    rope = nn.RoPE(8, traditional=False, base=10_000)
    inputs = mx.random.normal((3, 2, 1, 8))
    batched = rope_safe(rope, inputs, offset=17)

    for row in range(3):
        expected = rope(inputs[row : row + 1], offset=17)
        assert mx.allclose(batched[row : row + 1], expected, atol=1e-5)


def test_runner_chains_native_single_request_decode() -> None:
    runner_class = make_arkasr_mlx_runner_class()
    runner = object.__new__(runner_class)
    runner.model = _tiny_model()
    runner._req_token_ids = {"req": [1]}
    runner._req_caches = {"req": runner.model.make_cache()}
    runner._decode_step_ct = 0
    runner._clear_steps = 0

    first = runner.decode_batch_start(["req"])
    second = runner.decode_batch_start_chained(first)
    mx.eval(second.lazy_tokens)
    runner.decode_batch_finalize(first)
    runner.decode_batch_finalize(second)

    assert first.lazy_tokens.shape == (1,)
    assert second.lazy_tokens.shape == (1,)
    assert runner._req_caches["req"][0].offset == 2
    assert len(runner._req_token_ids["req"]) == 3


def test_native_mlx_sanitize_checkpoint_layout() -> None:
    model = _tiny_model()
    weights = {
        "lm_head.weight": mx.zeros((64, 32)),
        "model.embed_tokens.weight": mx.zeros((64, 32)),
        "audio_encoder.whisper.conv1.weight": mx.zeros((16, 8, 3)),
        "audio_encoder.whisper.conv2.weight": mx.zeros((16, 16, 3)),
    }

    sanitized = model.sanitize(weights)

    assert "lm_head.weight" not in sanitized
    assert sanitized["audio_encoder.whisper.conv1.weight"].shape == (16, 3, 8)
    assert sanitized["audio_encoder.whisper.conv2.weight"].shape == (16, 3, 16)


def test_native_mlx_quantizes_text_only() -> None:
    from mlx import nn
    from mlx_lm.utils import quantize_model

    model, config = quantize_model(
        _tiny_model(),
        {},
        group_size=32,
        bits=4,
    )

    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, nn.QuantizedLinear)
    assert isinstance(model.audio_encoder.whisper.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(model.audio_encoder.adapting[0], nn.Linear)
    assert config["quantization"]["group_size"] == 32
    assert config["quantization"]["bits"] == 4


def test_native_mlx_audio_parity_with_torch() -> None:
    torch = pytest.importorskip("torch")

    from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter as TorchAdapter
    from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig

    config = ArkasrConfig(
        whisper_config={
            "d_model": 16,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "num_mel_bins": 8,
            "max_source_positions": 64,
        },
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=128,
        merge_factor=2,
        audio_token_id=10,
    )
    torch.manual_seed(0)
    torch_adapter = TorchAdapter(config).eval()
    model = _tiny_model()
    weights = {}
    for key, value in torch_adapter.state_dict().items():
        array = value.detach().cpu().float().numpy()
        if key.startswith("whisper.conv") and array.ndim == 3:
            array = array.transpose(0, 2, 1)
        weights[key] = mx.array(array)
    model.audio_encoder.load_weights(list(weights.items()))

    generator = torch.Generator().manual_seed(0)
    mel = torch.randn((2, 8, 20), generator=generator)
    mask = torch.tensor([[1.0] * 20, [1.0] * 15 + [0.0] * 5])
    with torch.no_grad():
        expected = torch_adapter(mel, attention_mask=mask).numpy()
    actual = model.audio_encoder(mx.array(mel.numpy()), mx.array(mask.numpy()))

    mx.eval(actual)
    assert actual.shape == (2, 5, 32)
    assert np.allclose(np.asarray(actual), expected, atol=1e-5)


def test_native_mlx_audio_features_require_single_request() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="one request"):
        model.get_audio_features(mx.zeros((2, 8, 20)), None)
