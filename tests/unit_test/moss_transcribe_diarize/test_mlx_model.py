# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.moss_transcribe_diarize.mlx.config import (  # noqa: E402
    ModelConfig,
)
from sglang_omni.models.moss_transcribe_diarize.mlx.model import (  # noqa: E402
    MossTranscribeDiarizeModel,
)
from sglang_omni.models.moss_transcribe_diarize.mlx.runner import (  # noqa: E402
    MossTranscribeDiarizeMlxModelRunner,
)
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (  # noqa: E402
    VQAdaptor,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "audio_config": {
                "num_mel_bins": 4,
                "d_model": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "max_source_positions": 8,
            },
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "tie_word_embeddings": True,
            },
            "audio_token_id": 10,
            "audio_merge_size": 2,
            "adaptor_input_dim": 16,
        }
    )


@pytest.fixture(params=["cpu", "gpu"])
def mlx_device(request):
    previous = mx.default_device()
    device = mx.cpu if request.param == "cpu" else mx.gpu
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    mx.set_default_device(device)
    mx.random.seed(42)
    yield 3e-5 if device == mx.cpu else 5e-3
    mx.set_default_device(previous)


def test_mlx_audio_matches_torch_across_trimmed_windows(mlx_device) -> None:
    torch.manual_seed(42)
    config = _tiny_config()
    torch_config = WhisperConfig(**vars(config.audio_config))
    encoder = WhisperEncoder(torch_config).eval()
    adaptor = VQAdaptor(16, 8).eval()
    model = MossTranscribeDiarizeModel(config)
    weights = {
        f"model.whisper_encoder.{name}": mx.array(value.detach().numpy())
        for name, value in encoder.state_dict().items()
    }
    weights.update(
        {
            f"model.vq_adaptor.{name}": mx.array(value.detach().numpy())
            for name, value in adaptor.state_dict().items()
        }
    )
    model.load_weights(list(model.sanitize(weights).items()), strict=False)

    features = torch.randn(2, 4, 16)
    lengths = torch.tensor([3, 2], dtype=torch.long)
    mapping = torch.tensor([0, 0], dtype=torch.long)
    with torch.inference_mode():
        encoded = encoder(features, return_dict=True).last_hidden_state
        joined = torch.cat([encoded[0:1, :6], encoded[1:2, :4]], dim=1)
        expected = adaptor(joined.reshape(1, 5, 16))[0]

    actual = model.get_audio_features(
        mx.array(features.numpy()),
        mx.array(lengths.numpy()),
        mx.array(mapping.numpy()),
    )[0]
    np.testing.assert_allclose(
        np.array(actual), expected.numpy(), atol=mlx_device, rtol=mlx_device
    )


def test_mlx_prefill_scatters_audio_around_time_marker_tokens(mlx_device) -> None:
    runner = object.__new__(MossTranscribeDiarizeMlxModelRunner)
    runner.model = MossTranscribeDiarizeModel(_tiny_config())
    item = SimpleNamespace(
        feature=torch.zeros((1, 4, 16)),
        model_specific_data={
            "audio_feature_lengths": torch.tensor([4]),
            "audio_chunk_mapping": torch.tensor([0]),
        },
        pad_value=999,
    )
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(audio_token_id=10, mm_items=[item])
    )

    input_ids, embeddings = runner._audio_prefill_inputs(
        req, [1, 999, 999, 7, 999, 999, 2]
    )

    mx.eval(embeddings)
    assert input_ids.tolist() == [[1, 10, 10, 7, 10, 10, 2]]
    assert embeddings.shape == (1, 7, 8)

    cache = runner.model.make_cache()
    first = runner.model._forward_last_logits(embeddings, cache=cache)
    token = mx.argmax(first[:, -1], axis=-1)[:, None]
    decoded = runner.model(token, cache=cache)
    full = mx.concatenate([embeddings, runner.model.model.embed_tokens(token)], axis=1)
    expected = runner.model._forward_last_logits(full)
    np.testing.assert_allclose(
        np.array(decoded), np.array(expected), atol=mlx_device, rtol=mlx_device
    )
    assert cache[0].offset == 8


def test_mlx_weight_mapping_matches_hugging_face_checkpoint() -> None:
    model = MossTranscribeDiarizeModel(_tiny_config())
    result = model.sanitize(
        {
            "model.whisper_encoder.conv1.weight": mx.ones((8, 4, 3)),
            "model.vq_adaptor.layers.0.weight": mx.ones((8, 16)),
            "model.language_model.embed_tokens.weight": mx.ones((32, 8)),
        }
    )

    assert result["whisper_encoder.conv1.weight"].shape == (8, 3, 4)
    assert "vq_adaptor.linear1.weight" in result
    assert "model.embed_tokens.weight" in result


def test_mlx_whisper_casts_features_to_checkpoint_dtype() -> None:
    encoder = MossTranscribeDiarizeModel(_tiny_config()).whisper_encoder
    encoder.set_dtype(mx.bfloat16)

    output = encoder(mx.zeros((1, 4, 16), dtype=mx.float32))
    mx.eval(output)

    assert output.dtype == mx.bfloat16
