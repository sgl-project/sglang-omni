# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from torch import nn
from transformers import PretrainedConfig

import sglang_omni.models.firered_asr2.sglang_model as implementation


class FakeLanguageModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.received = []
        self.embedding = SimpleNamespace(weight=torch.empty(1, 24))

    def get_input_embeddings(self) -> SimpleNamespace:
        return self.embedding

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.received = list(weights)


@pytest.fixture
def model(
    monkeypatch: pytest.MonkeyPatch,
) -> implementation.FireRedASR2ForConditionalGeneration:
    monkeypatch.setattr(implementation, "Qwen2ForCausalLM", FakeLanguageModel)
    config = PretrainedConfig(
        audio_encoder_conf={"idim": 80, "n_layers_enc": 1, "n_head": 2, "d_model": 16},
        hidden_size=24,
        encoder_downsample_rate=2,
    )
    return implementation.FireRedASR2ForConditionalGeneration(config).eval()


def test_weight_loading_checks_audio_and_delegates_language_weights(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    weights = [(key, value.clone()) for key, value in model.state_dict().items()]
    weights.append(("llm.model.embed_tokens.weight", torch.randn(4, 24)))
    model.load_weights(iter(weights))
    assert model.language_model.received[0][0] == "model.embed_tokens.weight"


def test_missing_audio_weights_fail_instead_of_using_random_parameters(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    with pytest.raises(ValueError, match="Missing FireRed audio weights"):
        model.load_weights(
            iter([("llm.model.embed_tokens.weight", torch.randn(4, 24))])
        )


def test_unknown_weights_fail(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    with pytest.raises(ValueError, match="Unexpected FireRed checkpoint weight"):
        model.load_weights(iter([("wrong.weight", torch.zeros(1))]))


def test_saved_positional_encoding_is_required(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    weights = [
        (key, value)
        for key, value in model.state_dict().items()
        if key != "encoder.positional_encoding.pe"
    ]
    with pytest.raises(ValueError, match="encoder.positional_encoding.pe"):
        model.load_weights(iter(weights))


def test_audio_microbatching_preserves_order_and_lengths(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    torch.manual_seed(3)
    items = [
        SimpleNamespace(
            feature=torch.randn(frames, 80),
            num_audio_tokens=implementation.num_audio_tokens(frames),
        )
        for frames in [21, 72, 39]
    ]
    model.encoder_max_batch_size = 1
    singles = model.get_audio_feature(items)
    model.encoder_max_batch_size = 3
    batch = model.get_audio_feature(items)
    assert batch.shape == (sum(item.num_audio_tokens for item in items), 24)
    torch.testing.assert_close(batch, singles, atol=3e-6, rtol=3e-5)


def test_audio_placeholder_mismatch_fails(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    item = SimpleNamespace(feature=torch.randn(21, 80), num_audio_tokens=99)
    with pytest.raises(ValueError, match="Audio token length mismatch"):
        model.get_audio_feature([item])


def test_bfloat16_decoder_keeps_audio_weights_and_compute_in_float32(
    model: implementation.FireRedASR2ForConditionalGeneration,
) -> None:
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        mixed = implementation.FireRedASR2ForConditionalGeneration(model.config).eval()
    finally:
        torch.set_default_dtype(original)
    assert all(p.dtype == torch.float32 for p in mixed.encoder.parameters())
    assert all(p.dtype == torch.float32 for p in mixed.encoder_projector.parameters())
    assert mixed.encoder.positional_encoding.pe.dtype == torch.float32
    item = SimpleNamespace(feature=torch.randn(21, 80), num_audio_tokens=3)
    assert mixed.get_audio_feature([item]).dtype == torch.bfloat16
