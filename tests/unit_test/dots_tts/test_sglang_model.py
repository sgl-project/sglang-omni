from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import nn

from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel


def test_weight_loader_routes_every_checkpoint_namespace() -> None:
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: list[tuple[str, torch.Tensor]] = []

        def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
            self.received = list(weights)

    class _Flow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for root in (
                "patch_encoder",
                "hidden_proj",
                "latent_proj",
                "coordinate_proj",
                "xvec_proj",
                "velocity_field_predictor",
                "eos_proj",
            ):
                setattr(self, root, nn.Linear(1, 1, bias=False))

    model = DotsTTSSGLangModel.__new__(DotsTTSSGLangModel)
    nn.Module.__init__(model)
    model.qwen2 = _Backbone()
    model.flow = _Flow()
    flow_weights = [
        (name, torch.full_like(parameter, float(index + 1)))
        for index, (name, parameter) in enumerate(model.flow.named_parameters())
    ]
    codec_weights = [
        (f"{root}.unused", torch.ones(1))
        for root in (
            "audio_encoder",
            "dec_mi_layer",
            "decoder",
            "enc_mi_layer",
            "model",
            "post_proj",
            "pre_proj",
            "resample",
        )
    ]

    loaded = model.load_weights(
        [
            ("llm.model.embed_tokens.weight", torch.tensor([11.0])),
            *flow_weights,
            *codec_weights,
        ]
    )

    assert [name for name, _tensor in model.qwen2.received] == [
        "model.embed_tokens.weight"
    ]
    assert loaded == {
        *(f"flow.{name}" for name, _tensor in flow_weights),
    }
    for index, (_name, parameter) in enumerate(model.flow.named_parameters()):
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, float(index + 1)),
        )

    name, parameter = next(iter(model.flow.named_parameters()))
    partial_weight = torch.full_like(parameter, 99.0)
    assert model.load_weights([(name, partial_weight)]) == {f"flow.{name}"}
    torch.testing.assert_close(parameter, partial_weight)

    with pytest.raises(AssertionError, match="Unexpected dots.tts checkpoint weight"):
        model.load_weights([("new_acoustic_block.weight", torch.ones(1, 1))])


class _TransformerOnlyBackbone(nn.Module):
    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__()
        self.hidden = hidden
        self.model_kwargs: dict | None = None

    def model(self, **kwargs) -> torch.Tensor:
        self.model_kwargs = kwargs
        return self.hidden

    def forward(self, *args, **kwargs):
        raise AssertionError("dots.tts must not enter the lm_head logits path")


def _forward_model(hidden: torch.Tensor) -> DotsTTSSGLangModel:
    model = DotsTTSSGLangModel.__new__(DotsTTSSGLangModel)
    nn.Module.__init__(model)
    model.qwen2 = _TransformerOnlyBackbone(hidden)
    return model


def test_forward_prefill_skips_lm_head_and_keeps_full_hidden_rows() -> None:
    hidden = torch.arange(12.0).reshape(6, 2)
    model = _forward_model(hidden)
    embeds = torch.zeros(6, 2)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        input_embeds=embeds,
        extend_seq_lens=torch.tensor([4, 2]),
    )

    output = model.forward(
        input_ids=torch.zeros(6, dtype=torch.long),
        positions=torch.arange(6),
        forward_batch=batch,
    )

    assert isinstance(output, LogitsProcessorOutput)
    torch.testing.assert_close(output.hidden_states, hidden)
    assert output.next_token_logits.shape == (2, 1)
    assert model.qwen2.model_kwargs is not None
    assert model.qwen2.model_kwargs["input_embeds"] is embeds


def test_forward_decode_skips_lm_head_and_returns_per_request_rows() -> None:
    hidden = torch.arange(6.0).reshape(3, 2)
    model = _forward_model(hidden)
    embeds = torch.zeros(3, 2)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        input_embeds=embeds,
    )

    output = model.forward(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        forward_batch=batch,
        input_embeds=embeds,
    )

    assert isinstance(output, LogitsProcessorOutput)
    torch.testing.assert_close(output.hidden_states, hidden)
    assert output.next_token_logits.shape == (3, 1)
