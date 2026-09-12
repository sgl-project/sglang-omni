# SPDX-License-Identifier: Apache-2.0
"""Check real loader destinations, format guards, and flat-encoder execution."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.fun_asr.checkpoint import canonical_weight_name
from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoConfig
from sglang_omni.models.fun_asr.sglang_model import (
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
    FunAsrNanoForConditionalGeneration,
)


@pytest.mark.parametrize(
    "source,target",
    [
        ("stem.input", "layers.0.input"),
        ("layers.0.input", "layers.1.input"),
        ("layers.48.input", "layers.49.input"),
        ("timestamp_prediction_layers.0.input", "layers.50.input"),
        ("timestamp_prediction_layers.19.input", "layers.69.input"),
        ("layer_norm.weight", "layers.49.final_layernorm.weight"),
        ("timestamp_prediction_layer_norm.bias", "layers.69.final_layernorm.bias"),
    ],
)
def test_split_boundaries(source, target):
    assert (
        canonical_weight_name(
            "model.audio_tower." + source, layout="split", num_blocks=50, tp_blocks=20
        )
        == "model.audio_tower." + target
    )


@pytest.mark.parametrize(
    "layout,source",
    [
        ("flat", "stem.self_attn.q_proj.weight"),
        ("split", "layers.0.input_layernorm.weight"),
    ],
)
def test_mixed_layout_rejected(layout, source):
    with pytest.raises(ValueError, match="disagrees"):
        canonical_weight_name(
            "model.audio_tower." + source, layout=layout, num_blocks=50, tp_blocks=20
        )


@pytest.mark.parametrize("flat", [False, True])
def test_nested_config_counts_and_roundtrip(flat):
    audio = dict(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=3 if flat else 2,
        num_mel_bins=2,
        num_stacked_frames=3,
    )
    audio[
        "num_timestamp_prediction_layers" if flat else "num_timestamp_prediction_blocks"
    ] = 1
    config = FunAsrNanoConfig(
        audio_config=audio,
        adaptor_config=dict(
            hidden_size=8,
            intermediate_size=5,
            projector_hidden_size=12,
            num_attention_heads=2,
        ),
        text_config=dict(hidden_size=8, num_attention_heads=2),
    )
    assert config.encoder_config.encoder_layers == 2
    assert config.encoder_config.num_timestamp_prediction_blocks == 1
    assert config.encoder_config.input_size == 6
    assert config.adaptor_ffn_dim == 5
    assert config.adaptor_intermediate_size == 12
    restored = FunAsrNanoConfig.from_dict(config.to_dict())
    assert restored.checkpoint_layout == ("flat" if flat else "split")
    assert restored.adaptor_ffn_dim == 5


def tiny_model(layout="flat"):
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        checkpoint_layout=layout,
        encoder_config=SimpleNamespace(
            encoder_layers=2, num_timestamp_prediction_blocks=1
        ),
        text_config=SimpleNamespace(tie_word_embeddings=False),
    )
    model.audio_tower = FunAsrNanoAudioEncoder(
        input_size=6,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
        dropout_rate=0,
        attention_dropout_rate=0,
        activation_dropout_rate=0,
    )
    model.multi_modal_projector = FunAsrNanoAdaptor(
        encoder_dim=8, llm_dim=8, ffn_dim=12, num_layers=1, attention_heads=2
    )
    return model.eval()


def split_key(name):
    """Test export fixture: spell out the three blocks and two boundary norms."""
    name = "model." + name
    if name.startswith("model.audio_tower."):
        for source, target in [
            ("layers.1.final_layernorm.", "layer_norm."),
            ("layers.2.final_layernorm.", "timestamp_prediction_layer_norm."),
            ("layers.0.", "stem."),
            ("layers.1.", "layers.0."),
            ("layers.2.", "timestamp_prediction_layers.0."),
        ]:
            if name.startswith("model.audio_tower." + source):
                name = name.replace(source, target, 1)
                break
    else:
        name = name.replace("multi_modal_projector.layers.", "audio_adaptor.blocks.")
    for source, target in [
        (".input_layernorm.", ".self_attn_layer_norm."),
        (".post_attention_layernorm.", ".final_layer_norm."),
        (".self_attn.o_proj.", ".self_attn.out_proj."),
        (".self_attn.fsmn.", ".feedforward_sequential_memory."),
        (".mlp.fc1.", ".fc1."),
        (".mlp.fc2.", ".fc2."),
    ]:
        name = name.replace(source, target)
    return name


@pytest.mark.parametrize("layout", ["flat", "split"])
def test_full_audio_load_and_output(layout):
    source, target = tiny_model(), tiny_model(layout)
    weights = [
        ("model." + name if layout == "flat" else split_key(name), tensor.clone())
        for name, tensor in source.state_dict().items()
    ]
    target.load_weights(reversed(weights))
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], tensor, rtol=0, atol=0)
    x = torch.randn(2, 7, 6)
    mask = torch.tensor([[[1] * 7], [[1] * 4 + [0] * 3]], dtype=x.dtype)
    with torch.no_grad():
        torch.testing.assert_close(
            target.audio_tower(x, mask), source.audio_tower(x, mask), rtol=0, atol=0
        )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "shape", "unknown_bias"])
def test_loader_rejects_incomplete_or_ambiguous_weights(failure):
    model = tiny_model()
    weights = [
        ("model." + name, tensor.clone()) for name, tensor in model.state_dict().items()
    ]
    if failure == "missing":
        weights.pop()
    elif failure == "duplicate":
        weights.append(weights[0])
    elif failure == "shape":
        weights[0] = (weights[0][0], torch.zeros(1))
    else:
        weights.append(("model.audio_tower.missing.bias", torch.zeros(1)))
    with pytest.raises(ValueError):
        model.load_weights(weights)


def test_flat_model_keeps_all_valid_audio_frames():
    model = tiny_model()
    item = SimpleNamespace(
        feature=torch.randn(1, 6, 17), feature_attention_mask=torch.ones(1, 17)
    )
    assert model.get_audio_feature([item]).shape == (17, 8)
    model.config.checkpoint_layout = "split"
    assert model.get_audio_feature([item]).shape == (3, 8)


@pytest.mark.parametrize("flat", [False, True])
def test_feature_extractor_reads_layout_and_lfr_fields(tmp_path, flat):
    import json

    from sglang_omni.models.fun_asr.configuration_fun_asr import (
        FunAsrNanoFeatureExtractor,
    )

    audio = {
        "num_hidden_layers": 3 if flat else 2,
        (
            "num_timestamp_prediction_layers"
            if flat
            else "num_timestamp_prediction_blocks"
        ): 1,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "fun_asr_nano", "audio_config": audio})
    )
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor": {
                    "feature_extractor_type": "FunAsrNanoFeatureExtractor",
                    "feature_size": 40,
                    "num_frames_lfr": 3,
                    "stride_lfr": 2,
                }
            }
        )
    )
    extractor = FunAsrNanoFeatureExtractor.from_pretrained(
        tmp_path, local_files_only=True
    )
    assert extractor.lfr_m == 3
    assert extractor.lfr_n == 2
    assert extractor.legacy_audio_lengths is not flat


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_encoder_and_projector_match_native_hf(dtype):
    # Optional reference check: serving remains on the repo's pinned Transformers.
    reference = pytest.importorskip(
        "transformers.models.fun_asr_nano.modeling_fun_asr_nano"
    )
    configs = pytest.importorskip(
        "transformers.models.fun_asr_nano.configuration_fun_asr_nano"
    )
    torch.manual_seed(9)
    model = tiny_model().to(dtype=dtype)
    audio_config = configs.FunAsrNanoEncoderConfig(
        num_mel_bins=2,
        num_stacked_frames=3,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=3,
        num_timestamp_prediction_layers=1,
        fsmn_kernel_size=3,
    )
    audio_config._attn_implementation = "eager"
    encoder = reference.FunAsrNanoEncoder(audio_config).eval().to(dtype=dtype)
    encoder.load_state_dict(model.audio_tower.state_dict(), strict=True)
    adaptor_config = configs.FunAsrNanoAdaptorConfig(
        hidden_size=8,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        projector_hidden_size=12,
    )
    adaptor_config._attn_implementation = "eager"
    projector = (
        reference.FunAsrNanoMultiModalProjector(
            SimpleNamespace(audio_config=audio_config, adaptor_config=adaptor_config)
        )
        .eval()
        .to(dtype=dtype)
    )
    projector.load_state_dict(model.multi_modal_projector.state_dict(), strict=True)
    x = torch.randn(2, 7, 6, dtype=dtype)
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
    with torch.no_grad():
        ours = model.audio_tower(x, mask[:, None, :])
        theirs = encoder(x, mask).last_hidden_state
        valid = mask.bool()
        tolerance = 2e-5 if dtype == torch.float32 else 0.04
        torch.testing.assert_close(
            ours[valid], theirs[valid], atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            model.multi_modal_projector(ours, mask[:, None, :])[valid],
            projector(theirs, mask)[valid],
            atol=tolerance,
            rtol=tolerance,
        )
