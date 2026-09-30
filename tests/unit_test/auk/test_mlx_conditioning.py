"""CPU parity for native MLX text and reference-audio conditioning."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

mx = pytest.importorskip("mlx.core")

from transformers import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniVisionEncoderConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)

from sglang_omni.models.auk.mlx import conditioning
from sglang_omni.models.auk.mlx.conditioning import (
    AuKMlxConditionEncoder,
    AuKMlxConditionModel,
    AuKMlxTextEncoder,
)
from sglang_omni.models.auk.mlx.conditioning_audio import AuKMlxAudioEncoder


@pytest.fixture(autouse=True)
def cpu_stream():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture
def text_config():
    return Qwen2_5OmniTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000,
            "mrope_section": [1, 1, 2],
        },
        attn_implementation="eager",
    )


@pytest.fixture
def audio_config():
    return Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=24,
        d_model=16,
        output_dim=32,
        n_window=4,
        max_source_positions=8,
        attn_implementation="eager",
    )


@pytest.fixture
def thinker_config(text_config, audio_config):
    return Qwen2_5OmniThinkerConfig(
        text_config=text_config,
        audio_config=audio_config,
        vision_config=Qwen2_5OmniVisionEncoderConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=24,
            num_heads=2,
            out_hidden_size=32,
        ),
        audio_token_index=60,
        image_token_index=61,
        video_token_index=62,
        audio_start_token_id=58,
        audio_end_token_id=59,
        vision_start_token_id=57,
        attn_implementation="eager",
    )


def copy_weights(reference, native):
    weights = []
    for name, tensor in reference.state_dict().items():
        if "audio_bos_eos_token" in name or name.startswith(("visual.", "lm_head.")):
            continue
        value = mx.array(tensor.detach().numpy())
        if name.endswith(("conv1.weight", "conv2.weight")):
            value = value.transpose(0, 2, 1)
        weights.append((name, value))
    native.load_weights(weights, strict=True)
    native.eval()


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("sliding_window", [None, 3])
def test_all_text_layers_match_transformers(text_config, padding_side, sliding_window):
    if sliding_window is not None:
        text_config.sliding_window = sliding_window
        text_config.layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
    torch.manual_seed(3)
    reference = Qwen2_5OmniThinkerTextModel(text_config).eval()
    native = AuKMlxTextEncoder(text_config)
    copy_weights(reference, native)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [0, 0, 8, 9, 10, 11]])
    if padding_side == "right":
        ids[1] = ids[1].roll(-2)
    mask = ids != 0
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(~mask, 1)
    with torch.inference_mode():
        output = reference(
            input_ids=ids,
            attention_mask=mask.long(),
            position_ids=positions[None].expand(3, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
    actual = np.array(
        native(native.embed_tokens(mx.array(ids.numpy())), mx.array(mask.numpy()))
    )
    expected = torch.stack(output.hidden_states, dim=1).numpy()
    assert actual.shape == (2, text_config.num_hidden_layers + 1, 6, 32)
    for index, valid in enumerate(mask.numpy()):
        np.testing.assert_allclose(
            actual[index][:, valid], expected[index][:, valid], atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("lengths", [[3], [7], [8], [9], [16], [17], [11, 17, 6]])
def test_audio_windows_and_pooling_match_transformers(audio_config, lengths):
    torch.manual_seed(5)
    reference = Qwen2_5OmniAudioEncoder(audio_config).eval()
    native = AuKMlxAudioEncoder(audio_config)
    copy_weights(reference, native)
    features = torch.randn(audio_config.num_mel_bins, sum(lengths))
    with torch.inference_mode():
        expected = reference(
            features,
            feature_lens=torch.tensor(lengths),
            aftercnn_lens=(torch.tensor(lengths) + 1) // 2,
        ).last_hidden_state.numpy()
    actual = np.array(native(mx.array(features.numpy()), lengths))
    assert actual.shape[0] == sum((length + 1) // 4 for length in lengths)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_mixed_reference_audio_matches_transformers(thinker_config):
    torch.manual_seed(7)
    reference = Qwen2_5OmniThinkerForConditionalGeneration(thinker_config).eval()
    native = AuKMlxConditionModel(thinker_config)
    copy_weights(reference, native)
    ids = torch.tensor(
        [
            [2, 58, 60, 60, 60, 59, 8, 9, 0, 0],
            [2, 58, 60, 60, 60, 60, 59, 9, 10, 11],
            [0, 0, 0, 0, 0, 3, 4, 5, 6, 7],
        ]
    )
    mask = ids != 0
    features = torch.randn(2, thinker_config.audio_config.num_mel_bins, 20)
    feature_mask = torch.tensor([[1] * 11 + [0] * 9, [1] * 17 + [0] * 3])
    with torch.inference_mode():
        output = reference(
            input_ids=ids,
            attention_mask=mask.long(),
            input_features=features,
            feature_attention_mask=feature_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    actual = np.array(
        native(ids.numpy(), mask.numpy(), features.numpy(), feature_mask.numpy())
    )
    expected = torch.stack(output.hidden_states, dim=1).numpy()
    for index, valid in enumerate(mask.numpy()):
        np.testing.assert_allclose(
            actual[index][:, valid], expected[index][:, valid], atol=3e-5, rtol=3e-5
        )


def test_bfloat16_preserves_float32_text_residuals(thinker_config):
    torch.manual_seed(41)
    text_config = thinker_config.text_config
    text_config.num_hidden_layers = 12
    text_config.layer_types = ["full_attention"] * text_config.num_hidden_layers
    reference = (
        Qwen2_5OmniThinkerForConditionalGeneration(thinker_config)
        .eval()
        .to(torch.bfloat16)
        .float()
    )
    reference.set_attn_implementation("sdpa")
    native = AuKMlxConditionModel(thinker_config)
    copy_weights(reference, native)
    native.set_dtype(mx.bfloat16)
    ids = torch.tensor([[2, 58, 60, 60, 60, 59, 8, 9], [0, 0, 0, 0, 3, 4, 5, 6]])
    mask = ids != 0
    features = torch.randn(1, thinker_config.audio_config.num_mel_bins, 14)
    feature_mask = torch.tensor([[1] * 11 + [0] * 3])
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = reference(
            input_ids=ids,
            attention_mask=mask.long(),
            input_features=features,
            feature_attention_mask=feature_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    expected = torch.stack(output.hidden_states, dim=1)
    hidden = native(ids.numpy(), mask.numpy(), features.numpy(), feature_mask.numpy())
    assert hidden.dtype == mx.float32
    assert expected.dtype == torch.float32
    actual = np.array(hidden)
    for index, valid in enumerate(mask.numpy()):
        target = expected.numpy()[index][:, valid]
        errors = np.linalg.norm(actual[index][:, valid] - target, axis=-1)
        relative_errors = errors / np.maximum(np.linalg.norm(target, axis=-1), 1e-8)
        assert np.max(relative_errors) < 1e-2


def test_batch_encoder_returns_only_valid_tokens(thinker_config):
    model = AuKMlxConditionModel(thinker_config)
    ids = np.array([[2, 3, 4, 5], [0, 0, 6, 7]])
    inputs = {"input_ids": ids, "attention_mask": ids != 0}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return ["first", "second"]

        def __call__(self, *, text, audio, padding, return_tensors):
            assert text == ["first", "second"]
            assert audio is None
            assert padding and return_tensors == "np"
            return inputs

    encoder = object.__new__(AuKMlxConditionEncoder)
    encoder.model = model
    encoder.processor = Processor()
    results = encoder.encode_batch(
        [[{"role": "user"}], [{"role": "user"}]], [None, None]
    )
    together = np.array(model(**inputs))
    for index, (hidden, mask) in enumerate(results):
        valid = inputs["attention_mask"][index]
        np.testing.assert_array_equal(np.array(hidden), together[index][:, valid])
        np.testing.assert_array_equal(np.array(mask), np.ones(valid.sum(), dtype=bool))


@pytest.mark.parametrize("messages,audios", [([], []), ([[{}]], [])])
def test_batch_encoder_rejects_misaligned_requests(messages, audios):
    encoder = object.__new__(AuKMlxConditionEncoder)
    with pytest.raises(ValueError, match="equally sized nonempty"):
        encoder.encode_batch(messages, audios)


def test_conditioning_rejects_missing_or_misaligned_audio(thinker_config):
    native = AuKMlxConditionModel(thinker_config)
    ids = np.array([[2, 58, 60, 60, 60, 59, 8]])
    mask = np.ones_like(ids)
    features = np.ones((1, thinker_config.audio_config.num_mel_bins, 8), np.float32)
    with pytest.raises(ValueError, match="require reference audio"):
        native(ids, mask)
    with pytest.raises(ValueError, match="require their attention mask"):
        native(ids, mask, features)
    with pytest.raises(ValueError, match="placeholder token counts"):
        native(ids, mask, features, np.ones((1, 8)))
    with pytest.raises(ValueError, match="text and audio only"):
        native(np.array([[thinker_config.image_token_id]]), np.ones((1, 1)))


@pytest.mark.parametrize("lengths,frames", [([2], 2), ([4], 5), ([], 0)])
def test_audio_encoder_rejects_invalid_lengths(audio_config, lengths, frames):
    native = AuKMlxAudioEncoder(audio_config)
    with pytest.raises(ValueError):
        native(mx.zeros((audio_config.num_mel_bins, frames)), lengths)


def test_conditioner_rejects_unqualified_dtype():
    with pytest.raises(ValueError, match="float32 and bfloat16"):
        AuKMlxConditionEncoder("unused", dtype=mx.float16)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_original_checkpoint_loading(
    thinker_config, tmp_path, monkeypatch, sharded, dtype
):
    torch.manual_seed(19)
    reference = Qwen2_5OmniThinkerForConditionalGeneration(thinker_config).eval()
    checkpoint = {
        f"thinker.{name}": value.contiguous()
        for name, value in reference.state_dict().items()
    }
    checkpoint["talker.unused.weight"] = torch.ones((3, 3))
    if sharded:
        filenames = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        weight_map = {
            name: filenames[index % 2] for index, name in enumerate(checkpoint)
        }
        for filename in filenames:
            save_file(
                {
                    name: value
                    for name, value in checkpoint.items()
                    if weight_map[name] == filename
                },
                tmp_path / filename,
            )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    else:
        save_file(checkpoint, tmp_path / "model.safetensors")
    monkeypatch.setattr(
        conditioning.Qwen2_5OmniConfig,
        "from_pretrained",
        lambda path: SimpleNamespace(thinker_config=thinker_config),
    )
    monkeypatch.setattr(
        conditioning.Qwen2_5OmniProcessor, "from_pretrained", lambda path: object()
    )
    encoder = AuKMlxConditionEncoder(str(tmp_path), dtype=dtype)
    expected = AuKMlxConditionModel(thinker_config)
    copy_weights(reference, expected)
    expected.set_dtype(dtype)
    ids = np.array([[2, 58, 60, 60, 60, 59, 8]])
    mask = np.ones_like(ids)
    features = np.ones((1, thinker_config.audio_config.num_mel_bins, 11), np.float32)
    feature_mask = np.ones((1, 11))
    actual = encoder.model(ids, mask, features, feature_mask)
    target = expected(ids, mask, features, feature_mask)
    np.testing.assert_array_equal(np.array(actual), np.array(target))
