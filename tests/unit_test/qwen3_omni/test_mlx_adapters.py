"""Small native-model checks for component ownership and request state."""

import json
import os
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig as ModelConfig,
)

from sglang_omni.models.qwen3_omni.mlx.audio import Qwen3OmniMlxAudioEncoder
from sglang_omni.models.qwen3_omni.mlx.code2wav import Qwen3OmniMlxCode2Wav
from sglang_omni.models.qwen3_omni.mlx.encoders import (
    MlxAudioEncoder,
    MlxCode2Wav,
    MlxImageEncoder,
)
from sglang_omni.models.qwen3_omni.mlx.language import LanguageModel
from sglang_omni.models.qwen3_omni.mlx.loading import load_component
from sglang_omni.models.qwen3_omni.mlx.policy import apply_mlx_policy
from sglang_omni.models.qwen3_omni.mlx.runner import OmniMlxRunner
from sglang_omni.models.qwen3_omni.mlx.talker import Talker
from sglang_omni.models.qwen3_omni.mlx.talker_runner import MlxTalkerModelRunner
from sglang_omni.models.qwen3_omni.mlx.tensors import request_positions
from sglang_omni.models.qwen3_omni.mlx.vision import Qwen3OmniMlxVisionEncoder
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    new_stream = mx.new_thread_local_stream
    monkeypatch.setattr(
        mx, "new_thread_local_stream", lambda device: new_stream(mx.cpu)
    )
    yield
    mx.set_default_device(previous)


@pytest.fixture
def checkpoint(tmp_path):
    text = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [16, 8, 8]},
    )
    raw = dict(
        model_type="qwen3_omni_moe",
        thinker_config=dict(
            text_config=text,
            vision_config={"deepstack_visual_indexes": [0]},
            audio_config={},
            image_token_id=120,
            video_token_id=121,
            audio_token_id=122,
        ),
        talker_config=dict(text_config=text, accept_hidden_layer=1),
        code2wav_config={},
        quantization={
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "thinker.language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
        },
    )
    config = ModelConfig.from_dict(raw)
    model = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    nn.quantize(
        model,
        bits=4,
        group_size=64,
        class_predicate=lambda name, module: (
            {"bits": 8, "group_size": 64}
            if name == "model.embed_tokens"
            else hasattr(module, "to_quantized")
        ),
    )
    weights = {
        f"thinker.language_model.{name}": value
        for name, value in tree_flatten(model.parameters())
    }
    weights["code2wav.unowned.weight"] = mx.ones((1,))
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path


@pytest.fixture
def full_checkpoint(checkpoint):
    raw = json.loads((checkpoint / "config.json").read_text())
    text = raw["thinker_config"]["text_config"]
    raw["thinker_config"]["vision_config"] = dict(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=1,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        hidden_act="gelu_pytorch_tanh",
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    raw["thinker_config"]["audio_config"] = dict(
        d_model=64,
        encoder_attention_heads=1,
        encoder_layers=2,
        encoder_ffn_dim=128,
        num_mel_bins=8,
        downsample_hidden_size=8,
        max_source_positions=128,
        output_dim=64,
        activation_function="gelu",
        scale_embedding=False,
        n_window=50,
        n_window_infer=400,
        conv_chunksize=2,
    )
    raw["talker_config"].update(
        text_config={**text, "shared_expert_intermediate_size": 64},
        thinker_hidden_size=64,
        num_code_groups=4,
        code_predictor_config={
            **text,
            "num_code_groups": 4,
            "attention_bias": False,
        },
    )
    raw["code2wav_config"] = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        rms_norm_eps=1e-5,
        rope_theta=10000,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=128,
        sliding_window=4,
        codebook_size=128,
        num_quantizers=4,
        upsampling_ratios=[2],
        decoder_dim=32,
        upsample_rates=[2, 2],
        layer_scale_initial_scale=0.01,
    )
    config = ModelConfig.from_dict(raw)
    models = {
        "thinker.vision_tower": Qwen3OmniMlxVisionEncoder(
            config.thinker_config.vision_config
        ),
        "thinker.audio_tower": Qwen3OmniMlxAudioEncoder(
            config.thinker_config.audio_config
        ),
        "talker": Talker(config.talker_config),
        "code2wav": Qwen3OmniMlxCode2Wav(config.code2wav_config),
    }
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    weights.pop("code2wav.unowned.weight")
    for prefix, model in models.items():
        nn.quantize(
            model,
            bits=4,
            group_size=64,
            class_predicate=lambda name, module: (
                hasattr(module, "to_quantized") and module.weight.shape[-1] % 64 == 0
            ),
        )
        weights.update(
            (f"{prefix}.{name}", value)
            for name, value in tree_flatten(model.parameters())
        )
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    (checkpoint / "config.json").write_text(json.dumps(raw))
    return checkpoint


def test_native_towers_and_predictor_load_and_execute_across_threads(full_checkpoint):
    def load():
        mx.set_default_device(mx.cpu)
        return {
            component: load_component(str(full_checkpoint), component)
            for component in ("vision", "audio", "talker", "code2wav")
        }

    with ThreadPoolExecutor(max_workers=1) as pool:
        models = pool.submit(load).result()
    vision, residuals = models["vision"](
        mx.ones((4, 24)), grid_thw=mx.array([[1, 2, 2]])
    )
    assert vision.shape == (1, 64)
    assert len(residuals) == 1 and residuals[0].shape == (1, 64)
    audio = models["audio"](mx.ones((8, 108)), feature_lens=mx.array([103, 5]))
    assert audio.shape == (15, 64)
    talker = models["talker"]
    logits, hidden = talker(
        inputs_embeds=mx.ones((1, 2, 64)),
        position_ids=mx.broadcast_to(mx.arange(2)[None, None], (3, 1, 2)),
        past_key_values=talker.make_cache(),
    )
    feedback, codes = talker.predict_codes(
        mx.argmax(logits[:, -1], axis=-1)[:, None], hidden[:, -1:], 0.0, 1.0
    )
    assert codes.shape == (1, 4) and feedback.shape == (1, 1, 64)
    waveform = models["code2wav"](codes[:, :, None])
    assert waveform.shape == (1, 1, 8)
    mx.eval(vision, residuals, audio, feedback, codes, waveform)
    assert bool(mx.all(mx.isfinite(waveform)))


def test_native_factories_construct_with_optional_imports_blocked(full_checkpoint):
    script = """
import importlib.abc
import sys
import mlx.core as mx
import torch

class BlockOptionalModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx_vlm", "mlx_audio"}:
            raise ImportError("Optional model library forbidden: " + fullname)
        return None

sys.meta_path.insert(0, BlockOptionalModels())
mx.set_default_device(mx.cpu)
new_stream = mx.new_thread_local_stream
mx.new_thread_local_stream = lambda device: new_stream(mx.cpu)
from sglang_omni.models.qwen3_omni.mlx.encoders import (
    MlxImageEncoder, MlxAudioEncoder, MlxCode2Wav,
)
from sglang_omni.models.qwen3_omni.mlx.runner import OmniMlxRunner, OmniTalkerMlxRunner
from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (
    MlxTalkerPrefillBuilder, MlxTalkerProjectionAdapter,
)
path = sys.argv[1]
image = MlxImageEncoder(path)
audio = MlxAudioEncoder(path)
vocoder = MlxCode2Wav(path)
thinker = OmniMlxRunner(path, disable_radix_cache=True, pool_size=64)
talker = OmniTalkerMlxRunner(path, disable_radix_cache=True, pool_size=64)
projection = MlxTalkerProjectionAdapter(talker.model)
builder = MlxTalkerPrefillBuilder(
    model=projection, model_path=path,
    audio_token_id=122, image_token_id=120, video_token_id=121,
    tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=3,
    im_start_token_id=4, im_end_token_id=5,
    system_token_id=6, user_token_id=7, assistant_token_id=8,
    codec_bos_id=9, codec_nothink_id=10, codec_think_bos_id=11,
    codec_think_eos_id=12, codec_pad_id=13, speaker_map={"ethan":14},
    device=torch.device("cpu"),
)
assert builder.load_prompt_token_embeddings(torch.tensor([1, 2])).shape == (2, 64)
assert builder.get_tts_special_embeds()[0].shape == (1, 64)
assert image(pixel_values=torch.ones(4, 24), image_grid_thw=torch.tensor([[1,2,2]]))["image_embeds"].shape == (1,64)
assert audio(input_features=torch.ones(8,5), audio_feature_lengths=torch.tensor([5]))["audio_embeds"].shape == (1,64)
assert vocoder(torch.zeros(1,4,1,dtype=torch.long)).shape == (1,1,8)
thinker.clear()
talker.clear()
assert not any(name.split(".")[0] in {"mlx_vlm", "mlx_audio"} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(full_checkpoint)],
        env={**os.environ, "SGLANG_USE_MLX": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr


def test_native_models_reuse_mlx_lm_decoder_primitives(full_checkpoint):
    from mlx_lm.models.llama import Attention as LlamaAttention
    from mlx_lm.models.qwen3 import TransformerBlock
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    language = load_component(str(full_checkpoint), "language")
    talker = load_component(str(full_checkpoint), "talker")
    vocoder = load_component(str(full_checkpoint), "code2wav")
    assert isinstance(language.layers[0].mlp, Qwen3MoeSparseMoeBlock)
    assert isinstance(
        language.layers[0].mlp.switch_mlp.gate_proj, QuantizedSwitchLinear
    )
    assert isinstance(talker.code_predictor.model.layers[0], TransformerBlock)
    assert isinstance(talker.model.codec_embedding, nn.QuantizedEmbedding)
    assert isinstance(vocoder.pre_transformer.layers[0].self_attn, LlamaAttention)


def test_three_axis_positions_rotate_each_selected_frequency(checkpoint):
    model = load_component(str(checkpoint), "language")
    attention = model.layers[0].self_attn
    attention.q_proj = nn.Linear(64, 64, bias=False)
    attention.k_proj = nn.Linear(64, 64, bias=False)
    attention.q_proj.weight = mx.eye(64)
    attention.k_proj.weight = mx.eye(64)
    attention.q_norm.weight = mx.ones((64,))
    attention.k_norm.weight = mx.ones((64,))
    cache = model.make_cache()[0]
    attention(
        mx.ones((1, 1, 64)),
        cache=cache,
        position_ids=mx.array([[[2]], [[3]], [[5]]]),
    )
    selected = np.array([2, 3, 5, 2], dtype=np.float32)
    frequencies = 10000.0 ** (-np.arange(0, 8, 2, dtype=np.float32) / 64)
    angles = selected * frequencies
    expected = (np.cos(angles) - np.sin(angles)) / np.sqrt(1 + 1e-6)
    np.testing.assert_allclose(np.array(cache.keys)[0, 0, 0, :4], expected, atol=1e-6)


def test_talker_router_accumulates_low_precision_experts_in_float32(full_checkpoint):
    from mlx.utils import tree_map

    config = ModelConfig.from_dict(
        json.loads((full_checkpoint / "config.json").read_text())
    )
    model = Talker(config.talker_config).layers[0].mlp
    model.update(tree_map(lambda tensor: tensor.astype(mx.float16), model.parameters()))
    output = model(mx.ones((1, 2, 64), dtype=mx.float16))
    # note (Copilot): CPU GatherMM only evaluates float32; inspect the fp16 graph's result dtype.
    assert output.dtype == mx.float32


def test_configured_attention_bias_is_loaded_strictly(full_checkpoint):
    raw = json.loads((full_checkpoint / "config.json").read_text())
    raw.pop("quantization")
    raw["thinker_config"]["text_config"]["attention_bias"] = True
    raw["talker_config"]["text_config"]["attention_bias"] = True
    raw["talker_config"]["code_predictor_config"]["attention_bias"] = True
    config = ModelConfig.from_dict(raw)
    language = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    talker = Talker(config.talker_config)
    weights = {}
    for prefix, model in (("thinker.language_model", language), ("talker", talker)):
        weights.update(
            (f"{prefix}.{name}", value)
            for name, value in tree_flatten(model.parameters())
        )
    weights["thinker.language_model.model.layers.0.self_attn.q_proj.bias"] = mx.full(
        (64,), 2.0
    )
    mx.save_safetensors(str(full_checkpoint / "weights.safetensors"), weights)
    (full_checkpoint / "config.json").write_text(json.dumps(raw))
    language = load_component(str(full_checkpoint), "language")
    talker = load_component(str(full_checkpoint), "talker")
    np.testing.assert_array_equal(
        np.array(language.layers[0].self_attn.q_proj.bias), np.full((64,), 2.0)
    )
    assert talker.code_predictor.model.layers[0].self_attn.o_proj.bias.shape == (64,)


def test_deepstack_is_added_before_accepted_layer_capture(checkpoint):
    config = ModelConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    model = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    for layer in model.layers:
        layer.load_weights(
            [
                (name, mx.zeros_like(value))
                for name, value in tree_flatten(layer.parameters())
            ]
        )
    residuals = mx.zeros((1, 3, 1, 64))
    residuals[:, 1, 0, :] = 7
    result = model(
        mx.array([[1, 2, 3]]),
        inputs_embeds=mx.ones((1, 3, 64)),
        position_ids=mx.broadcast_to(mx.arange(3)[None, None], (3, 1, 3)),
        deepstack_visual_embeds=residuals,
        output_hidden_state_idx=1,
    )
    np.testing.assert_array_equal(np.array(result.hidden_states)[0, :, 0], [1, 8, 1])


def test_variable_audio_lengths_do_not_mix_samples(full_checkpoint):
    encoder = load_component(str(full_checkpoint), "audio")
    first = mx.linspace(-1, 1, 8 * 103).reshape(8, 103)
    second = mx.linspace(2, 3, 8 * 5).reshape(8, 5)
    combined = encoder(
        mx.concatenate([first, second], axis=-1), feature_lens=mx.array([103, 5])
    )
    separate = mx.concatenate(
        [
            encoder(first, feature_lens=mx.array([103])),
            encoder(second, feature_lens=mx.array([5])),
        ]
    )
    np.testing.assert_allclose(np.array(combined), np.array(separate), atol=2e-5)


def test_audio_preserves_float32_features_with_bfloat16_convolutions(full_checkpoint):
    raw = json.loads((full_checkpoint / "config.json").read_text())
    encoder = Qwen3OmniMlxAudioEncoder(
        ModelConfig.from_dict(raw).thinker_config.audio_config
    )
    encoder.set_dtype(mx.bfloat16)
    features = mx.linspace(-1, 1, 8 * 16).reshape(1, 8, 16)
    expected = features[..., None]
    for convolution in (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3):
        expected = nn.gelu(convolution(expected))
    actual = encoder.convolve(features)
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("temporal", [1, 2])
def test_quantized_vision_positions_preserve_fractional_interpolation(
    full_checkpoint, temporal
):
    encoder = load_component(str(full_checkpoint), "vision")
    table = nn.Embedding(16, 64)
    table.weight = mx.ones((16, 64))
    encoder.pos_embed = table.to_quantized(group_size=64, bits=4)
    assert encoder.pos_embed.weight.dtype == mx.uint32
    positions = encoder.interpolate_position_embeddings(mx.array([[temporal, 6, 6]]))
    assert positions.shape == (temporal * 36, 64)
    np.testing.assert_allclose(np.array(positions), 1.0, atol=1e-6)


def individual_expert_weights(weights, prefix, *, keep_stacked=False):
    result = dict(weights)
    for name, value in weights.items():
        if name.startswith(f"{prefix}switch_mlp."):
            if not keep_stacked:
                result.pop(name)
            projection = name.removeprefix(f"{prefix}switch_mlp.")
            for index in range(value.shape[0]):
                result[f"{prefix}experts.{index}.{projection}"] = value[index]
    return result


def test_individual_experts_reject_existing_stacked_weights(checkpoint):
    path = checkpoint / "weights.safetensors"
    weights = individual_expert_weights(
        mx.load(str(path)),
        "thinker.language_model.model.layers.0.mlp.",
        keep_stacked=True,
    )
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="Duplicate expert"):
        load_component(str(checkpoint), "language")


def test_individual_quantized_experts_after_dense_layer_preserve_metadata(checkpoint):
    raw = json.loads((checkpoint / "config.json").read_text())
    raw["thinker_config"]["text_config"]["mlp_only_layers"] = [0]
    config = ModelConfig.from_dict(raw)
    original = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    nn.quantize(
        original,
        bits=4,
        group_size=64,
        class_predicate=lambda name, module: (
            {"bits": 8, "group_size": 64}
            if name in ("model.embed_tokens", "model.layers.1.mlp.switch_mlp.gate_proj")
            else hasattr(module, "to_quantized")
        ),
    )
    weights = {
        f"thinker.language_model.{name}": value
        for name, value in tree_flatten(original.parameters())
    }
    prefix = "thinker.language_model.model.layers.1.mlp."
    weights = individual_expert_weights(weights, prefix)
    for index in range(2):
        raw["quantization"][f"{prefix}experts.{index}.gate_proj"] = {
            "bits": 8,
            "group_size": 64,
        }
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    (checkpoint / "config.json").write_text(json.dumps(raw))
    restored = load_component(str(checkpoint), "language")
    assert restored.layers[1].mlp.switch_mlp.gate_proj.bits == 8
    for (_, expected), (_, actual) in zip(
        tree_flatten(original.parameters()),
        tree_flatten(restored.parameters()),
        strict=True,
    ):
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
    logits = restored(
        mx.array([[1, 2]]),
        position_ids=mx.broadcast_to(mx.arange(2)[None, None], (3, 1, 2)),
    ).logits
    assert bool(mx.all(mx.isfinite(logits)))


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_expert", "Incomplete"),
        ("missing_biases", "Inconsistent"),
        ("mixed_bits", "Incompatible individual expert quantization"),
        ("non_affine", "affine"),
        ("metadata_collision", "Duplicate expert target"),
    ],
)
def test_individual_experts_reject_incomplete_or_conflicting_groups(
    checkpoint, problem, message
):
    path = checkpoint / "weights.safetensors"
    prefix = "thinker.language_model.model.layers.0.mlp."
    weights = individual_expert_weights(mx.load(str(path)), prefix)
    raw = json.loads((checkpoint / "config.json").read_text())
    if problem == "missing_expert":
        for field in ("weight", "scales", "biases"):
            weights.pop(f"{prefix}experts.1.gate_proj.{field}")
    elif problem == "missing_biases":
        weights.pop(f"{prefix}experts.1.gate_proj.biases")
    elif problem == "mixed_bits":
        raw["quantization"][f"{prefix}experts.1.gate_proj"] = {
            "bits": 8,
            "group_size": 64,
        }
    elif problem == "non_affine":
        for index in range(2):
            raw["quantization"][f"{prefix}experts.{index}.gate_proj"] = {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
            }
    else:
        raw["quantization"][f"{prefix}switch_mlp.gate_proj"] = {
            "bits": 4,
            "group_size": 64,
        }
    mx.save_safetensors(str(path), weights)
    (checkpoint / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_component(str(checkpoint), "language")


def test_individual_unquantized_experts_load_strictly(checkpoint):
    raw = json.loads((checkpoint / "config.json").read_text())
    raw.pop("quantization")
    config = ModelConfig.from_dict(raw)
    original = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    weights = {
        f"thinker.language_model.{name}": value
        for name, value in tree_flatten(original.parameters())
    }
    for index in range(2):
        weights = individual_expert_weights(
            weights, f"thinker.language_model.model.layers.{index}.mlp."
        )
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    (checkpoint / "config.json").write_text(json.dumps(raw))
    restored = load_component(str(checkpoint), "language")
    for (_, expected), (_, actual) in zip(
        tree_flatten(original.parameters()),
        tree_flatten(restored.parameters()),
        strict=True,
    ):
        np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_vision_activation_respects_serialized_tanh_gelu(full_checkpoint):
    model = load_component(str(full_checkpoint), "vision")
    mlp = model.blocks[0].mlp
    hidden = mx.linspace(-2, 2, 64).reshape(1, 64)
    projected = mlp.linear_fc1(hidden)
    expected = mlp.linear_fc2(nn.gelu_approx(projected))
    np.testing.assert_array_equal(np.array(mlp(hidden)), np.array(expected))
    assert not np.array_equal(
        np.array(expected), np.array(mlp.linear_fc2(nn.gelu(projected)))
    )


def test_fused_quantized_experts_preserve_per_module_bits(checkpoint):
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    prefix = "thinker.language_model.model.layers.0.mlp."
    for projection in ("gate_proj", "up_proj"):
        quantized = mx.quantize(mx.ones((2, 64, 64)), bits=8, group_size=64)
        for suffix, tensor in zip(("weight", "scales", "biases"), quantized):
            weights[f"{prefix}switch_mlp.{projection}.{suffix}"] = tensor
    for suffix in ("weight", "scales", "biases"):
        gate = weights.pop(f"{prefix}switch_mlp.gate_proj.{suffix}")
        up = weights.pop(f"{prefix}switch_mlp.up_proj.{suffix}")
        down = weights.pop(f"{prefix}switch_mlp.down_proj.{suffix}")
        weights[f"{prefix}experts.gate_up_proj.{suffix}"] = mx.concatenate(
            [gate, up], axis=-2
        )
        weights[f"{prefix}experts.down_proj.{suffix}"] = down
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    raw = json.loads((checkpoint / "config.json").read_text())
    raw["quantization"][f"{prefix}experts.gate_up_proj"] = {"bits": 8, "group_size": 64}
    (checkpoint / "config.json").write_text(json.dumps(raw))
    model = load_component(str(checkpoint), "language")
    assert model.layers[0].mlp.switch_mlp.gate_proj.bits == 8
    assert model.layers[0].mlp.switch_mlp.up_proj.bits == 8
    assert model.layers[0].mlp.switch_mlp.down_proj.bits == 4
    logits = model(
        mx.array([[1, 2]]),
        position_ids=mx.broadcast_to(mx.arange(2)[None, None], (3, 1, 2)),
    ).logits
    assert bool(mx.all(mx.isfinite(logits)))


@pytest.fixture
def tied_checkpoint(checkpoint):
    raw = json.loads((checkpoint / "config.json").read_text())
    raw.pop("quantization")
    raw["thinker_config"]["text_config"]["tie_word_embeddings"] = True
    config = ModelConfig.from_dict(raw)
    model = LanguageModel(config.thinker_config.text_config, config.thinker_config)
    weights = {
        f"thinker.language_model.{name}": value
        for name, value in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    (checkpoint / "config.json").write_text(json.dumps(raw))
    return checkpoint


@pytest.mark.parametrize("official_namespace", [False, True])
def test_tied_embeddings_accept_identical_redundant_head(
    tied_checkpoint, official_namespace
):
    original = load_component(str(tied_checkpoint), "language")
    path = tied_checkpoint / "weights.safetensors"
    weights = mx.load(str(path))
    weights["thinker.language_model.lm_head.weight"] = weights[
        "thinker.language_model.model.embed_tokens.weight"
    ]
    if official_namespace:
        weights = {
            name.replace("thinker.language_model.", "thinker.", 1): value
            for name, value in weights.items()
        }
    else:
        pass
    mx.save_safetensors(str(path), weights)
    restored = load_component(str(tied_checkpoint), "language")
    assert not hasattr(restored, "lm_head")
    tokens = mx.array([[1, 2]])
    positions = mx.broadcast_to(mx.arange(2)[None, None], (3, 1, 2))
    np.testing.assert_array_equal(
        np.array(restored(tokens, position_ids=positions).logits),
        np.array(original(tokens, position_ids=positions).logits),
    )


@pytest.mark.parametrize("conflict", ["values", "shape", "dtype", "missing_embedding"])
def test_tied_embeddings_reject_conflicting_head_alias(tied_checkpoint, conflict):
    path = tied_checkpoint / "weights.safetensors"
    weights = mx.load(str(path))
    name = "thinker.language_model.model.embed_tokens.weight"
    head = weights[name]
    if conflict == "values":
        head = head + 1
    elif conflict == "shape":
        head = head[:-1]
    elif conflict == "dtype":
        head = head.astype(mx.bfloat16)
    else:
        weights.pop(name)
    weights["thinker.language_model.lm_head.weight"] = head
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="Conflicting tied embedding alias"):
        load_component(str(tied_checkpoint), "language")


def test_tied_alias_does_not_discard_unrecognized_head_tensors(tied_checkpoint):
    path = tied_checkpoint / "weights.safetensors"
    weights = mx.load(str(path))
    weights["thinker.language_model.lm_head.weight"] = weights[
        "thinker.language_model.model.embed_tokens.weight"
    ]
    weights["thinker.language_model.lm_head.bias"] = mx.zeros((128,))
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="lm_head.bias"):
        load_component(str(tied_checkpoint), "language")


def test_strict_component_load_rejects_unexpected_weight(checkpoint):
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    weights["thinker.language_model.unexpected.weight"] = mx.ones((1,))
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    with pytest.raises(ValueError, match="unexpected|Unexpected|not in model"):
        load_component(str(checkpoint), "language")


def test_component_index_does_not_open_unowned_shards(checkpoint):
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    weight_map = {
        name: "weights.safetensors"
        for name in weights
        if name.startswith("thinker.language_model.")
    }
    weight_map["code2wav.unowned.weight"] = "absent-unowned.safetensors"
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    model = load_component(str(checkpoint), "language")
    assert model.model.embed_tokens.bits == 8


def test_official_language_namespace_retains_embedding_override(checkpoint):
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    weights = {
        name.replace("thinker.language_model.", "thinker."): value
        for name, value in weights.items()
    }
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    model = load_component(str(checkpoint), "language")
    embedding = load_component(str(checkpoint), "embedding")
    assert model.model.embed_tokens.bits == embedding.bits == 8
    np.testing.assert_array_equal(
        np.array(model.model.embed_tokens(mx.array([1, 2]))),
        np.array(embedding(mx.array([1, 2]))),
    )


def test_talker_completion_retires_native_and_feedback_state(full_checkpoint):
    from sglang_omni.models.qwen3_omni.mlx.runner import OmniTalkerMlxRunner
    from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue

    native = OmniTalkerMlxRunner(
        str(full_checkpoint), disable_radix_cache=True, pool_size=64
    )
    native.position_deltas["speech"] = 4
    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(_mlx_runner=native)
    data = SimpleNamespace(
        req=SimpleNamespace(omni_mlx_prompt_hidden=torch.ones(1, 64)),
        pending_feedback_queue=deque([torch.ones(64)]),
        pending_text_queue=PendingTextTensorQueue(),
        decode_input_embeds=[torch.ones(64)],
    )
    runner.on_request_finished("speech", data)
    assert native.position_deltas == {}
    assert data.req.omni_mlx_prompt_hidden is None
    assert not data.pending_feedback_queue and not data.decode_input_embeds
    assert not data.pending_text_queue


def make_request(model_inputs=None, positions=None, delta=0):
    return SimpleNamespace(
        omni_model_inputs=model_inputs or {},
        multimodal_inputs=(
            None
            if positions is None
            else SimpleNamespace(
                mrope_positions=positions, mrope_position_delta=torch.tensor([delta])
            )
        ),
    )


def prefill(runner, request, tokens, request_id="first"):
    return runner.prefill_start(
        req_id=request_id,
        new_token_ids=tokens,
        full_token_ids=tokens,
        prefix_slot_ids=[],
        new_slot_ids=list(range(len(tokens))),
        req_pool_idx=0,
        req=request,
    )


def test_component_ownership_and_per_layer_quantization(checkpoint):
    model = load_component(str(checkpoint), "language")
    assert model.model.embed_tokens.bits == 8
    assert model.model.layers[0].self_attn.q_proj.bits == 4
    assert isinstance(model, LanguageModel)
    assert not hasattr(model, "code2wav")
    embedding = load_component(str(checkpoint), "embedding")
    assert embedding.bits == 8


def test_strict_component_load_rejects_missing_weight(checkpoint):
    weights = mx.load(str(checkpoint / "weights.safetensors"))
    weights.pop("thinker.language_model.model.norm.weight")
    mx.save_safetensors(str(checkpoint / "weights.safetensors"), weights)
    with pytest.raises(ValueError):
        load_component(str(checkpoint), "language")


def test_explicit_positions_capture_decode_and_cleanup(checkpoint):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    positions = torch.tensor([[0, 4, 5], [0, 2, 3], [0, 6, 7]])
    request = make_request(
        {
            "image_embeds": torch.ones(1, 64),
            "deepstack_visual_embeds": [torch.ones(1, 64)],
            "pad_values": {"image": 1000000},
        },
        positions=positions,
        delta=5,
    )
    pending = prefill(runner, request, [2, 1000000, 3])
    runner.prefill_finalize(pending)
    assert pending.full_token_ids == [2, 120, 3]
    assert request.omni_mlx_prompt_hidden.shape == (3, 64)
    assert request.omni_mlx_capture_layer == 1
    decoded = runner.decode_batch_start(["first"])
    runner.decode_batch_finalize(decoded)
    assert (
        runner._req_caches["first"][0].offset == 4
    )  # noqa: leading-underscore - native cache state
    runner.remove_request("first")
    assert not runner.position_deltas
    assert not runner._req_caches  # noqa: leading-underscore - native cache state
    assert not runner._req_token_ids  # noqa: leading-underscore - native token state
    second = prefill(runner, make_request(), [2, 3], request_id="second")
    runner.prefill_finalize(second)
    runner.clear()
    assert not runner.position_deltas
    assert not runner._req_caches  # noqa: leading-underscore - native cache state


@pytest.mark.parametrize(
    "features,layers",
    [
        (torch.ones(2, 64), [torch.ones(2, 64)]),
        (torch.ones(1, 64), []),
        (torch.ones(1, 64), [torch.ones(2, 64)]),
    ],
)
def test_invalid_media_counts_raise(checkpoint, features, layers):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    with pytest.raises(ValueError):
        prefill(
            runner,
            make_request(
                {
                    "image_embeds": features,
                    "deepstack_visual_embeds": layers,
                }
            ),
            [2, 120, 3],
        )
    assert not runner.position_deltas


def test_positions_require_exact_coverage():
    with pytest.raises(ValueError, match="cover all"):
        request_positions(make_request(positions=torch.zeros(3, 2)), 3)
    positions, delta = request_positions(make_request(), 3)
    np.testing.assert_array_equal(np.array(positions), [[[0, 1, 2]]] * 3)
    assert delta == 0


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_running_requests", 2),
        ("disable_radix_cache", False),
        ("chunked_prefill_size", 256),
        ("enable_mixed_chunk", True),
        ("mlx_enable_sampling", True),
        ("cuda_graph_backend_prefill", "breakable"),
        ("cuda_graph_backend_prefill", "piecewise"),
        ("disable_decode_cuda_graph", False),
        ("tp_size", 2),
    ],
)
def test_unsupported_scheduling_modes_fail(option, value):
    with pytest.raises(ValueError, match="requires"):
        apply_mlx_policy({"tp_size": 1}, {option: value})


def test_policy_uses_bounded_native_worker():
    overrides = {"tp_size": 1}
    apply_mlx_policy(overrides, None)
    assert overrides["max_running_requests"] == 1
    assert overrides["max_total_tokens"] == 8192
    assert overrides["disable_radix_cache"]
    assert overrides["disable_overlap_schedule"]


def test_text_prefill_accepts_absent_multimodal_sidecar(checkpoint):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    request = make_request()
    request.omni_model_inputs = None
    pending = prefill(runner, request, [2, 3])
    assert runner.prefill_finalize(pending) >= 0
    runner.remove_request("first")


def test_mixed_visual_prompt_requires_both_residuals(checkpoint):
    runner = OmniMlxRunner(str(checkpoint), disable_radix_cache=True, pool_size=64)
    request = make_request(
        {
            "image_embeds": torch.ones(1, 64),
            "video_embeds": torch.ones(1, 64),
            "image_deepstack_visual_embeds": [torch.ones(1, 64)],
        }
    )
    with pytest.raises(ValueError, match="video requires"):
        prefill(runner, request, [2, 120, 121])
    request.omni_model_inputs["video_deepstack_visual_embeds"] = [
        torch.full((1, 64), 2.0)
    ]
    pending = prefill(runner, request, [2, 121, 120])
    runner.prefill_finalize(pending)
    assert request.omni_mlx_prompt_hidden.shape == (3, 64)
    runner.remove_request("first")


def test_audio_adapter_rejects_holey_mask():
    encoder = MlxAudioEncoder.__new__(MlxAudioEncoder)
    with pytest.raises(ValueError, match="valid frame prefixes"):
        encoder(
            input_features=torch.ones(1, 128, 3),
            feature_attention_mask=torch.tensor([[1, 0, 1]]),
        )


def test_audio_adapter_reuses_shared_packing(monkeypatch):
    from sglang_omni.models.qwen3_omni.mlx import encoders

    pack = encoders.pack_padded_audio_features
    calls = []

    def recording_pack(features, mask, lengths):
        calls.append(lengths.tolist())
        return pack(features, mask, lengths)

    monkeypatch.setattr(encoders, "pack_padded_audio_features", recording_pack)
    encoder = MlxAudioEncoder.__new__(MlxAudioEncoder)
    features = torch.arange(20).reshape(2, 2, 5).float()
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    expected = pack(features, mask.bool(), mask.sum(-1))
    rows = int(
        encoders._get_feat_extract_output_lengths(mask.sum(-1)).sum()
    )  # noqa: leading-underscore - Transformers length contract

    def tower(packed, feature_lens):
        np.testing.assert_array_equal(np.array(packed), expected.numpy())
        assert feature_lens.tolist() == [5, 3]
        return mx.zeros((rows, 64))

    encoder.audio_tower = tower
    output = encoder(input_features=features, feature_attention_mask=mask)
    assert calls == [[5, 3]]
    assert output["audio_embeds"].shape == (rows, 64)


def test_mlx_special_embeddings_use_shared_cache(monkeypatch):
    from sglang_omni.models.qwen3_omni.components.talker_prefill import (
        TalkerPrefillBuilder,
    )
    from sglang_omni.models.qwen3_omni.mlx.talker_prefill import MlxTalkerPrefillBuilder

    assert (
        MlxTalkerPrefillBuilder.get_tts_special_embeds
        is TalkerPrefillBuilder.get_tts_special_embeds
    )
    builder = MlxTalkerPrefillBuilder.__new__(MlxTalkerPrefillBuilder)
    builder.tts_special_cache = None
    builder.tts_bos_token_id, builder.tts_eos_token_id, builder.tts_pad_token_id = (
        1,
        2,
        3,
    )
    builder.model = SimpleNamespace(text_projection=lambda rows: rows + 1)
    calls = []

    def embeddings(token_ids):
        calls.append(token_ids.tolist())
        return torch.zeros(3, 64)

    monkeypatch.setattr(builder, "load_prompt_token_embeddings", embeddings)
    first = builder.get_tts_special_embeds()
    assert builder.get_tts_special_embeds() is first
    assert calls == [[1, 2, 3]]
    assert all(torch.equal(row, torch.ones(1, 64)) for row in first)


def test_vocoder_rejects_invalid_codes():
    vocoder = MlxCode2Wav.__new__(MlxCode2Wav)
    vocoder.config = SimpleNamespace(num_quantizers=16, codebook_size=2048)
    with pytest.raises(ValueError, match="invalid codec"):
        vocoder(torch.full((1, 16, 2), 2150))


def test_video_encoder_accepts_shared_timing_sidecar():
    encoder = MlxImageEncoder.__new__(MlxImageEncoder)
    encoder.spatial_merge_size = 2
    encoder.visual = lambda pixels, grid_thw: (mx.ones((1, 64)), [mx.ones((1, 64))])
    result = encoder(
        pixel_values_videos=torch.ones(4, 64),
        video_grid_thw=torch.tensor([[1, 2, 2]]),
        video_second_per_grid=torch.tensor([2.0]),
        use_audio_in_video=True,
    )
    assert result["video_token_counts"].tolist() == [1]
    assert result["video_embeds"].shape == (1, 64)
    assert result["deepstack_visual_embeds_video"][0].shape == (1, 64)


def test_talker_preserves_greedy_code_prediction_after_sampling_normalization():
    class FakeTalker:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=64))
        predictor_temperature = None

        def __call__(self, **kwargs):
            return mx.zeros((1, 1, 3072)), mx.zeros((1, 1, 64))

        def predict_codes(self, tokens, hidden, **kwargs):
            self.predictor_temperature = kwargs["temperature"]
            return mx.zeros((1, 1, 64)), mx.zeros((1, 16), dtype=mx.int32)

    model = FakeTalker()
    native = SimpleNamespace(model=model, _req_caches={"speech": []})
    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(_mlx_runner=native)
    runner.outbox = Queue()
    sampling = SamplingParams(temperature=0.0)
    assert sampling.temperature == 1.0
    assert sampling.top_k == 1
    data = SimpleNamespace(
        suppress_tokens=[2150],
        req=SimpleNamespace(sampling_params=sampling),
        output_ids=[],
        pending_feedback_queue=deque(),
        stage_payload=SimpleNamespace(request=SimpleNamespace(params={"stream": True})),
    )
    result = runner.speech_step(
        SimpleNamespace(request_id="speech", data=data),
        mx.zeros((1, 1, 64)),
        mx.zeros((3, 1, 1), dtype=mx.int32),
    )
    assert model.predictor_temperature == 0
    assert result.next_token_ids.shape == (1,)
    emitted = runner.outbox.get_nowait()
    assert emitted.target == "code2wav"
    assert emitted.metadata == {"stream": True}
    assert emitted.data.shape == (16,)
    assert len(data.pending_feedback_queue) == 1


def test_prompt_capture_uses_existing_stream_and_is_consumed_once():
    hidden = torch.ones(3, 64)
    request = SimpleNamespace(
        inflight_middle_chunks=0,
        omni_mlx_prompt_hidden=hidden,
        omni_mlx_capture_layer=24,
    )
    data = SimpleNamespace(
        req=request,
        stage_payload=StagePayload(
            request_id="speech",
            request=OmniRequest(
                inputs=[],
                params={"stream": True},
                metadata={"output_modalities": ["text", "audio"]},
            ),
            data={},
        ),
    )
    builder = make_thinker_stream_output_builder(speech_enabled=True)
    messages = builder("speech", data, SimpleNamespace(data=17))
    assert [message.target for message in messages] == ["decode", "talker_ar"]
    assert messages[0].data.tolist() == [17]
    assert messages[1].data is hidden
    assert messages[1].metadata == {"token_id": 17, "prompt_hidden_layer": 24}
    assert request.omni_mlx_prompt_hidden is None
    following = builder("speech", data, SimpleNamespace(data=18))
    assert following[1].data.tolist() == [18]


def run_sampled_talker_step(logits, sampling, *, suppress=(), history=(), steps=1):
    class FakeTalker:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=64))

        def __call__(self, **kwargs):
            return logits[None, None, :], mx.zeros((1, 1, 64))

        def predict_codes(self, tokens, hidden, **kwargs):
            codes = mx.concatenate([tokens, mx.zeros((1, 15), dtype=mx.int32)], axis=-1)
            return mx.zeros((1, 1, 64)), codes

    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(
        _mlx_runner=SimpleNamespace(model=FakeTalker(), _req_caches={"speech": []})
    )
    runner.outbox = Queue()
    data = SimpleNamespace(
        suppress_tokens=list(suppress),
        req=SimpleNamespace(sampling_params=sampling, output_ids=list(history)),
        output_ids=[],
        pending_feedback_queue=deque(),
        stage_payload=SimpleNamespace(request=SimpleNamespace(params={"stream": True})),
    )
    for _ in range(steps):
        result = runner.speech_step(
            SimpleNamespace(request_id="speech", data=data),
            mx.zeros((1, 1, 64)),
            mx.zeros((3, 1, 1), dtype=mx.int32),
        )
        data.req.output_ids.append(result.next_token_ids.item())
        assert data.output_ids == []
    return result


def test_talker_penalty_reads_live_history_before_terminalization():
    result = run_sampled_talker_step(
        mx.array([3.0, 2.0, 0.0]),
        SamplingParams(temperature=0.0, repetition_penalty=2.0),
        steps=2,
    )
    assert result.next_token_ids.item() == 1


@pytest.mark.parametrize("params", [{"seed": 17}, {"sampling_seed": 29}, {}])
def test_talker_prefill_uses_shared_resolved_seed(monkeypatch, params):
    from contextlib import nullcontext

    from sglang_omni.models.qwen3_omni.mlx import talker_prefill
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.pipeline.stage.stream_queue import StreamItem
    from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue

    fake_builder = SimpleNamespace(
        build_prompt_prefill=lambda *args, **kwargs: {
            "input_embeds": torch.zeros(2, 64),
            "input_ids": [1, 2],
            "pending_text_queue": PendingTextTensorQueue(),
            "tts_pad_embed": torch.zeros(64),
            "tts_eos_embed": torch.zeros(64),
            "prompt_model_inputs": {},
        },
        append_text_chunk=lambda *args: None,
        mark_thinker_done=lambda *args: None,
    )
    monkeypatch.setattr(
        talker_prefill, "MlxTalkerPrefillBuilder", lambda **kwargs: fake_builder
    )
    builder, *_ = make_talker_scheduler_adapters(
        tokenizer=None,
        codec_vocab_size=3072,
        model=SimpleNamespace(config=SimpleNamespace(codec_eos_token_id=2150)),
        model_path="unused",
        thinker_config=None,
    )
    payload = StagePayload(
        request_id="seeded-speech",
        request=OmniRequest(inputs=[], params=params),
        data={},
    )
    payload.prefetched_chunks = [StreamItem(0, torch.tensor([3]), "thinker")]
    payload.prefetched_stream_done = True
    data = builder(payload)
    resolved = data.req.sampling_params.sampling_seed
    assert resolved == params.get("seed", params.get("sampling_seed", resolved))
    assert resolved is not None
    data.req.prefix_indices = []
    data.req.extend_range = SimpleNamespace(length=2)
    native = SimpleNamespace(
        position_deltas={}, _req_caches={}, _acquire_cache=lambda: []
    )
    runner = MlxTalkerModelRunner.__new__(MlxTalkerModelRunner)
    runner.tp_worker = SimpleNamespace(_mlx_runner=native)
    monkeypatch.setattr(runner, "mlx_stream_context", nullcontext)
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.talker_model_runner.QwenTalkerModelRunner.projected_prefill_slice",
        lambda **kwargs: torch.zeros(2, 64),
    )
    monkeypatch.setattr(
        runner,
        "speech_step",
        lambda *args: mx.random.uniform(shape=(4,)).tolist(),
    )
    mx.random.seed(resolved)
    expected = mx.random.uniform(shape=(4,)).tolist()
    mx.random.seed(999)
    actual = runner.custom_prefill_forward(
        None, None, [SimpleNamespace(request_id=payload.request_id, data=data)]
    )
    assert actual == expected


def test_talker_nucleus_is_normalized_and_shift_invariant(monkeypatch):
    from mlx_lm.sample_utils import apply_top_p, make_sampler

    from sglang_omni.models.qwen3_omni.mlx import talker_runner

    received = []

    def recording_sampler(**kwargs):
        actual_sampler = make_sampler(**kwargs)

        def sample(logprobs):
            received.append(np.array(logprobs))
            nucleus = np.array(apply_top_p(logprobs, kwargs["top_p"]))
            assert np.isfinite(nucleus).tolist() == [[True, True, False]]
            return actual_sampler(logprobs)

        return sample

    monkeypatch.setattr(talker_runner, "make_sampler", recording_sampler)
    for shift in (0.0, 100.0, -100.0):
        result = run_sampled_talker_step(
            mx.array([-10.0, -11.0, -12.0]) + shift,
            SamplingParams(temperature=0.8, top_p=0.9, top_k=2),
        )
        assert result.next_token_ids.item() in (0, 1)
    expected = torch.log_softmax(torch.tensor([[-10.0, -11.0, -12.0]]), dim=-1)
    for logprobs in received:
        np.testing.assert_allclose(logprobs, expected.numpy(), atol=1e-5)
        np.testing.assert_allclose(np.exp(logprobs).sum(), 1.0, atol=1e-5)


def test_talker_normalizes_after_suppression_and_repetition_penalty(monkeypatch):
    from mlx_lm.sample_utils import make_sampler

    from sglang_omni.models.qwen3_omni.mlx import talker_runner

    def recording_sampler(**kwargs):
        actual_sampler = make_sampler(**kwargs)

        def sample(logprobs):
            expected = torch.log_softmax(
                torch.tensor([[-float("inf"), -20.0, -11.0]]), dim=-1
            )
            np.testing.assert_allclose(np.array(logprobs), expected.numpy(), atol=1e-5)
            return actual_sampler(logprobs)

        return sample

    monkeypatch.setattr(talker_runner, "make_sampler", recording_sampler)
    result = run_sampled_talker_step(
        mx.array([100.0, -10.0, -11.0]),
        SamplingParams(temperature=0.8, top_p=0.9, top_k=2, repetition_penalty=2.0),
        suppress=(0,),
        history=(1,),
    )
    assert result.next_token_ids.item() == 2


@pytest.mark.parametrize(
    ("top_k", "expected_top_k"), [(-1, 0), (3072, 0), (4096, 0), (8, 8)]
)
def test_talker_translates_normalized_top_k_for_real_sampler(
    monkeypatch, top_k, expected_top_k
):
    from mlx_lm.sample_utils import make_sampler

    from sglang_omni.models.qwen3_omni.mlx import talker_runner

    observed = []

    def recording_sampler(**kwargs):
        observed.append(kwargs)
        return make_sampler(**kwargs)

    monkeypatch.setattr(talker_runner, "make_sampler", recording_sampler)
    sampling = SamplingParams(temperature=0.8, top_p=0.9, top_k=top_k)
    if top_k == -1:
        assert sampling.top_k == 1073741824
    else:
        pass
    logits = mx.full((3072,), -float("inf"))
    logits[:3] = mx.array([-10.0, -11.0, -12.0])
    result = run_sampled_talker_step(logits, sampling)
    assert observed == [{"temp": 0.8, "top_p": 0.9, "top_k": expected_top_k}]
    assert result.next_token_ids.item() in (0, 1)


@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize("component", ["language", "embedding"])
def test_component_resolves_checkpoint_without_network(
    checkpoint, monkeypatch, source, component
):
    from sglang_omni.models import weight_loader

    downloads = []
    hub_id = f"mlx-community/{checkpoint.name}"

    def cached_snapshot(model_path, *, local_files_only):
        assert model_path == hub_id
        downloads.append(model_path)
        return str(checkpoint)

    monkeypatch.setattr(weight_loader, "snapshot_download", cached_snapshot)
    model_path = str(checkpoint) if source == "local" else hub_id
    model = load_component(model_path, component)
    assert model is not None
    assert downloads == ([] if source == "local" else [model_path])


@pytest.mark.parametrize("source", ["local", "hub"])
def test_runner_uses_resolved_checkpoint_for_component_and_config(
    checkpoint, monkeypatch, source
):
    from sglang_omni.models import weight_loader

    downloads = []
    hub_id = f"mlx-community/{checkpoint.name}"

    def cached_snapshot(model_path, *, local_files_only):
        assert model_path == hub_id
        downloads.append(model_path)
        return str(checkpoint)

    monkeypatch.setattr(weight_loader, "snapshot_download", cached_snapshot)
    model_path = str(checkpoint) if source == "local" else hub_id
    runner = OmniMlxRunner(
        model_path, disable_radix_cache=True, pool_size=64, trust_remote_code=False
    )
    assert runner.model_path == str(checkpoint)
    assert runner.trust_remote_code is False
    assert runner.accept_hidden_layer == 1
    assert runner.placeholder_token_ids == {"image": 120, "video": 121, "audio": 122}
    assert downloads == ([] if source == "local" else [model_path])
    pending = prefill(runner, make_request(), [2, 3])
    assert runner.prefill_finalize(pending) >= 0
    runner.remove_request("first")
