# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen3-ASR native Hugging Face flat config and weight loading."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers.common import get_hf_text_config
from sglang.srt.utils.hf_transformers.config import get_config
from sglang.test.parallel_groups import parallel_scope, publish
from transformers import AutoConfig, Qwen3ForCausalLM
from transformers.models.qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRConfig as UpstreamHFQwen3ASRConfig,
)

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.qwen3_asr import torch_mps_runner
from sglang_omni.models.qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRProcessor,
    register_qwen3_asr_hf_config,
)
from sglang_omni.models.qwen3_asr.encoder_service import text_hidden_size
from sglang_omni.models.qwen3_asr.sglang_model import Qwen3ASRForConditionalGeneration


def qwen3_asr_1_7b_hf_config_dict() -> dict[str, object]:
    """Real config.json shape from Qwen/Qwen3-ASR-1.7B-hf."""
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "audio_config": {
            "activation_dropout": 0,
            "activation_function": "gelu",
            "attention_dropout": 0,
            "conv_chunksize": 500,
            "d_model": 1024,
            "downsample_hidden_size": 480,
            "dropout": 0,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layers": 24,
            "initializer_range": 0.02,
            "max_position_embeddings": 13,
            "model_type": "qwen3_asr_encoder",
            "n_window": 50,
            "n_window_infer": 800,
            "num_key_value_heads": 16,
            "num_mel_bins": 128,
            "output_dim": 2048,
            "scale_embedding": False,
        },
        "audio_token_id": 151676,
        "dtype": "bfloat16",
        "eos_token_id": [151643, 151645],
        "initializer_range": 0.02,
        "model_type": "qwen3_asr",
        "pad_token_id": 151645,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": None,
            "eos_token_id": None,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_types": ["full_attention"] * 28,
            "max_position_embeddings": 65536,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "pad_token_id": None,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "sliding_window": None,
            "tie_word_embeddings": True,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        },
        "tie_word_embeddings": True,
        "timestamp_token_id": 151705,
        "token_classification_bias": False,
        "transformers_version": "5.13.0.dev0",
    }


def oruk_fourier_2b_config_dict() -> dict[str, object]:
    """Real config.json shape from oruk/oruk-fourier-2b."""
    config_dict = qwen3_asr_1_7b_hf_config_dict()
    config_dict["auto_map"] = {"AutoConfig": "configuration_qwen3_asr.Qwen3ASRConfig"}
    config_dict["transformers_version"] = "5.21.0.dev0"
    return config_dict


def legacy_thinker_1_7b_config_dict() -> dict[str, object]:
    """Real config.json shape from legacy Qwen/Qwen3-ASR-1.7B."""
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "architectures": ["Qwen3ASRThinkerForConditionalGeneration"],
            "audio_config": {
                "activation_dropout": 0,
                "activation_function": "gelu",
                "attention_dropout": 0,
                "conv_chunksize": 500,
                "d_model": 1024,
                "downsample_hidden_size": 480,
                "dropout": 0,
                "encoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "encoder_layers": 24,
                "initializer_range": 0.02,
                "max_source_positions": 1500,
                "model_type": "qwen3_asr_audio_encoder",
                "n_window": 50,
                "n_window_infer": 800,
                "num_mel_bins": 128,
                "output_dim": 2048,
                "scale_embedding": False,
            },
            "audio_end_token_id": 151670,
            "audio_start_token_id": 151669,
            "audio_token_id": 151676,
            "dtype": "bfloat16",
            "initializer_range": 0.02,
            "model_type": "qwen3_asr_thinker",
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "initializer_range": 0.02,
                "intermediate_size": 6144,
                "max_position_embeddings": 65536,
                "model_type": "qwen3_asr_text",
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "interleaved": True,
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000,
                "tie_word_embeddings": True,
                "use_cache": True,
                "vocab_size": 151936,
            },
        },
        "transformers_version": "4.57.6",
    }


def tiny_audio_config_dict() -> dict[str, object]:
    return {
        "model_type": "qwen3_asr_encoder",
        "d_model": 32,
        "encoder_layers": 1,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 64,
        "downsample_hidden_size": 8,
        "num_mel_bins": 16,
        "output_dim": 64,
        "n_window": 50,
        "n_window_infer": 100,
        "conv_chunksize": 500,
        "max_position_embeddings": 13,
    }


def tiny_text_config_dict(tie_word_embeddings: bool = True) -> dict[str, object]:
    return {
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "tie_word_embeddings": tie_word_embeddings,
    }


def build_checkpoint_tensors(
    *,
    format_kind: str,
    tie_word_embeddings: bool,
    include_lm_head: bool,
    seed: int = 42,
) -> list[tuple[str, torch.Tensor]]:
    """Build deterministic full-coverage checkpoint tensors for legacy or native HF layout."""
    generator = torch.Generator().manual_seed(seed)

    def _rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    audio_prefix = (
        "thinker.audio_tower." if format_kind == "legacy" else "model.audio_tower."
    )
    llm_prefix = (
        "thinker.model." if format_kind == "legacy" else "model.language_model."
    )
    proj1_prefix = (
        "thinker.audio_tower.proj1."
        if format_kind == "legacy"
        else "model.multi_modal_projector.linear_1."
    )
    proj2_prefix = (
        "thinker.audio_tower.proj2."
        if format_kind == "legacy"
        else "model.multi_modal_projector.linear_2."
    )
    lm_head_name = (
        "thinker.lm_head.weight" if format_kind == "legacy" else "lm_head.weight"
    )

    weights = [
        (f"{audio_prefix}conv2d1.weight", _rand(8, 1, 3, 3)),
        (f"{audio_prefix}conv2d1.bias", _rand(8)),
        (f"{audio_prefix}conv2d2.weight", _rand(8, 8, 3, 3)),
        (f"{audio_prefix}conv2d2.bias", _rand(8)),
        (f"{audio_prefix}conv2d3.weight", _rand(8, 8, 3, 3)),
        (f"{audio_prefix}conv2d3.bias", _rand(8)),
        (f"{audio_prefix}conv_out.weight", _rand(32, 16)),
        (f"{audio_prefix}layers.0.self_attn.q_proj.weight", _rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.q_proj.bias", _rand(32)),
        (f"{audio_prefix}layers.0.self_attn.k_proj.weight", _rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.k_proj.bias", _rand(32)),
        (f"{audio_prefix}layers.0.self_attn.v_proj.weight", _rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.v_proj.bias", _rand(32)),
        (f"{audio_prefix}layers.0.self_attn.out_proj.weight", _rand(32, 32)),
        (f"{audio_prefix}layers.0.self_attn.out_proj.bias", _rand(32)),
        (f"{audio_prefix}layers.0.self_attn_layer_norm.weight", _rand(32)),
        (f"{audio_prefix}layers.0.self_attn_layer_norm.bias", _rand(32)),
        (f"{audio_prefix}layers.0.fc1.weight", _rand(64, 32)),
        (f"{audio_prefix}layers.0.fc1.bias", _rand(64)),
        (f"{audio_prefix}layers.0.fc2.weight", _rand(32, 64)),
        (f"{audio_prefix}layers.0.fc2.bias", _rand(32)),
        (f"{audio_prefix}layers.0.final_layer_norm.weight", _rand(32)),
        (f"{audio_prefix}layers.0.final_layer_norm.bias", _rand(32)),
        (f"{audio_prefix}ln_post.weight", _rand(32)),
        (f"{audio_prefix}ln_post.bias", _rand(32)),
        (f"{proj1_prefix}weight", _rand(32, 32)),
        (f"{proj1_prefix}bias", _rand(32)),
        (f"{proj2_prefix}weight", _rand(64, 32)),
        (f"{proj2_prefix}bias", _rand(64)),
        (f"{llm_prefix}embed_tokens.weight", _rand(128, 64)),
        (f"{llm_prefix}layers.0.self_attn.q_norm.weight", _rand(16)),
        (f"{llm_prefix}layers.0.self_attn.k_norm.weight", _rand(16)),
        (f"{llm_prefix}layers.0.self_attn.q_proj.weight", _rand(64, 64)),
        (f"{llm_prefix}layers.0.self_attn.k_proj.weight", _rand(32, 64)),
        (f"{llm_prefix}layers.0.self_attn.v_proj.weight", _rand(32, 64)),
        (f"{llm_prefix}layers.0.self_attn.o_proj.weight", _rand(64, 64)),
        (f"{llm_prefix}layers.0.mlp.gate_proj.weight", _rand(128, 64)),
        (f"{llm_prefix}layers.0.mlp.up_proj.weight", _rand(128, 64)),
        (f"{llm_prefix}layers.0.mlp.down_proj.weight", _rand(64, 128)),
        (f"{llm_prefix}layers.0.input_layernorm.weight", _rand(64)),
        (f"{llm_prefix}layers.0.post_attention_layernorm.weight", _rand(64)),
        (f"{llm_prefix}norm.weight", _rand(64)),
        (f"{llm_prefix}rotary_emb.inv_freq", _rand(8)),
        ("talker.unused.weight", _rand(4, 4)),
        ("code2wav.unused.weight", _rand(4, 4)),
    ]

    if include_lm_head:
        lm_head_weight = _rand(128, 64)
        if tie_word_embeddings:
            lm_head_weight = torch.full((128, 64), -999.0, dtype=torch.float32)
        weights.append((lm_head_name, lm_head_weight))

    return weights


def build_cpu_model(
    config: Qwen3ASRConfig | UpstreamHFQwen3ASRConfig,
    *,
    pp_size: int = 1,
    pp_rank: int = 0,
) -> Qwen3ASRForConditionalGeneration:
    pp_group = SimpleNamespace(
        rank_in_group=pp_rank,
        world_size=pp_size,
        is_first_rank=(pp_rank == 0),
        is_last_rank=(pp_rank == pp_size - 1),
    )
    with parallel_scope(pp_size=pp_size, pp_rank=pp_rank, pp_group=pp_group):
        model = Qwen3ASRForConditionalGeneration(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


@pytest.fixture
def cpu_runtime_context() -> Iterator[None]:
    reset_context()
    publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
    try:
        yield
    finally:
        reset_context()


@pytest.mark.parametrize(
    ("config_factory", "trust_remote_code"),
    [
        (qwen3_asr_1_7b_hf_config_dict, False),
        (oruk_fourier_2b_config_dict, False),
        (oruk_fourier_2b_config_dict, True),
    ],
)
def test_native_hf_config_loading_and_roundtrip(
    tmp_path: Path,
    config_factory: object,
    trust_remote_code: bool,
) -> None:
    """Flat HF Qwen3-ASR configs parse cleanly without falling back to default sizes."""
    register_qwen3_asr_hf_config()
    (tmp_path / "config.json").write_text(json.dumps(config_factory()))

    config = get_config(str(tmp_path), trust_remote_code=trust_remote_code)
    model_config = ModelConfig(
        model_path=str(tmp_path), trust_remote_code=trust_remote_code
    )
    ModelWorker.apply_arch_override(model_config, "Qwen3ASRForConditionalGeneration")

    assert isinstance(config, Qwen3ASRConfig)
    assert config.audio_config.encoder_layers == 24
    assert config.audio_config.d_model == 1024
    assert config.audio_config.output_dim == 2048
    assert config.audio_config.encoder_attention_heads == 16
    assert config.audio_config.max_source_positions == 1500
    assert config.audio_config is config.thinker_config.audio_config

    resolved_text_config = get_hf_text_config(config)
    assert resolved_text_config is config.text_config
    assert resolved_text_config.num_hidden_layers == 28
    assert resolved_text_config.hidden_size == 2048
    assert resolved_text_config.num_attention_heads == 16
    assert resolved_text_config.num_key_value_heads == 8
    assert resolved_text_config.head_dim == 128
    assert resolved_text_config.vocab_size == 151936
    assert resolved_text_config.tie_word_embeddings is True
    assert resolved_text_config.dtype == torch.bfloat16
    assert resolved_text_config.eos_token_id == [151643, 151645]
    assert resolved_text_config.pad_token_id == 151645
    assert config.audio_token_id == 151676

    assert model_config.hidden_size == 2048
    assert model_config.num_hidden_layers == 28
    assert model_config.model_is_mrope is False
    assert model_config.dtype == torch.bfloat16
    assert model_config.hf_eos_token_id == {151643, 151645}
    assert text_hidden_size(SimpleNamespace(config=config)) == 2048

    copied = copy.deepcopy(config)
    assert copied.audio_config.d_model == 1024
    assert copied.text_config.hidden_size == 2048
    roundtripped = Qwen3ASRConfig.from_dict(config.to_dict())
    assert roundtripped.audio_config.encoder_layers == 24
    assert roundtripped.text_config.num_hidden_layers == 28
    assert roundtripped.text_config.tie_word_embeddings is True

    direct = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=False)
    assert isinstance(direct, Qwen3ASRConfig)
    assert direct.audio_config.d_model == 1024


def test_legacy_thinker_config_backwards_compat(tmp_path: Path) -> None:
    """Legacy thinker_config checkpoints keep exact dimensions and MRoPE flag."""
    register_qwen3_asr_hf_config()
    (tmp_path / "config.json").write_text(json.dumps(legacy_thinker_1_7b_config_dict()))

    config = get_config(str(tmp_path), trust_remote_code=False)
    model_config = ModelConfig(model_path=str(tmp_path), trust_remote_code=False)
    ModelWorker.apply_arch_override(model_config, "Qwen3ASRForConditionalGeneration")

    assert isinstance(config, Qwen3ASRConfig)
    assert config.thinker_config.audio_config.encoder_layers == 24
    assert config.thinker_config.audio_config.d_model == 1024
    assert config.thinker_config.audio_config.output_dim == 2048
    assert config.thinker_config.audio_start_token_id == 151669
    assert config.thinker_config.audio_end_token_id == 151670
    assert config.audio_token_id == 151676
    assert config.thinker_config.text_config.hidden_size == 2048
    assert model_config.model_is_mrope is True
    assert model_config.dtype == torch.bfloat16


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_top_level_tie_word_embeddings_propagation_does_not_mutate_input(
    tie_word_embeddings: bool,
) -> None:
    text_dict = tiny_text_config_dict()
    text_dict.pop("tie_word_embeddings")
    snapshot = dict(text_dict)

    config = Qwen3ASRConfig(
        audio_config=tiny_audio_config_dict(),
        text_config=text_dict,
        tie_word_embeddings=tie_word_embeddings,
    )

    assert config.text_config.tie_word_embeddings is tie_word_embeddings
    assert text_dict == snapshot


def test_model_worker_apply_arch_override_supports_flat_upstream_hf_config() -> None:
    upstream_config = UpstreamHFQwen3ASRConfig(
        audio_config=tiny_audio_config_dict(),
        text_config=tiny_text_config_dict(tie_word_embeddings=True),
        tie_word_embeddings=True,
    )
    assert not hasattr(upstream_config, "thinker_config")
    model_config = SimpleNamespace(
        hf_config=upstream_config,
        hf_text_config=None,
        num_attention_heads=0,
        num_key_value_heads=0,
        hidden_size=0,
        num_hidden_layers=0,
        num_attention_layers=0,
    )

    ModelWorker.apply_arch_override(model_config, "Qwen3ASRForConditionalGeneration")

    assert model_config.hf_text_config is upstream_config.text_config
    assert model_config.hidden_size == 64
    assert model_config.num_hidden_layers == 1
    assert model_config.num_attention_heads == 4
    assert model_config.num_key_value_heads == 2
    assert model_config.num_attention_layers == 1


def test_native_hf_tied_weights_match_legacy_thinker_bit_for_bit(
    cpu_runtime_context: None,
) -> None:
    """Native HF tied checkpoints load bit-for-bit identically to legacy thinker checkpoints."""
    native_config = Qwen3ASRConfig(
        audio_config=tiny_audio_config_dict(),
        text_config=tiny_text_config_dict(tie_word_embeddings=True),
        tie_word_embeddings=True,
        dtype="float32",
    )
    legacy_config = Qwen3ASRConfig(
        thinker_config={
            "audio_config": tiny_audio_config_dict(),
            "text_config": tiny_text_config_dict(tie_word_embeddings=True),
            "dtype": "float32",
        }
    )

    native_model = build_cpu_model(native_config)
    legacy_model = build_cpu_model(legacy_config)

    native_weights = build_checkpoint_tensors(
        format_kind="native_hf",
        tie_word_embeddings=True,
        include_lm_head=False,
        seed=123,
    )
    legacy_weights = build_checkpoint_tensors(
        format_kind="legacy",
        tie_word_embeddings=True,
        include_lm_head=True,
        seed=123,
    )

    native_model.load_weights(native_weights)
    legacy_model.load_weights(legacy_weights)

    native_params = dict(native_model.named_parameters(remove_duplicate=False))
    legacy_params = dict(legacy_model.named_parameters(remove_duplicate=False))
    assert set(native_params.keys()) == set(legacy_params.keys())

    for name, native_param in native_params.items():
        assert (
            native_param.abs().sum().item() > 0.0
        ), f"Parameter {name} stayed zero after native HF weight loading"
        torch.testing.assert_close(
            native_param, legacy_params[name], rtol=0.0, atol=0.0
        )

    weights_by_name = dict(native_weights)
    torch.testing.assert_close(
        native_params["audio_tower.proj1.weight"],
        weights_by_name["model.multi_modal_projector.linear_1.weight"],
    )
    torch.testing.assert_close(
        native_params["audio_tower.proj2.bias"],
        weights_by_name["model.multi_modal_projector.linear_2.bias"],
    )
    torch.testing.assert_close(
        native_model.language_model.lm_head.weight,
        weights_by_name["model.language_model.embed_tokens.weight"],
    )
    torch.testing.assert_close(
        legacy_model.language_model.lm_head.weight,
        weights_by_name["model.language_model.embed_tokens.weight"],
    )


@pytest.mark.parametrize("format_kind", ["native_hf", "legacy"])
def test_native_hf_and_legacy_untied_lm_head(
    cpu_runtime_context: None,
    format_kind: str,
) -> None:
    if format_kind == "native_hf":
        config = Qwen3ASRConfig(
            audio_config=tiny_audio_config_dict(),
            text_config=tiny_text_config_dict(tie_word_embeddings=False),
            tie_word_embeddings=False,
            dtype="float32",
        )
    else:
        config = Qwen3ASRConfig(
            thinker_config={
                "audio_config": tiny_audio_config_dict(),
                "text_config": tiny_text_config_dict(tie_word_embeddings=False),
                "dtype": "float32",
            }
        )
    model = build_cpu_model(config)
    weights = build_checkpoint_tensors(
        format_kind=format_kind,
        tie_word_embeddings=False,
        include_lm_head=True,
        seed=77,
    )
    weights_by_name = dict(weights)
    model.load_weights(weights)

    embed_key = (
        "model.language_model.embed_tokens.weight"
        if format_kind == "native_hf"
        else "thinker.model.embed_tokens.weight"
    )
    lm_head_key = (
        "lm_head.weight" if format_kind == "native_hf" else "thinker.lm_head.weight"
    )
    assert model.language_model.lm_head is not model.language_model.model.embed_tokens
    torch.testing.assert_close(
        model.language_model.model.embed_tokens.weight,
        weights_by_name[embed_key],
    )
    torch.testing.assert_close(
        model.language_model.lm_head.weight,
        weights_by_name[lm_head_key],
    )


def test_native_hf_tied_embeddings_on_pp_last_rank(
    cpu_runtime_context: None,
) -> None:
    """PP last rank populates tied lm_head from embed_tokens when checkpoint omits lm_head."""
    text_dict = tiny_text_config_dict(tie_word_embeddings=True)
    text_dict["num_hidden_layers"] = 2
    config = Qwen3ASRConfig(
        audio_config=tiny_audio_config_dict(),
        text_config=text_dict,
        tie_word_embeddings=True,
        dtype="float32",
    )
    model = build_cpu_model(config, pp_size=2, pp_rank=1)
    weights = build_checkpoint_tensors(
        format_kind="native_hf",
        tie_word_embeddings=True,
        include_lm_head=False,
        seed=99,
    )
    weights_by_name = dict(weights)
    model.load_weights(weights)

    torch.testing.assert_close(
        model.language_model.lm_head.weight,
        weights_by_name["model.language_model.embed_tokens.weight"],
    )


def test_upstream_hf_config_without_thinker_config_attribute(
    cpu_runtime_context: None,
) -> None:
    """Direct upstream HF Qwen3ASRConfig initializes audio encoder and loads all weights."""
    upstream_config = UpstreamHFQwen3ASRConfig(
        audio_config=tiny_audio_config_dict(),
        text_config=tiny_text_config_dict(tie_word_embeddings=True),
        tie_word_embeddings=True,
    )
    assert not hasattr(upstream_config, "thinker_config")
    assert not hasattr(upstream_config.audio_config, "max_source_positions")

    model = build_cpu_model(upstream_config)
    assert model.audio_tower.max_source_positions == 1500

    weights = build_checkpoint_tensors(
        format_kind="native_hf",
        tie_word_embeddings=True,
        include_lm_head=False,
        seed=55,
    )
    model.load_weights(weights)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        assert (
            parameter.abs().sum().item() > 0.0
        ), f"Parameter {name} stayed zero for upstream HF Qwen3ASRConfig"


@pytest.mark.parametrize(
    ("format_kind", "tie_word_embeddings", "include_lm_head"),
    [
        ("native_hf", True, False),
        ("native_hf", False, True),
        ("legacy", True, True),
    ],
)
def test_torch_mps_runner_loads_native_hf_and_legacy_checkpoints(
    tmp_path: Path,
    format_kind: str,
    tie_word_embeddings: bool,
    include_lm_head: bool,
) -> None:
    """Torch MPS loader parses flat and legacy configs and ties omitted lm_head weights."""
    text_dict = tiny_text_config_dict(tie_word_embeddings=tie_word_embeddings)
    if format_kind == "native_hf":
        root_config: dict[str, object] = {
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "model_type": "qwen3_asr",
            "audio_config": tiny_audio_config_dict(),
            "text_config": {
                **text_dict,
                "eos_token_id": None,
                "pad_token_id": None,
            },
            "tie_word_embeddings": tie_word_embeddings,
            "eos_token_id": [126, 127],
            "pad_token_id": 127,
        }
    else:
        root_config = {
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "model_type": "qwen3_asr",
            "thinker_config": {
                "audio_config": tiny_audio_config_dict(),
                "text_config": text_dict,
            },
        }
    (tmp_path / "config.json").write_text(json.dumps(root_config))

    weights = build_checkpoint_tensors(
        format_kind=format_kind,
        tie_word_embeddings=tie_word_embeddings,
        include_lm_head=include_lm_head,
        seed=31,
    )
    tensor_map = {name: tensor for name, tensor in weights}
    save_file(tensor_map, str(tmp_path / "model.safetensors"))

    resolved_config = torch_mps_runner.text_config(tmp_path)
    assert resolved_config.hidden_size == 64
    assert resolved_config.tie_word_embeddings is tie_word_embeddings
    if format_kind == "native_hf":
        assert resolved_config.eos_token_id == [126, 127]
        assert resolved_config.pad_token_id == 127

    with torch.device("meta"):
        language_model = Qwen3ForCausalLM(resolved_config)
    torch_mps_runner.load_language_weights(language_model, tmp_path)

    embed_key = (
        "model.language_model.embed_tokens.weight"
        if format_kind == "native_hf"
        else "thinker.model.embed_tokens.weight"
    )
    torch.testing.assert_close(
        language_model.model.embed_tokens.weight,
        tensor_map[embed_key],
    )
    if format_kind == "native_hf" and tie_word_embeddings:
        torch.testing.assert_close(
            language_model.lm_head.weight,
            tensor_map[embed_key],
        )
    else:
        lm_head_key = (
            "lm_head.weight" if format_kind == "native_hf" else "thinker.lm_head.weight"
        )
        torch.testing.assert_close(
            language_model.lm_head.weight,
            tensor_map[lm_head_key],
        )
    assert torch.isfinite(language_model.model.rotary_emb.inv_freq).all()


def test_processor_from_pretrained_forwards_revision() -> None:
    sentinel_feature_extractor = SimpleNamespace(sampling_rate=16000)
    sentinel_tokenizer = SimpleNamespace(pad_token_id=151643)
    with (
        mock.patch(
            "sglang_omni.models.qwen3_asr.configuration_qwen3_asr."
            "AutoFeatureExtractor.from_pretrained",
            return_value=sentinel_feature_extractor,
        ) as mock_feature_extractor,
        mock.patch(
            "sglang_omni.models.qwen3_asr.configuration_qwen3_asr."
            "AutoTokenizer.from_pretrained",
            return_value=sentinel_tokenizer,
        ) as mock_tokenizer,
        mock.patch.object(Qwen3ASRProcessor, "__init__", return_value=None) as init,
    ):
        Qwen3ASRProcessor.from_pretrained(
            "oruk/oruk-fourier-2b",
            trust_remote_code=False,
            revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        )
        mock_feature_extractor.assert_called_once_with(
            "oruk/oruk-fourier-2b",
            trust_remote_code=False,
            revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        )
        mock_tokenizer.assert_called_once_with(
            "oruk/oruk-fourier-2b",
            trust_remote_code=False,
            revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        )
        init.assert_called_once_with(
            feature_extractor=sentinel_feature_extractor,
            tokenizer=sentinel_tokenizer,
        )
