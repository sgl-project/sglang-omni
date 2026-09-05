# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from sglang_omni.models.breeze_tts.checkpoint import backbone_weights, load_component
from sglang_omni.models.breeze_tts.depth_decoder import BreezeDepthDecoder
from sglang_omni.models.breeze_tts.frontend import BreezeFrontend
from sglang_omni.models.breeze_tts.hf_config import BreezeConfig, register_breeze_config
from sglang_omni.models.breeze_tts.sampling import (
    SamplingConfig,
    apply_cfg,
    sample_logits,
)


def test_backbone_uses_nested_qwen_config_and_roundtrips(tiny_config, tmp_path):
    register_breeze_config()
    (tmp_path / "config.json").write_text(json.dumps(tiny_config))
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, BreezeConfig)
    assert config.rms_norm_eps == 1e-6  # outer legacy Llama epsilon is 1e-5
    assert config.rope_parameters["rope_theta"] == 1000000
    assert config.vocab_size == tiny_config["vocab_size"] + 1
    assert config.eos_token_id == tiny_config["vocab_size"]
    config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    assert restored.vocab_size == config.vocab_size  # don't add EOS twice
    assert restored.depth_decoder_config == config.depth_decoder_config


@pytest.mark.parametrize(
    "scale,expected", [(0, [2.0, 4.0]), (1, [3.0, 1.0]), (4, [6.0, -8.0])]
)
def test_guidance_endpoints_and_extrapolation(scale, expected):
    actual = apply_cfg(torch.tensor([[3.0, 1.0], [2.0, 4.0]]), scale)
    torch.testing.assert_close(actual, torch.tensor([expected]))


def test_reserved_codes_suppressed_but_backbone_eos_preserved():
    logits = torch.tensor([[0.0, 2.0, 0.0, 0.0, 100.0, 101.0, 90.0]])
    params = SamplingConfig(temperature=0)
    rng = torch.Generator().manual_seed(0)
    assert sample_logits(logits, params, rng, codebook_size=4).item() == 1
    assert (
        sample_logits(logits, params, rng, codebook_size=4, eos_token_id=6).item() == 6
    )


def test_request_rng_is_independent_of_interleaving_and_global_rng():
    scores = torch.zeros(1, 8)
    params = SamplingConfig(top_k=0)
    a, b = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    other = torch.Generator().manual_seed(17)
    expected = [
        sample_logits(scores, params, a, codebook_size=8).item() for _ in range(12)
    ]
    state = torch.random.get_rng_state().clone()
    actual = []
    for _ in range(12):
        sample_logits(scores, params, other, codebook_size=8)
        actual.append(sample_logits(scores, params, b, codebook_size=8).item())
    assert expected == actual
    assert torch.equal(torch.random.get_rng_state(), state)


def test_repetition_penalty_applies_only_to_generated_history_once():
    scores = torch.tensor([[0.0, 3.0, 2.0, 0.0]])
    params = SamplingConfig(temperature=0, repetition_penalty=2)
    assert (
        sample_logits(
            scores, params, torch.Generator(), history=[1, 1], codebook_size=4
        ).item()
        == 2
    )
    assert scores.tolist() == [[0.0, 3.0, 2.0, 0.0]]  # don't mutate CFG branch logits


def test_depth_cached_decode_matches_full_sequence_and_resets(tiny_config):
    torch.manual_seed(5)
    model = BreezeDepthDecoder(tiny_config["depth_decoder_config"]).eval()
    torch.nn.init.normal_(model.codebooks_head.weight, std=0.1)
    hidden = torch.randn(2, 16)
    first = torch.tensor([3])
    params = SamplingConfig(temperature=0, cfg_scale=4)
    actual = model.decode_frame(
        hidden, first, params, torch.Generator(), codebook_size=8
    )
    # Independent, non-cached depth forward guards cache reset, the initial
    # hidden/c0 layout, head index and the c(k) embedding offset.
    prefix = [
        hidden.unsqueeze(1),
        model.model.embed_tokens(first.expand(2)).unsqueeze(1),
    ]
    expected = [first.item()]
    with torch.no_grad():
        for codebook in range(1, 4):
            all_embeds = model.model.inputs_embeds_projector(torch.cat(prefix, dim=1))
            out = model.model(
                inputs_embeds=all_embeds, use_cache=False
            ).last_hidden_state[:, -1]
            logits = out @ model.codebooks_head.weight[codebook - 1]
            token = apply_cfg(logits, params.cfg_scale)[:, :8].argmax(-1)
            expected.append(token.item())
            prefix.append(
                model.model.embed_tokens(token.expand(2) + codebook * 11).unsqueeze(1)
            )
    assert actual.tolist() == expected
    # An unrelated generation must not leave any frame/depth KV state behind.
    model.decode_frame(
        hidden * 2,
        torch.tensor([2]),
        replace(params, cfg_scale=0),
        torch.Generator(),
        codebook_size=8,
    )
    assert (
        model.decode_frame(
            hidden, first, params, torch.Generator(), codebook_size=8
        ).tolist()
        == expected
    )
    expected_embed = sum(model.model.embed_tokens(actual[i] + i * 11) for i in range(4))
    torch.testing.assert_close(model.embed_frames(actual), expected_embed)


def test_backbone_mapping_filters_auxiliary_weights_and_requires_all_layers(
    tiny_config,
):
    from transformers import Qwen3Config, Qwen3Model

    backbone = Qwen3Model(Qwen3Config(**tiny_config["backbone_config"]))
    tensors = {
        "backbone_model." + name: value
        for name, value in backbone.state_dict().items()
        if not name.startswith("embed_tokens.")
    }
    tensors["lm_head.weight"] = torch.zeros(12, 16)
    tensors["codec_model.unused.weight"] = torch.zeros(1)
    actual = dict(backbone_weights(tensors.items(), 2))
    torch.testing.assert_close(
        actual["model.layers.0.self_attn.q_proj.weight"],
        backbone.layers[0].self_attn.q_proj.weight,
    )
    assert "codec_model.unused.weight" not in actual
    assert "model.embed_tokens.weight" not in actual
    del tensors["backbone_model.layers.1.self_attn.q_proj.weight"]
    with pytest.raises(ValueError, match="missing backbone weights"):
        list(backbone_weights(tensors.items(), 2))


def test_meta_frontend_strict_loading_without_legacy_codec(tiny_config, tmp_path):
    model = BreezeFrontend(tiny_config).eval()
    tensors = {
        f"text_encoder.{k}": v for k, v in model.text_encoder.state_dict().items()
    }
    tensors["text_encoder_proj.weight"] = model.text_encoder_proj.weight
    tensors["depth_decoder.model.embed_tokens.weight"] = model.audio_embeddings.weight
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(tiny_config))
    loaded = BreezeFrontend.from_checkpoint(str(tmp_path), "cpu")
    assert all(not tensor.is_meta for tensor in loaded.buffers())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        expected = (
            model.to(torch.bfloat16).text_encoder(input_ids=ids).last_hidden_state
        )
        actual = loaded.text_encoder(input_ids=ids).last_hidden_state
    torch.testing.assert_close(actual, expected)
    # An incomplete checkpoint must fail, not leave randomly initialized weights.
    del tensors["text_encoder_proj.weight"]
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_component(model.text_encoder_proj, str(tmp_path), "text_encoder_proj.")
