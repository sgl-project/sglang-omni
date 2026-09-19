# SPDX-License-Identifier: Apache-2.0
"""Torch/MPS parity for MOSS-TD's non-contiguous audio prefill."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.models.moss_transcribe_diarize.torch_mps_runner import (
    MossTranscribeDiarizeTorchMpsModelRunner,
    load_language_model,
)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_mps_scatter_and_cache_match_full_forward(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(42)
    language_model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                tie_word_embeddings=True,
            )
        )
        .eval()
        .to(device)
    )
    features = torch.randn(4, 8, device=device)
    logits = []
    hook = language_model.register_forward_hook(
        lambda _model, _args, output: logits.append(
            output.logits[:, -1].detach().clone()
        )
    )
    forward_marker = object()

    class FakeMossModel:
        @staticmethod
        def _get_audio_feature_uncached(items, forward_batch):
            assert len(items) == 1
            assert forward_batch is forward_marker
            return features

    runner = object.__new__(MossTranscribeDiarizeTorchMpsModelRunner)
    runner.device = torch.device(device)
    runner._past_key_values = {}
    runner.prefill_chunk_size = 3
    runner.model = FakeMossModel()
    runner.model.language_model = language_model
    runner._next_token_result = lambda tokens: tokens
    item = SimpleNamespace(
        feature=torch.zeros(1, 1),
        audio_feature_lengths=torch.tensor([4]),
        pad_value=999,
    )
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(mm_items=[item], audio_token_id=10),
        sampling_params=SimpleNamespace(max_new_tokens=8),
    )
    requests = [SimpleNamespace(request_id="one", data=SimpleNamespace(req=req))]

    first = runner.custom_prefill_forward(
        forward_marker,
        SimpleNamespace(input_ids=torch.tensor([1, 999, 999, 7, 999, 999, 2])),
        requests,
    )
    second = runner.custom_decode_forward(
        None, SimpleNamespace(input_ids=first), requests
    )
    hook.remove()

    with torch.inference_mode():
        ids = torch.tensor([[1, 10, 10, 7, 10, 10, 2]], device=device)
        embeddings = language_model.model.embed_tokens(ids)
        embeddings[0, torch.tensor([1, 2, 4, 5], device=device)] = features
        expected_first = language_model(inputs_embeds=embeddings).logits[:, -1]
        torch.testing.assert_close(logits[-2], expected_first, atol=1e-4, rtol=1e-4)
        assert torch.equal(first, expected_first.argmax(dim=-1))
        embeddings = torch.cat(
            [embeddings, language_model.model.embed_tokens(first.reshape(1, 1))],
            dim=1,
        )
        expected = language_model(inputs_embeds=embeddings).logits[:, -1]
    torch.testing.assert_close(logits[-1], expected, atol=1e-4, rtol=1e-4)
    assert torch.equal(second, expected.argmax(dim=-1))
    runner.on_request_finished("one", None)
    assert not runner._past_key_values


def test_torch_mps_microbatches_encoder_windows() -> None:
    batch_sizes = []

    class FakeMossModel:
        @staticmethod
        def _get_audio_feature_uncached(items, _forward_batch):
            item = items[0]
            batch_sizes.append(item.feature.shape[0])
            assert item.audio_chunk_mapping.tolist() == [0] * len(item.feature)
            return torch.repeat_interleave(
                item.feature, item.audio_feature_lengths, dim=0
            )

    runner = object.__new__(MossTranscribeDiarizeTorchMpsModelRunner)
    runner.encoder_window_batch_size = 2
    runner.model = FakeMossModel()
    item = SimpleNamespace(
        feature=torch.arange(5, dtype=torch.float32).reshape(5, 1),
        audio_feature_lengths=torch.tensor([1, 2, 1, 2, 3]),
    )

    output = runner._get_audio_feature(item, None)

    assert batch_sizes == [2, 2, 1]
    torch.testing.assert_close(
        output, torch.repeat_interleave(item.feature, item.audio_feature_lengths, dim=0)
    )


def test_torch_mps_loads_official_language_weight_names(tmp_path) -> None:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=True,
    )
    expected = Qwen3ForCausalLM(config).eval()
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config.to_dict()}))
    weights = {
        "model.language_model." + name.removeprefix("model."): value
        for name, value in expected.state_dict().items()
        if name.startswith("model.")
    }
    save_file(weights, tmp_path / "model.safetensors")

    actual = load_language_model(tmp_path)

    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected_logits = expected(input_ids).logits
        actual_logits = actual(input_ids).logits
    torch.testing.assert_close(actual_logits, expected_logits)
