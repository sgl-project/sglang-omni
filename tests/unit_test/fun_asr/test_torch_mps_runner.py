# SPDX-License-Identifier: Apache-2.0
"""Torch runner parity and request-cache lifecycle."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_runner_cache_lifecycle_and_flat_audio_features(device):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from sglang_omni.models.fun_asr.torch_mps_runner import FunASRTorchMpsModelRunner

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(42)
    model = (
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
    runner = object.__new__(FunASRTorchMpsModelRunner)
    runner.device = torch.device(device)
    runner._past_key_values = {}
    features = torch.randn(2, 8, device=device)
    runner.model = SimpleNamespace(
        language_model=model, get_audio_feature=lambda items: features
    )
    runner._next_token_result = lambda tokens: tokens
    item = SimpleNamespace(feature=torch.zeros(1), pad_value=999)
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(mm_items=[item], audio_token_id=10)
    )
    requests = [SimpleNamespace(request_id="one", data=SimpleNamespace(req=req))]
    first = runner.custom_prefill_forward(
        None, SimpleNamespace(input_ids=torch.tensor([1, 999, 999, 2])), requests
    )
    assert "one" in runner._past_key_values
    second = runner.custom_decode_forward(
        None, SimpleNamespace(input_ids=first), requests
    )
    with torch.inference_mode():
        embeds = model.model.embed_tokens(torch.tensor([[1, 10, 10, 2]], device=device))
        embeds[0, 1:3] = features
        embeds = torch.cat(
            [embeds, model.model.embed_tokens(first.reshape(1, 1))], dim=1
        )
        expected = model(inputs_embeds=embeds).logits[:, -1].argmax(dim=-1)
    assert torch.equal(second, expected)
    runner.on_request_finished("one", None)
    assert not runner._past_key_values
    runner._past_key_values["one"] = object()
    runner.abort_request("one")
    runner.abort_request("one")
    assert not runner._past_key_values
    with pytest.raises(RuntimeError, match="no cache"):
        runner.custom_decode_forward(None, SimpleNamespace(input_ids=first), requests)


def test_language_checkpoint_loads_tied_weights_strictly(tmp_path):
    import json

    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from sglang_omni.models.fun_asr.torch_mps_runner import load_language_model

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
        name.replace("model.", "model.language_model.", 1): value
        for name, value in expected.state_dict().items()
        if name != "lm_head.weight"
    }
    save_file(weights, tmp_path / "model.safetensors")
    actual = load_language_model(tmp_path)
    assert (
        actual.lm_head.weight.data_ptr() == actual.model.embed_tokens.weight.data_ptr()
    )
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value)
    weights.pop("model.language_model.norm.weight")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_language_model(tmp_path)
