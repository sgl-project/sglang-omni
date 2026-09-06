# SPDX-License-Identifier: Apache-2.0
"""Torch runner parity and request-cache lifecycle."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("batched_audio", [False, True])
def test_torch_runner_cache_lifecycle_and_audio_features(device, batched_audio):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from sglang_omni.models.qwen3_asr.torch_mps_runner import (
        Qwen3ASRTorchMpsModelRunner,
    )

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
    runner = object.__new__(Qwen3ASRTorchMpsModelRunner)
    runner.device = torch.device(device)
    runner._past_key_values = {}
    features = torch.randn(2, 8, device=device)
    runner.model = SimpleNamespace(
        language_model=model,
        get_audio_feature=lambda items: (
            features.unsqueeze(0) if batched_audio else features
        ),
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


def test_shared_runner_retains_model_specific_errors():
    from sglang_omni.model_runner.audio_torch_mps import AudioTorchMpsModelRunner

    class OtherAudioRunner(AudioTorchMpsModelRunner):
        model_name = "Other ASR"

    runner = object.__new__(OtherAudioRunner)
    with pytest.raises(
        RuntimeError, match="Other ASR Torch MPS.*max_running_requests=1"
    ):
        runner._one_request([])
    req = SimpleNamespace(multimodal_inputs=None)
    request = SimpleNamespace(data=SimpleNamespace(req=req))
    with pytest.raises(
        ValueError, match="Other ASR Torch MPS requires exactly one audio item"
    ):
        runner.custom_prefill_forward(None, None, [request])


@pytest.mark.parametrize("token_ids", [[1, 2], [1, 999, 2, 999]])
def test_torch_runner_rejects_missing_or_disjoint_audio_span(token_ids):
    from sglang_omni.models.qwen3_asr.torch_mps_runner import (
        Qwen3ASRTorchMpsModelRunner,
    )

    runner = object.__new__(Qwen3ASRTorchMpsModelRunner)
    item = SimpleNamespace(feature=torch.zeros(1), pad_value=999)
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(mm_items=[item], audio_token_id=10)
    )
    request = SimpleNamespace(data=SimpleNamespace(req=req))
    with pytest.raises(
        ValueError,
        match="Qwen3-ASR Torch MPS.*(no audio placeholders|must be contiguous)",
    ):
        runner.custom_prefill_forward(
            None, SimpleNamespace(input_ids=torch.tensor(token_ids)), [request]
        )
