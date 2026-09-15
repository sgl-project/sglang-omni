# SPDX-License-Identifier: Apache-2.0
"""Torch/MPS parity for MOSS-TD's non-contiguous audio prefill."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.layers import linear
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models import whisper
from transformers import Qwen3Config, Qwen3ForCausalLM, WhisperConfig

from sglang_omni.models.moss_transcribe_diarize.hf_config import (
    MossTranscribeDiarizeConfig,
)
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
    VQAdaptor,
)
from sglang_omni.models.moss_transcribe_diarize.torch_mps_runner import (
    MossTranscribeDiarizeTorchMpsModelRunner,
    load_language_model,
)


@pytest.fixture(params=["cpu", "mps"])
def moss_runner(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> MossTranscribeDiarizeTorchMpsModelRunner:
    device = request.param
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(42)
    parallel = SimpleNamespace(tp_size=1, tp_rank=0)
    monkeypatch.setattr(whisper, "get_parallel", lambda: parallel)
    monkeypatch.setattr(linear, "get_parallel", lambda: parallel)
    monkeypatch.setattr(linear, "get_tp_group", lambda: None)
    model = MossTranscribeDiarizeForConditionalGeneration.__new__(
        MossTranscribeDiarizeForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.config = MossTranscribeDiarizeConfig(audio_merge_size=2)
    model.whisper_encoder = whisper.WhisperEncoder(
        WhisperConfig(
            num_mel_bins=4,
            d_model=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            max_source_positions=8,
        )
    )
    model.vq_adaptor = VQAdaptor(16, 8)
    model.compiled_encoder = None
    model.encoder_graph_runner = None
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    model.language_model = (
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
    model.eval().to(device)
    runner = object.__new__(MossTranscribeDiarizeTorchMpsModelRunner)
    runner.device = torch.device(device)
    runner.past_key_values = {}
    runner.prefill_chunk_size = 3
    runner.encoder_window_batch_size = 2
    runner.model = model
    return runner


def test_microbatched_audio_and_chunked_prefill_match_full_forward(
    moss_runner: MossTranscribeDiarizeTorchMpsModelRunner,
) -> None:
    runner = moss_runner
    device = runner.device
    language_model = runner.model.language_model
    audio_item = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.randn(3, 4, 16),
        model_specific_data={"audio_feature_lengths": torch.tensor([1, 2, 1])},
        pad_value=999,
    )
    features = runner.model.get_audio_feature_uncached([audio_item], None)
    logits = []
    hook = language_model.register_forward_hook(
        lambda _model, _args, output: logits.append(
            output.logits[:, -1].detach().clone()
        )
    )
    batch_sizes: list[int] = []
    encoder_hook = runner.model.whisper_encoder.register_forward_pre_hook(
        lambda model, inputs: batch_sizes.append(inputs[0].shape[0])
    )
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(mm_items=[audio_item], audio_token_id=10),
        sampling_params=SimpleNamespace(max_new_tokens=8),
    )
    requests = [SimpleNamespace(request_id="one", data=SimpleNamespace(req=req))]

    first = runner.custom_prefill_forward(
        None,
        SimpleNamespace(input_ids=torch.tensor([1, 999, 999, 7, 999, 999, 2])),
        requests,
    ).next_token_ids
    second = runner.custom_decode_forward(
        None, SimpleNamespace(input_ids=first), requests
    ).next_token_ids
    hook.remove()
    encoder_hook.remove()
    assert max(batch_sizes) <= runner.encoder_window_batch_size

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
    assert not runner.past_key_values


def test_torch_mps_loads_official_language_weight_names(tmp_path: Path) -> None:
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
