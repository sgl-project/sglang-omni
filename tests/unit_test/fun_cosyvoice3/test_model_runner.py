# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)


def test_cosyvoice3_runner_collects_speech_tokens_and_skips_eos() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    requests = [
        SimpleNamespace(data=SimpleNamespace(output_codes=[])),
        SimpleNamespace(data=SimpleNamespace(output_codes=[])),
    ]
    result = SimpleNamespace(next_token_ids=torch.tensor([[EOS_ID], [13]]))

    runner._collect_tokens(result, None, None, requests)

    assert requests[0].data.output_codes == []
    assert [code.item() for code in requests[1].data.output_codes] == [13]
    assert requests[1].data.output_codes[0].dtype == torch.long


def test_cosyvoice3_runner_skips_all_control_tokens() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    requests = [SimpleNamespace(data=SimpleNamespace(output_codes=[]))]

    runner._collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([VOCAB_SIZE + 3])),
        None,
        None,
        requests,
    )

    assert requests[0].data.output_codes == []


def test_cosyvoice3_runner_samples_before_prefill_and_decode_collection() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)

    assert runner.sample_before_post_prefill(None, None, []) is True
    assert runner.sample_before_post_decode(None, None, []) is True


def test_cosyvoice3_load_weights_maps_custom_and_backbone_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the loader path, not only the standalone key mapper."""
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    speech_embedding = torch.nn.Parameter(torch.zeros(2, 3))
    decoder = torch.nn.Parameter(torch.zeros(2, 3))
    model._cached_params_dict = {
        "speech_embedding.weight": speech_embedding,
        "llm_decoder.weight": decoder,
    }
    forwarded = []
    monkeypatch.setattr(
        "sglang.srt.models.qwen2.Qwen2ForCausalLM.load_weights",
        lambda _self, weights: forwarded.extend(weights),
    )

    speech_value = torch.ones(2, 3)
    decoder_value = torch.full((2, 3), 2.0)
    model.load_weights(
        [
            ("speech_embedding.weight", speech_value),
            ("llm_decoder.weight", decoder_value),
            ("llm.model.lm_head.weight", torch.ones(2, 3)),
            ("llm.model.model.layers.0.weight", torch.full((3, 3), 3.0)),
        ]
    )

    assert torch.equal(speech_embedding, speech_value)
    assert torch.equal(decoder, decoder_value)
    assert len(forwarded) == 1
    assert forwarded[0][0] == "model.layers.0.weight"
    assert torch.equal(forwarded[0][1], torch.full((3, 3), 3.0))


def test_cosyvoice3_runner_builds_prefill_embedding_slice_after_prefix() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = torch.nn.Linear(3, 3, bias=False)
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    extend_range=SimpleNamespace(length=2), prefix_indices=[99]
                ),
                prompt_input_embeds=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            )
        )
    ]
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    result = runner._build_prefill_input_embeds(forward_batch, requests)

    assert torch.equal(
        result, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
    )
