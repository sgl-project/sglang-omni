# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.request_builders import build_sglang_talker_request
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer


@pytest.mark.parametrize("minimum", [None, 0, 1, 32])
def test_chat_forwards_explicit_minimum_and_omits_unset_value(minimum):
    request = ChatCompletionRequest(
        model="qwen3-omni",
        messages=[{"role": "user", "content": "Read this aloud."}],
        talker_min_new_tokens=minimum,
    )
    result = _build_chat_generate_request(request)
    if minimum is None:
        assert "talker_min_new_tokens" not in result.extra_params
    else:
        assert result.extra_params["talker_min_new_tokens"] == minimum


def test_chat_rejects_negative_minimum_before_generation():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ChatCompletionRequest(
            model="qwen3-omni",
            messages=[{"role": "user", "content": "Read this aloud."}],
            talker_min_new_tokens=-1,
        )


@pytest.mark.parametrize("minimum", [0, 1, 32])
def test_talker_builder_preserves_minimum_and_codec_stop_token(minimum):
    data = build_sglang_talker_request(
        torch.zeros(2, 4),
        tokenizer=FakeQwenTokenizer(),
        codec_vocab_size=4096,
        codec_eos_id=4095,
        max_new_tokens=32,
        min_new_tokens=minimum,
    )
    assert data.req.sampling_params.min_new_tokens == minimum
    assert data.req.sampling_params.max_new_tokens == 32
    assert data.req.eos_token_ids == {4095}
    assert 4095 in data.req.sampling_params.stop_token_ids


@pytest.mark.parametrize("minimum", [-1, 33])
def test_talker_builder_rejects_invalid_minimum_before_scheduling(minimum):
    with pytest.raises(ValueError, match="min_new_tokens"):
        build_sglang_talker_request(
            torch.zeros(2, 4),
            tokenizer=FakeQwenTokenizer(),
            codec_vocab_size=4096,
            max_new_tokens=32,
            min_new_tokens=minimum,
        )


def test_talker_builder_default_minimum_does_not_force_longer_output():
    data = build_sglang_talker_request(
        torch.zeros(2, 4), tokenizer=FakeQwenTokenizer(), codec_vocab_size=4096
    )
    assert data.req.sampling_params.min_new_tokens == 0


@pytest.mark.parametrize("largest_stop", [5, 7])
def test_codec_sampler_masks_stop_and_eos_only_below_each_rows_minimum(largest_stop):
    stop_mask = torch.zeros(4, 8, dtype=torch.bool)
    stop_mask[:, [5, 7]] = True
    talker = SimpleNamespace(
        _repetition_penalties=torch.ones(4, 1),
        _repetition_mask=torch.zeros(4, 8, dtype=torch.bool),
        _suppress_mask=torch.zeros(4, 8, dtype=torch.bool),
        _sampling_output_lens=torch.tensor([0, 0, 1, 2]),
        _sampling_min_new_tokens=torch.tensor([0, 1, 1, 3]),
        _min_new_token_stop_mask=stop_mask,
        _sampler=None,
    )
    logits = torch.zeros(4, 8)
    logits[:, 6] = 50
    logits[:, [5, 7]] = 90
    logits[:, largest_stop] = 100
    original = logits.clone()
    first = Qwen3OmniTalker._sample_decode_tokens(talker, logits, None)
    assert first.tolist() == [largest_stop, 6, largest_stop, 6]
    talker._sampling_output_lens += 1
    second = Qwen3OmniTalker._sample_decode_tokens(talker, logits, None)
    assert second.tolist() == [largest_stop] * 4
    assert torch.equal(logits, original)
