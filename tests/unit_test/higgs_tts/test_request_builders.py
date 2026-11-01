# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from sglang.srt.mem_cache.radix_cache import RadixKey

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload


def test_higgs_reference_audio_namespaces_radix_cache() -> None:
    prompt = [-100, -100, 42]
    first = request_builders.build_sglang_higgs_request(
        HiggsTtsState(
            prompt_token_ids=prompt,
            reference_codes_delayed=[[1, 2], [3, 4]],
        ),
        request_id="first",
    ).req
    second = request_builders.build_sglang_higgs_request(
        HiggsTtsState(
            prompt_token_ids=prompt,
            reference_codes_delayed=[[5, 6], [7, 8]],
        ),
        request_id="second",
    ).req

    assert first.origin_input_ids == second.origin_input_ids
    assert first.extra_key != second.extra_key
    with pytest.raises(ValueError, match="matching extra_key"):
        RadixKey(first.origin_input_ids, first.extra_key).match(
            RadixKey(second.origin_input_ids, second.extra_key)
        )


def test_higgs_scheduler_adapters_clamp_cap_and_record_engine_time(
    monkeypatch,
) -> None:
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(
        request_builders,
        "perf_counter",
        lambda: next(ticks),
    )
    request_builder, result_adapter = request_builders.make_higgs_scheduler_adapters(
        max_new_tokens_cap=2048,
    )
    state = HiggsTtsState(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=4096,
    )
    payload = StagePayload(
        request_id="req-higgs",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )

    data = request_builder(payload)
    data.output_codes.append(torch.tensor([1, 2, 3], dtype=torch.long))
    result = result_adapter(data)

    assert data.max_new_tokens == 2048
    assert data.req.sampling_params.max_new_tokens == 2048
    assert result.data["completion_tokens"] == 1
    assert result.data["engine_time_s"] == 2.5


def test_higgs_result_adapter_reads_output_code_buffer() -> None:
    _, result_adapter = request_builders.make_higgs_scheduler_adapters()
    state = HiggsTtsState(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=8,
        num_codebooks=3,
    )
    payload = StagePayload(
        request_id="req-higgs",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )
    data = request_builders.build_sglang_higgs_request(
        state, request_id=payload.request_id
    )
    data.stage_payload = payload
    data.output_codes.append(torch.tensor([9, 9, 9], dtype=torch.long))
    data.output_code_buffer = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    data.output_code_count = 2

    result = result_adapter(data)

    assert result.data["output_codes_delayed"] == [[1, 2, 3], [4, 5, 6]]
    assert result.data["completion_tokens"] == 2


def test_text_length_budget_basic_and_clamping() -> None:
    # 0 tokens -> minimum floor (50) or base (1.5 * 2.0 * 25 = 75)
    assert request_builders.text_length_budget(0) == 75
    assert request_builders.text_length_budget(0, min_tokens=100) == 100

    # 2 tokens ("Hello world"): ceil(2.0 * (1.5 + 0.4 * 2) * 25) = 115
    assert request_builders.text_length_budget(2) == 115

    # 25 tokens (~5s speech, SeedTTS EN benchmark target text):
    # ceil(2.0 * (1.5 + 0.4 * 25) * 25) = ceil(2.0 * 11.5 * 25) = 575
    assert request_builders.text_length_budget(25) == 575

    # Long text (200 tokens): ceil(2.0 * (1.5 + 0.4 * 200) * 25) = 4075 -> capped at 2048
    assert request_builders.text_length_budget(200) == 2048
    assert request_builders.text_length_budget(200, max_cap=1000) == 1000


def test_text_length_budget_safety_factor() -> None:
    # Custom safety factor
    # 10 tokens: (1.5 + 4.0) * 25 = 137.5
    assert request_builders.text_length_budget(10, safety_factor=1.0) == 138
    assert request_builders.text_length_budget(10, safety_factor=3.0) == 413

    with pytest.raises(ValueError, match="safety_factor must be > 0"):
        request_builders.text_length_budget(10, safety_factor=0.0)

    with pytest.raises(ValueError, match="safety_factor must be > 0"):
        request_builders.text_length_budget(10, safety_factor=-1.5)


def test_estimate_text_tokens_en_zh() -> None:
    assert request_builders.estimate_text_tokens("") == 0
    assert request_builders.estimate_text_tokens("   ") == 0

    # Short English text: "Hello world" (10 non-space chars) -> ceil(10 * 0.28) = 3
    en_tokens = request_builders.estimate_text_tokens("Hello world")
    assert en_tokens == 3

    # Chinese text: 6 chars -> 6 tokens
    zh_tokens = request_builders.estimate_text_tokens("你好，世界！")
    assert zh_tokens == 6

    # Mixed text
    mixed_tokens = request_builders.estimate_text_tokens("Hello 世界")
    # 2 CJK chars + 5 ascii chars (ceil(5 * 0.28) = 2) -> 4
    assert mixed_tokens == 4


def test_calculate_higgs_max_new_tokens_with_and_without_tokenizer() -> None:
    # Direct int token count
    assert request_builders.calculate_higgs_max_new_tokens(2) == 115

    # String without tokenizer
    budget_no_tok = request_builders.calculate_higgs_max_new_tokens("Hello world")
    assert budget_no_tok > 50

    # String with mock tokenizer
    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [101, 102, 103, 104, 105]  # 5 tokens

    mock_tok = MockTokenizer()
    # 5 tokens: ceil(2.0 * (1.5 + 0.4 * 5) * 25) = ceil(2.0 * 3.5 * 25) = 175
    budget_with_tok = request_builders.calculate_higgs_max_new_tokens(
        "any text", tokenizer=mock_tok
    )
    assert budget_with_tok == 175

    # Empty string
    assert request_builders.calculate_higgs_max_new_tokens("") == 75
