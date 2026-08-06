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
        "_perf_counter",
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


def test_higgs_result_adapter_propagates_conversion_failure(
    monkeypatch,
) -> None:
    _, result_adapter = request_builders.make_higgs_scheduler_adapters()
    state = HiggsTtsState(prompt_token_ids=[1], max_new_tokens=8)
    payload = StagePayload(
        request_id="req-higgs-failed-result",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )
    data = request_builders.build_sglang_higgs_request(
        state, request_id=payload.request_id
    )
    data.stage_payload = payload

    def fail_result_conversion(_state, _data) -> None:
        raise RuntimeError("result conversion failed")

    monkeypatch.setattr(
        request_builders,
        "apply_higgs_result",
        fail_result_conversion,
    )

    with pytest.raises(RuntimeError, match="result conversion failed"):
        result_adapter(data)
