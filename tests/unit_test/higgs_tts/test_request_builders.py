# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload


def test_higgs_scheduler_adapters_clamp_cap_and_record_engine_time(
    monkeypatch,
) -> None:
    ticks = iter([10.0, 12.5])
    reset_calls: list[str] = []
    monkeypatch.setattr(
        request_builders,
        "_perf_counter",
        lambda: next(ticks),
    )
    request_builder, result_adapter = request_builders.make_higgs_scheduler_adapters(
        SimpleNamespace(reset_request=reset_calls.append),
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
    assert reset_calls == ["req-higgs"]


def test_higgs_result_adapter_reads_output_code_buffer() -> None:
    reset_calls: list[str] = []
    _, result_adapter = request_builders.make_higgs_scheduler_adapters(
        SimpleNamespace(reset_request=reset_calls.append),
    )
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
    assert reset_calls == ["req-higgs"]
