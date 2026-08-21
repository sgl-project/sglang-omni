# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 AR streaming output builder contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
    make_cosyvoice3_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(
    *,
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="req-cosy",
        request=OmniRequest(inputs="你好", params=params or {}),
        data=data if data is not None else {},
    )


_DEFAULT_REQ = object()


def _data(
    payload: StagePayload,
    *,
    codes: list[int],
    req: Any = _DEFAULT_REQ,
) -> CosyVoice3SGLangRequestData:
    if req is _DEFAULT_REQ:
        req = SimpleNamespace(inflight_middle_chunks=0)
    return CosyVoice3SGLangRequestData(
        req=req,
        stage_payload=payload,
        output_codes=[torch.tensor([code], dtype=torch.long) for code in codes],
    )


def test_stream_builder_emits_only_new_codes_in_order() -> None:
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={"stream": True})
    data = _data(payload, codes=[7])

    messages = builder("req-cosy", data, None)

    assert len(messages) == 1
    message = messages[0]
    assert message.request_id == "req-cosy"
    assert message.type == "stream"
    assert message.target == "vocoder"
    assert message.data.tolist() == [7]
    assert message.data.ndim == 1
    assert message.data.dtype == torch.long
    assert message.data.device.type == "cpu"
    assert message.metadata == {
        "modality": "audio_codes",
        "stream": True,
        "sample_rate": 24000,
        "row_index": 0,
    }

    # No new sampled codes: exactly-once cursor must not re-emit.
    assert builder("req-cosy", data, None) == []

    data.output_codes.append(torch.tensor([8], dtype=torch.long))
    data.output_codes.append(torch.tensor([9], dtype=torch.long))
    messages = builder("req-cosy", data, None)
    assert len(messages) == 1
    assert messages[0].data.tolist() == [8, 9]
    assert messages[0].metadata["row_index"] == 1


def test_stream_builder_skips_non_streaming_requests() -> None:
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={"stream": False}, params={"stream": False})
    data = _data(payload, codes=[7])

    assert builder("req-cosy", data, None) == []
    assert data.stream_output_code_count == 0


def test_stream_builder_state_dict_is_authoritative_stream_gate() -> None:
    # The prepared state folds tts_params.stream; params alone may stay empty.
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={"stream": True}, params={})
    data = _data(payload, codes=[7])

    messages = builder("req-cosy", data, None)
    assert len(messages) == 1
    assert messages[0].data.tolist() == [7]


def test_stream_builder_state_dict_stream_false_overrides_params() -> None:
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={"stream": False}, params={"stream": True})
    data = _data(payload, codes=[7])

    assert builder("req-cosy", data, None) == []
    assert data.stream_output_code_count == 0


def test_stream_builder_falls_back_to_request_params_for_unprepared_payloads() -> None:
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={}, params={"stream": True})
    data = _data(payload, codes=[7])

    messages = builder("req-cosy", data, None)
    assert len(messages) == 1
    assert messages[0].data.tolist() == [7]


def test_stream_builder_suppresses_emission_during_chunked_prefill() -> None:
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={"stream": True})
    req = SimpleNamespace(inflight_middle_chunks=2)
    data = _data(payload, codes=[7], req=req)

    assert builder("req-cosy", data, None) == []
    assert data.stream_output_code_count == 0

    # Emission resumes with the full backlog once the prompt chunks drain.
    req.inflight_middle_chunks = 0
    messages = builder("req-cosy", data, None)
    assert len(messages) == 1
    assert messages[0].data.tolist() == [7]
    assert messages[0].metadata["row_index"] == 0


def test_stream_builder_tolerates_missing_scheduler_request() -> None:
    builder = make_cosyvoice3_stream_output_builder()
    payload = _payload(data={"stream": True})
    data = _data(payload, codes=[7], req=None)

    messages = builder("req-cosy", data, None)
    assert len(messages) == 1
    assert messages[0].data.tolist() == [7]
