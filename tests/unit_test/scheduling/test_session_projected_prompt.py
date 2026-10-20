# SPDX-License-Identifier: Apache-2.0
"""A projected talker prompt rides a native session as the unit's one embedding span."""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionBridge,
    BridgeSession,
    SessionUnit,
)
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
    session_projected_prefill_rows,
)

RETAINED = [7, 8, 9]
WIDTH = 4
MAX_REQUEST_TOKENS = 4096


def adapter_request(ids: list[int]) -> Req:
    request = Req("unit-1", "", array("q", ids), SamplingParams(), vocab_size=64)
    request.tokenizer = None
    return request


def bridge_with_unit(ids: list[int]) -> tuple[ARSessionBridge, SessionUnit]:
    identity = SessionIdentity("talker")
    scheduler = Mock()
    scheduler.model_config.vocab_size = 64
    scheduler.max_req_len = MAX_REQUEST_TOKENS

    def create_req(tokenized, tokenizer, vocab_size, eos_token_ids):
        return SimpleNamespace(
            origin_input_ids=[*RETAINED, *tokenized.input_ids],
            to_finish=None,
            skip_radix_cache_insert=False,
            logprob_start_len=0,
        )

    scheduler.session_controller.get.return_value = SimpleNamespace(
        session_id="native", create_req=create_req, req_nodes={}
    )
    adapter = Mock()
    adapter.plan_history_cut.return_value = None
    bridge = ARSessionBridge(scheduler, adapter)
    chunk = TimedChunk("text", 0, 0, 0, {"text": "hi"}, format="text_delta")
    unit = SessionUnit(
        request_id="unit-1", session_identity=identity, stages=("talker",), chunk=chunk
    )
    bridge.sessions[identity.id] = BridgeSession(session_identity=identity, unit=unit)
    bridge.units_by_request_id["unit-1"] = unit
    return bridge, unit


def test_a_projected_prompt_becomes_the_unit_span_after_the_retained_prefix():
    bridge, unit = bridge_with_unit([1, 2])
    prompt = torch.arange(2 * WIDTH, dtype=torch.float32).reshape(2, WIDTH)
    data = SGLangARRequestData(
        req=adapter_request([1, 2]),
        prefill_input_embeds=prompt,
        input_embeds_are_projected=True,
    )
    bridge.create_session_request(StagePayload("unit-1", OmniRequest(None), {}), data)
    assert data.prefill_input_embeds is None
    assert [(span.start, span.end) for span in unit.embedding_spans] == [
        (len(RETAINED), len(RETAINED) + 2)
    ]
    assert torch.equal(data.session_embedding_spans[0].input_embeds, prompt)
    assert (unit.native_offset, unit.prompt_length) == (len(RETAINED), 2)
    assert data.req.skip_radix_cache_insert is True


@pytest.mark.parametrize(
    "rows, own_spans",
    [(3, []), (2, [EmbeddingSpan(start=0, end=1, input_embeds=torch.zeros(1, WIDTH))])],
)
def test_a_projected_prompt_must_be_the_only_span_over_all_the_ids(rows, own_spans):
    bridge, _ = bridge_with_unit([1, 2])
    data = SGLangARRequestData(
        req=adapter_request([1, 2]),
        prefill_input_embeds=torch.zeros(rows, WIDTH),
        unit_embedding_spans=own_spans,
        input_embeds_are_projected=True,
    )
    with pytest.raises(ValueError, match="one span"):
        bridge.create_session_request(
            StagePayload("unit-1", OmniRequest(None), {}), data
        )


def test_a_completed_projected_unit_keeps_its_decode_rows_for_rebuilds():
    bridge, unit = bridge_with_unit([1, 2])
    data = SGLangARRequestData(
        req=adapter_request([1, 2]),
        prefill_input_embeds=torch.zeros(2, WIDTH),
        input_embeds_are_projected=True,
    )
    bridge.create_session_request(StagePayload("unit-1", OmniRequest(None), {}), data)
    data.decode_input_embeds = [torch.full((WIDTH,), 1.0), torch.full((WIDTH,), 2.0)]
    bridge.complete("unit-1", data)
    session = bridge.sessions["talker"]
    assert session.unit is None and "unit-1" not in bridge.units_by_request_id
    assert [(span.start, span.end) for span in session.embedding_spans] == [
        (3, 5),
        (5, 7),
    ]
    assert session.embedding_spans[1].input_embeds[1].tolist() == [2.0] * WIDTH


def test_a_completed_token_unit_keeps_no_decode_rows():
    bridge, unit = bridge_with_unit([1, 2])
    data = SGLangARRequestData(req=adapter_request([1, 2]))
    bridge.create_session_request(StagePayload("unit-1", OmniRequest(None), {}), data)
    data.decode_input_embeds = [torch.zeros(WIDTH)]
    bridge.complete("unit-1", data)
    assert bridge.sessions["talker"].embedding_spans == []


def test_projected_session_prefill_rows_come_from_the_spans_and_fed_back_tokens():
    spans = [
        EmbeddingSpan(start=3, end=5, input_embeds=torch.ones(2, WIDTH)),
        EmbeddingSpan(start=5, end=7, input_embeds=torch.full((2, WIDTH), 2.0)),
    ]
    data = SGLangARRequestData(
        req=SimpleNamespace(
            rid="unit-2",
            extend_range=SimpleNamespace(start=4, end=7),
            get_fill_ids=lambda: [0, 0, 5, 0, 0, 0, 0],
        ),
        session_embedding_spans=spans,
    )

    def embed_tokens(token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.float32)[:, None].repeat(1, WIDTH)

    rows = session_projected_prefill_rows(
        data, embed_tokens, torch.device("cpu"), torch.float32
    )
    assert rows.tolist() == [[1.0] * WIDTH, [2.0] * WIDTH, [2.0] * WIDTH]
    data.req.extend_range = SimpleNamespace(start=2, end=7)
    rows = session_projected_prefill_rows(
        data, embed_tokens, torch.device("cpu"), torch.float32
    )
    assert rows[:2].tolist() == [[5.0] * WIDTH, [1.0] * WIDTH]


def test_a_unit_keeps_the_stage_payload_its_adapter_built():
    bridge, unit = bridge_with_unit([1, 2])
    payload = StagePayload("unit-1", OmniRequest(None), {})
    built = StagePayload("unit-1", OmniRequest(None, params={"stream": True}), {})
    data = SGLangARRequestData(req=adapter_request([1, 2]), stage_payload=built)
    bridge.create_session_request(payload, data)
    assert data.stage_payload is built
    bridge, unit = bridge_with_unit([1, 2])
    data = SGLangARRequestData(req=adapter_request([1, 2]))
    bridge.create_session_request(payload, data)
    assert data.stage_payload is payload
