# SPDX-License-Identifier: Apache-2.0
"""The talker session speaks each reply as one request and relays its text to code2wav."""

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.qwen3_omni import talker_session
from sglang_omni.models.qwen3_omni.talker_session import (
    CODE2WAV_STAGE,
    QwenTalkerSessionAdapter,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import CONTROL, REPLY_TRUNCATED, TEXT_DELTA

IDENTITY = SessionIdentity("session")
PARAMS = {
    "talker_max_new_tokens": 4096,
    "talker_temperature": 0.9,
    "talker_top_k": 50,
    "talker_top_p": 1.0,
    "talker_repetition_penalty": 1.05,
}


class Tokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(ord("a") + token_id) for token_id in token_ids)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> QwenTalkerSessionAdapter:
    prefill_builder = Mock(codec_bos_id=7)
    config = SimpleNamespace(audio_token_id=102, image_token_id=100, video_token_id=101)
    built = Mock()
    monkeypatch.setattr(talker_session, "build_talker_request_data", built)
    result = QwenTalkerSessionAdapter(prefill_builder, Tokenizer(), 3072, 2150, config)
    result.outbox = queue.Queue()
    result.open(IDENTITY, OmniRequest(None))
    return result


def token(token_id: int) -> StreamItem:
    return StreamItem(0, torch.tensor([token_id]), "thinker", {"token_id": token_id})


def unit(turn: int, prefetched: list[int]) -> tuple[TimedChunk, StagePayload]:
    chunk = TimedChunk(CONTROL, 0, 0, 4, {}, format="qwen.unit", turn=turn)
    payload = StagePayload(f"unit-{turn}", OmniRequest(None, params=PARAMS), {})
    payload.prefetched_chunks = [token(token_id) for token_id in prefetched]
    return chunk, payload


def relayed(adapter: QwenTalkerSessionAdapter) -> list[tuple[str, str, int]]:
    out = []
    while not adapter.outbox.empty():
        message = adapter.outbox.get_nowait()
        chunk = TimedChunk.from_dict(message.metadata["chunk"])
        assert message.target == CODE2WAV_STAGE and chunk.format == TEXT_DELTA
        assert message.data.numel() == 0
        out.append((message.request_id, chunk.payload["text"], chunk.turn))
    return out


def test_a_reply_is_one_request_whose_text_reaches_code2wav(
    adapter: QwenTalkerSessionAdapter,
) -> None:
    chunk, payload = unit(1, [0, 1])
    assert adapter.prepare_unit(IDENTITY, chunk, payload).reset_history is True
    request_data = adapter.build(IDENTITY, chunk, payload)
    assert request_data.stage_payload.request.params["stream"] is True
    assert "stream" not in payload.request.params
    request_data.stage_payload = payload
    call = talker_session.build_talker_request_data.call_args
    assert call.kwargs["thinker_config"] is None
    sampling = call.kwargs["resolve_sampling_config"](PARAMS)
    assert sampling["codec_eos_id"] == 2150 and 2150 not in sampling["suppress_tokens"]
    assert relayed(adapter) == [("unit-1", "a", 1), ("unit-1", "b", 1)]
    adapter.append_text_chunk(request_data, token(2))
    adapter.prefill_builder.append_text_chunk.assert_called_once()
    assert relayed(adapter) == [("unit-1", "c", 1)]
    adapter.mark_thinker_done(request_data)
    adapter.prefill_builder.mark_thinker_done.assert_called_once_with(request_data)
    adapter.append_text_chunk(request_data, token(3))
    assert relayed(adapter) == []


def flagged(adapter: QwenTalkerSessionAdapter) -> list[tuple[str, bool]]:
    out = []
    while not adapter.outbox.empty():
        message = adapter.outbox.get_nowait()
        assert message.target == CODE2WAV_STAGE
        assert message.data.numel() == 0
        out.append((message.request_id, message.metadata[REPLY_TRUNCATED]))
    return out


def test_a_reply_too_short_to_speak_relays_its_text_and_builds_no_request(
    adapter: QwenTalkerSessionAdapter,
) -> None:
    chunk, payload = unit(1, [0])
    result = adapter.complete_without_generation(IDENTITY, chunk, payload)
    assert result.request.params["stream"] is True
    assert relayed(adapter) == [("unit-1", "a", 1)]
    talker_session.build_talker_request_data.assert_not_called()


def test_multimodal_prompts_get_their_positions_on_the_native_request(
    monkeypatch: pytest.MonkeyPatch, adapter: QwenTalkerSessionAdapter
) -> None:
    monkeypatch.setattr(talker_session, "talker_can_use_linear_mrope", lambda *a: False)
    positions = torch.zeros(3, 4, dtype=torch.long)
    monkeypatch.setattr(
        talker_session, "compute_mrope_positions", lambda *a: (positions, 2)
    )
    request = SimpleNamespace(origin_input_ids=[1, 2, 3, 4], multimodal_inputs=None)
    request_data = SimpleNamespace(req=request, talker_model_inputs={})
    adapter.bind_request(IDENTITY, request_data, SimpleNamespace())
    assert request.multimodal_inputs.mrope_positions is positions
    assert request.multimodal_inputs.mrope_position_delta == 2


def test_a_reply_cut_at_the_talker_token_budget_flags_it_for_code2wav(
    adapter: QwenTalkerSessionAdapter,
) -> None:
    """A reply the talker could not finish must say so, or code2wav ends it as a natural stop."""
    chunk, payload = unit(1, [])
    request_data = adapter.build(IDENTITY, chunk, payload)
    adapter.mark_thinker_done(request_data)
    request_data.finish_reason = "length"
    assert adapter.result(IDENTITY, request_data).request_id == "unit-1"
    assert flagged(adapter) == [("unit-1", True)]
    assert adapter.replies == {}


def test_a_reply_cut_before_its_thinker_ended_still_flags_it(
    adapter: QwenTalkerSessionAdapter,
) -> None:
    """The talker can use up its budget while text still streams; the flag must not wait for the thinker."""
    chunk, payload = unit(1, [])
    request_data = adapter.build(IDENTITY, chunk, payload)
    request_data.finish_reason = "length"
    adapter.result(IDENTITY, request_data)
    assert flagged(adapter) == [("unit-1", True)]
    assert adapter.replies == {}


def test_a_reply_that_stopped_on_its_own_puts_no_flag(
    adapter: QwenTalkerSessionAdapter,
) -> None:
    chunk, payload = unit(1, [])
    request_data = adapter.build(IDENTITY, chunk, payload)
    adapter.mark_thinker_done(request_data)
    request_data.finish_reason = "stop"
    adapter.result(IDENTITY, request_data)
    assert flagged(adapter) == []
