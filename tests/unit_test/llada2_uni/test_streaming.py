# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.llada2_uni.components.streaming_detokenizer import (
    LLaDA2StreamingDetokenizeScheduler,
)
from sglang_omni.models.llada2_uni.config import LLaDA2UniPipelineConfig
from sglang_omni.models.llada2_uni.request_builders import (
    make_dllm_thinker_stream_output_builder,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload, StreamMessage
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class _ByteTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        *,
        special_token_ids: set[int] | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self._vocab = vocab
        self._special = special_token_ids or set()
        self.eos_token_id = eos_token_id

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        chunks = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self._special:
                continue
            chunks.append(self._vocab[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")


def _payload(
    *,
    stream: bool,
    output_ids: list[int] | None = None,
    finish_reason: str = "stop",
    finish_reason_data: dict[str, object] | None = None,
) -> StagePayload:
    output_ids = list(output_ids or [])
    thinker_out = {
        "output_ids": output_ids,
        "is_final": True,
        "finish_reason": finish_reason,
    }
    if finish_reason_data is not None:
        thinker_out["finish_reason_data"] = finish_reason_data
    return StagePayload(
        request_id="req",
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={
            "prompt": {"input_ids": torch.tensor([[101, 102]])},
            "thinker_out": thinker_out,
        },
    )


def _stream_item(data: object, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=data,
        from_stage="thinker",
        metadata={"modality": "text"},
    )


def _drain(scheduler: LLaDA2StreamingDetokenizeScheduler) -> list[OutgoingMessage]:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


class _FakeCoordinator:
    def __init__(self, messages: list[StreamMessage | CompleteMessage]) -> None:
        self._messages = messages

    async def stream(self, request_id, omni_request):
        del request_id, omni_request
        for message in self._messages:
            yield message


def test_llada2_streaming_topology_routes_thinker_to_decode() -> None:
    config = LLaDA2UniPipelineConfig(model_path="dummy")
    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    decode = next(stage for stage in config.stages if stage.name == "decode")

    assert thinker.stream_to == ["decode"]
    assert decode.can_accept_stream_before_payload is True


def test_dllm_stream_builder_emits_only_for_streaming_requests() -> None:
    builder = make_dllm_thinker_stream_output_builder()
    req_data = SimpleNamespace(stage_payload=_payload(stream=False))

    assert builder("req", req_data, [1, 2]) == []

    req_data.stage_payload = _payload(stream=True)
    messages = builder("req", req_data, [1, 2])

    assert len(messages) == 1
    assert messages[0].target == "decode"
    assert messages[0].metadata == {"modality": "text"}
    assert torch.equal(messages[0].data, torch.tensor([1, 2]))


def test_accepted_blocks_emit_append_only_text_and_slim_final() -> None:
    tokenizer = _ByteTokenizer({1: b"hello", 2: b" ", 3: b"world"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    payload = _payload(stream=True, output_ids=[1, 2, 3])
    scheduler._on_streaming_new_request("req", payload)

    scheduler._on_chunk("req", _stream_item(torch.tensor([1, 2])))
    scheduler._on_chunk("req", _stream_item(torch.tensor([3]), chunk_id=1))
    scheduler._on_done("req")

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream", "result"]
    assert [message.data["text"] for message in messages[:-1]] == ["hello ", "world"]
    result = messages[-1].data.data
    assert "text" not in result
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert result["events"][0]["payload"]["text"] == "hello world"


def test_utf8_sequence_can_span_accepted_blocks() -> None:
    tokenizer = _ByteTokenizer({1: b"\xe4", 2: b"\xbd", 3: b"\xa0"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    scheduler._on_streaming_new_request(
        "req", _payload(stream=True, output_ids=[1, 2, 3])
    )

    scheduler._on_chunk("req", _stream_item(torch.tensor([1, 2])))
    assert _drain(scheduler) == []

    scheduler._on_chunk("req", _stream_item(torch.tensor([3]), chunk_id=1))
    messages = _drain(scheduler)
    assert len(messages) == 1
    assert messages[0].data["text"] == "\u4f60"


def test_eos_and_special_tokens_do_not_emit_text() -> None:
    tokenizer = _ByteTokenizer(
        {1: b"ok", 2: b"<eos>", 3: b"<special>"},
        special_token_ids={2, 3},
        eos_token_id=2,
    )
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=2)
    scheduler._on_streaming_new_request("req", _payload(stream=True, output_ids=[1, 2]))

    scheduler._on_chunk("req", _stream_item(torch.tensor([1, 3, 2])))

    messages = _drain(scheduler)
    assert len(messages) == 1
    assert messages[0].data["text"] == "ok"


def test_done_before_payload_finalizes_when_payload_arrives() -> None:
    tokenizer = _ByteTokenizer({1: b"ready"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)

    scheduler._on_chunk("req", _stream_item(torch.tensor([1])))
    scheduler._on_done("req")
    assert [message.type for message in _drain(scheduler)] == ["stream"]

    scheduler._on_streaming_new_request("req", _payload(stream=True, output_ids=[1]))
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["result"]
    assert "req" not in scheduler._text_states


def test_terminal_payload_flushes_missing_stream_suffix() -> None:
    tokenizer = _ByteTokenizer({1: b"first", 2: b" second"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    scheduler._on_streaming_new_request("req", _payload(stream=True, output_ids=[1, 2]))
    scheduler._on_chunk("req", _stream_item(torch.tensor([1])))
    _drain(scheduler)

    scheduler._on_done("req")

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert messages[0].data["text"] == " second"


def test_non_streaming_request_preserves_full_text_result() -> None:
    tokenizer = _ByteTokenizer({1: b"full", 2: b" text"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    message = IncomingMessage(
        request_id="req",
        type="new_request",
        data=_payload(stream=False, output_ids=[1, 2]),
    )

    scheduler._handle_new_request_batch([message])

    messages = _drain(scheduler)
    assert len(messages) == 1
    assert messages[0].type == "result"
    assert messages[0].data.data["text"] == "full text"


def test_non_streaming_request_trims_matched_stop_string() -> None:
    tokenizer = _ByteTokenizer(
        {1: b"The ", 2: b"uppercase", 3: b" English", 4: b" ignored"}
    )
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    message = IncomingMessage(
        request_id="req",
        type="new_request",
        data=_payload(
            stream=False,
            output_ids=[1, 2, 3, 4],
            finish_reason_data={"type": "stop", "matched": "uppercase English"},
        ),
    )

    scheduler._handle_new_request_batch([message])

    messages = _drain(scheduler)
    assert len(messages) == 1
    assert messages[0].data.data["text"] == "The "
    assert messages[0].data.data["usage"]["completion_tokens"] == 4


def test_non_streaming_request_trims_matched_stop_token() -> None:
    tokenizer = _ByteTokenizer({1: b"visible", 2: b"<stop>"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    message = IncomingMessage(
        request_id="req",
        type="new_request",
        data=_payload(
            stream=False,
            output_ids=[1, 2],
            finish_reason_data={"type": "stop", "matched": 2},
        ),
    )

    scheduler._handle_new_request_batch([message])

    messages = _drain(scheduler)
    assert messages[0].data.data["text"] == "visible"


def test_length_finish_does_not_trim_matching_text() -> None:
    tokenizer = _ByteTokenizer({1: b"before ", 2: b"stop", 3: b" after"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    message = IncomingMessage(
        request_id="req",
        type="new_request",
        data=_payload(
            stream=False,
            output_ids=[1, 2, 3],
            finish_reason="length",
            finish_reason_data={"type": "length", "length": 3},
        ),
    )

    scheduler._handle_new_request_batch([message])

    messages = _drain(scheduler)
    assert messages[0].data.data["text"] == "before stop after"


def test_streaming_terminal_stop_only_flushes_text_before_match() -> None:
    tokenizer = _ByteTokenizer(
        {1: b"The ", 2: b"uppercase", 3: b" English", 4: b" ignored"}
    )
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    scheduler._on_streaming_new_request(
        "req",
        _payload(
            stream=True,
            output_ids=[1, 2, 3, 4],
            finish_reason_data={"type": "stop", "matched": "uppercase English"},
        ),
    )

    scheduler._on_done("req")

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert messages[0].data["text"] == "The "
    result = messages[1].data.data
    assert "text" not in result
    assert result["events"][0]["payload"]["text"] == "The "


def test_streaming_terminal_stop_preserves_emitted_safe_prefix() -> None:
    tokenizer = _ByteTokenizer({1: b"safe ", 2: b"stop", 3: b" here"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    scheduler._on_streaming_new_request(
        "req",
        _payload(
            stream=True,
            output_ids=[1, 2, 3],
            finish_reason_data={"type": "stop", "matched": "stop here"},
        ),
    )
    scheduler._on_chunk("req", _stream_item(torch.tensor([1])))
    first_messages = _drain(scheduler)
    assert [message.data["text"] for message in first_messages] == ["safe "]

    scheduler._on_done("req")

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["result"]
    assert messages[0].data.data["events"][0]["payload"]["text"] == "safe "


def test_client_stream_aggregates_llada_deltas_once() -> None:
    tokenizer = _ByteTokenizer({1: b"hello", 2: b" ", 3: b"world"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    scheduler._on_streaming_new_request(
        "req", _payload(stream=True, output_ids=[1, 2, 3])
    )
    scheduler._on_chunk("req", _stream_item(torch.tensor([1, 2])))
    scheduler._on_chunk("req", _stream_item(torch.tensor([3]), chunk_id=1))
    scheduler._on_done("req")

    coordinator_messages: list[StreamMessage | CompleteMessage] = []
    for message in _drain(scheduler):
        if message.type == "stream":
            coordinator_messages.append(
                StreamMessage(
                    request_id=message.request_id,
                    from_stage="decode",
                    chunk=message.data,
                    stage_name="decode",
                    modality="text",
                )
            )
        else:
            coordinator_messages.append(
                CompleteMessage(
                    request_id=message.request_id,
                    from_stage="decode",
                    success=True,
                    result=message.data.data,
                )
            )

    client = Client(coordinator=_FakeCoordinator(coordinator_messages))

    async def collect():
        return [
            chunk
            async for chunk in client.completion_stream(
                GenerateRequest(prompt="ignored", stream=True),
                request_id="req",
            )
        ]

    chunks = asyncio.run(collect())

    assert "".join(chunk.text or "" for chunk in chunks) == "hello world"
    assert chunks[-1].text in (None, "")
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 3


def test_abort_clears_incremental_text_state() -> None:
    tokenizer = _ByteTokenizer({1: b"partial"})
    scheduler = LLaDA2StreamingDetokenizeScheduler(tokenizer, eos_token_id=None)
    scheduler._on_chunk("req", _stream_item(torch.tensor([1])))

    scheduler.abort("req")

    assert "req" not in scheduler._text_states
