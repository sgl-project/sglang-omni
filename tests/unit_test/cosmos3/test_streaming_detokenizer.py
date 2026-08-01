# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import torch

from sglang_omni.models.cosmos3.components.streaming_detokenizer import (
    Cosmos3StreamingDetokenizer,
)
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    pieces: ClassVar[dict[int, str]] = {1: "hello", 2: " ", 3: "world"}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)


def _payload(*, stream: bool) -> StagePayload:
    state = Cosmos3PipelineState(
        prompt={
            "input_ids": torch.tensor([8, 9]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "prompt_text": "prompt",
        },
        text_out={
            "output_ids": [1, 2, 3],
            "finish_reason": "stop",
            "is_final": True,
        },
    )
    return StagePayload(
        request_id="decode-request",
        request=OmniRequest(inputs=None, params={"stream": stream}),
        data=state.to_dict(),
    )


def test_non_streaming_payload_decodes_complete_text() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_new_request("decode-request", _payload(stream=False))
    message = scheduler.outbox.get_nowait()

    assert message.type == "result"
    assert message.data.data == {
        "modality": "text",
        "text": "hello world",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }


def test_streaming_tokens_emit_deltas_and_slim_terminal_result() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())
    payload = _payload(stream=True)

    scheduler._on_stream_chunk(
        "decode-request", SimpleNamespace(data=torch.tensor([1]))
    )
    stream_message = scheduler.outbox.get_nowait()
    assert stream_message.type == "stream"
    assert stream_message.data["text"] == "hello"

    scheduler._on_new_request("decode-request", payload)
    assert scheduler.outbox.empty()
    scheduler._on_stream_done("decode-request")
    result_message = scheduler.outbox.get_nowait()

    assert result_message.type == "result"
    assert "text" not in result_message.data.data
    assert result_message.data.data["usage"]["completion_tokens"] == 3


def test_stream_done_before_terminal_payload_does_not_deadlock() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_stream_done("decode-request")
    scheduler._on_new_request("decode-request", _payload(stream=True))

    assert scheduler.outbox.get_nowait().type == "result"
