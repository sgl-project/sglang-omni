# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import torch

from sglang_omni.models.cosmos3.components.streaming_detokenizer import (
    _STATE_MAX,
    Cosmos3StreamingDetokenizer,
)
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage


class _FakeTokenizer:
    pieces: ClassVar[dict[int, str]] = {
        1: "hello",
        2: " ",
        3: "world",
        5: "hi###�",
        6: "�",
        10: " gam",
        11: "ma",
    }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)


def _payload(
    *,
    stream: bool,
    request_id: str = "decode-request",
    matched_stop: int | str | None = None,
) -> StagePayload:
    text_out: dict[str, Any] = {
        "output_ids": [1, 2, 3],
        "finish_reason": "stop",
        "is_final": True,
    }
    if matched_stop is not None:
        text_out["matched_stop"] = matched_stop
    state = Cosmos3PipelineState(
        prompt={
            "input_ids": torch.tensor([8, 9]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "prompt_text": "prompt",
        },
        text_out=text_out,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params={"stream": stream}),
        data=state.to_dict(),
    )


def _stream_item(
    token_id: int = 1, metadata: dict[str, Any] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(data=torch.tensor([token_id]), metadata=metadata)


def _terminal_flush_item(token_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        data=torch.tensor(token_ids), metadata={"terminal_flush": True}
    )


def _drain_outbox(
    scheduler: Cosmos3StreamingDetokenizer,
) -> list[OutgoingMessage]:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


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


def test_non_streaming_trims_matched_stop_string() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_new_request(
        "decode-request",
        _payload(stream=False, matched_stop="world"),
    )
    message = scheduler.outbox.get_nowait()

    assert message.data.data["text"] == "hello "
    assert message.data.data["usage"]["completion_tokens"] == 3


def test_non_streaming_trims_matched_stop_token_id() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_new_request(
        "decode-request",
        _payload(stream=False, matched_stop=3),
    )
    message = scheduler.outbox.get_nowait()

    assert message.data.data["text"] == "hello "
    assert message.data.data["usage"]["completion_tokens"] == 3


def test_streaming_terminal_flush_trims_matched_stop_string() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_stream_chunk("decode-request", _stream_item(5))
    assert scheduler.outbox.empty()

    scheduler._on_new_request(
        "decode-request",
        _payload(stream=True, matched_stop="###"),
    )
    scheduler._on_stream_done("decode-request")

    messages = _drain_outbox(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert messages[0].data["text"] == "hi"


def test_streaming_terminal_flush_drops_matched_stop_token_id() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_stream_chunk("decode-request", _stream_item(6))
    assert scheduler.outbox.empty()

    scheduler._on_new_request(
        "decode-request",
        _payload(stream=True, matched_stop=6),
    )
    scheduler._on_stream_done("decode-request")

    messages = _drain_outbox(scheduler)
    assert [message.type for message in messages] == ["result"]


def test_terminal_flush_chunk_buffers_and_matched_stop_is_never_streamed() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_stream_chunk("decode-request", _stream_item(1))
    live_message = scheduler.outbox.get_nowait()
    assert live_message.data["text"] == "hello"

    scheduler._on_stream_chunk("decode-request", _terminal_flush_item([10, 11]))
    assert scheduler.outbox.empty()

    scheduler._on_new_request(
        "decode-request",
        _payload(stream=True, matched_stop=" gamma"),
    )
    scheduler._on_stream_done("decode-request")

    messages = _drain_outbox(scheduler)
    assert [message.type for message in messages] == ["result"]


def test_terminal_flush_chunk_emits_held_text_without_matched_stop() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_stream_chunk("decode-request", _terminal_flush_item([10]))
    assert scheduler.outbox.empty()

    scheduler._on_new_request("decode-request", _payload(stream=True))
    scheduler._on_stream_done("decode-request")

    messages = _drain_outbox(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert messages[0].data["text"] == " gam"


def test_streaming_tokens_emit_deltas_and_slim_terminal_result() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())
    payload = _payload(stream=True)

    scheduler._on_stream_chunk("decode-request", _stream_item())
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


def test_abort_then_stream_chunk_does_not_resurrect() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())
    request_id = "aborted-request"

    scheduler.abort(request_id)
    scheduler._on_stream_chunk(request_id, _stream_item())
    scheduler._on_stream_done(request_id)

    assert request_id in scheduler._aborted
    assert request_id not in scheduler._state
    assert request_id not in scheduler._done_seen
    assert scheduler.outbox.empty()


def test_abort_then_new_request_does_not_resume() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())
    request_id = "aborted-request"

    scheduler.abort(request_id)
    scheduler._on_new_request(
        request_id,
        _payload(stream=True, request_id=request_id),
    )
    scheduler._on_stream_chunk(request_id, _stream_item())
    scheduler._on_stream_done(request_id)

    assert request_id not in scheduler._state
    assert request_id not in scheduler._done_seen
    assert scheduler.outbox.empty()


def test_abort_clears_state_and_done_seen() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    scheduler._on_stream_chunk("state-request", _stream_item())
    _drain_outbox(scheduler)
    assert "state-request" in scheduler._state
    scheduler.abort("state-request")
    assert "state-request" not in scheduler._state
    assert "state-request" in scheduler._aborted

    scheduler._on_stream_done("done-request")
    assert "done-request" in scheduler._done_seen
    scheduler.abort("done-request")
    assert "done-request" not in scheduler._done_seen
    assert "done-request" in scheduler._aborted


def test_eviction_spares_live_and_done_entries() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    for index in range(_STATE_MAX):
        scheduler._ensure_state(f"request-{index}")
    now = time.monotonic()
    for index in range(5000):
        scheduler._state[f"request-{index}"].last_seen = now - 1000.0
    for index in range(5000, 5100):
        state = scheduler._state[f"request-{index}"]
        state.last_seen = now - 1000.0
        state.done = True

    scheduler._ensure_state("trigger")

    assert all(
        f"request-{index}" not in scheduler._state for index in range(0, 5000, 499)
    )
    assert all(f"request-{index}" in scheduler._state for index in range(5000, 5100))
    assert "request-9999" in scheduler._state
    assert "trigger" in scheduler._state
    assert len(scheduler._state) == _STATE_MAX + 1 - 5000


def test_eviction_throttled() -> None:
    scheduler = Cosmos3StreamingDetokenizer(_FakeTokenizer())

    for index in range(_STATE_MAX):
        scheduler._ensure_state(f"request-{index}")
    scheduler._state["request-0"].last_seen = time.monotonic() - 1000.0
    scheduler._last_evict_s = time.monotonic()

    scheduler._ensure_state("trigger-1")
    scheduler._ensure_state("trigger-2")
    assert "request-0" in scheduler._state

    scheduler._last_evict_s -= 1.0
    scheduler._ensure_state("trigger-3")
    assert "request-0" not in scheduler._state


def test_abort_waits_for_in_flight_handler_then_blocks_late_messages() -> None:
    in_decode = threading.Event()
    release_decode = threading.Event()
    abort_started = threading.Event()
    abort_done = threading.Event()

    class _BlockingTokenizer(_FakeTokenizer):
        def decode(
            self,
            token_ids: Any,
            skip_special_tokens: bool = True,
        ) -> str:
            in_decode.set()
            if not release_decode.wait(timeout=2.0):
                raise TimeoutError("test did not release tokenizer decode")
            return super().decode(token_ids, skip_special_tokens=skip_special_tokens)

    scheduler = Cosmos3StreamingDetokenizer(_BlockingTokenizer())
    request_id = "concurrent-abort"

    handler_thread = threading.Thread(
        target=scheduler._on_stream_chunk,
        args=(request_id, _stream_item()),
    )

    def run_abort() -> None:
        abort_started.set()
        scheduler.abort(request_id)
        abort_done.set()

    abort_thread = threading.Thread(target=run_abort)
    handler_thread.start()
    assert in_decode.wait(timeout=1.0)
    abort_thread.start()
    assert abort_started.wait(timeout=1.0)
    try:
        assert not abort_done.wait(timeout=0.1)
    finally:
        release_decode.set()

    handler_thread.join(timeout=2.0)
    abort_thread.join(timeout=2.0)
    assert not handler_thread.is_alive()
    assert not abort_thread.is_alive()
    assert abort_done.is_set()
    assert request_id in scheduler._aborted
    assert request_id not in scheduler._state

    before_late_messages = _drain_outbox(scheduler)
    assert [message.type for message in before_late_messages] == ["stream"]

    scheduler._on_stream_chunk(request_id, _stream_item())
    scheduler._on_new_request(
        request_id,
        _payload(stream=False, request_id=request_id),
    )
    scheduler._on_stream_done(request_id)

    assert request_id not in scheduler._state
    assert request_id not in scheduler._done_seen
    assert scheduler.outbox.empty()
