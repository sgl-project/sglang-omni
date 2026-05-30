# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


def _payload(request_id: str, *, stream: bool = False) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={"request_id": request_id},
    )


class _TestStreamingScheduler(StreamingSimpleScheduler):
    def __init__(self, *, max_batch_size: int = 4, max_batch_wait_ms: int = 0):
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.stream_state: set[str] = set()
        super().__init__(
            self._compute,
            batch_compute_fn=self._compute_batch,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return bool(payload.request.params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        del payload
        self.stream_state.add(request_id)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        self.stream_state.add(request_id)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data={"chunk": item.data},
                metadata={"modality": "test"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        return [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data={"done": request_id},
            )
        ]

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_state.discard(request_id)

    def _compute(self, payload: StagePayload) -> StagePayload:
        self.single_calls.append(payload.request_id)
        payload.data = {"single": payload.request_id}
        return payload

    def _compute_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        self.batch_calls.append([payload.request_id for payload in payloads])
        for payload in payloads:
            payload.data = {"batch": payload.request_id}
        return payloads


def _drain_results(scheduler: StreamingSimpleScheduler) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def test_streaming_simple_scheduler_batches_non_streaming_requests() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    scheduler.inbox.put(IncomingMessage("b", "new_request", _payload("b")))
    scheduler.inbox.put(IncomingMessage("c", "new_request", _payload("c")))

    batch = scheduler._collect_new_request_batch(first)
    scheduler._handle_new_request_batch(batch)

    assert scheduler.batch_calls == [["a", "b", "c"]]
    assert [msg.request_id for msg in _drain_results(scheduler)] == ["a", "b", "c"]


def test_streaming_simple_scheduler_keeps_streaming_request_out_of_batch() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    scheduler.inbox.put(
        IncomingMessage("stream", "new_request", _payload("stream", stream=True))
    )
    scheduler.inbox.put(IncomingMessage("b", "new_request", _payload("b")))

    batch = scheduler._collect_new_request_batch(first)

    assert [msg.request_id for msg in batch] == ["a"]
    assert scheduler._next_message().request_id == "stream"
    assert scheduler._next_message().request_id == "b"


def test_streaming_simple_scheduler_done_before_payload_finalizes_later() -> None:
    scheduler = _TestStreamingScheduler()

    scheduler._on_done("req")
    scheduler._on_streaming_new_request("req", _payload("req", stream=True))

    out = scheduler.outbox.get_nowait()
    assert out.type == "result"
    assert out.data == {"done": "req"}
    assert "req" not in scheduler._pending_done
    assert "req" not in scheduler.stream_state


def test_streaming_simple_scheduler_ignores_late_non_streaming_done() -> None:
    scheduler = _TestStreamingScheduler()

    scheduler._handle_new_request_batch(
        [IncomingMessage("req", "new_request", _payload("req", stream=False))]
    )
    scheduler._on_done("req")

    assert scheduler.outbox.get_nowait().type == "result"
    assert "req" not in scheduler._pending_done


def test_streaming_simple_scheduler_abort_clears_all_stream_state() -> None:
    scheduler = _TestStreamingScheduler()
    scheduler._stream_payloads["req"] = _payload("req", stream=True)
    scheduler._pending_done.add("req")
    scheduler.stream_state.add("req")

    scheduler.abort("req")

    assert "req" not in scheduler._stream_payloads
    assert "req" not in scheduler._pending_done
    assert "req" not in scheduler.stream_state
    assert "req" in scheduler._aborted_request_ids


def test_streaming_simple_scheduler_keeps_queued_control_message_out_of_batch() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    chunk = StreamItem(chunk_id=0, data="x", from_stage="source")
    scheduler.inbox.put(IncomingMessage("stream", "stream_chunk", chunk))
    scheduler.inbox.put(IncomingMessage("b", "new_request", _payload("b")))

    batch = scheduler._collect_new_request_batch(first)

    assert [msg.request_id for msg in batch] == ["a"]
    next_msg = scheduler._next_message()
    assert next_msg.request_id == "stream"
    assert next_msg.type == "stream_chunk"
    assert scheduler._next_message().request_id == "b"
