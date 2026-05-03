# SPDX-License-Identifier: Apache-2.0
"""V1 OmniScheduler inbox/outbox and streaming regression tests."""

from __future__ import annotations

import queue
from types import SimpleNamespace

from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.omni_scheduler import OmniScheduler


def _scheduler_shell(**kwargs) -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.tp_size = 1
    scheduler._aborted_request_ids = set()
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._stream_chunk_handler = kwargs.get("stream_chunk_handler")
    scheduler._stream_done_handler = kwargs.get("stream_done_handler")
    scheduler._request_builder = kwargs.get("request_builder")
    scheduler._result_adapter = kwargs.get("result_adapter")
    return scheduler


def test_v1_scheduler_buffers_stream_chunks_until_request_arrives() -> None:
    seen_chunks = []

    def request_builder(payload):
        req_data = SimpleNamespace(req=SimpleNamespace(rid=payload.request_id))
        seen_chunks.extend(payload.prefetched_chunks)
        return req_data

    scheduler = _scheduler_shell(request_builder=request_builder)
    chunk = SimpleNamespace(data="audio", metadata={"idx": 0})
    payload = SimpleNamespace(request_id="req-1")

    scheduler.inbox.put(
        IncomingMessage(request_id="req-1", type="stream_chunk", data=chunk)
    )
    scheduler.recv_requests()
    scheduler.process_input_requests([payload])

    assert seen_chunks == [chunk]
    assert scheduler._pending_stream_chunks == {}
    assert [req.rid for req in scheduler.waiting_queue] == ["req-1"]


def test_v1_scheduler_marks_prefetched_stream_done_on_late_request() -> None:
    done_seen = []

    def mark_done(req_data):
        req_data.done = True
        done_seen.append(req_data.req.rid)

    def request_builder(payload):
        return SimpleNamespace(req=SimpleNamespace(rid=payload.request_id))

    scheduler = _scheduler_shell(
        request_builder=request_builder,
        stream_done_handler=mark_done,
    )
    payload = SimpleNamespace(request_id="req-2")

    scheduler.inbox.put(IncomingMessage(request_id="req-2", type="stream_done"))
    scheduler.recv_requests()
    scheduler.process_input_requests([payload])

    assert done_seen == ["req-2"]
    assert "req-2" not in scheduler._pending_stream_done


def test_v1_scheduler_abort_drops_pending_stream_state_and_inbox_messages() -> None:
    scheduler = _scheduler_shell(request_builder=lambda payload: payload)
    scheduler._pending_stream_chunks["drop-me"] = [object()]
    scheduler._pending_stream_done.add("drop-me")
    scheduler._deferred_request_payloads["drop-me"] = object()
    scheduler.waiting_queue = [
        SimpleNamespace(rid="drop-me"),
        SimpleNamespace(rid="keep-me"),
    ]
    scheduler.inbox.put(IncomingMessage(request_id="drop-me", type="new_request"))
    scheduler.inbox.put(IncomingMessage(request_id="keep-me", type="new_request"))

    scheduler.abort("drop-me")

    assert "drop-me" in scheduler._aborted_request_ids
    assert "drop-me" not in scheduler._pending_stream_chunks
    assert "drop-me" not in scheduler._pending_stream_done
    assert "drop-me" not in scheduler._deferred_request_payloads
    assert [req.rid for req in scheduler.waiting_queue] == ["keep-me"]
    retained = scheduler.inbox.get_nowait()
    assert retained.request_id == "keep-me"
    assert scheduler.inbox.empty()


def test_v1_scheduler_stream_output_emits_result_outbox_message() -> None:
    def result_adapter(data):
        return {"ids": data.output_ids, "finish_reason": data.finish_reason}

    scheduler = _scheduler_shell(result_adapter=result_adapter)
    req_data = SimpleNamespace(output_ids=[], finish_reason=None)
    req = SimpleNamespace(
        rid="req-result",
        output_ids=[10, 11],
        finished=lambda: True,
        finished_reason=SimpleNamespace(to_json=lambda: {"type": "stop"}),
        _omni_data=req_data,
    )

    scheduler.stream_output([req])

    msg = scheduler.outbox.get_nowait()
    assert msg.request_id == "req-result"
    assert msg.type == "result"
    assert msg.data == {"ids": [10, 11], "finish_reason": "stop"}
    assert scheduler.outbox.empty()
