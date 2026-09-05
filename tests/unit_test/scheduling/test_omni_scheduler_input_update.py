# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import deque
from queue import Queue
from types import SimpleNamespace

from sglang_omni.proto.messages import InputUpdateMessage
from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _update(
    request_id: str = "request-1",
    *,
    seq_no: int = 0,
) -> InputUpdateMessage:
    return InputUpdateMessage(
        request_id=request_id,
        session_id="session-1",
        turn_id="turn-1",
        seq_no=seq_no,
        token_ids=(7,),
        byte_count=1,
    )


def _scheduler() -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler.inbox = Queue()
    scheduler.outbox = Queue()
    scheduler.tp_size = 1
    scheduler.is_entry_rank = True
    scheduler._request_admission_lock = threading.RLock()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._abort_callback = None
    scheduler._mark_running_request_aborted = lambda request_id: False
    scheduler._release_immediate_request_resources = lambda request_id: None
    return scheduler


def test_recv_requests_broadcasts_and_dispatches_input_updates_in_order(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    scheduler.tp_size = 2
    scheduler.tp_group = SimpleNamespace(rank=0, ranks=[0, 1])
    scheduler.tp_cpu_group = object()
    received: list[InputUpdateMessage] = []
    scheduler._on_input_update = lambda request_id, message: received.append(message)

    first = _update(seq_no=0)
    second = _update(seq_no=1)
    payload = SimpleNamespace(request_id="request-1")
    scheduler.inbox.put(IncomingMessage("request-1", "input_update", first))
    scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload))
    scheduler.inbox.put(IncomingMessage("request-1", "input_update", second))
    broadcasts: list[list[IncomingMessage]] = []

    def _broadcast(messages, rank, group, *, src):
        assert rank == 0
        assert group is scheduler.tp_cpu_group
        assert src == 0
        broadcasts.append(messages)
        return messages

    monkeypatch.setattr(omni_scheduler_module, "broadcast_pyobj", _broadcast)

    assert scheduler.recv_requests() == [payload]
    assert received == [first, second]
    assert [message.type for message in broadcasts[0]] == [
        "input_update",
        "new_request",
        "input_update",
    ]


def test_unsupported_input_update_fails_only_target_request() -> None:
    scheduler = _scheduler()
    keep = SimpleNamespace(request_id="request-keep")
    scheduler.inbox.put(
        IncomingMessage("request-bad", "input_update", _update("request-bad"))
    )
    scheduler.inbox.put(IncomingMessage("request-keep", "new_request", keep))

    assert scheduler.recv_requests() == [keep]
    output = scheduler.outbox.get_nowait()
    assert output.request_id == "request-bad"
    assert output.type == "error"
    assert "does not support" in str(output.data)
    assert "request-bad" in scheduler._aborted_request_ids
    assert "request-keep" not in scheduler._aborted_request_ids
