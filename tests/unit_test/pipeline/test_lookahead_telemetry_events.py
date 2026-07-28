# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the thinker lookahead telemetry events.

Drives the real ``_event_loop_async_decode`` / ``_run_batch_launch`` /
``_run_batch_resolve`` with stubbed heavy deps (same pattern as
``test_async_decode.py``) and asserts the three event types reach the
request-event recorder — and that nothing is emitted when the recorder
is inactive.
"""

from __future__ import annotations

import json
import queue
import threading
import types

import pytest

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class _FakeBatch:
    def __init__(self, n):
        self.reqs = [
            types.SimpleNamespace(rid=f"req-{i}", finished=lambda: False)
            for i in range(n)
        ]

    def copy(self):
        return self


def _read_events(tmp_path):
    events = []
    for f in tmp_path.glob("events_*.jsonl"):
        for line in f.read_text().splitlines():
            events.append(json.loads(line))
    return events


@pytest.fixture
def recorder(tmp_path):
    rec = get_recorder()
    rec.start("test", str(tmp_path), "thinker")
    yield tmp_path
    rec.stop()


def _new_loop_scheduler(events_taken, min_bs=2):
    s = OmniScheduler.__new__(OmniScheduler)
    s._admin_lock = threading.Lock()
    s._admin_queue = queue.Queue()
    s.chunked_req = None
    s.is_mixed_chunk = False
    s.page_size = 1
    s.running_batch = types.SimpleNamespace(batch_is_full=False)
    s.server_args = types.SimpleNamespace(disable_radix_cache=False)
    s.token_to_kv_pool_allocator = types.SimpleNamespace(free=lambda _: None)
    s.waiting_queue = []
    s._running = True
    s._engine_paused = False
    s._async_pending = None
    s.async_decode_min_batch_size = min_bs
    s.cur_batch = None
    s.last_batch = None
    s.recv_requests = lambda: []
    s._take_deferred_request_payloads = lambda: []
    s.process_input_requests = lambda r: None
    s._batch_is_decode = lambda b: True
    s.self_check_during_idle = lambda: events_taken.append("idle")
    s.self_check_during_busy = lambda: None
    s._run_batch_launch = lambda b: ("sched_output", "pending_step")
    s._resolve_and_process = lambda pb, ps, pstep: events_taken.append("resolve")
    s._resolve_pending_async = OmniScheduler._resolve_pending_async.__get__(s)
    s.run_batch = lambda b: object()
    s.process_batch_result = lambda b, r: None
    return s


def _drive(s, seq):
    batches = [None if n is None else _FakeBatch(n) for n in seq]
    state = {"i": 0}

    def gnb():
        i = state["i"]
        state["i"] += 1
        if i >= len(batches) - 1:
            s._running = False
        return batches[i] if i < len(batches) else None

    s.get_next_batch_to_run = gnb
    s._event_loop_async_decode()


def test_decision_events_emitted_with_routing_outcome(recorder):
    taken = []
    s = _new_loop_scheduler(taken, min_bs=2)
    _drive(s, [1, 2, 2, None])

    decisions = [
        e
        for e in _read_events(recorder)
        if e["event_name"] == "thinker_lookahead_decision"
    ]
    # one decision per decode batch (idle iteration emits nothing)
    assert [d["metadata"]["bs"] for d in decisions] == [1, 2, 2]
    assert [d["metadata"]["use_lookahead"] for d in decisions] == [
        False,  # bs1 below min_bs -> fast path
        True,
        True,
    ]
    assert all(d["metadata"]["min_bs"] == 2 for d in decisions)
    assert all(d["stage"] == "thinker" for d in decisions)
    assert decisions[0]["request_id"] == "req-0"


def test_launch_and_resolve_events_carry_bs_and_query_counters(recorder):
    s = OmniScheduler.__new__(OmniScheduler)
    s.forward_ct = 0
    s._emit_prefill_start_for_batch = lambda b: None  # not under test
    s._build_sched_output = lambda b: "sched_output"
    s._emit_stream_output = lambda so, mo, skip_rids=(): None
    s._req_is_retracted = lambda r: False
    s._model_runner = types.SimpleNamespace(
        execute_launch=lambda so: "pending_step",
        execute_resolve=lambda p: types.SimpleNamespace(can_run_cuda_graph=True),
        _async_query_hit=7,
        _async_query_miss=3,
    )
    batch = _FakeBatch(4)
    sched_output, pending = s._run_batch_launch(batch)
    pending = types.SimpleNamespace(
        batch_result=types.SimpleNamespace(next_token_ids=object()),
        scheduler_output=types.SimpleNamespace(requests=[]),
    )
    s._run_batch_resolve(batch, sched_output, pending)

    events = _read_events(recorder)
    launches = [e for e in events if e["event_name"] == "thinker_lookahead_launch"]
    resolves = [e for e in events if e["event_name"] == "thinker_lookahead_resolve"]
    assert len(launches) == 1 and launches[0]["metadata"]["bs"] == 4
    assert len(resolves) == 1
    assert resolves[0]["metadata"]["query_hit_total"] == 7
    assert resolves[0]["metadata"]["query_miss_total"] == 3


def test_no_events_and_no_errors_when_recorder_inactive(tmp_path):
    assert not get_recorder().is_active()
    taken = []
    s = _new_loop_scheduler(taken, min_bs=2)
    _drive(s, [1, 2, None])  # must not raise, must not write anything
    assert list(tmp_path.glob("events_*.jsonl")) == []
