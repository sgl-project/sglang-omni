# SPDX-License-Identifier: Apache-2.0
from array import array
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionRef, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class TestAdapter(ARSessionAdapter):
    __test__ = False

    def __init__(self, **hooks):
        self.__dict__.update(hooks)


def payload(op, rid="r", epoch=0):
    return StagePayload(
        request_id=rid,
        request=OmniRequest(
            inputs=None,
            metadata={
                "omni_session": {
                    "op": op,
                    "ref": asdict(SessionRef("s", epoch=epoch)),
                    "stages": ["ar"],
                    "chunk": asdict(TimedChunk("text", 0, 0, 0, "x")),
                    "output_limits": {"chunks": 2, "bytes": 1024},
                }
            },
        ),
        data={"relayed": True},
    )


def data(rid="r", ids=(1, 2)):
    params = SamplingParams(max_new_tokens=2, temperature=0)
    return SGLangARRequestData(
        req=Req(rid, None, array("q", ids), params, vocab_size=32)
    )


class BridgeHarness:
    def __init__(self, monkeypatch):
        from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

        self.scheduler = s = _construct_omni_scheduler(monkeypatch)
        s.model_config.vocab_size = 32
        s.max_running_requests = 2
        s.server_args.mem_fraction_static = None
        s.tree_cache = SimpleNamespace(
            release_session=lambda _: self.events.append("slot released"),
            release_radix_session=lambda _: None,
            slots={},
        )
        s.session_controller = SessionController(s.tree_cache)
        self.bridge = ARSessionBridge(s, TestAdapter())
        s._session_bridge = self.bridge
        self.events = []
        s._release_request_kv_cache = lambda req: self.events.append("release")
        s._shutdown_callback = lambda: self.events.append("resources")
        s._event_loop_normal = lambda: None

    def native(self, sid="s"):
        return self.scheduler.session_controller.get(sid)

    def inflight(self, sid="s"):
        return self.native(sid)._inflight

    def occupy_native(self):
        self.native()._inflight = True

    def append(self, rid="r", ids=(1, 2), sid="s"):
        p = payload("append", rid)
        p.request.metadata["omni_session"]["ref"]["session_id"] = sid
        self.bridge.accept(p)
        d = data(rid, ids)
        self.bridge.materialize(p, d)
        return d

    def finish(self, d):
        d.req.output_ids = array("q", [3, 4])
        d.req.finished_reason = FINISH_LENGTH(2)
        d.req.session.finish_req(d.req)
        self.bridge.complete(d.req.rid)

    def trace_drain(self, callback=None):
        self.scheduler._resolve_pending_async = callback or (
            lambda: self.events.append("drain")
        )

    def retain_slot(self, d, row=None):
        from sglang.srt.session.streaming_session import SessionSlot

        slot = SessionSlot()
        slot.kv.req_pool_idx = row
        slot.kv.kv_allocated_len = 2
        slot.restore_to_req(d.req)
        self.scheduler.tree_cache.slots["s"] = slot
        return slot

    def capacity(self, rows, tokens):
        s = self.scheduler
        s.tree_cache.evictable_size = lambda: 0
        s.req_to_token_pool = SimpleNamespace(free_slots=list(range(rows)))
        s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: tokens)

    def enqueue(self, d, queued=False):
        self.bridge.requests[d.req.rid].enqueued = True
        d.req._omni_data = d
        d.req._omni_terminal_claimed = False
        if queued:
            self.scheduler.waiting_queue = [d.req]
        else:
            batch = SimpleNamespace(reqs=[d.req])
            self.scheduler.running_batch = batch
            self.scheduler.cur_batch = self.scheduler.last_batch = batch

    def reject_append(self, reason):
        s = self.scheduler

        def build(ref, chunk, p):
            d = data(p.request_id)
            d.enforce_request_limits = reason == "input_length"
            if reason == "priority":
                d.req.priority = 1
            return d

        self.bridge.adapter = TestAdapter(build=build)
        # Note (Junnan Li): Admission must fail after native history is materialized.
        self.bridge.capacity_error = lambda rid: None
        s._release_request_kv_cache = lambda req: pytest.fail("unadmitted KV released")
        s.max_req_len = 3 if reason == "length" else 63
        s.max_req_input_len = 3 if reason == "input_length" else 63
        s.abort_on_priority_when_disabled = reason == "priority"
        if reason == "queue":
            s.max_queued_requests = 0
            # Note (Junnan Li): The queue fills after the initial admission check.
            s._waiting_queue_is_full = lambda: False

    def async_steps(self, case):
        s = self.scheduler
        self.blocked = True
        self.case = case
        self.bridge.command(payload("open"))
        self.request = self.append()
        self.enqueue(self.request)
        req = self.request.req
        batch = SimpleNamespace(reqs=[req])
        batch.copy = lambda: SimpleNamespace(reqs=[req])

        def step(name):
            def wait():
                if self.blocked and name == case.failing_step:
                    self.events.append(name + " wait failed")
                    raise RuntimeError("device wait failed")
                self.events.append(name + " waited")

            return SimpleNamespace(event=SimpleNamespace(synchronize=wait))

        self.current = (batch, "current", step("current"))
        self.previous = (batch, "previous", step("previous"))
        s._async_pending = self.previous if case.launch_first else self.current

        def collect(batch, name, pending, **kwargs):
            assert s._async_pending is None or name == "previous"
            self.events.append(name)
            s._running = False
            if case.collect_error:
                raise ValueError("collect failed")
            return SimpleNamespace(next_token_ids=None)

        s._run_batch_resolve = collect
        s.process_batch_result = lambda *args: self.events.append("process")

        def failure(batch, error):
            if not case.collect_error:
                pytest.fail("failed wait reached cleanup")
            assert str(error) == "collect failed"
            self.events.append("cleanup")
            s._resolve_pending_async()

        s._handle_batch_failure = failure
        if case.launch_first:
            s._running = True
            s._model_runner = None
            s.async_decode_min_batch_size = 1
            s._process_admin_requests = lambda: None
            s.recv_requests = lambda: []
            s._take_deferred_request_payloads = lambda: []
            s.process_input_requests = lambda _: None
            s.get_next_batch_to_run = lambda: batch
            s._batch_is_decode = lambda _: True
            s._run_batch_launch = lambda _: self.current[1:]

    def run_pending(self):
        entries = {
            "launch": self.scheduler._event_loop_async_decode,
            "shutdown": self.scheduler.start,
            "close": lambda: self.bridge.command(payload("close", "close")),
            "resolve": self.scheduler._resolve_pending_async,
        }
        entries[self.case.entry]()

    @property
    def pending(self):
        return self.scheduler._async_pending, self.scheduler._async_previous

    def retry_pending(self):
        self.blocked = False
        self.scheduler._resolve_pending_async()


@pytest.fixture
def bridge_env(monkeypatch):
    return BridgeHarness(monkeypatch)
