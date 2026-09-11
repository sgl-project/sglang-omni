# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from tests.unit_test.fixtures.ar_session import TestAdapter, bridge, data, payload


def test_cancelled_close_retains_cleanup_intent(monkeypatch):
    from sglang_omni.scheduling.messages import IncomingMessage
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    s._session_bridge = ARSessionBridge(s, TestAdapter())
    s.session_controller = bridge().scheduler.session_controller
    s.tree_cache = s.session_controller.tree_cache
    s.process_input_requests([payload("open")])
    s.outbox.get_nowait()
    s._aborted_request_ids.add("close")
    s.inbox.put(IncomingMessage("close", "new_request", payload("close", "close")))
    s._drain_inbox_for_request("close")
    s.process_input_requests(s.recv_requests())
    assert not s._session_bridge.owners
    assert s.outbox.empty()


def test_shutdown_callback_runs_even_if_native_cleanup_fails(monkeypatch):
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    events = []
    s._event_loop_normal = lambda: None
    s._resolve_pending_async = lambda: None
    s._session_bridge = SimpleNamespace(
        shutdown=lambda: (_ for _ in ()).throw(RuntimeError("native failure"))
    )
    s._shutdown_callback = lambda: events.append("resources")
    with pytest.raises(RuntimeError, match="native failure"):
        s.start()
    assert events == ["resources"]


def test_request_abort_fences_output_before_async_drain():
    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    b.materialize(payload("append"), d)
    calls = []
    b.scheduler.waiting_queue = []
    b.scheduler.chunked_req = None
    b.scheduler._release_request_kv_cache = lambda req: calls.append("release")
    b.scheduler._run_abort_callback = lambda rid: None
    b.requests["r"].enqueued = True
    d.req._omni_data = None

    def abort(rid, **kw):
        b.scheduler._aborted_request_ids.add(rid)
        d.req.to_finish = object()
        calls.append(kw)

    b.scheduler.abort = abort

    def drain():
        assert "r" in b.scheduler._aborted_request_ids
        assert d.req.to_finish is not None
        calls.append("drain")

    b.scheduler._resolve_pending_async = drain
    b.cancel("r")
    assert calls == [{}, "drain", "release"]
    assert not b.requests


def test_idle_close_drains_completed_lookahead_before_slot_release():
    b = bridge()
    b.command(payload("open"))
    events = []
    b.scheduler._resolve_pending_async = lambda: events.append("drain")
    b.scheduler.session_controller.tree_cache.release_session = (
        lambda sid: events.append("release")
    )
    b.command(payload("close"))
    assert events == ["drain", "release"]


def test_failed_gpu_wait_retains_native_cleanup_ownership():
    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    b.materialize(payload("append"), d)
    b.scheduler.waiting_queue = []
    b.scheduler.chunked_req = None
    b.scheduler._release_request_kv_cache = lambda req: None
    b.requests["r"].enqueued = True
    d.req._omni_data = None

    def abort(rid):
        if b.cancelling != rid:
            b.cancel(rid)

    b.scheduler.abort = abort

    def fail():
        raise RuntimeError("device wait failed")

    b.scheduler._async_pending = (
        None,
        None,
        SimpleNamespace(event=SimpleNamespace(synchronize=fail)),
    )
    with pytest.raises(RuntimeError, match="device wait failed"):
        b.command(payload("close", "close"))
    assert b.requests["r"].active == "r"
    assert d.req.session._inflight
    b.scheduler._async_pending[2].event.synchronize = lambda: None

    def resolve():
        b.scheduler._async_pending = None

    b.scheduler._resolve_pending_async = resolve
    assert b.command(payload("close", "retry")).data == {"closed": True}
    assert not b.owners and not b.requests
    assert not b.scheduler.session_controller.sessions


@pytest.mark.parametrize("shutdown", [False, True])
def test_regular_async_wait_failure_preserves_native_owner_and_retries(
    monkeypatch, shutdown
):
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    b = ARSessionBridge(s, TestAdapter())
    s.model_config.vocab_size = 32
    s._session_bridge = b
    s.session_controller = bridge().scheduler.session_controller
    s.tree_cache = s.session_controller.tree_cache
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    b.materialize(payload("append"), d)
    req = d.req
    b.requests["r"].enqueued = True
    req._omni_data = d
    req._omni_terminal_claimed = False
    batch = SimpleNamespace(reqs=[req])
    s.running_batch = s.cur_batch = s.last_batch = batch
    events = []
    blocked = True

    def wait():
        if blocked:
            events.append("wait failed")
            raise RuntimeError("device wait failed")
        events.append("wait succeeded")

    step = SimpleNamespace(event=SimpleNamespace(synchronize=wait))
    pending = (batch, None, step)
    s._async_pending = pending

    def resolve(*args, **kwargs):
        step.event.synchronize()
        return SimpleNamespace(next_token_ids=None)

    s._run_batch_resolve = resolve
    s.process_batch_result = lambda *args: events.append("process")
    s._release_request_kv_cache = lambda req: events.append("release")
    s._shutdown_callback = lambda: events.append("resources")
    s._event_loop_normal = lambda: None
    with pytest.raises(RuntimeError, match="device wait failed"):
        s.start() if shutdown else s._resolve_pending_async()
    assert "release" not in events and "process" not in events
    assert s._async_pending is pending
    assert b.requests["r"].native_owned and req.session._inflight
    assert len(s.session_controller.sessions) == 1
    blocked = False
    assert b.command(payload("close", "retry")).data == {"closed": True}
    assert s._async_pending is None
    assert not b.owners and not b.requests and not s.session_controller.sessions
    assert events.count("release") == 1


@pytest.mark.parametrize("failed_step", ["previous", "current"])
def test_launch_first_previous_wait_failure_retains_both_steps(
    monkeypatch, failed_step
):
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    s._session_bridge = SimpleNamespace(owners={})
    events = []
    blocked = True

    def wait_previous():
        if blocked and failed_step == "previous":
            raise RuntimeError("previous device wait failed")
        events.append("previous waited")

    req = SimpleNamespace(
        rid="r",
        session=SimpleNamespace(streaming=True),
        finished=lambda: False,
        is_retracted=False,
    )
    batch = SimpleNamespace(reqs=[req], copy=lambda: SimpleNamespace(reqs=[req]))
    previous = (
        batch,
        "previous",
        SimpleNamespace(event=SimpleNamespace(synchronize=wait_previous)),
    )

    def wait_current():
        if blocked and failed_step == "current":
            raise RuntimeError("current device wait failed")
        events.append("current waited")

    current_step = SimpleNamespace(event=SimpleNamespace(synchronize=wait_current))
    s._async_pending = previous
    s._running = True
    s._model_runner = None
    s.async_decode_min_batch_size = 1
    s._process_admin_requests = lambda: None
    s.recv_requests = lambda: []
    s._take_deferred_request_payloads = lambda: []
    s.process_input_requests = lambda _: None
    s.get_next_batch_to_run = lambda: batch
    s._batch_is_decode = lambda _: True
    s._run_batch_launch = lambda _: ("current", current_step)

    def resolve(batch, output, step, **kwargs):
        step.event.synchronize()
        events.append(output)
        s._running = False
        return SimpleNamespace(next_token_ids=None)

    s._run_batch_resolve = resolve
    s.process_batch_result = lambda *args: None
    s._handle_batch_failure = lambda *args: pytest.fail("failed wait reached cleanup")
    with pytest.raises(RuntimeError, match="device wait failed"):
        s._event_loop_async_decode()
    assert s._async_pending[2] is current_step
    assert s._async_previous is previous
    assert "previous" not in events and "current" not in events
    blocked = False
    s._resolve_pending_async()
    assert events.index("previous") < events.index("current")
    assert s._async_pending is None and s._async_previous is None


def test_post_wait_collect_error_does_not_recollect_or_release_twice(monkeypatch):
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    s._session_bridge = SimpleNamespace(owners={})
    events = []
    req = SimpleNamespace(
        rid="r", session=None, finished=lambda: False, is_retracted=False
    )
    batch = SimpleNamespace(reqs=[req])
    step = SimpleNamespace(
        event=SimpleNamespace(synchronize=lambda: events.append("wait"))
    )
    s._async_pending = (batch, None, step)

    def collect(*args, **kwargs):
        assert s._async_pending is None
        events.append("collect")
        raise ValueError("collect failed")

    def failure(batch, error):
        assert str(error) == "collect failed"
        events.append("cleanup")
        s._resolve_pending_async()

    s._run_batch_resolve = collect
    s._handle_batch_failure = failure
    s._resolve_pending_async()
    assert events == ["wait", "collect", "cleanup"]
    assert s._async_pending is None and s._async_previous is None


def test_cancel_does_not_abort_active_core_request():
    b = bridge()
    b.command(payload("open"))
    native = b.scheduler.session_controller.get("s")
    assert native is not None and native.timeout is None
    b.accept(payload("append"))
    b.materialize(payload("append"), data())
    with pytest.raises(RuntimeError, match="unit completion"):
        b.command(payload("abort", epoch=1))
    assert native._inflight
    assert b.owners["s"].ref.epoch == 0


def test_cancel_at_boundary_preserves_native_session():
    b = bridge()
    b.command(payload("open"))
    native = b.scheduler.session_controller.get("s")
    b.command(payload("abort", epoch=1))
    assert b.scheduler.session_controller.get("s") is native
    assert b.owners["s"].ref.epoch == 1


def test_adapter_state_follows_core_session_lifetime():
    from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter

    calls = []

    class Adapter(ARSessionAdapter):
        def open(self, ref, request):
            calls.append(("open", ref.session_id))

        def close(self, ref):
            calls.append(("close", ref.session_id))

    b = bridge()
    b.adapter = Adapter()
    b.command(payload("open"))
    b.command(payload("abort", epoch=1))
    assert calls == [("open", "s")]
    b.command(payload("close", epoch=1))
    b.command(payload("close", epoch=1))
    assert calls == [("open", "s"), ("close", "s")]


def test_queued_cancel_detaches_request_without_releasing_session_kv():
    from sglang.srt.session.streaming_session import SessionSlot

    b = bridge()
    b.command(payload("open"))
    p = payload("append")
    b.accept(p)
    d = data()
    b.materialize(p, d)
    slot = SessionSlot()
    slot.kv.req_pool_idx = 1
    slot.kv.kv_allocated_len = 2
    slot.restore_to_req(d.req)
    b.scheduler.tree_cache.slots["s"] = slot
    b.scheduler.waiting_queue = [d.req]
    b.requests["r"].enqueued = True

    def abort(rid):
        assert not d.req.kv.holds_kv
        assert d.req.kv is not slot.kv

    b.scheduler.abort = abort
    b.cancel("r")
    assert slot.kv.holds_kv
    assert slot.kv.kv_allocated_len == 2
    assert not b.requests
