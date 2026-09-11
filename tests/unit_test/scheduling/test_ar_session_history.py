# SPDX-License-Identifier: Apache-2.0
from array import array
from types import SimpleNamespace

import pytest

from sglang_omni.proto.session import SessionRef
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from tests.unit_test.fixtures.ar_session import TestAdapter, bridge, data, payload


def test_materializes_native_history_preserving_sidecars():
    b = bridge()
    b.command(payload("open"))
    p = payload("append")
    b.accept(p)
    d = data()
    sidecar = object()
    d.model_inputs = {"marker": sidecar}
    b.materialize(p, d)
    assert d.req.session is b.scheduler.session_controller.get(
        b.native_id(SessionRef("s"))
    )
    assert d.model_inputs["marker"] is sidecar
    assert b.scheduler.session_controller.get(b.native_id(SessionRef("s")))._inflight
    b.rollback("r")
    assert not d.req.session._inflight


def test_one_inflight_rejection_does_not_clear_first_owner():
    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    b.materialize(payload("append"), d)
    with pytest.raises(ValueError, match="active"):
        b.accept(payload("append", "other"))
    assert d.req.session._inflight


def test_session_embeddings_explicitly_rejected():
    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    d.prefill_input_embeds = object()
    with pytest.raises(ValueError, match="embedding"):
        b.materialize(payload("append"), d)
    assert not b.scheduler.session_controller.get(
        b.native_id(SessionRef("s"))
    )._inflight


def test_native_history_rolls_back_failed_append():
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    b.materialize(payload("append"), d)
    native = d.req.session
    d.req.output_ids = array("q", [3, 4])
    d.req.finished_reason = FINISH_LENGTH(2)
    native.finish_req(d.req)
    b.complete("r")
    b.accept(payload("append", "bad"))
    failed = data("bad", [9])
    b.materialize(payload("append", "bad"), failed)
    assert list(failed.req.origin_input_ids) == [1, 2, 3, 4, 9]
    b.rollback("bad")
    b.accept(payload("append", "next"))
    good = data("next", [8])
    b.materialize(payload("append", "next"), good)
    assert list(good.req.origin_input_ids) == [1, 2, 3, 4, 8]


def test_materialization_drains_previous_unit_lookahead():
    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    events = []
    b.scheduler._resolve_pending_async = lambda: events.append("drain")
    b.materialize(payload("append"), data())
    assert events == ["drain"]


def test_native_rejection_does_not_rollback_another_inflight_owner(monkeypatch):
    import sglang.srt.managers.schedule_batch as native_batch

    monkeypatch.setattr(
        native_batch, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )
    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    native = b.scheduler.session_controller.get(b.native_id(SessionRef("s")))
    native._inflight = True
    with pytest.raises(ValueError, match="native session rejected"):
        b.materialize(payload("append"), data())
    b.rollback("r")
    assert native._inflight


@pytest.mark.parametrize("rejection", ["length", "input_length", "priority", "queue"])
def test_production_admission_rejection_rolls_back_native_history(
    monkeypatch, rejection
):
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH
    from sglang.srt.runtime_context import get_context

    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    s.model_config.vocab_size = 32
    s.server_args.mem_fraction_static = None
    s.session_controller = bridge().scheduler.session_controller
    s.tree_cache = s.session_controller.tree_cache
    built = []

    def build(ref, chunk, p):
        d = data(p.request_id)
        d.enforce_request_limits = rejection == "input_length"
        if rejection == "priority":
            d.req.priority = 1
        built.append(d)
        return d

    b = ARSessionBridge(s, TestAdapter(build=build))
    s._session_bridge = b
    # Note (Junnan Li): Exercise post-build rejection after native history materialization.
    b.capacity_error = lambda rid: None
    b.command(payload("open"))
    b.accept(payload("append", "first"))
    previous = data("first")
    b.materialize(payload("append", "first"), previous)
    native = previous.req.session
    previous.req.output_ids = array("q", [3, 4])
    previous.req.finished_reason = FINISH_LENGTH(2)
    native.finish_req(previous.req)
    b.complete("first")
    sid = b.native_id(SessionRef("s"))
    retained_slot = object()
    s.tree_cache.slots[sid] = retained_slot
    s._release_request_kv_cache = lambda req: pytest.fail("unadmitted KV released")
    s.max_req_len = 3 if rejection == "length" else 63
    s.max_req_input_len = 3 if rejection == "input_length" else 63
    s.abort_on_priority_when_disabled = rejection == "priority"
    if rejection == "queue":
        s.max_queued_requests = 0
        # Note (Junnan Li): The queue fills after the initial admission check.
        s._waiting_queue_is_full = lambda: False
    with get_context().override_server_args(weight_version="test"):
        s.process_input_requests([payload("append")])
    msg = s.outbox.get_nowait()
    assert msg.type == "error"
    assert s.tree_cache.slots[sid] is retained_slot
    b.accept(payload("append", "next"))
    following = data("next", [9])
    b.materialize(payload("append", "next"), following)
    assert list(following.req.origin_input_ids) == [1, 2, 3, 4, 9]
    b.rollback("next")
    s.tree_cache.slots.clear()
    assert b.command(payload("close", "close")).data == {"closed": True}


@pytest.mark.parametrize("available, exhausted", [(2, False), (1, True)])
def test_admission_counts_retained_kv_from_native_session_slot(available, exhausted):
    from sglang.srt.session.streaming_session import SessionSlot

    b = bridge()
    b.command(payload("open"))
    p = payload("append")
    b.accept(p)
    d = data()
    b.materialize(p, d)
    slot = SessionSlot()
    slot.kv.kv_allocated_len = 2
    slot.restore_to_req(d.req)
    b.scheduler.tree_cache.slots["s"] = slot
    b.scheduler.tree_cache.evictable_size = lambda: 0
    b.scheduler.req_to_token_pool = SimpleNamespace(free_slots=[0, 1])
    b.scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: available
    )
    error = b.capacity_error(p.request_id)
    assert error == ("session KV capacity exhausted" if exhausted else None)


@pytest.mark.parametrize("free_slots, exhausted", [(2, True), (3, False)])
def test_admission_reserves_rows_for_other_unallocated_sessions(free_slots, exhausted):
    b = bridge()
    for sid, rid in [("s", "r"), ("other", "other-r")]:
        opened = payload("open", "open-" + rid)
        opened.request.metadata["omni_session"]["ref"]["session_id"] = sid
        b.command(opened)
        p = payload("append", rid)
        p.request.metadata["omni_session"]["ref"]["session_id"] = sid
        b.accept(p)
        b.materialize(p, data(rid))
    b.scheduler.tree_cache.evictable_size = lambda: 0
    b.scheduler.req_to_token_pool = SimpleNamespace(free_slots=list(range(free_slots)))
    b.scheduler.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 8)
    error = b.capacity_error("r")
    assert error == (
        "session request slot capacity exhausted (one admission slot reserved)"
        if exhausted
        else None
    )
