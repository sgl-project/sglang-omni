# SPDX-License-Identifier: Apache-2.0
"""Unit tests for voice-fusion atomic prefill admission gating in
``OmniScheduler.get_next_batch_to_run`` — see the "Co-batching" section of
``docs/voice_fusion_design.md`` for why this gate exists (there is no
scheduler-side atomic admission upstream, and a naive attempt to trim an
already-``prepare_for_extend``-processed batch corrupts it) and exactly what
it does and does not guarantee.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _req(rid, input_len, max_new_tokens):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(input_len)),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
    )


def _scheduler(
    *,
    available,
    evictable=0,
    running_reqs=None,
    cur_batch_reqs=None,
    last_batch_reqs=None,
    page_size=1,
    chunked_prefill_size=None,
    max_prefill_tokens=None,
    slot_capacity=10_000,
    chunked_req=None,
):
    s = object.__new__(OmniScheduler)
    s.waiting_queue = []
    s._fusion_group_members = {}
    s._fusion_group_withhold_ticks = {}
    s._aborted_request_ids = set()
    s._request_admission_lock = threading.RLock()
    s.page_size = page_size
    s.chunked_prefill_size = chunked_prefill_size
    s.max_prefill_tokens = max_prefill_tokens
    s.chunked_req = chunked_req
    s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: available)
    s.tree_cache = SimpleNamespace(evictable_size=lambda: evictable)
    s.running_batch = SimpleNamespace(reqs=running_reqs or [])
    s.cur_batch = SimpleNamespace(reqs=cur_batch_reqs or []) if cur_batch_reqs else None
    s.last_batch = (
        SimpleNamespace(reqs=last_batch_reqs or []) if last_batch_reqs else None
    )
    # Mirrors upstream's `pp_max_micro_batch_size - running_bs` shape closely
    # enough to test that the gate derives `running_bs` from the same
    # widened in-flight set the token estimate uses, not just
    # `len(running_batch.reqs)` alone (see
    # test_slot_budget_accounts_for_not_yet_merged_cur_batch below).
    s.get_num_allocatable_reqs = lambda running_bs: max(0, slot_capacity - running_bs)
    s.abort = mock.Mock()
    s._emit_request_error = mock.Mock()
    return s


def _register_group(s, reqs):
    group_rids = {r.rid for r in reqs}
    for r in reqs:
        s._fusion_group_members[r.rid] = group_rids


# --------------------------------------------------------------------------- #
# _estimate_available_prefill_tokens
# --------------------------------------------------------------------------- #
def test_estimate_available_subtracts_running_reservations():
    running = [_req("run1", 10, 100), _req("run2", 10, 50)]
    s = _scheduler(available=1000, evictable=200, running_reqs=running)
    assert s._estimate_available_prefill_tokens() == 1000 + 200 - 100 - 50


def test_estimate_available_also_reserves_for_cur_and_last_batch():
    # Fable finding #1: upstream folds cur_batch/last_batch into
    # running_batch INSIDE the call this gate runs before -- at gate time
    # those reqs may not be in running_batch yet, so they must be reserved
    # against separately, not ignored.
    running = [_req("run1", 10, 100)]
    cur = [_req("cur1", 10, 30)]
    last = [_req("last1", 10, 20)]
    s = _scheduler(
        available=1000,
        running_reqs=running,
        cur_batch_reqs=cur,
        last_batch_reqs=last,
    )
    assert s._estimate_available_prefill_tokens() == 1000 - 100 - 30 - 20


def test_estimate_available_dedupes_overlapping_rid_across_batches():
    # Same rid appearing in both running_batch and cur_batch (already
    # folded in) must not be reserved twice.
    dup = _req("dup", 10, 100)
    s = _scheduler(available=1000, running_reqs=[dup], cur_batch_reqs=[dup])
    assert s._estimate_available_prefill_tokens() == 1000 - 100


def test_estimate_available_floors_at_zero():
    running = [_req("run1", 10, 10_000)]
    s = _scheduler(available=100, running_reqs=running)
    assert s._estimate_available_prefill_tokens() == 0


def test_estimate_available_capped_by_chunked_prefill_size():
    # Fable finding #3: chunked prefill IS enabled for this deployment
    # (engine_builder.py sets chunked_prefill_size=8192) and is an
    # independent budget dimension upstream enforces alongside the
    # KV-token one.
    s = _scheduler(available=100_000, chunked_prefill_size=500)
    assert s._estimate_available_prefill_tokens() == 500


def test_estimate_available_capped_by_max_prefill_tokens():
    s = _scheduler(available=100_000, max_prefill_tokens=300)
    assert s._estimate_available_prefill_tokens() == 300


def test_estimate_available_defaults_to_zero_on_pool_read_failure():
    s = _scheduler(available=0)

    def _boom():
        raise RuntimeError("pool unavailable")

    s.token_to_kv_pool_allocator.available_size = _boom
    assert s._estimate_available_prefill_tokens() == 0


# --------------------------------------------------------------------------- #
# _fusion_group_prefill_cost / _prefill_in_flight_reqs
# --------------------------------------------------------------------------- #
def test_group_prefill_cost_sums_members_plus_page_size():
    s = _scheduler(available=0, page_size=4)
    members = [_req("a", 100, 50), _req("b", 80, 50)]
    assert s._fusion_group_prefill_cost(members) == (100 + 50 + 4) + (80 + 50 + 4)


def test_prefill_in_flight_reqs_unions_running_cur_and_last_batch():
    running = [_req("run1", 1, 1)]
    cur = [_req("cur1", 1, 1)]
    last = [_req("last1", 1, 1)]
    s = _scheduler(
        available=0, running_reqs=running, cur_batch_reqs=cur, last_batch_reqs=last
    )
    assert {r.rid for r in s._prefill_in_flight_reqs()} == {"run1", "cur1", "last1"}


def test_prefill_in_flight_reqs_dedupes_by_rid():
    dup = _req("dup", 1, 1)
    s = _scheduler(available=0, running_reqs=[dup], cur_batch_reqs=[dup])
    assert [r.rid for r in s._prefill_in_flight_reqs()] == ["dup"]


# --------------------------------------------------------------------------- #
# _reorder_queue_for_atomic_fusion_admission / _restore_queue_after_...
# --------------------------------------------------------------------------- #
def test_no_fusion_traffic_is_a_no_op():
    s = _scheduler(available=1000)
    ordinary = [_req("o1", 5, 5), _req("o2", 5, 5)]
    s.waiting_queue = list(ordinary)

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert withheld == []
    assert s.waiting_queue == ordinary


def test_incomplete_group_is_withheld_entirely_no_tick_charged():
    s = _scheduler(available=10_000)
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    ordinary = _req("o1", 5, 5)
    # The follower hasn't arrived yet -- only the leader sits in the queue.
    s.waiting_queue = [leader, ordinary]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert withheld == [leader]
    assert s.waiting_queue == [ordinary]
    # Incomplete groups always eventually resolve on their own (build
    # finishes, or the decode-time backstop cascades an abort) -- no
    # withhold-tick counter should be charged for this reason.
    assert s._fusion_group_withhold_ticks == {}
    s.abort.assert_not_called()


def test_complete_affordable_group_moves_to_front_ahead_of_ordinary():
    s = _scheduler(available=10_000)
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    o1, o2 = _req("o1", 5, 5), _req("o2", 5, 5)
    # Deliberately interleaved in the raw queue.
    s.waiting_queue = [o1, leader, o2, follower]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert withheld == []
    # Group members contiguous at the front, in their original relative
    # order; ordinary requests follow, in their original relative order.
    assert s.waiting_queue == [leader, follower, o1, o2]


def test_complete_group_over_token_budget_is_withheld_and_charged_a_tick():
    s = _scheduler(available=50)  # far too little for the group below
    leader = _req("leader", 1000, 1000)
    follower = _req("leader#fuse1", 1000, 1000)
    _register_group(s, [leader, follower])
    ordinary = _req("o1", 5, 5)
    s.waiting_queue = [leader, follower, ordinary]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert set(withheld) == {leader, follower}
    assert s.waiting_queue == [ordinary]
    gid = frozenset({"leader", "leader#fuse1"})
    assert s._fusion_group_withhold_ticks[gid] == 1
    s.abort.assert_not_called()


def test_complete_group_over_slot_budget_is_withheld():
    s = _scheduler(available=100_000, slot_capacity=1)
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    s.waiting_queue = [leader, follower]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert set(withheld) == {leader, follower}
    assert s.waiting_queue == []


def test_slot_budget_accounts_for_not_yet_merged_cur_batch():
    # Regression for the exact staleness Fable's re-review caught: at gate
    # time, upstream has NOT yet folded self.cur_batch (last tick's just-
    # admitted prefill batch) into self.running_batch -- computing the slot
    # budget from `len(running_batch.reqs)` alone would overestimate free
    # slots for one tick and let a group through that upstream's own
    # (correctly, cur_batch-inclusive) running_bs would still split.
    running = [_req(f"run{i}", 1, 0) for i in range(3)]
    cur = [_req(f"cur{i}", 1, 0) for i in range(4)]  # not yet merged
    # slot_capacity=8: running-only view sees allocatable=8-3=5 (group of 3
    # fits); the correct (running+cur unioned) view sees allocatable=8-7=1
    # (group of 3 does NOT fit) -- this test pins the correct behavior.
    s = _scheduler(
        available=100_000,
        slot_capacity=8,
        running_reqs=running,
        cur_batch_reqs=cur,
    )
    leader = _req("leader", 1, 0)
    f1 = _req("leader#fuse1", 1, 0)
    f2 = _req("leader#fuse2", 1, 0)
    _register_group(s, [leader, f1, f2])
    s.waiting_queue = [leader, f1, f2]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert set(withheld) == {leader, f1, f2}
    assert s.waiting_queue == []


def test_chunked_req_in_flight_withholds_every_group():
    s = _scheduler(available=100_000, chunked_req=object())
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    ordinary = _req("o1", 5, 5)
    s.waiting_queue = [leader, follower, ordinary]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    assert set(withheld) == {leader, follower}
    assert s.waiting_queue == [ordinary]


def test_two_groups_reserve_against_a_shared_running_budget():
    # Each group alone fits comfortably, but not both at once this tick.
    s = _scheduler(available=1500, page_size=0)
    a_leader, a_follower = _req("a", 500, 0), _req("a#fuse1", 500, 0)
    b_leader, b_follower = _req("b", 500, 0), _req("b#fuse1", 500, 0)
    _register_group(s, [a_leader, a_follower])
    _register_group(s, [b_leader, b_follower])
    s.waiting_queue = [a_leader, a_follower, b_leader, b_follower]

    withheld = s._reorder_queue_for_atomic_fusion_admission()

    # cost(a) = 1000 < 1500 -> admitted, remaining budget 500.
    # cost(b) = 1000 >= 500 -> withheld.
    assert {r.rid for r in withheld} == {"b", "b#fuse1"}
    assert [r.rid for r in s.waiting_queue] == ["a", "a#fuse1"]


def test_restore_puts_withheld_reqs_back_at_the_front():
    s = _scheduler(available=1000)
    kept = _req("kept", 5, 5)
    s.waiting_queue = [kept]
    withheld = [_req("late1", 5, 5), _req("late2", 5, 5)]

    s._restore_queue_after_atomic_fusion_admission(withheld)

    assert s.waiting_queue == [*withheld, kept]


def test_restore_is_a_no_op_for_empty_withheld_list():
    s = _scheduler(available=1000)
    kept = _req("kept", 5, 5)
    s.waiting_queue = [kept]

    s._restore_queue_after_atomic_fusion_admission([])

    assert s.waiting_queue == [kept]


def test_restore_drops_reqs_aborted_during_the_withhold_window():
    # Fable finding #5: abort() runs on a different thread (Stage's own
    # event loop) and can fire for a member of a currently-withheld group
    # while it sits outside self.waiting_queue -- abort()'s own queue
    # filter won't find it there, so restore must independently check
    # _aborted_request_ids or the aborted req gets silently resurrected.
    s = _scheduler(available=1000)
    late1, late2 = _req("late1", 5, 5), _req("late2", 5, 5)
    s._aborted_request_ids.add("late1")

    s._restore_queue_after_atomic_fusion_admission([late1, late2])

    assert s.waiting_queue == [late2]


# --------------------------------------------------------------------------- #
# Give-up path: a group withheld for too many consecutive ticks is aborted
# with a client-visible error instead of withheld forever.
# --------------------------------------------------------------------------- #
def test_group_withheld_past_the_limit_is_aborted_and_excluded_from_withheld():
    s = _scheduler(available=50)  # never affordable
    leader = _req("leader", 1000, 1000)
    follower = _req("leader#fuse1", 1000, 1000)
    _register_group(s, [leader, follower])

    withheld = None
    for _ in range(OmniScheduler._MAX_FUSION_WITHHOLD_TICKS):
        s.waiting_queue = [leader, follower]
        withheld = s._reorder_queue_for_atomic_fusion_admission()

    # Every tick up to and including the limit charged a tick; the LAST one
    # crosses the threshold and gives up.
    assert withheld == []  # given up on -> excluded, not carried forward
    assert s.abort.call_count == 1
    aborted_rid = s.abort.call_args[0][0]
    assert aborted_rid in {"leader", "leader#fuse1"}
    assert s._emit_request_error.call_count == 2
    emitted_rids = {call.args[0] for call in s._emit_request_error.call_args_list}
    assert emitted_rids == {"leader", "leader#fuse1"}
    gid = frozenset({"leader", "leader#fuse1"})
    assert gid not in s._fusion_group_withhold_ticks


def test_withhold_tick_counter_resets_once_a_group_is_admitted():
    s = _scheduler(available=50)
    leader = _req("leader", 1000, 1000)
    follower = _req("leader#fuse1", 1000, 1000)
    _register_group(s, [leader, follower])

    s.waiting_queue = [leader, follower]
    s._reorder_queue_for_atomic_fusion_admission()
    gid = frozenset({"leader", "leader#fuse1"})
    assert s._fusion_group_withhold_ticks[gid] == 1

    # Budget frees up -> admitted this tick -> counter must be dropped, not
    # merely left stale, so a later rough patch starts counting from zero.
    s.token_to_kv_pool_allocator.available_size = lambda: 100_000
    s.waiting_queue = [leader, follower]
    s._reorder_queue_for_atomic_fusion_admission()
    assert gid not in s._fusion_group_withhold_ticks


# --------------------------------------------------------------------------- #
# get_next_batch_to_run: the withhold/restore wrap around the upstream call
# --------------------------------------------------------------------------- #
def test_get_next_batch_to_run_withholds_then_restores_around_upstream_call():
    s = _scheduler(available=10_000)
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    # The follower is missing -> the whole group is withheld for this call.
    ordinary = _req("o1", 5, 5)
    s.waiting_queue = [leader, ordinary]

    observed_queue_during_upstream = []

    def fake_upstream(self):
        observed_queue_during_upstream.extend(self.waiting_queue)
        return None

    with mock.patch.object(
        omni_scheduler_module._Upstream, "get_next_batch_to_run", fake_upstream
    ):
        result = s.get_next_batch_to_run()

    assert result is None
    # Upstream never saw the incomplete group's leader.
    assert observed_queue_during_upstream == [ordinary]
    # Restored afterward so the leader isn't lost while its sibling is still
    # being built.
    assert s.waiting_queue == [leader, ordinary]


def test_get_next_batch_to_run_lets_a_complete_affordable_group_through():
    s = _scheduler(available=10_000)
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    s.waiting_queue = [leader, follower]

    observed_queue_during_upstream = []

    def fake_upstream(self):
        observed_queue_during_upstream.extend(self.waiting_queue)
        self.waiting_queue = []  # simulate upstream admitting everything
        return "a-batch"

    with mock.patch.object(
        omni_scheduler_module._Upstream, "get_next_batch_to_run", fake_upstream
    ):
        result = s.get_next_batch_to_run()

    # batch="a-batch" has no .reqs, so the post-admission visibility scan
    # short-circuits via `not reqs` and returns it straight through.
    assert result == "a-batch"
    assert observed_queue_during_upstream == [leader, follower]
    # Nothing withheld -> restore is a no-op, upstream's own mutation stands.
    assert s.waiting_queue == []


def test_get_next_batch_to_run_restores_even_if_upstream_raises():
    s = _scheduler(available=10_000)
    leader = _req("leader", 10, 10)
    follower = _req("leader#fuse1", 10, 10)
    _register_group(s, [leader, follower])
    ordinary = _req("o1", 5, 5)
    s.waiting_queue = [leader, ordinary]  # incomplete group -> withheld

    def fake_upstream(self):
        raise RuntimeError("boom")

    with mock.patch.object(
        omni_scheduler_module._Upstream, "get_next_batch_to_run", fake_upstream
    ):
        try:
            s.get_next_batch_to_run()
            assert False, "expected RuntimeError to propagate"
        except RuntimeError:
            pass

    # The withheld leader must not be lost just because upstream blew up.
    assert s.waiting_queue == [leader, ordinary]


# --------------------------------------------------------------------------- #
# Lock reentrancy: _request_admission_lock is an RLock specifically so the
# give-up path's call into abort() (which itself takes the same lock, see
# OmniScheduler.abort) does not deadlock while this gate already holds it.
# --------------------------------------------------------------------------- #
def test_give_up_path_does_not_deadlock_against_a_reentrant_abort():
    s = _scheduler(available=50)
    leader = _req("leader", 1000, 1000)
    follower = _req("leader#fuse1", 1000, 1000)
    _register_group(s, [leader, follower])

    acquired_reentrantly = []

    def fake_abort(rid):
        # Mirrors the real OmniScheduler.abort(), which itself does
        # `with self._request_admission_lock:` -- proves the lock held
        # around this whole method is the reentrant RLock, not a plain Lock.
        with s._request_admission_lock:
            acquired_reentrantly.append(rid)

    s.abort = fake_abort

    for _ in range(OmniScheduler._MAX_FUSION_WITHHOLD_TICKS):
        s.waiting_queue = [leader, follower]
        s._reorder_queue_for_atomic_fusion_admission()

    assert len(acquired_reentrantly) == 1
