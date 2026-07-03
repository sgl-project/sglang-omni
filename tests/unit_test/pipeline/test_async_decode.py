# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the async-decode (one-step lookahead) state machine.

The heavy sub-steps (_build_forward_batch / _prepare_and_forward / _finalize)
and the model-specific hooks are stubbed, and torch.cuda.Event is patched, so
these run CPU-only. The pinned ping-pong test is CUDA-guarded.

Pending ownership lives with the CALLER (execute_launch returns a handle,
execute_resolve takes it) because launch-first scheduling has two steps
momentarily in flight.
"""

from __future__ import annotations

import queue
import threading
import types
from unittest import mock

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput, RequestOutput


class _StubRunner(ModelRunner):
    """ModelRunner with mocked sub-steps; exercises only execute_launch/resolve."""

    def __init__(self):
        self._async_enabled = True
        self._staging_slot = 0
        self._host_staging_buffers = []
        self._async_query_hit = 0
        self._async_query_miss = 0
        self.launch_calls = 0
        self.resolve_calls = 0
        self.finalize_calls = 0
        self.last_resolved_buf = None
        self.last_prepare_is_lookahead = None
        self.last_skip_rids = None

    def _build_forward_batch(self, scheduler_output):
        sb = types.SimpleNamespace(is_prefill_only=False, output_ids=None)
        return types.SimpleNamespace(), sb, types.SimpleNamespace(), False  # decode

    def _prepare_and_forward(
        self,
        forward_batch,
        schedule_batch,
        requests,
        is_prefill,
        *,
        is_lookahead=False,
    ):
        self.last_prepare_is_lookahead = is_lookahead
        return types.SimpleNamespace(
            next_token_ids=object(),
            logits_output=types.SimpleNamespace(next_token_logits=None),
            can_run_cuda_graph=False,
        )

    def post_decode_launch(self, result, forward_batch, requests):
        self.launch_calls += 1
        return f"hostbuf-{self.launch_calls}"

    def post_decode_resolve(
        self, launch_buf, result, forward_batch, schedule_batch, requests
    ):
        self.resolve_calls += 1
        self.last_resolved_buf = launch_buf

    def _finalize(
        self,
        batch_result,
        forward_batch,
        schedule_batch,
        model_worker_batch,
        scheduler_output,
        set_output_ids=True,
        skip_rids=None,
    ):
        self.finalize_calls += 1
        self.last_set_output_ids = set_output_ids
        self.last_skip_rids = skip_rids or set()
        return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})


def _patch_event(ready: bool):
    class _FakeEvent:
        def __init__(self):
            self.synced = False

        def record(self):
            pass

        def query(self):
            return ready

        def synchronize(self):
            self.synced = True

    return mock.patch("torch.cuda.Event", _FakeEvent)


def _sched_output(n):
    req_stub = types.SimpleNamespace(finished=lambda: False)
    return types.SimpleNamespace(
        requests=[
            types.SimpleNamespace(
                request_id=f"r{i}",
                data=types.SimpleNamespace(req=req_stub),
            )
            for i in range(n)
        ],
        batch_data=object(),
    )


def test_launch_returns_handle_resolve_consumes_it():
    r = _StubRunner()
    with _patch_event(ready=True):
        step = r.execute_launch(_sched_output(2))
        assert step is not None and step.n_real == 2
        out = r.execute_resolve(step)
    assert out is not None
    assert (r.launch_calls, r.resolve_calls, r.finalize_calls) == (1, 1, 1)
    assert (r._async_query_hit, r._async_query_miss) == (1, 0)
    assert r.last_prepare_is_lookahead is True
    # resolve must NOT re-publish output_ids: under launch-first it runs one
    # step behind on the LIVE running batch, whose output_ids the current launch
    # already set at the right length. Re-stamping the lagged step's tokens
    # leaves a stale-length output_ids -> input_ids/seq_lens mismatch once a req
    # finishes mid-batch (the bs>1 replay crash). The launch publishes it.
    assert r.last_set_output_ids is False


def test_two_launches_return_distinct_handles():
    # launch-first keeps two steps in flight; both must be independent handles
    r = _StubRunner()
    with _patch_event(ready=True):
        s1 = r.execute_launch(_sched_output(1))
        s2 = r.execute_launch(_sched_output(1))
        assert s1 is not s2 and s1.launch_buf != s2.launch_buf
        # resolve in order N-1 then N
        r.execute_resolve(s1)
        assert r.last_resolved_buf == s1.launch_buf
        r.execute_resolve(s2)
        assert r.last_resolved_buf == s2.launch_buf


def test_resolve_none_returns_none():
    # Warmup / drained: nothing to resolve.
    r = _StubRunner()
    assert r.execute_resolve(None) is None
    assert r.finalize_calls == 0


def test_query_miss_falls_back_to_synchronize():
    r = _StubRunner()
    with _patch_event(ready=False):
        step = r.execute_launch(_sched_output(1))
        r.execute_resolve(step)
    assert step.event.synced is True
    assert (r._async_query_hit, r._async_query_miss) == (0, 1)


def test_resolve_recomputes_finished_overrun_skip_rids():
    r = _StubRunner()
    keep_req = types.SimpleNamespace(finished=lambda: False)
    skip_req = types.SimpleNamespace(finished=lambda: True)
    sched_output = types.SimpleNamespace(
        requests=[
            types.SimpleNamespace(
                request_id="keep",
                data=types.SimpleNamespace(req=keep_req),
            ),
            types.SimpleNamespace(
                request_id="skip",
                data=types.SimpleNamespace(req=skip_req),
            ),
        ],
        batch_data=object(),
    )
    with _patch_event(ready=True):
        step = r.execute_launch(sched_output)
        r.execute_resolve(step)
    assert r.last_skip_rids == {"skip"}


def test_resolve_skips_retracted_row():
    """A request retracted (KV freed, returned to waiting) while its lagged step
    was in flight must be skipped at resolve, exactly like a prior-step finish.

    This guards the shared async-resolve path (Higgs and MOSS-TTS-Local both use
    base ModelRunner.execute_resolve): without overlap, upstream would otherwise
    append + check_finished the retracted req and re-free its already-freed KV
    (a double-free assertion). Crash-fix only; faithful frame accounting for a
    retracted req is out of scope here.
    """
    r = _StubRunner()
    keep_req = types.SimpleNamespace(finished=lambda: False, is_retracted=False)
    retracted_req = types.SimpleNamespace(finished=lambda: False, is_retracted=True)
    sched_output = types.SimpleNamespace(
        requests=[
            types.SimpleNamespace(
                request_id="keep",
                data=types.SimpleNamespace(req=keep_req),
            ),
            types.SimpleNamespace(
                request_id="retracted",
                data=types.SimpleNamespace(req=retracted_req),
            ),
        ],
        batch_data=object(),
    )
    with _patch_event(ready=True):
        step = r.execute_launch(sched_output)
        r.execute_resolve(step)
    assert r.last_skip_rids == {"retracted"}


def test_base_lookahead_eligible_default_true():
    """Base runner default: any batch is lookahead-eligible. A model with a
    sync-only collect fallback overrides this; the scheduler's async gate
    consults it alongside the min-batch-size and is-decode checks.
    """
    r = _StubRunner()
    assert r.lookahead_eligible(types.SimpleNamespace(reqs=[])) is True
    assert r.lookahead_eligible(None) is True


def test_finalize_skips_overrun_bookkeeping_and_extras():
    class _OutputProcessor:
        def process(self, batch_result, scheduler_output):
            del batch_result
            return {
                req.request_id: RequestOutput(
                    request_id=req.request_id, extra={"seen": req.request_id}
                )
                for req in scheduler_output.requests
            }

    runner = ModelRunner.__new__(ModelRunner)
    runner.output_processor = _OutputProcessor()
    batch_result = types.SimpleNamespace(
        next_token_ids=torch.tensor([1, 2]),
        logits_output=None,
        can_run_cuda_graph=False,
    )
    schedule_batch = types.SimpleNamespace(is_prefill_only=False, output_ids=None)
    model_worker_batch = types.SimpleNamespace()
    keep_data = types.SimpleNamespace(generation_steps=0, extra_model_outputs={})
    skip_data = types.SimpleNamespace(generation_steps=0, extra_model_outputs={})
    scheduler_output = types.SimpleNamespace(
        requests=[
            types.SimpleNamespace(request_id="keep", data=keep_data),
            types.SimpleNamespace(request_id="skip", data=skip_data),
        ]
    )

    runner._finalize(
        batch_result,
        types.SimpleNamespace(),
        schedule_batch,
        model_worker_batch,
        scheduler_output,
        skip_rids={"skip"},
    )

    assert keep_data.generation_steps == 1
    assert keep_data.extra_model_outputs == {"seen": "keep"}
    assert skip_data.generation_steps == 0
    assert skip_data.extra_model_outputs == {}


def test_finalize_unions_finalize_skip_rids_hook():
    # finalize_skip_rids() (default empty on base) is unioned into skip_rids
    # inside _finalize, so a model can suppress generation_steps for rows it
    # sampled but must not count (e.g. non-final chunked prefill) even when the
    # caller passes no skip_rids.
    class _OutputProcessor:
        def process(self, batch_result, scheduler_output):
            del batch_result
            return {
                req.request_id: RequestOutput(request_id=req.request_id, extra={})
                for req in scheduler_output.requests
            }

    runner = ModelRunner.__new__(ModelRunner)
    runner.output_processor = _OutputProcessor()
    runner.finalize_skip_rids = lambda scheduler_output: {"chunk"}
    batch_result = types.SimpleNamespace(
        next_token_ids=torch.tensor([1, 2]),
        logits_output=None,
        can_run_cuda_graph=False,
    )
    schedule_batch = types.SimpleNamespace(is_prefill_only=False, output_ids=None)
    normal_data = types.SimpleNamespace(generation_steps=0, extra_model_outputs={})
    chunk_data = types.SimpleNamespace(generation_steps=0, extra_model_outputs={})
    scheduler_output = types.SimpleNamespace(
        requests=[
            types.SimpleNamespace(request_id="normal", data=normal_data),
            types.SimpleNamespace(request_id="chunk", data=chunk_data),
        ]
    )

    runner._finalize(
        batch_result,
        types.SimpleNamespace(),
        schedule_batch,
        types.SimpleNamespace(),
        scheduler_output,
    )

    # The hook rid is skipped with no skip_rids arg; the normal row advances.
    assert normal_data.generation_steps == 1
    assert chunk_data.generation_steps == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires CUDA")
def test_host_staging_pingpong():
    r = _StubRunner()
    dev = torch.zeros(8, 18)
    b0 = r._next_host_staging(dev)
    b1 = r._next_host_staging(dev)
    b2 = r._next_host_staging(dev)
    assert len(r._host_staging_buffers) == 2
    assert b0 is b2 and b0 is not b1  # ping-pong between exactly 2 buffers
    assert b0.is_pinned() and tuple(b0.shape) == (8, 18) and b0.dtype == dev.dtype


def test_batch_is_decode():
    decode = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(
            is_decode=lambda: True, is_extend=lambda: False
        )
    )
    extend = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(
            is_decode=lambda: False, is_extend=lambda: True
        )
    )
    assert OmniScheduler._batch_is_decode(decode) is True
    assert OmniScheduler._batch_is_decode(extend) is False
    assert (
        OmniScheduler._batch_is_decode(types.SimpleNamespace(forward_mode=None))
        is False
    )


def test_async_pending_batch_getattr_safe():
    # OmniScheduler.__getattr__ raises for unset attrs; _async_pending_batch
    # must tolerate that (test fixtures may bypass __init__).
    s = OmniScheduler.__new__(OmniScheduler)
    assert s._async_pending_batch() is None
    s._async_pending = ("batchX", "sched_out", "pending_step")
    assert s._async_pending_batch() == "batchX"


# ---------------------------------------------------------------------------
# Fast path: bs < async_decode_min_batch_size bypasses the lookahead and runs a
# plain synchronous step (avoids the bs=1 overhead regression). Drives the real
# _event_loop_async_decode with stubbed deps over a scripted batch-size sequence
# that exercises the 1 -> 2 -> 2 -> 1 -> 1 transitions, incl. the bs>=2 -> bs=1
# drain.
# ---------------------------------------------------------------------------


class _FakeBatch:
    def __init__(self, n):
        # real ScheduleBatch.reqs are Reqs with .finished(); none finish here
        self.reqs = [types.SimpleNamespace(finished=lambda: False) for _ in range(n)]

    def copy(self):
        return self


def _new_scheduler_for_async_loop():
    s = OmniScheduler.__new__(OmniScheduler)
    s._admin_lock = threading.Lock()
    s._admin_queue = queue.Queue()
    return s


def _drive_loop(seq, min_bs=2):
    """Run the real event loop over `seq` (each item = bs int, or None for idle)
    and return the ordered list of path events taken."""
    events = []
    s = _new_scheduler_for_async_loop()
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
    s.self_check_during_idle = lambda: events.append("idle")
    s.self_check_during_busy = lambda: None

    def launch(b):
        events.append("launch")
        return ("sched_output", "pending_step")

    s._run_batch_launch = launch
    s._resolve_and_process = lambda pb, ps, pstep: events.append("resolve")
    # use the REAL drain helper so the bs>=2 -> bs=1 transition is exercised
    s._resolve_pending_async = OmniScheduler._resolve_pending_async.__get__(s)

    def run_batch(b):
        events.append("sync")
        return object()  # not _FAILED_BATCH_RESULT

    s.run_batch = run_batch
    s.process_batch_result = lambda b, r: None

    batches = [None if n is None else _FakeBatch(n) for n in seq]
    state = {"i": 0}

    def gnb():
        i = state["i"]
        state["i"] += 1
        if i >= len(batches) - 1:
            s._running = False  # stop after the final scripted item
        return batches[i] if i < len(batches) else None

    s.get_next_batch_to_run = gnb
    s._event_loop_async_decode()
    return events, s


def test_fast_path_bs1_bypasses_lookahead_and_drains_on_transition():
    # bs sequence: 1, 2, 2, 1, 1, idle
    events, s = _drive_loop([1, 2, 2, 1, 1, None], min_bs=2)
    assert events == [
        "sync",  # bs1: fast path (no pending to drain)
        "launch",  # bs2: lookahead, no prev pending
        "launch",
        "resolve",  # bs2: lookahead launch + resolve prev
        "resolve",
        "sync",  # bs1: DRAIN the in-flight bs2 step, then sync
        "sync",  # bs1: fast path, nothing to drain
        "idle",  # empty
    ]
    # the in-flight step was drained -> no pending left stranded
    assert s._async_pending is None


def test_fast_path_threshold_one_keeps_all_decode_on_lookahead():
    # min_bs=1 -> even bs=1 uses lookahead (fast path disabled). The trailing
    # empty step drains the last in-flight launch before going idle.
    events, _ = _drive_loop([1, 1, None], min_bs=1)
    assert events == ["launch", "launch", "resolve", "resolve", "idle"]


def test_fast_path_threshold_four_routes_bs1_to_3_sync():
    # min_bs=4 -> bs=3 still bypasses (sync); bs=4 uses lookahead; the trailing
    # empty step drains the bs=4 launch.
    events, _ = _drive_loop([3, 4, None], min_bs=4)
    assert events == ["sync", "launch", "resolve", "idle"]


# ---------------------------------------------------------------------------
# Stale-batch overrun regression: the fast-path `batch` is built (get_next_batch
# _to_run, top of loop) BEFORE the in-flight lookahead step is drained. If the
# drain finishes a req that is also present in that batch, running the batch
# again re-frees its KV cache (process_batch_result_decode -> release_kv_cache
# -> pop_committed_kv_cache asserts "Committed KV cache already freed"). This is
# the talker async-ON crash at bs>=2; the talker is hit because it marks no
# early (sampler) finish, so every finish is detected only in the resolve half.
# ---------------------------------------------------------------------------


class _DFReq:
    def __init__(self, name):
        self.name = name
        self._done = False

    def finished(self):
        return self._done


class _DFBatch:
    """ScheduleBatch stand-in: shares Req objects on copy() (as the real
    .copy() does) and drops finished reqs on filter_batch() (as the real one
    does when keep_indices is None)."""

    def __init__(self, reqs):
        self.reqs = list(reqs)

    def copy(self):
        return _DFBatch(self.reqs)

    def filter_batch(self, keep_indices=None):
        if keep_indices is None:
            keep_indices = [i for i, r in enumerate(self.reqs) if not r.finished()]
        self.reqs = [self.reqs[i] for i in keep_indices]

    def is_empty(self):
        return not self.reqs


def test_fast_path_does_not_double_free_req_finished_by_drain():
    victim = _DFReq("victim")
    other = _DFReq("other")
    running = [victim, other]  # both in flight at the start
    freed = set()
    double_freed = []

    def release_kv(req):
        if req.name in freed:
            double_freed.append(req.name)
        freed.add(req.name)

    s = _new_scheduler_for_async_loop()
    s._running = True
    s._engine_paused = False
    s._async_pending = None
    s.async_decode_min_batch_size = 2
    s.cur_batch = None
    s.last_batch = None
    s.recv_requests = lambda: []
    s._take_deferred_request_payloads = lambda: []
    s.process_input_requests = lambda r: None
    s._batch_is_decode = lambda b: True
    s.self_check_during_idle = lambda: None
    s.self_check_during_busy = lambda: None
    s._run_batch_launch = lambda b: ("sched_output", "pending_step")
    # real drain helper -> exercises the real fast-path ordering under test
    s._resolve_pending_async = OmniScheduler._resolve_pending_async.__get__(s)

    # Resolving a step finishes the next scheduled req and frees its KV (mirrors
    # process_batch_result_decode -> release_kv_cache). other finishes first
    # (bs 2 -> 1), then victim finishes in the bs=1 fast-path drain.
    finish_order = [other, victim]

    def resolve_and_process(pb, ps, pstep):
        if finish_order:
            r = finish_order.pop(0)
            r._done = True
            release_kv(r)
            if r in running:
                running.remove(r)

    s._resolve_and_process = resolve_and_process

    s.run_batch = lambda b: object()  # not _FAILED_BATCH_RESULT

    def process_batch_result(b, r):
        # process_batch_result_decode frees any req that is finished() at this
        # step. For a stale batch carrying a req the drain already finished,
        # this is the double free.
        for req in b.reqs:
            if req.finished():
                release_kv(req)

    s.process_batch_result = process_batch_result

    state = {"i": 0}

    def gnb():
        state["i"] += 1
        if not running:
            s._running = False
            return None
        return _DFBatch(list(running))

    s.get_next_batch_to_run = gnb
    s._event_loop_async_decode()

    assert not double_freed, f"KV double-freed (stale fast-path batch): {double_freed}"


def _scaffold_async_loop(*, async_pending=None):
    s = _new_scheduler_for_async_loop()
    s._running = True
    s._engine_paused = False
    s._async_pending = async_pending
    s.async_decode_min_batch_size = 2
    s.cur_batch = None
    s.last_batch = None
    s.recv_requests = lambda: []
    s._take_deferred_request_payloads = lambda: []
    s.process_input_requests = lambda r: None
    s._batch_is_decode = lambda b: True
    s.self_check_during_idle = lambda: None
    s.self_check_during_busy = lambda: None
    s._resolve_pending_async = OmniScheduler._resolve_pending_async.__get__(s)
    return s


def test_async_path_launch_failure_calls_handle_batch_failure():
    failures = []
    s = _scaffold_async_loop()

    def launch(b):
        raise RuntimeError("launch boom")

    s._run_batch_launch = launch
    s._resolve_and_process = lambda *a, **kw: None
    s._handle_batch_failure = lambda b, exc: failures.append((b, type(exc), str(exc)))

    batch = _FakeBatch(2)
    batches = [batch]
    state = {"i": 0}

    def gnb():
        i = state["i"]
        state["i"] += 1
        if i >= 0:
            s._running = False
        return batches[i] if i < len(batches) else None

    s.get_next_batch_to_run = gnb
    s._event_loop_async_decode()

    assert failures == [(batch, RuntimeError, "launch boom")]
    # launch failed before _async_pending was set; prev state preserved.
    assert s._async_pending is None


def test_async_path_resolve_failure_calls_handle_batch_failure():
    failures = []
    prev_batch = _FakeBatch(2)
    s = _scaffold_async_loop(
        async_pending=(prev_batch, "prev_sched", "prev_step"),
    )

    s._run_batch_launch = lambda b: ("sched_output", "pending_step")

    def resolve(pb, ps, pstep):
        raise RuntimeError("resolve boom")

    s._resolve_and_process = resolve
    s._handle_batch_failure = lambda b, exc: failures.append((b, type(exc), str(exc)))

    new_batch = _FakeBatch(2)
    batches = [new_batch]
    state = {"i": 0}

    def gnb():
        i = state["i"]
        state["i"] += 1
        if i >= 0:
            s._running = False
        return batches[i] if i < len(batches) else None

    s.get_next_batch_to_run = gnb
    s._event_loop_async_decode()

    assert failures == [(prev_batch, RuntimeError, "resolve boom")]
    # launch succeeded; _async_pending was rotated to the new batch.
    assert s._async_pending is not None
    assert s._async_pending[0] is new_batch


def test_drain_resolve_failure_calls_handle_batch_failure():
    failures = []
    stranded_batch = _FakeBatch(2)
    s = OmniScheduler.__new__(OmniScheduler)
    s._async_pending = (stranded_batch, "sched", "step")

    def resolve(pb, ps, pstep):
        raise RuntimeError("drain boom")

    s._resolve_and_process = resolve
    s._handle_batch_failure = lambda b, exc: failures.append((b, type(exc), str(exc)))

    OmniScheduler._resolve_pending_async(s)

    assert failures == [(stranded_batch, RuntimeError, "drain boom")]
    assert s._async_pending is None
