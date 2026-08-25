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
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
)
from tests.unit_test.fakes import FakeExecutionBridge

_STUB_DEVICE = torch.device("cpu")


class _StubRunner(ModelRunner):
    """ModelRunner with mocked sub-steps; exercises only execute_launch/resolve."""

    def __init__(self):
        self.device = _STUB_DEVICE
        self._async_enabled = True
        self._execution_bridge = FakeExecutionBridge(_STUB_DEVICE)
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
        sb = types.SimpleNamespace(is_prefill_only=False, input_ids=None)
        sb.copy = lambda: sb
        self.last_schedule_batch = sb
        return types.SimpleNamespace(), sb, False  # decode

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
            next_token_ids=torch.tensor([17], dtype=torch.long),
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
        scheduler_output,
        skip_rids=None,
    ):
        self.finalize_calls += 1
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

    return mock.patch.object(torch.get_device_module(_STUB_DEVICE), "Event", _FakeEvent)


def test_launch_event_comes_from_the_runners_device_module():
    """execute_launch must build the event on the runner's own device backend.

    Guards against a regression to a hardcoded backend (e.g. torch.cuda.Event or
    torch.cpu.Event): a fixed-backend call would not construct the patched class.
    The device is this host's live backend rather than a literal, because an
    uncompiled backend's Event can be referenced but not instantiated ("dummy
    base class Event"), so a hardcoded one would fail on other builds' CI.
    """
    from sglang_omni.platforms import current_platform

    accel = torch.device(current_platform.device_type)
    runner = _StubRunner()
    runner.device = accel
    runner._execution_bridge = FakeExecutionBridge(accel)

    seen = []
    real_event = torch.get_device_module(accel).Event

    class _RecordingEvent(real_event):  # type: ignore[misc,valid-type]
        def __new__(cls, *a, **k):
            seen.append(cls)
            return super().__new__(cls)

    with mock.patch.object(torch.get_device_module(accel), "Event", _RecordingEvent):
        pending = runner.execute_launch(_sched_output(1))

    assert seen, "no event built from the runner's device module"
    assert pending is not None
    assert isinstance(pending.event, real_event)


def _sched_output(n):
    req_stub = types.SimpleNamespace(finished=lambda: False, is_retracted=False)
    return SchedulerOutput(
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
        assert step is not None
        out = r.execute_resolve(step)
    assert out is not None
    assert (r.launch_calls, r.resolve_calls, r.finalize_calls) == (1, 1, 1)
    assert (r._async_query_hit, r._async_query_miss) == (1, 0)
    assert r.last_prepare_is_lookahead is True
    assert len(r._execution_bridge.published) == 1
    published_batch, published_ids = r._execution_bridge.published[0]
    assert published_batch is r.last_schedule_batch
    assert torch.equal(published_ids, torch.tensor([17]))
    # resolve must NOT re-publish the token rail: under launch-first it runs
    # one step behind on the LIVE running batch, whose rail the current launch
    # already published at the right length. Re-stamping the lagged step's
    # tokens leaves a stale-length rail -> input_ids/seq_lens mismatch once a
    # req finishes mid-batch (the bs>1 replay crash). The launch publishes it.


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
    keep_req = types.SimpleNamespace(finished=lambda: False, is_retracted=False)
    skip_req = types.SimpleNamespace(finished=lambda: True, is_retracted=False)
    sched_output = SchedulerOutput(
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
    sched_output = SchedulerOutput(
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


def test_base_lookahead_eligible_gates_output_history_sampling():
    """The base default launch samples with host ``req.output_ids`` one step
    stale (and before the sglang penalizer state is cumulated), so any
    output-history-dependent sampling term — repetition / frequency / presence
    penalty, or ``min_new_tokens`` EOS suppression — would systematically
    diverge from the sync path. Those batches must route synchronously. The
    scheduler's async gate consults this alongside the min-batch-size /
    is-decode checks.
    """

    def _sp(**overrides):
        params = dict(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            min_new_tokens=0,
        )
        params.update(overrides)
        return types.SimpleNamespace(**params)

    def _req(*, custom_logit_processor=None, **sampling_overrides):
        return types.SimpleNamespace(
            sampling_params=_sp(**sampling_overrides),
            custom_logit_processor=custom_logit_processor,
        )

    def _batch(*reqs):
        return types.SimpleNamespace(reqs=list(reqs))

    runner = _StubRunner()
    history_free = _req(custom_logit_processor=None)
    history_dependent = _req(custom_logit_processor="processor")

    # Empty batch and all-history-free rows are eligible.
    assert runner.lookahead_eligible(_batch()) is True
    assert runner.lookahead_eligible(_batch(history_free)) is True
    # Any output-history-dependent term on any row forces sync.
    assert runner.lookahead_eligible(_batch(_req(repetition_penalty=1.05))) is False
    assert runner.lookahead_eligible(_batch(_req(frequency_penalty=0.5))) is False
    assert runner.lookahead_eligible(_batch(_req(presence_penalty=0.5))) is False
    assert runner.lookahead_eligible(_batch(_req(min_new_tokens=4))) is False
    assert runner.lookahead_eligible(_batch(history_dependent)) is False
    # A single tainted row taints the whole batch.
    assert (
        runner.lookahead_eligible(_batch(_req(), _req(frequency_penalty=0.1))) is False
    )
    assert runner.lookahead_eligible(_batch(history_free, history_dependent)) is False


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
        scheduler_output,
    )

    # The hook rid is skipped with no skip_rids arg; the normal row advances.
    assert normal_data.generation_steps == 1
    assert chunk_data.generation_steps == 0


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires CUDA")
def test_host_staging_pingpong():
    r = _StubRunner()
    b0 = r._next_host_staging((8, 18), torch.float32)
    b1 = r._next_host_staging((8, 18), torch.float32)
    b2 = r._next_host_staging((8, 18), torch.float32)
    assert len(r._host_staging_buffers) == 2
    assert b0 is b2 and b0 is not b1  # ping-pong between exactly 2 buffers
    assert b0.is_pinned() and tuple(b0.shape) == (8, 18) and b0.dtype == torch.float32


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires CUDA")
def test_default_launch_resolve_pinned_snapshot_pingpong():
    """Base (plain-LM) launch/resolve: launch snapshots the sampled ids into a
    pinned buffer, resolve points ``result.next_token_ids`` at that snapshot,
    and the ping-pong depth matches the "at most one in-flight step" invariant
    — launch(N+1) must not write the buffer resolve(N) will read, and
    launch(N+2) may reuse it only after resolve(N) consumed it.
    """
    r = ModelRunner.__new__(ModelRunner)
    r._staging_slot = 0
    r._host_staging_buffers = []
    reqs = [object(), object()]

    def _result(vals):
        return types.SimpleNamespace(
            next_token_ids=torch.tensor(vals, device="cuda"), logits_output=None
        )

    r1, r2 = _result([11, 12]), _result([21, 22])
    buf1 = r.post_decode_launch(r1, None, reqs)
    buf2 = r.post_decode_launch(r2, None, reqs)
    assert buf1 is not buf2
    torch.cuda.synchronize()  # stands in for the recorded launch event wait
    r.post_decode_resolve(buf1, r1, None, None, reqs)
    r.post_decode_resolve(buf2, r2, None, None, reqs)
    assert r1.next_token_ids.tolist() == [11, 12]
    assert r2.next_token_ids.tolist() == [21, 22]
    # resolve hands downstream a pinned host view, so the output processor's
    # .tolist() cannot trigger a GPU sync
    assert r1.next_token_ids.device.type == "cpu" and r1.next_token_ids.is_pinned()
    buf3 = r.post_decode_launch(_result([31, 32]), None, reqs)
    assert buf3 is buf1


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires CUDA")
def test_default_launch_staging_grows_then_slices_to_smaller_batch():
    r = ModelRunner.__new__(ModelRunner)
    r._staging_slot = 0
    r._host_staging_buffers = []
    big = types.SimpleNamespace(
        next_token_ids=torch.tensor([1, 2, 3, 4], device="cuda"), logits_output=None
    )
    r.post_decode_launch(big, None, [object()] * 4)
    small = types.SimpleNamespace(
        next_token_ids=torch.tensor([7, 8], device="cuda"), logits_output=None
    )
    buf_small = r.post_decode_launch(small, None, [object()] * 2)
    assert buf_small.numel() == 4  # capacity retained; no realloc on shrink
    torch.cuda.synchronize()
    r.post_decode_resolve(buf_small, small, None, None, [object()] * 2)
    assert small.next_token_ids.tolist() == [7, 8]


def _real_radix_pools(size=64):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache

    kv = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=size, dtype=torch.float16, device="cpu", kvcache=kv, need_sort=False
    )
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    return allocator, req_to_token_pool, cache


def _decoding_req(allocator, req_to_token_pool, rid, prompt, outputs):
    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(rid, "", list(prompt), SamplingParams(max_new_tokens=8))
    req_to_token_pool.alloc([req])
    n = len(prompt)
    slots = allocator.alloc(n)
    req_to_token_pool.write((req.req_pool_idx, slice(0, n)), slots.to(torch.int32))
    req.kv = ReqKvInfo(kv_allocated_len=n, swa_evicted_seqlen=0)
    req.kv_committed_len = n
    req.output_ids = [outputs[0]]
    for tok in outputs[1:]:
        _commit_step_slot(allocator, req_to_token_pool, req)
        req.output_ids.append(tok)
    return req


def _commit_step_slot(allocator, req_to_token_pool, req):
    slot = allocator.alloc(1)
    pos = req.kv.kv_allocated_len
    req_to_token_pool.write(
        (req.req_pool_idx, slice(pos, pos + 1)), slot.to(torch.int32)
    )
    req.kv.kv_allocated_len += 1
    req.kv_committed_len += 1
    return int(slot[0])


class _StaleDecodeBatch:
    def __init__(self, reqs, out_cache_loc):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.reqs = list(reqs)
        self.out_cache_loc = out_cache_loc
        self.forward_mode = ForwardMode.DECODE
        self.decoding_reqs = None

    def filter_batch(self, keep_indices=None):
        self.reqs = [self.reqs[i] for i in keep_indices]
        self.out_cache_loc = None


def test_drop_stale_overrun_leaves_drained_rows_single_owned():
    from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
    from sglang.srt.mem_cache.common import release_kv_cache
    from sglang.srt.mem_cache.radix_cache import MatchPrefixParams, RadixKey
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(page_size=1):
        allocator, req_to_token_pool, cache = _real_radix_pools()
        total = allocator.available_size()
        survivor = _decoding_req(allocator, req_to_token_pool, "s", [1, 2], [20])
        finished = _decoding_req(allocator, req_to_token_pool, "f", [3, 4], [30, 31])
        retracted = _decoding_req(allocator, req_to_token_pool, "r", [5, 6], [40])
        step_slots = [
            _commit_step_slot(allocator, req_to_token_pool, req)
            for req in (survivor, finished, retracted)
        ]

        finished.finished_reason = FINISH_MATCHED_TOKEN(matched=31)
        release_kv_cache(finished, cache)
        retracted.is_retracted = True
        release_kv_cache(retracted, cache, is_insert=False)

        s = OmniScheduler.__new__(OmniScheduler)
        s.page_size = 1
        s.server_args = types.SimpleNamespace(disable_radix_cache=False)
        s.token_to_kv_pool_allocator = allocator
        batch = _StaleDecodeBatch(
            [survivor, finished, retracted], torch.tensor(step_slots)
        )
        out = s._drop_stale_overrun(batch)

        assert out is batch
        assert [req.rid for req in out.reqs] == ["s"]
        assert out.out_cache_loc.tolist() == [step_slots[0]]

        survivor_slots = survivor.kv.kv_allocated_len
        assert (
            allocator.available_size() + cache.evictable_size()
            == total - survivor_slots
        )
        free = allocator.free_pages.tolist()
        cached = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([3, 4, 30, 31]))
        ).device_indices.tolist()
        assert step_slots[0] not in free and step_slots[0] not in cached
        assert free.count(step_slots[1]) == 0 and cached.count(step_slots[1]) == 1
        assert free.count(step_slots[2]) == 1 and step_slots[2] not in cached

        assert (
            s._drop_stale_overrun(
                _StaleDecodeBatch([finished, retracted], torch.tensor(step_slots[1:]))
            )
            is None
        )
        assert (
            allocator.available_size() + cache.evictable_size()
            == total - survivor_slots
        )
        clean = _StaleDecodeBatch([survivor], torch.tensor(step_slots[:1]))
        assert s._drop_stale_overrun(clean) is clean


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


def test_async_pending_batch_uses_initialized_state():
    s = OmniScheduler.__new__(OmniScheduler)
    s._async_pending = None
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
        self.reqs = [
            types.SimpleNamespace(
                finished=lambda: False,
                is_retracted=False,
                sampling_params=types.SimpleNamespace(
                    repetition_penalty=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    min_new_tokens=0,
                ),
                custom_logit_processor=None,
            )
            for _ in range(n)
        ]
        self.out_cache_loc = torch.arange(n)

    def copy(self):
        return self

    def filter_batch(self, keep_indices):
        self.reqs = [self.reqs[i] for i in keep_indices]
        self.out_cache_loc = None


def _new_scheduler_for_async_loop():
    s = OmniScheduler.__new__(OmniScheduler)
    s._admin_lock = threading.Lock()
    s._admin_queue = queue.Queue()
    s._request_admission_lock = threading.RLock()
    s._pending_request_builds = {}
    s._pending_request_admissions = {}
    s._model_runner = None
    s.chunked_req = None
    s.is_mixed_chunk = False
    s.page_size = 1
    s.running_batch = types.SimpleNamespace(batch_is_full=False)
    s.server_args = types.SimpleNamespace(disable_radix_cache=False)
    s.token_to_kv_pool_allocator = types.SimpleNamespace(free=lambda _: None)
    s.waiting_queue = []
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


def test_custom_logit_processor_transitions_async_sync_async():
    events = []
    s = _scaffold_async_loop()
    s._model_runner = _StubRunner()
    launch_events = iter(("launch async N", "launch async N+2"))

    def launch_async(batch):
        events.append(next(launch_events))
        return "sched_output", "pending_step"

    s._run_batch_launch = launch_async
    s._resolve_and_process = lambda *args: events.append("resolve N")
    s.run_batch = lambda batch: events.append("run sync N+1") or object()
    s.process_batch_result = lambda batch, result: None

    timestamp_batch = _FakeBatch(2)
    timestamp_batch.reqs[1].custom_logit_processor = "processor"
    batches = [_FakeBatch(2), timestamp_batch, _FakeBatch(2)]
    state = {"i": 0}

    def get_next_batch_to_run():
        i = state["i"]
        state["i"] += 1
        if i == len(batches) - 1:
            s._running = False
        return batches[i]

    s.get_next_batch_to_run = get_next_batch_to_run
    s._event_loop_async_decode()

    assert events == [
        "launch async N",
        "resolve N",
        "run sync N+1",
        "launch async N+2",
    ]


@pytest.mark.parametrize(
    (
        "is_mixed_chunk",
        "batch_is_full",
        "prefill_source",
        "has_pending_at_schedule",
    ),
    [
        pytest.param(True, False, "waiting", False, id="mixed-waiting-prefill"),
        pytest.param(False, False, "waiting", True, id="non-mixed-prefill"),
        pytest.param(True, True, "chunked", False, id="mixed-chunked-prefill"),
    ],
)
def test_pending_decode_drain_order_for_prefill(
    is_mixed_chunk,
    batch_is_full,
    prefill_source,
    has_pending_at_schedule,
):
    events = []
    pending_during_schedule = []
    pending_batch = _FakeBatch(2)
    pending = (pending_batch, "prev_sched", "prev_step")
    s = _scaffold_async_loop(async_pending=pending)
    s.is_mixed_chunk = is_mixed_chunk
    s.running_batch.batch_is_full = batch_is_full
    s.waiting_queue = [object()] if prefill_source == "waiting" else []
    s.chunked_req = object() if prefill_source == "chunked" else None
    s._batch_is_decode = lambda batch: False
    s._resolve_and_process = lambda *args: events.append("resolve")
    s.process_batch_result = lambda batch, result: None

    def run_batch(batch):
        events.append("prefill")
        return object()

    s.run_batch = run_batch

    def get_next_batch_to_run():
        events.append("schedule")
        pending_during_schedule.append(s._async_pending)
        s._running = False
        return _FakeBatch(1)

    s.get_next_batch_to_run = get_next_batch_to_run
    s._event_loop_async_decode()

    expected_events = (
        ["schedule", "resolve", "prefill"]
        if has_pending_at_schedule
        else ["resolve", "schedule", "prefill"]
    )
    assert events == expected_events
    expected_pending = pending if has_pending_at_schedule else None
    assert pending_during_schedule[0] is expected_pending


def test_full_running_batch_keeps_lookahead_with_waiting_requests():
    events = []
    pending_batch = _FakeBatch(2)
    pending = (pending_batch, "prev_sched", "prev_step")
    s = _scaffold_async_loop(async_pending=pending)
    s.is_mixed_chunk = True
    s.running_batch.batch_is_full = True
    s.waiting_queue = [object()]
    s._resolve_and_process = lambda *args: events.append("resolve")

    def launch(batch):
        events.append("launch")
        return "sched_output", "pending_step"

    s._run_batch_launch = launch

    def get_next_batch_to_run():
        events.append(("schedule", s._async_pending is pending))
        s._running = False
        return _FakeBatch(2)

    s.get_next_batch_to_run = get_next_batch_to_run
    s._event_loop_async_decode()

    assert events == [("schedule", True), "launch", "resolve"]


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
        self.is_retracted = False

    def finished(self):
        return self._done


class _DFBatch:
    """ScheduleBatch stand-in: shares Req objects on copy() (as the real
    .copy() does) and drops finished reqs on filter_batch() (as the real one
    does when keep_indices is None)."""

    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.forward_mode = types.SimpleNamespace(
            is_decode=lambda: True, is_extend=lambda: False
        )
        self.out_cache_loc = torch.arange(100, 100 + len(reqs))
        self.decoding_reqs = None

    def copy(self):
        return _DFBatch(self.reqs)

    def filter_batch(self, keep_indices=None):
        if keep_indices is None:
            keep_indices = [i for i, r in enumerate(self.reqs) if not r.finished()]
        self.reqs = [self.reqs[i] for i in keep_indices]
        self.out_cache_loc = None

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


class _MixedBatch:
    def __init__(self, lens, done, lp_start_lens=None, logprob=None):
        self.forward_mode = types.SimpleNamespace(
            is_decode=lambda: False, is_extend=lambda: True
        )
        logprob = logprob or [False] * len(lens)
        self.reqs = [
            types.SimpleNamespace(
                finished=lambda d=d: d,
                return_logprob=lp,
                is_retracted=False,
            )
            for d, lp in zip(done, logprob)
        ]
        self.extend_lens = list(lens)
        self.extend_num_tokens = sum(lens)
        self.prefix_lens = [10 * (i + 1) for i in range(len(lens))]
        self.extend_logprob_start_lens = (
            list(lp_start_lens) if lp_start_lens else [0] * len(lens)
        )
        self.out_cache_loc = torch.arange(100, 100 + sum(lens))
        self.input_ids = torch.arange(sum(lens))
        self.input_embeds = None
        self.replace_embeds = None
        self.mix_running_indices = None
        self.prefill_input_ids_cpu = None
        self.decoding_reqs = None
        self.return_logprob = any(logprob)
        if self.return_logprob:
            self.extend_input_logprob_token_ids = torch.tensor(
                [
                    100 * (i + 1) + k
                    for i, (length, start) in enumerate(
                        zip(self.extend_lens, self.extend_logprob_start_lens)
                    )
                    for k in range(length - start)
                ]
            )
        else:
            self.extend_input_logprob_token_ids = None

    def filter_batch(self, keep_indices):
        self.reqs = [self.reqs[i] for i in keep_indices]
        self.out_cache_loc = None
        self.return_logprob = any(r.return_logprob for r in self.reqs)


def _drop_stale_scheduler():
    return OmniScheduler.__new__(OmniScheduler)


def test_drop_stale_overrun_mixed_reslices_per_token():
    s = _drop_stale_scheduler()
    batch = _MixedBatch(lens=[3, 1, 1], done=[False, True, False])
    out = s._drop_stale_overrun(batch)
    assert out is batch
    assert out.out_cache_loc.tolist() == [100, 101, 102, 104]
    assert out.input_ids.tolist() == [0, 1, 2, 4]
    assert out.extend_lens == [3, 1]
    assert out.extend_num_tokens == 4
    assert out.prefix_lens == [10, 30]
    assert out.extend_input_logprob_token_ids is None


def test_drop_stale_overrun_extend_multitoken_drop():
    s = _drop_stale_scheduler()
    batch = _MixedBatch(lens=[2, 3], done=[True, False])
    out = s._drop_stale_overrun(batch)
    assert out.out_cache_loc.tolist() == [102, 103, 104]
    assert out.input_ids.tolist() == [2, 3, 4]
    assert out.extend_lens == [3]
    assert out.extend_num_tokens == 3
    assert out.prefix_lens == [20]


def test_drop_stale_overrun_reslices_deferred_prefill_tokens():
    s = _drop_stale_scheduler()
    batch = _MixedBatch(lens=[2, 3], done=[True, False])
    batch.prefill_input_ids_cpu = batch.input_ids
    batch.input_ids = None

    out = s._drop_stale_overrun(batch)

    assert out.input_ids is None
    assert out.prefill_input_ids_cpu.tolist() == [2, 3, 4]


def test_drop_stale_overrun_rejects_mixed_deferred_prefill():
    s = _drop_stale_scheduler()
    batch = _MixedBatch(lens=[2, 3], done=[True, False])
    batch.mix_running_indices = torch.tensor([1])

    with pytest.raises(RuntimeError, match="mixed chunked-prefill"):
        s._drop_stale_overrun(batch)


def test_drop_stale_overrun_reslices_logprob_token_ids():
    s = _drop_stale_scheduler()
    batch = _MixedBatch(
        lens=[3, 2, 2],
        done=[False, True, False],
        lp_start_lens=[1, 0, 2],
        logprob=[True, True, True],
    )
    assert batch.extend_input_logprob_token_ids.tolist() == [100, 101, 200, 201]
    out = s._drop_stale_overrun(batch)
    assert out.extend_input_logprob_token_ids.tolist() == [100, 101]
    assert out.extend_lens == [3, 2]
    assert out.extend_logprob_start_lens == [1, 2]


def test_drop_stale_overrun_drops_last_logprob_req():
    s = _drop_stale_scheduler()
    batch = _MixedBatch(lens=[2, 2], done=[True, False], logprob=[True, False])
    out = s._drop_stale_overrun(batch)
    assert out.return_logprob is False
    assert out.extend_input_logprob_token_ids is None


def test_drop_stale_overrun_filters_decoding_reqs():
    # dropped folded-decode row must also be removed from decoding_reqs
    s = _drop_stale_scheduler()
    batch = _MixedBatch(lens=[3, 1, 1], done=[False, True, False])
    batch.decoding_reqs = [batch.reqs[1], batch.reqs[2]]
    live_decode = batch.reqs[2]
    out = s._drop_stale_overrun(batch)
    assert out.decoding_reqs == [live_decode]
