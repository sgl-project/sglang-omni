# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_continuation import (
    decode_continuation,
    encode_continuation,
)
from sglang_omni.scheduling.pd_kv_adapter import ReservedAllocation
from sglang_omni.scheduling.pd_runtime import (
    DecodeRequestPoolExhausted,
    PDDecodeAdmission,
    continuation_from_req,
    req_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class _ReqPool:
    def __init__(self):
        self.req_to_token = torch.zeros((4, 32), dtype=torch.int64)

    def alloc(self, reqs):
        for index, req in enumerate(reqs):
            req.req_pool_idx = index
        return torch.arange(len(reqs), dtype=torch.int64)

    def write(self, key, value):
        self.req_to_token[key] = value

    def free(self, req):
        req.req_pool_idx = None


class _FailingReqPool(_ReqPool):
    def __init__(self):
        super().__init__()
        self.freed = False

    def write(self, key, value):
        raise RuntimeError("mapping write failed")

    def free(self, req):
        self.freed = True
        super().free(req)


class _CapacityReqPool(_ReqPool):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.active: dict[int, Req] = {}

    def alloc(self, reqs):
        if len(self.active) + len(reqs) > self.capacity:
            return None
        indices = []
        for req in reqs:
            index = next(index for index in range(4) if index not in self.active)
            req.req_pool_idx = index
            self.active[index] = req
            indices.append(index)
        return torch.tensor(indices, dtype=torch.int64)

    def free(self, req):
        self.active.pop(req.req_pool_idx, None)
        super().free(req)


def _prefill_req() -> Req:
    sampling = SamplingParams(
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
        stop_token_ids={2},
        sampling_seed=17,
    )
    req = Req(
        rid="request-1",
        origin_input_text="",
        origin_input_ids=array("q", [10, 11, 12]),
        sampling_params=sampling,
        vocab_size=128,
        eos_token_ids={2},
    )
    req.output_ids.append(42)
    req.cached_tokens = 1
    req.cached_tokens_device = 1
    payload = StagePayload(
        request_id=req.rid,
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={"state": "generic"},
    )
    data = SGLangARRequestData(
        req=req,
        output_ids=req.output_ids,
        stage_payload=payload,
        input_embeds_are_projected=False,
    )
    req._omni_data = data
    return req


def _decode_admission(request_id: str, slot: int) -> PDDecodeAdmission:
    source = _prefill_req()
    source.rid = request_id
    continuation = continuation_from_req(source, f"transfer-{request_id}")
    allocation = ReservedAllocation(
        slots=torch.tensor([slot, slot + 1, slot + 2], dtype=torch.int64),
        page_indices=(slot, slot + 1, slot + 2),
        seq_len=3,
    )
    return PDDecodeAdmission(continuation, allocation)


def _decode_scheduler(
    admissions: list[PDDecodeAdmission],
    pool: _ReqPool,
    *,
    state_restorer=None,
) -> tuple[OmniScheduler, list[torch.Tensor]]:
    freed: list[torch.Tensor] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_ready_queue = queue.Queue()
    for admission in admissions:
        scheduler._pd_ready_queue.put(admission)
    scheduler._pd_deferred_admission = None
    scheduler._pd_admission_lock = threading.Lock()
    scheduler._pd_state_restorer = state_restorer
    scheduler._aborted_request_ids = set()
    scheduler.req_to_token_pool = pool
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=freed.append)
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()
    scheduler.is_entry_rank = True
    return scheduler, freed


def test_continuation_reconstructs_req_and_transferred_mapping() -> None:
    source = _prefill_req()
    continuation = continuation_from_req(source, "transfer-1")
    continuation = decode_continuation(encode_continuation(continuation))
    pool = _ReqPool()
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )

    rebuilt = req_from_continuation(
        continuation,
        allocation,
        req_to_token_pool=pool,
    )

    assert rebuilt.rid == source.rid
    assert list(rebuilt.origin_input_ids) == [10, 11, 12]
    assert list(rebuilt.output_ids) == [42]
    assert rebuilt.sampling_params.sampling_seed == 17
    assert rebuilt.sampling_params.stop_token_ids == {2}
    assert rebuilt.prefix_indices.tolist() == [7, 8, 9]
    assert rebuilt.kv.kv_allocated_len == 3
    assert pool.req_to_token[0, :3].tolist() == [7, 8, 9]
    assert rebuilt._omni_data.stage_payload == source._omni_data.stage_payload


def test_generic_pd_state_adapter_projects_and_restores_state() -> None:
    source = _prefill_req()
    source._omni_data.return_logprob = True
    source._omni_data.output_token_logprobs = [[-0.25, 42]]
    source._omni_data.stage_payload.data = {"position": torch.tensor([3, 4])}

    def state_builder(req):
        payload = req._omni_data.stage_payload
        projected = StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={"position": payload.data["position"].tolist()},
        )
        return (
            projected.to_dict(),
            {"schema": "test-adapter-v1", "delta": [-2]},
            list(req.origin_input_ids),
        )

    restored = []

    def state_restorer(req, data, resume):
        restored.append((req.rid, data.stage_payload.data, resume))

    continuation = continuation_from_req(source, "transfer-1", state_builder)
    continuation = decode_continuation(encode_continuation(continuation))
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )
    rebuilt = req_from_continuation(
        continuation,
        allocation,
        req_to_token_pool=_ReqPool(),
        state_restorer=state_restorer,
    )

    assert rebuilt._omni_data.return_logprob is True
    assert rebuilt.return_logprob is False
    assert rebuilt._omni_data.output_token_logprobs == [[-0.25, 42]]
    assert restored == [
        (
            source.rid,
            {"position": [3, 4]},
            {"schema": "test-adapter-v1", "delta": [-2]},
        )
    ]


def test_reconstruction_failure_releases_request_pool_slot() -> None:
    continuation = continuation_from_req(_prefill_req(), "transfer-1")
    pool = _FailingReqPool()
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )

    with pytest.raises(RuntimeError, match="mapping write failed"):
        req_from_continuation(
            continuation,
            allocation,
            req_to_token_pool=pool,
        )

    assert pool.freed


def test_request_pool_exhaustion_has_a_retryable_signal() -> None:
    continuation = continuation_from_req(_prefill_req(), "transfer-1")
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )

    with pytest.raises(DecodeRequestPoolExhausted):
        req_from_continuation(
            continuation,
            allocation,
            req_to_token_pool=_CapacityReqPool(capacity=0),
        )


def test_committed_decode_admission_enters_waiting_queue() -> None:
    pool = _ReqPool()
    scheduler, freed = _decode_scheduler([_decode_admission("request-1", 7)], pool)

    scheduler._drain_pd_admissions()

    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    assert freed == []
    admitted = scheduler.outbox.get_nowait()
    assert admitted.type == "pd_admitted"
    assert admitted.request_id == "request-1"


def test_aborted_decode_admission_releases_transferred_slots() -> None:
    admission = _decode_admission("request-1", 7)
    scheduler, freed = _decode_scheduler([admission], _ReqPool())
    scheduler._aborted_request_ids = {"request-1"}

    scheduler._drain_pd_admissions()

    assert scheduler.waiting_queue == []
    assert freed == [admission.allocation.slots]
    assert scheduler.outbox.empty()


def test_decode_admission_retries_pool_exhaustion_without_releasing_kv() -> None:
    admission = _decode_admission("request-1", 7)
    pool = _CapacityReqPool(capacity=0)
    scheduler, freed = _decode_scheduler([admission], pool)

    scheduler._drain_pd_admissions()

    assert scheduler._pd_deferred_admission is admission
    assert scheduler.waiting_queue == []
    assert freed == []
    assert scheduler.outbox.empty()

    pool.capacity = 1
    scheduler._drain_pd_admissions()
    scheduler._drain_pd_admissions()

    assert scheduler._pd_deferred_admission is None
    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    assert [list(req.output_ids) for req in scheduler.waiting_queue] == [[42]]
    assert freed == []
    assert [scheduler.outbox.get_nowait().request_id] == ["request-1"]
    assert scheduler.outbox.empty()


def test_decode_admission_preserves_fifo_across_capacity_waves() -> None:
    admissions = [
        _decode_admission("request-1", 7),
        _decode_admission("request-2", 10),
        _decode_admission("request-3", 13),
    ]
    pool = _CapacityReqPool(capacity=1)
    scheduler, freed = _decode_scheduler(admissions, pool)
    admitted = []

    for expected in ("request-1", "request-2", "request-3"):
        scheduler._drain_pd_admissions()
        admitted.append(scheduler.outbox.get_nowait().request_id)
        req = scheduler.waiting_queue.pop(0)
        assert req.rid == expected
        assert list(req.output_ids) == [42]
        pool.free(req)

    scheduler._drain_pd_admissions()

    assert admitted == ["request-1", "request-2", "request-3"]
    assert scheduler._pd_deferred_admission is None
    assert scheduler.waiting_queue == []
    assert freed == []
    assert scheduler.outbox.empty()


def test_permanent_decode_reconstruction_error_is_not_retried(monkeypatch) -> None:
    admission = _decode_admission("request-1", 7)

    def fail_reconstruction(*_args, **_kwargs):
        raise ValueError("invalid continuation state")

    monkeypatch.setattr(
        omni_scheduler_module,
        "req_from_continuation",
        fail_reconstruction,
    )
    scheduler, freed = _decode_scheduler([admission], _CapacityReqPool(capacity=1))

    scheduler._drain_pd_admissions()
    scheduler._drain_pd_admissions()

    assert scheduler._pd_deferred_admission is None
    assert freed == [admission.allocation.slots]
    error = scheduler.outbox.get_nowait()
    assert error.type == "error"
    assert error.metadata == {"pd_pre_admission": True}
    assert isinstance(error.data, ValueError)
    assert scheduler.outbox.empty()


def test_abort_while_decode_admission_is_deferred_releases_once() -> None:
    admission = _decode_admission("request-1", 7)
    scheduler, freed = _decode_scheduler([admission], _CapacityReqPool(capacity=0))
    scheduler._drain_pd_admissions()
    scheduler._aborted_request_ids.add("request-1")

    scheduler._drain_pd_admissions()
    scheduler._drain_pd_admissions()

    assert scheduler._pd_deferred_admission is None
    assert freed == [admission.allocation.slots]
    assert scheduler.waiting_queue == []
    assert scheduler.outbox.empty()


def test_discard_decode_admissions_releases_each_allocation_once() -> None:
    admissions = [
        _decode_admission("request-1", 7),
        _decode_admission("request-2", 10),
    ]
    scheduler, freed = _decode_scheduler(admissions, _CapacityReqPool(capacity=0))
    scheduler._drain_pd_admissions()

    scheduler._discard_pd_admissions()
    scheduler._discard_pd_admissions()

    assert scheduler._pd_deferred_admission is None
    assert freed == [
        admissions[0].allocation.slots,
        admissions[1].allocation.slots,
    ]


def test_metrics_reporter_handles_multiple_reconstructed_requests() -> None:
    scheduler, _ = _decode_scheduler(
        [
            _decode_admission("request-1", 7),
            _decode_admission("request-2", 10),
        ],
        _CapacityReqPool(capacity=2),
    )
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[], retracted_queue=[], num_tokens_pre_allocated=0
    )
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.scheduler = scheduler

    scheduler._drain_pd_admissions()
    snapshots = [
        reporter._build_queued_request_metrics() for _ in scheduler.waiting_queue
    ]

    assert [req.rid for req in scheduler.waiting_queue] == [
        "request-1",
        "request-2",
    ]
    assert len(snapshots) == 2
    assert all(snapshot.num_decode_requests == 0 for snapshot in snapshots)


def test_prefill_handoff_reuses_request_mapping_and_detaches_batch() -> None:
    req = _prefill_req()
    pool = _ReqPool()
    pool.alloc([req])
    pool.write(
        (req.req_pool_idx, slice(0, 3)),
        torch.tensor([7, 8, 9], dtype=torch.int64),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.req_to_token_pool = pool
    scheduler.page_size = 1
    scheduler._pd_pool_id = "prefill:kv"
    scheduler._pd_partner = "decode"
    scheduler.tree_cache = object()
    scheduler.outbox = queue.Queue()
    batch = SimpleNamespace(reqs=[req])

    scheduler._queue_pd_prefill_handoffs(batch, {id(req)})

    assert batch.reqs == []
    handoff = scheduler.outbox.get_nowait()
    assert handoff.type == "pd_handoff"
    assert handoff.data.source_page_indices == (7, 8, 9)
    assert handoff.data.continuation.output_ids == [42]


def test_prefill_middle_chunk_waits_for_sampled_token(monkeypatch) -> None:
    req = _prefill_req()
    req.output_ids.pop()
    req.inflight_middle_chunks = 1
    pool = _ReqPool()
    pool.alloc([req])
    pool.write(
        (req.req_pool_idx, slice(0, 3)),
        torch.tensor([7, 8, 9], dtype=torch.int64),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_role = "prefill"
    scheduler.req_to_token_pool = pool
    scheduler.page_size = 1
    scheduler._pd_pool_id = "prefill:kv"
    scheduler._pd_partner = "decode"
    scheduler.tree_cache = object()
    scheduler.outbox = queue.Queue()
    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    def process_result(_scheduler, _batch, _result):
        if req.inflight_middle_chunks:
            req.inflight_middle_chunks -= 1
        else:
            req.output_ids.append(42)

    monkeypatch.setattr(
        omni_scheduler_module._Upstream,
        "process_batch_result",
        process_result,
    )

    scheduler.process_batch_result(batch, object())

    assert batch.reqs == [req]
    assert list(req.output_ids) == []
    assert req.req_pool_idx == 0
    assert not hasattr(req, "_pd_handoff_started")
    assert scheduler.outbox.empty()

    scheduler.process_batch_result(batch, object())

    assert batch.reqs == []
    handoff = scheduler.outbox.get_nowait().data
    assert handoff.continuation.output_ids == [42]
    assert list(req.output_ids) == [42]

    releases = []
    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.release_kv_cache",
        lambda released_req, tree_cache: releases.append((released_req, tree_cache)),
    )
    handoff.lease.release()
    handoff.lease.release()

    assert releases == [(req, scheduler.tree_cache)]


def test_continuation_rejects_missing_prefill_token() -> None:
    req = _prefill_req()
    req.output_ids.pop()

    with pytest.raises(ValueError, match="has no sampled output token"):
        continuation_from_req(req, "transfer-1")


def test_decode_scheduler_delegates_prebuilt_admission(monkeypatch) -> None:
    running = object()
    batch = object()
    calls = []

    def get_next(_scheduler, running_batch):
        calls.append(running_batch)
        return SimpleNamespace(running_batch=running, batch_to_run=batch)

    monkeypatch.setattr(
        omni_scheduler_module._Upstream,
        "get_next_disagg_decode_batch_to_run",
        get_next,
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_role = "decode"
    scheduler._pd_ready_queue = queue.SimpleQueue()
    scheduler._pd_deferred_admission = None
    scheduler._pd_admission_lock = threading.Lock()
    scheduler.running_batch = object()

    assert scheduler.get_next_batch_to_run() is batch
    assert calls
    assert scheduler.running_batch is running


def test_prefill_retries_admission_after_ack_returns_request_slot(monkeypatch) -> None:
    running = SimpleNamespace(batch_is_full=True, is_empty=lambda: True)

    def get_next(_scheduler, running_batch, _last_batch):
        assert running_batch.batch_is_full is False
        return SimpleNamespace(running_batch=running_batch, batch_to_run=None)

    monkeypatch.setattr(
        omni_scheduler_module._Upstream,
        "get_next_batch_to_run",
        get_next,
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_role = "prefill"
    scheduler._pd_ready_queue = queue.Queue()
    scheduler._pd_deferred_admission = None
    scheduler._pd_admission_lock = threading.Lock()
    scheduler.running_batch = running
    scheduler.last_batch = None
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)

    assert scheduler.get_next_batch_to_run() is None
    assert running.batch_is_full is False
