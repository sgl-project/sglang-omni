# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from array import array
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.pipeline.umm import CONTEXT_KEY, ContextRef
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.sglang_backend.request_data import (
    DllmRequest,
    SGLangDLLMRequestData,
)


class ReqDouble:
    _uncond_input_ids: list[int] | None = (
        None  # noqa: leading-underscore  # DLLM protocol
    )

    def __init__(self, *, rid: str = "req", block_size: int = 4) -> None:
        self.rid = rid
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.origin_input_ids = array("q", [1, 2])
        self.full_untruncated_fill_ids = self.origin_input_ids + array(
            "q", [-1] * block_size
        )
        self.extend_end = len(self.full_untruncated_fill_ids)
        self.output_ids: list[int] = []
        self.output_ids_through_stop: list[int] = self.output_ids
        self.finished_reason = None
        self.kv = ReqKvInfo(req_pool_idx=3)
        self.accepted_lengths: list[int] = []
        self.is_finished = False

    @property
    def seqlen(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)

    def update_finish_state(self, *, new_accepted_len: int = 1) -> None:
        self.accepted_lengths.append(new_accepted_len)

    def finished(self) -> bool:
        return self.is_finished or self.finished_reason is not None


def make_scheduler(
    *, fdfo: bool, block_size: int = 4, context_len: int = 4096
) -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler.dllm_config = SimpleNamespace(
        first_done_first_out_mode=fdfo,
        block_size=block_size,
    )
    scheduler.model_config = SimpleNamespace(context_len=context_len)
    scheduler.rid_to_req_data = {}
    scheduler.abort_lock = threading.Lock()
    scheduler.aborted_request_ids = set()
    scheduler.inbox = queue.Queue()
    scheduler.tp_rank = 0
    scheduler.tp_size = 1
    scheduler.running = True
    scheduler.waiting_queue = []
    scheduler.cond_to_unconds = {}
    scheduler.uncond_to_cond = {}
    scheduler.uncond_rids = set()
    scheduler.orphaned_uncond_rids = set()
    scheduler.result_adapter = lambda request_data: StagePayload("req", None, {})
    scheduler.outbox = SimpleNamespace(put=lambda value: None)
    return scheduler


@pytest.mark.parametrize("tp_rank,expected_errors", [(0, 1), (1, 0)])
def test_tp_request_rejection_is_reported_by_leader_only(
    tp_rank: int, expected_errors: int
) -> None:
    errors: list[object] = []
    scheduler = make_scheduler(fdfo=False)
    scheduler.tp_size = 2
    scheduler.tp_rank = tp_rank
    scheduler.outbox = SimpleNamespace(put=errors.append)
    scheduler.cond_to_unconds = {"request": []}

    scheduler.reject_waiting_request_group("request", "request group is too large")

    assert len(errors) == expected_errors


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_dllm_scheduler_owns_tp_work_broadcast(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, tp_rank: int
) -> None:
    monkeypatch.setattr(
        dllm_scheduler_module, "get_parallel", lambda: SimpleNamespace(tp_size=tp_size)
    )
    scheduler = DllmScheduler(
        tp_worker=SimpleNamespace(tp_rank=tp_rank),
        tree_cache=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=None,
        model_config=None,
        dllm_config=SimpleNamespace(block_size=128),
        request_builder=lambda payload: payload,
        result_adapter=lambda result: result,
    )

    assert scheduler.tp_rank == tp_rank
    assert scheduler.tp_size == tp_size
    assert scheduler.requires_tp_work_fanout is False


def test_model_worker_fdfo_forwards_carried_states_and_all_result_fields() -> None:
    carried_states = [{"round": 1}, None]
    next_states = [{"round": 2}, None]
    calls = []

    class Algorithm:
        fdfo = True

        def run(self, model_runner, forward_batch, algo_states):
            calls.append((model_runner, forward_batch, algo_states))
            return (
                "logits",
                [[10, 11, 12, 13], [20, 21, 22, 23]],
                [0, 4],
                next_states,
                True,
            )

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = Algorithm()
    worker.model_runner = object()
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(dllm_algo_state=carried_states[0]),
            SimpleNamespace(dllm_algo_state=carried_states[1]),
        ]
    )

    result = ModelWorker.forward_batch_generation(
        worker,
        "forward-batch",
        batch=batch,
    )

    assert calls == [(worker.model_runner, "forward-batch", carried_states)]
    assert result.next_token_ids == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]
    assert result.accept_length_per_req_cpu == [0, 4]
    assert result.dllm_algo_state == next_states
    assert result.can_run_cuda_graph is True


def test_model_worker_sync_dllm_accepts_five_field_result() -> None:
    class Algorithm:
        fdfo = False

        def run(self, model_runner, forward_batch, algo_states):
            assert algo_states is None
            return ("logits", [[10, 11]], None, None, False)

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = Algorithm()
    worker.model_runner = object()

    result = ModelWorker.forward_batch_generation(worker, "forward-batch")

    assert result.logits_output == "logits"
    assert result.next_token_ids == [[10, 11]]
    assert result.accept_length_per_req_cpu is None
    assert result.dllm_algo_state is None
    assert result.can_run_cuda_graph is False


def test_dllm_scheduler_event_loop_passes_schedule_batch_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(DllmScheduler)
    batch = SimpleNamespace(output_ids=None, reqs=[])
    forward_batch = SimpleNamespace()
    forwarded = []

    scheduler.running = True
    scheduler.drain_and_purge = lambda: scheduler.running
    scheduler.schedule_next_batch = lambda: batch
    scheduler.apply_results = lambda *_: None
    scheduler.apply_cfg_padding_metadata = lambda *_: None

    def stop_after_step(_batch) -> None:
        scheduler.running = False

    scheduler.post_step = stop_after_step
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(device="cpu"),
        forward_batch_generation=lambda forward_batch, *, batch: (
            forwarded.append((forward_batch, batch))
            or SimpleNamespace(next_token_ids=[])
        ),
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "resolve_deferred_prefill_inputs",
        lambda *_: None,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "DllmForwardBatch",
        SimpleNamespace(init_new=lambda *args, **kwargs: forward_batch),
    )

    scheduler._event_loop()  # noqa: leading-underscore  # production name

    assert forwarded == [(forward_batch, batch)]
    assert forward_batch.reqs == []


def test_fdfo_unresolved_block_carries_tokens_state_and_resident_kv() -> None:
    scheduler = make_scheduler(fdfo=True)
    req = ReqDouble()
    scheduler.staging_queue = [req]
    cache_calls = []
    free_calls = []
    scheduler.tree_cache = SimpleNamespace(
        checkpoint=lambda *args, **kwargs: cache_calls.append((args, kwargs))
    )
    scheduler.req_to_token_pool = SimpleNamespace(
        free=lambda value: free_calls.append(value)
    )
    batch = SimpleNamespace(
        reqs=[req],
        filter_batch=lambda **kwargs: None,
    )
    state = {"round": 2}
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=[0],
        dllm_algo_state=[state],
    )

    scheduler.apply_results(batch, result)
    scheduler.post_step(batch)

    assert req.dllm_incomplete_ids == array("q", [10, 11, 12, 13])
    assert req.dllm_algo_state is state
    assert req.output_ids == []
    assert req.accepted_lengths == []
    assert scheduler.staging_queue == [req]
    assert cache_calls == []
    assert free_calls == []
    assert req.kv.req_pool_idx == 3


def test_fdfo_resolved_block_commits_fill_ids_and_output_tokens() -> None:
    scheduler = make_scheduler(fdfo=True)
    req = ReqDouble()
    req.dllm_incomplete_ids = array("q", [7, 8, 9, 10])
    req.dllm_algo_state = {"round": 1}
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=[4],
        dllm_algo_state=[None],
    )

    scheduler.apply_results(batch, result)

    assert req.dllm_incomplete_ids == array("q")
    assert req.dllm_algo_state is None
    assert req.full_untruncated_fill_ids == array("q", [1, 2, 10, 11, 12, 13])
    assert req.output_ids == [10, 11, 12, 13]
    assert req.accepted_lengths == [4]


@pytest.mark.parametrize(
    ("context_len", "finish_reason"),
    [(9, "length"), (10, None)],
)
def test_block_that_cannot_be_followed_inside_the_context_finishes_by_length(
    context_len: int, finish_reason: str | None
) -> None:
    scheduler = make_scheduler(fdfo=False, context_len=context_len)
    outbox = []
    scheduler.outbox = SimpleNamespace(put=outbox.append)
    req = ReqDouble()
    req_data = SimpleNamespace(output_ids=None, finish_reason=None)
    scheduler.rid_to_req_data = {req.rid: req_data}
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler.apply_results(batch, result)

    assert req.finished() is (finish_reason is not None)
    assert req_data.finish_reason == finish_reason
    assert [message.type for message in outbox] == (
        ["result"] if finish_reason is not None else []
    )


def test_sync_dllm_result_commits_generated_suffix() -> None:
    scheduler = make_scheduler(fdfo=False)
    req = ReqDouble()
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler.apply_results(batch, result)

    assert req.full_untruncated_fill_ids == array("q", [1, 2, -1, -1, 10, 11])
    assert req.output_ids == [10, 11]
    assert req.accepted_lengths == [2]


def test_request_that_fails_to_build_gets_an_error_and_the_next_is_queued() -> None:
    scheduler = make_scheduler(fdfo=False)
    scheduler.abort_lock = threading.Lock()
    scheduler.aborted_request_ids = set()
    scheduler.waiting_queue = []
    scheduler.staging_queue = []
    scheduler.inbox = queue.Queue()
    outbox = []
    scheduler.outbox = SimpleNamespace(put=outbox.append)
    build_error = ValueError("temperature must be a non-negative finite number")

    def request_builder(payload: str) -> SimpleNamespace:
        if payload == "bad":
            raise build_error
        else:
            return SimpleNamespace(req=ReqDouble(rid="good"))

    scheduler.request_builder = request_builder
    for request_id in ("bad", "good"):
        scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=request_id)
        )

    scheduler.drain_and_purge()

    assert [(m.request_id, m.type, m.data) for m in outbox] == [
        ("bad", "error", build_error)
    ]
    assert [req.rid for req in scheduler.waiting_queue] == ["good"]


@pytest.fixture
def chunked_scheduler() -> DllmScheduler:
    scheduler = make_scheduler(fdfo=False, block_size=32)
    scheduler.dllm_config = DllmConfig(
        algorithm="LowConfidence",
        algorithm_config={},
        block_size=32,
        mask_id=9,
        max_running_requests=2,
    )
    scheduler.req_to_token_pool = ReqToTokenPool(2, 256, "cpu", False)
    scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        512, torch.float32, "cpu", None, False
    )
    scheduler.tree_cache = UnifiedRadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    scheduler.chunked_prefill_size = 32
    scheduler.model_config = None
    scheduler.staging_queue = []
    scheduler.retained_contexts = {}
    scheduler.contexts_to_retain = set()
    for rid in ("cond", "uncond"):
        params = SamplingParams(max_new_tokens=32, temperature=0.0)
        params.normalize(None)
        scheduler.waiting_queue.append(
            DllmRequest(
                rid,
                "",
                array("q", [1] * 128),
                params,
                dllm_config=scheduler.dllm_config,
            )
        )
    return scheduler


def test_cfg_chunked_admission_rolls_back_and_retries(
    chunked_scheduler: DllmScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = chunked_scheduler
    requests = scheduler.waiting_queue.copy()
    scheduler.cond_to_unconds = {"cond": ["uncond"]}
    scheduler.uncond_to_cond = {"uncond": "cond"}
    requests[1]._is_uncond = True  # noqa: leading-underscore  # DLLM protocol
    allocator = scheduler.token_to_kv_pool_allocator
    occupied = allocator.alloc(300)
    original_spans = [(req.prefix_len, req.extend_end) for req in requests]
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        SimpleNamespace(
            init_new=lambda **kwargs: SimpleNamespace(
                reqs=kwargs["reqs"], prepare_for_extend=lambda: None
            )
        ),
    )
    with get_context().override_server_args(page_size=1, max_prefill_tokens=64):
        assert scheduler.schedule_next_batch() is None
        assert scheduler.waiting_queue == requests
        assert [(req.prefix_len, req.extend_end) for req in requests] == original_spans
        allocator.free(occupied)
        batch = scheduler.schedule_next_batch()

    assert batch.reqs == requests
    assert all(req.extend_len == 32 for req in batch.reqs)


@pytest.mark.parametrize(
    ("lengths", "expected_rounds"),
    [
        (
            [40, 104],
            [
                {"cond": (0, 32), "uncond": (0, 32)},
                {"uncond": (32, 64)},
                {"uncond": (64, 96)},
                {"cond": (32, 64), "uncond": (96, 128)},
            ],
        ),
        (
            [8, 72],
            [
                {"uncond": (0, 32)},
                {"uncond": (32, 64)},
                {"cond": (0, 32), "uncond": (64, 96)},
            ],
        ),
    ],
    ids=["both-rows-encode-first", "conditional-decodes-at-admission"],
)
def test_cfg_rows_encode_at_their_own_length_then_decode_together(
    chunked_scheduler: DllmScheduler,
    monkeypatch: pytest.MonkeyPatch,
    lengths: list[int],
    expected_rounds: list[dict[str, tuple[int, int]]],
) -> None:
    scheduler = chunked_scheduler
    requests = scheduler.waiting_queue.copy()
    for request, length in zip(requests, lengths, strict=True):
        request.origin_input_ids = array("q", [1] * length)
    scheduler.cond_to_unconds = {"cond": ["uncond"]}
    scheduler.uncond_to_cond = {"uncond": "cond"}
    scheduler.uncond_rids = {"uncond"}
    scheduler.orphaned_uncond_rids = set()
    requests[1]._is_uncond = True  # noqa: leading-underscore  # DLLM protocol
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        SimpleNamespace(
            init_new=lambda **kwargs: SimpleNamespace(
                reqs=kwargs["reqs"],
                prepare_for_extend=lambda: None,
                filter_batch=lambda **kwargs: None,
            )
        ),
    )

    rounds = []
    with get_context().override_server_args(page_size=1, max_prefill_tokens=64):
        for _ in expected_rounds:
            batch = scheduler.schedule_next_batch()
            rounds.append(
                {req.rid: (req.prefix_len, req.extend_end) for req in batch.reqs}
            )
            scheduler.post_step(batch)

    assert rounds == expected_rounds
    assert not any(req.is_dllm_prefill() for req in batch.reqs)


def test_companion_in_another_block_column_is_rejected(
    chunked_scheduler: DllmScheduler,
) -> None:
    scheduler = chunked_scheduler
    conditional = scheduler.waiting_queue[0]

    with pytest.raises(ValueError, match="share a block column"):
        scheduler.create_uncond_companion(
            conditional, [1] * 129, 0, 0, "-uncond", mark_img=False
        )


def test_companion_keeps_the_span_of_its_padding(
    chunked_scheduler: DllmScheduler,
) -> None:
    scheduler = chunked_scheduler
    conditional = scheduler.waiting_queue[0]

    scheduler.create_uncond_companion(
        conditional, [1] * 96, 3, 40, "-uncond", mark_img=False
    )

    companion = scheduler.waiting_queue[-1]
    assert companion.rid == "cond-uncond"
    assert (
        companion._dllm_pad_start,  # noqa: leading-underscore  # DLLM protocol
        companion._dllm_pad_len,  # noqa: leading-underscore  # DLLM protocol
    ) == (40, 3)


def test_dllm_staging_admission_uses_dllm_config(
    chunked_scheduler: DllmScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = chunked_scheduler
    request = scheduler.waiting_queue[0]
    scheduler.waiting_queue = [request]
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        SimpleNamespace(
            init_new=lambda **kwargs: SimpleNamespace(
                reqs=kwargs["reqs"], prepare_for_extend=lambda: None
            )
        ),
    )

    with get_context().override_server_args(page_size=1, max_prefill_tokens=64):
        assert scheduler.schedule_next_batch() is not None
        assert scheduler.staging_queue == [request]
        batch = scheduler.schedule_next_batch()

    assert batch.reqs == [request]
    assert request.extend_len == scheduler.dllm_config.block_size


def test_abort_between_blocks_releases_cached_tokens(
    chunked_scheduler: DllmScheduler,
) -> None:
    scheduler = chunked_scheduler
    req = scheduler.waiting_queue.pop(0)
    scheduler.waiting_queue.clear()
    scheduler.staging_queue = [req]
    pool = scheduler.req_to_token_pool
    allocator = scheduler.token_to_kv_pool_allocator
    pool.alloc([req])
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 32
    pool.req_to_token[req.kv.req_pool_idx, :32] = allocator.alloc(32).int()
    req.extend_end = 32
    scheduler.post_step(SimpleNamespace(reqs=[req], filter_batch=lambda **kwargs: None))
    scheduler.aborted_request_ids = {req.rid}

    with get_context().override_server_args(page_size=1):
        scheduler.drain_and_purge()

    assert scheduler.staging_queue == []
    assert allocator.available_size() == 512
    assert pool.available_size() == 2
    assert req.kv.is_kv_released


@pytest.mark.parametrize("max_new_tokens,accepted", [(128, True), (129, False)])
def test_cfg_group_must_fit_physical_kv_pool(
    chunked_scheduler: DllmScheduler, max_new_tokens: int, accepted: bool
) -> None:
    scheduler = chunked_scheduler
    request = scheduler.waiting_queue[0]
    scheduler.waiting_queue.clear()
    request.sampling_params.max_new_tokens = max_new_tokens
    request._uncond_input_ids = list(
        request.origin_input_ids
    )  # noqa: leading-underscore  # DLLM protocol
    scheduler.request_builder = lambda payload: SimpleNamespace(req=request)
    errors = []
    scheduler.outbox = SimpleNamespace(put=errors.append)
    scheduler.inbox.put(
        IncomingMessage(request_id=request.rid, type="new_request", data=None)
    )

    with get_context().override_server_args(page_size=32, max_prefill_tokens=64):
        scheduler.drain_and_purge()

    if accepted:
        assert not errors
        assert [req.rid for req in scheduler.waiting_queue] == ["cond", "cond-uncond"]
    else:
        assert len(errors) == 1
        assert errors[0].request_id == "cond" and errors[0].type == "error"
        assert "requires 576 KV tokens" in errors[0].data
        assert not scheduler.waiting_queue and not scheduler.rid_to_req_data
        assert not scheduler.cond_to_unconds and not scheduler.uncond_to_cond
    assert scheduler.token_to_kv_pool_allocator.available_size() == 512


def run_tp_abort_rank(rank: int, rendezvous: str, stop_follower_early: bool) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=15),
    )
    try:
        scheduler = make_scheduler(fdfo=False)
        scheduler.tp_rank, scheduler.tp_size = rank, 2
        scheduler.abort_lock = threading.Lock()
        scheduler.aborted_request_ids = set()
        scheduler.inbox = queue.Queue()
        group_ids = ["cond", "cond-uncond", "cond-uncond-img"]
        scheduler.staging_queue = [ReqDouble(rid=rid) for rid in group_ids]
        scheduler.cond_to_unconds = {"cond": group_ids[1:]}
        scheduler.uncond_to_cond = dict.fromkeys(group_ids[1:], "cond")
        scheduler.uncond_rids = set(group_ids[1:])
        scheduler.request_builder = lambda rid: SimpleNamespace(req=ReqDouble(rid=rid))
        released: list[str] = []
        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(
                rank=rank, ranks=[0, 1], cpu_group=dist.group.WORLD
            )
        )
        scheduler.tree_cache = None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dllm_scheduler_module, "get_parallel", lambda: parallel)
            patch.setattr(
                dllm_scheduler_module,
                "release_kv_once",
                lambda req, cache: released.append(req.rid),
            )
            for step in range(3):
                if step == 0 and rank == 1:
                    scheduler.abort("cond")
                    if stop_follower_early:
                        scheduler.stop()
                elif step == 1 and rank == 0:
                    scheduler.abort("cond")
                    scheduler.inbox.put(
                        IncomingMessage("after", "new_request", "after")
                    )
                elif step == 2 and rank == 1:
                    scheduler.abort("after")
                assert scheduler.drain_and_purge()
                requests = scheduler.staging_queue or scheduler.waiting_queue
                assert [req.rid for req in requests] == (
                    group_ids if step == 0 else ["after"]
                )
                # Match a forward collective after each agreed scheduling boundary.
                value = torch.tensor([len(requests)])
                dist.all_reduce(value)
                assert value.item() == 2 * len(requests)

            if rank == 0:
                scheduler.stop()
            assert not scheduler.drain_and_purge()

        assert released == group_ids
        assert not scheduler.cond_to_unconds
        assert not scheduler.uncond_to_cond
        assert not scheduler.uncond_rids
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("stop_follower_early", [False, True])
def test_tp_abort_at_forward_boundary(
    tmp_path: Path, stop_follower_early: bool
) -> None:
    mp.spawn(
        run_tp_abort_rank,
        args=((tmp_path / "tp-rendezvous").as_uri(), stop_follower_early),
        nprocs=2,
        join=True,
    )


PHASE_POOL_TOKENS = 64
GENERATED_TOKEN_ID = 50


def build_phase_request(payload: StagePayload) -> SGLangDLLMRequestData:
    params = SamplingParams(
        max_new_tokens=payload.data["max_new_tokens"], temperature=0
    )
    params.normalize(None)
    request = DllmRequest(
        payload.request_id,
        "",
        array("q", payload.data["input_ids"]),
        params,
        dllm_config=payload.data["dllm_config"],
    )
    companion_ids = payload.data.get("uncond_ids")
    if companion_ids is not None:
        request._uncond_input_ids = (
            companion_ids  # noqa: leading-underscore  # DLLM protocol
        )
        request._uncond_pad_len = 0  # noqa: leading-underscore  # DLLM protocol
        request._uncond_pad_start = 0  # noqa: leading-underscore  # DLLM protocol
    else:
        pass
    data = SGLangDLLMRequestData(output_ids=request.output_ids, req=request)
    data.stage_payload = payload
    return data


def finish_phase(data: SGLangDLLMRequestData) -> StagePayload:
    payload = data.stage_payload
    context = (
        {CONTEXT_KEY: ContextRef("thinker", payload.request_id)}
        if payload.data["continues"]
        else {}
    )
    return StagePayload(payload.request_id, None, context)


@pytest.fixture
def radix_scheduler() -> DllmScheduler:
    with get_context().override_server_args(page_size=4, max_prefill_tokens=64):
        pool = ReqToTokenPool(3, PHASE_POOL_TOKENS, "cpu", False)
        allocator = PagedTokenToKVPoolAllocator(
            PHASE_POOL_TOKENS, 4, torch.float32, "cpu", None, False
        )
        tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=4,
                tree_components=(ComponentType.FULL,),
            )
        )
        yield DllmScheduler(
            SimpleNamespace(),
            tree_cache,
            pool,
            allocator,
            None,
            SimpleNamespace(
                context_len=PHASE_POOL_TOKENS, vocab_size=300, is_encoder_decoder=False
            ),
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={},
                block_size=4,
                mask_id=99,
                max_running_requests=3,
            ),
            request_builder=build_phase_request,
            result_adapter=finish_phase,
        )


def run_phase(
    scheduler: DllmScheduler,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    continues: bool = True,
    request_id: str = "story",
    uncond_ids: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Run one thinker phase through the real cache and return its block spans."""
    phase = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "continues": continues,
        "dllm_config": scheduler.dllm_config,
    }
    if uncond_ids is not None:
        phase["uncond_ids"] = uncond_ids
    else:
        pass
    scheduler.inbox.put(
        IncomingMessage(
            request_id=request_id,
            type="new_request",
            data=StagePayload(request_id, None, phase),
        )
    )
    spans = []
    for _ in range(32):
        scheduler.drain_and_purge()
        batch = scheduler.schedule_next_batch()
        if batch is None:
            continue
        else:
            pass
        spans.extend((req.prefix_len, req.extend_end) for req in batch.reqs)
        generated = [
            [GENERATED_TOKEN_ID]
            * max(req.extend_end - len(req.origin_input_ids) - len(req.output_ids), 0)
            for req in batch.reqs
        ]
        scheduler.apply_results(
            batch,
            SimpleNamespace(
                next_token_ids=generated,
                accept_length_per_req_cpu=None,
                dllm_algo_state=None,
            ),
        )
        scheduler.post_step(batch)
        if not scheduler.staging_queue:
            return spans
        else:
            pass
    raise AssertionError(f"phase of {request_id} never finished")


PROMPT_IDS = list(range(1, 11))
# The phase below generates six tokens, so the sequence fills four whole blocks.
CONTINUED_IDS = [*PROMPT_IDS, *[GENERATED_TOKEN_ID] * 6, 60, 61]


def evict_everything(scheduler: DllmScheduler) -> None:
    scheduler.tree_cache.evict(EvictParams(num_tokens=PHASE_POOL_TOKENS))


def test_next_phase_resumes_from_the_prefix_the_last_phase_pinned(
    radix_scheduler: DllmScheduler,
) -> None:
    scheduler = radix_scheduler

    first_spans = run_phase(scheduler, PROMPT_IDS, 6)
    evict_everything(scheduler)
    second_spans = run_phase(scheduler, CONTINUED_IDS, 2, continues=False)

    assert first_spans == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert second_spans == [(16, 20)]
    # The last phase leaves nothing pinned, so its whole sequence can be reclaimed.
    assert scheduler.retained_contexts == {}
    evict_everything(scheduler)
    assert scheduler.token_to_kv_pool_allocator.available_size() == PHASE_POOL_TOKENS


def test_abort_between_phases_releases_the_pinned_prefix(
    radix_scheduler: DllmScheduler,
) -> None:
    scheduler = radix_scheduler
    run_phase(scheduler, PROMPT_IDS, 6)

    scheduler.abort("story")
    scheduler.drain_and_purge()
    evict_everything(scheduler)

    assert scheduler.retained_contexts == {}
    assert scheduler.token_to_kv_pool_allocator.available_size() == PHASE_POOL_TOKENS


def test_admission_short_of_memory_gives_up_a_pinned_prefix(
    radix_scheduler: DllmScheduler,
) -> None:
    scheduler = radix_scheduler
    run_phase(scheduler, PROMPT_IDS, 6)
    other_ids = list(range(100, 140))

    first_spans = run_phase(scheduler, other_ids, 12, request_id="other")

    assert first_spans[0] == (0, 4)
    assert "story" not in scheduler.retained_contexts


def test_cfg_group_that_outgrows_the_free_pages_gives_up_a_pinned_prefix(
    radix_scheduler: DllmScheduler,
) -> None:
    """A pinned prefix could not be evicted, so a CFG group that decoded into the last free pages crashed the thinker."""
    scheduler = radix_scheduler
    run_phase(scheduler, PROMPT_IDS, 6)

    run_phase(
        scheduler,
        list(range(100, 120)),
        8,
        continues=False,
        request_id="other",
        uncond_ids=list(range(200, 220)),
    )

    assert "story" not in scheduler.retained_contexts


def test_abort_releases_a_cfg_row_that_never_ran(
    radix_scheduler: DllmScheduler,
) -> None:
    """An aborted conditional row that still waited for its companion kept the prefix its admission matched locked."""
    scheduler = radix_scheduler
    run_phase(scheduler, PROMPT_IDS, 6)
    scheduler.inbox.put(
        IncomingMessage(
            request_id="story",
            type="new_request",
            data=StagePayload(
                "story",
                None,
                {
                    "input_ids": CONTINUED_IDS,
                    "max_new_tokens": 2,
                    "continues": False,
                    "dllm_config": scheduler.dllm_config,
                    # A longer companion encodes alone while the conditional row waits.
                    "uncond_ids": list(range(200, 226)),
                },
            ),
        )
    )
    scheduler.drain_and_purge()
    batch = scheduler.schedule_next_batch()
    conditional = scheduler.rid_to_req_data["story"].req
    assert conditional not in batch.reqs and conditional.lock is not None

    scheduler.abort("story")
    scheduler.drain_and_purge()
    evict_everything(scheduler)

    assert scheduler.tree_cache.protected_size() == 0
    assert scheduler.token_to_kv_pool_allocator.available_size() == PHASE_POOL_TOKENS
