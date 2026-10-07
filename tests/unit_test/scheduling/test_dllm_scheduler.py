# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


class ReqDouble:
    def __init__(self, *, rid: str = "req", block_size: int = 4) -> None:
        self.rid = rid
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.origin_input_ids = array("q", [1, 2])
        self.full_untruncated_fill_ids = self.origin_input_ids + array(
            "q", [-1] * block_size
        )
        self.extend_range = SimpleNamespace(end=len(self.full_untruncated_fill_ids))
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
    scheduler.waiting_queue = []
    scheduler.cond_to_unconds = {}
    scheduler.uncond_to_cond = {}
    scheduler.uncond_rids = set()
    scheduler.orphaned_uncond_rids = set()
    scheduler.result_adapter = lambda value: value
    scheduler.outbox = SimpleNamespace(put=lambda value: None)
    return scheduler


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
    scheduler.drain_and_purge = lambda: None
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


def test_dllm_staging_admission_uses_dllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.runtime_context import get_context

    scheduler = make_scheduler(fdfo=True)
    scheduler.tree_cache = object()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    scheduler.chunked_prefill_size = 16
    scheduler.waiting_queue = []
    req = SimpleNamespace(
        rid="req",
        kv=SimpleNamespace(),
        inflight_middle_chunks=0,
        init_next_round_input=lambda: None,
    )
    scheduler.staging_queue = [req]
    created = {}

    class Adder:
        def __init__(self, *args, **kwargs) -> None:
            created["dllm_config"] = kwargs.get("dllm_config")
            self.can_run_list = []

        def add_dllm_staging_req(self, value):
            created["staging_req"] = value
            self.can_run_list.append(value)
            return dllm_scheduler_module.AddReqResult.CONTINUE

    class Batch:
        @staticmethod
        def init_new(**kwargs):
            created["batch_reqs"] = kwargs["reqs"]
            return SimpleNamespace(prepare_for_extend=lambda: None)

    monkeypatch.setattr(dllm_scheduler_module, "PrefillAdder", Adder)
    monkeypatch.setattr(dllm_scheduler_module, "ScheduleBatch", Batch)

    with get_context().override_server_args(page_size=1, max_prefill_tokens=16):
        batch = scheduler.schedule_next_batch()

    assert batch is not None
    assert created["dllm_config"] is scheduler.dllm_config
    assert created["staging_req"] is req
    assert created["batch_reqs"] == [req]


def test_fdfo_unresolved_block_carries_tokens_state_and_resident_kv() -> None:
    scheduler = make_scheduler(fdfo=True)
    req = ReqDouble()
    scheduler.staging_queue = [req]
    cache_calls = []
    free_calls = []
    scheduler.tree_cache = SimpleNamespace(
        cache_unfinished_req=lambda *args, **kwargs: cache_calls.append((args, kwargs))
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


def test_fdfo_result_requires_accept_lengths() -> None:
    scheduler = make_scheduler(fdfo=True)
    batch = SimpleNamespace(reqs=[ReqDouble()])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    with pytest.raises(AssertionError, match="missing accept lengths"):
        scheduler.apply_results(batch, result)


@pytest.mark.parametrize(
    ("context_len", "finish_reason"),
    [(9, "length"), (10, None)],
)
def test_block_that_cannot_be_followed_inside_the_context_finishes_by_length(
    context_len: int, finish_reason: str | None
) -> None:
    scheduler = make_scheduler(fdfo=False, context_len=context_len)
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
    scheduler.tree_cache = ChunkCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            page_size=1,
        )
    )
    scheduler.chunked_prefill_size = 32
    scheduler.model_config = None
    scheduler.staging_queue = []
    for rid in ("cond", "uncond"):
        params = SamplingParams(max_new_tokens=32, temperature=0.0)
        params.normalize(None)
        scheduler.waiting_queue.append(
            Req(
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
    original_ranges = [req.extend_range for req in requests]
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
        assert [req.extend_range for req in requests] == original_ranges
        allocator.free(occupied)
        batch = scheduler.schedule_next_batch()

    assert batch.reqs == requests
    assert all(req.extend_range.length == 32 for req in batch.reqs)


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
    req.set_extend_range(0, 32)
    scheduler.post_step(SimpleNamespace(reqs=[req], filter_batch=lambda **kwargs: None))
    scheduler.aborted_request_ids = {req.rid}

    with get_context().override_server_args(page_size=1):
        scheduler.drain_and_purge()

    assert scheduler.staging_queue == []
    assert allocator.available_size() == 512
    assert pool.available_size() == 2
    assert req.kv.is_kv_released
