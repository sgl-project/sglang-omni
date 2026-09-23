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
from sglang_omni.scheduling.message import IncomingMessage


class _ReqDouble:
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
        self._finished = False

    def update_finish_state(self, *, new_accepted_len: int = 1) -> None:
        self.accepted_lengths.append(new_accepted_len)

    def finished(self) -> bool:
        return self._finished


def _scheduler(*, fdfo: bool, block_size: int = 4) -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler.dllm_config = SimpleNamespace(
        first_done_first_out_mode=fdfo,
        block_size=block_size,
    )
    scheduler._rid_to_req_data = {}
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler.inbox = queue.Queue()
    scheduler.tp_rank = 0
    scheduler.tp_size = 1
    scheduler._running = True
    scheduler._waiting_queue = []
    scheduler._cond_to_unconds = {}
    scheduler._uncond_to_cond = {}
    scheduler._uncond_rids = set()
    scheduler._orphaned_uncond_rids = set()
    scheduler._result_adapter = lambda value: value
    scheduler.outbox = SimpleNamespace(put=lambda value: None)
    return scheduler


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

    class _Algorithm:
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
    worker.dllm_algorithm = _Algorithm()
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
    class _Algorithm:
        fdfo = False

        def run(self, model_runner, forward_batch, algo_states):
            assert algo_states is None
            return ("logits", [[10, 11]], None, None, False)

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = _Algorithm()
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

    scheduler._running = True
    scheduler.drain_and_purge = lambda: scheduler._running
    scheduler.schedule_next_batch = lambda: batch
    scheduler.apply_results = lambda *_: None
    scheduler.apply_cfg_padding_metadata = lambda *_: None

    def stop_after_step(_batch) -> None:
        scheduler._running = False

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

    scheduler._event_loop()

    assert forwarded == [(forward_batch, batch)]
    assert forward_batch.reqs == []


def test_dllm_staging_admission_uses_dllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.runtime_context import get_context

    scheduler = _scheduler(fdfo=True)
    scheduler.tree_cache = object()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    scheduler._chunked_prefill_size = 16
    scheduler._waiting_queue = []
    req = SimpleNamespace(
        rid="req",
        kv=SimpleNamespace(),
        inflight_middle_chunks=0,
        init_next_round_input=lambda: None,
    )
    scheduler._staging_queue = [req]
    created = {}

    class _Adder:
        def __init__(self, *args, **kwargs) -> None:
            created["dllm_config"] = kwargs.get("dllm_config")
            self.can_run_list = []

        def add_dllm_staging_req(self, value):
            created["staging_req"] = value
            self.can_run_list.append(value)
            return dllm_scheduler_module.AddReqResult.CONTINUE

    class _Batch:
        @staticmethod
        def init_new(**kwargs):
            created["batch_reqs"] = kwargs["reqs"]
            return SimpleNamespace(prepare_for_extend=lambda: None)

    monkeypatch.setattr(dllm_scheduler_module, "PrefillAdder", _Adder)
    monkeypatch.setattr(dllm_scheduler_module, "ScheduleBatch", _Batch)

    with get_context().override_server_args(page_size=1, max_prefill_tokens=16):
        batch = scheduler.schedule_next_batch()

    assert batch is not None
    assert created["dllm_config"] is scheduler.dllm_config
    assert created["staging_req"] is req
    assert created["batch_reqs"] == [req]


def test_fdfo_unresolved_block_carries_tokens_state_and_resident_kv() -> None:
    scheduler = _scheduler(fdfo=True)
    req = _ReqDouble()
    scheduler._staging_queue = [req]
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
    assert scheduler._staging_queue == [req]
    assert cache_calls == []
    assert free_calls == []
    assert req.kv.req_pool_idx == 3


def test_fdfo_resolved_block_commits_fill_ids_and_output_tokens() -> None:
    scheduler = _scheduler(fdfo=True)
    req = _ReqDouble()
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
    scheduler = _scheduler(fdfo=True)
    batch = SimpleNamespace(reqs=[_ReqDouble()])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    with pytest.raises(AssertionError, match="missing accept lengths"):
        scheduler.apply_results(batch, result)


def test_sync_dllm_result_commits_generated_suffix() -> None:
    scheduler = _scheduler(fdfo=False)
    req = _ReqDouble()
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
    scheduler = _scheduler(fdfo=False, block_size=32)
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
    scheduler._chunked_prefill_size = 32
    scheduler.model_config = None
    scheduler._staging_queue = []
    for rid in ("cond", "uncond"):
        params = SamplingParams(max_new_tokens=32, temperature=0.0)
        params.normalize(None)
        scheduler._waiting_queue.append(
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
    requests = scheduler._waiting_queue.copy()
    scheduler._cond_to_unconds = {"cond": ["uncond"]}
    scheduler._uncond_to_cond = {"uncond": "cond"}
    requests[1]._is_uncond = True
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
        assert scheduler._waiting_queue == requests
        assert [req.extend_range for req in requests] == original_ranges
        allocator.free(occupied)
        batch = scheduler.schedule_next_batch()

    assert batch.reqs == requests
    assert all(req.extend_range.length == 32 for req in batch.reqs)


def test_abort_between_blocks_releases_cached_tokens(
    chunked_scheduler: DllmScheduler,
) -> None:
    scheduler = chunked_scheduler
    req = scheduler._waiting_queue.pop(0)
    scheduler._waiting_queue.clear()
    scheduler._staging_queue = [req]
    pool = scheduler.req_to_token_pool
    allocator = scheduler.token_to_kv_pool_allocator
    pool.alloc([req])
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 32
    pool.req_to_token[req.kv.req_pool_idx, :32] = allocator.alloc(32).int()
    req.set_extend_range(0, 32)
    scheduler.post_step(SimpleNamespace(reqs=[req], filter_batch=lambda **kwargs: None))
    scheduler._aborted_request_ids = {req.rid}

    with get_context().override_server_args(page_size=1):
        scheduler.drain_and_purge()

    assert scheduler._staging_queue == []
    assert allocator.available_size() == 512
    assert pool.available_size() == 2
    assert req.kv.is_kv_released


def run_tp_abort_rank(rank: int, rendezvous: str, stop_follower_early: bool) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=15),
    )
    try:
        scheduler = _scheduler(fdfo=False)
        scheduler.tp_rank, scheduler.tp_size = rank, 2
        scheduler._abort_lock = threading.Lock()
        scheduler._aborted_request_ids = set()
        scheduler.inbox = queue.Queue()
        group_ids = ["cond", "cond-uncond", "cond-uncond-img"]
        scheduler._staging_queue = [_ReqDouble(rid=rid) for rid in group_ids]
        scheduler._cond_to_unconds = {"cond": group_ids[1:]}
        scheduler._uncond_to_cond = dict.fromkeys(group_ids[1:], "cond")
        scheduler._uncond_rids = set(group_ids[1:])
        scheduler._request_builder = lambda rid: SimpleNamespace(
            req=_ReqDouble(rid=rid)
        )
        released: list[str] = []
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                tp_group=SimpleNamespace(
                    rank=rank, ranks=[0, 1], cpu_group=dist.group.WORLD
                ),
            ),
        )
        scheduler.tree_cache = None
        with pytest.MonkeyPatch.context() as patch:
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
                requests = scheduler._staging_queue or scheduler._waiting_queue
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
        assert not scheduler._cond_to_unconds
        assert not scheduler._uncond_to_cond
        assert not scheduler._uncond_rids
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
