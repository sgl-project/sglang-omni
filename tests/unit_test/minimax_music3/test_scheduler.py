# SPDX-License-Identifier: Apache-2.0

from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler


class TokenAllocator:
    device = "cpu"
    page_size = 1

    def __init__(self, capacity: int):
        self.free_tokens = set(range(1, capacity + 1))

    def available_size(self):
        return len(self.free_tokens)

    def alloc(self, count: int):
        assert count <= self.available_size()
        indices = sorted(self.free_tokens)[:count]
        self.free_tokens.difference_update(indices)
        return torch.tensor(indices, dtype=torch.int32)

    def free_segments(self, segments):
        for indices, _ in segments:
            tokens = indices.tolist()
            assert len(set(tokens)) == len(tokens)
            assert self.free_tokens.isdisjoint(tokens)
            self.free_tokens.update(tokens)

    def check_decode_capacity(self, *, num_tokens, tree_cache):
        return self.available_size() >= num_tokens


@pytest.fixture
def runtime():
    with get_context().override_server_args(
        disable_radix_cache=True, page_size=1, retraction_policy="length"
    ):
        yield


def make_pair(index=0, *, prompt=8, generated=4, max_new=100):
    params = SamplingParams(max_new_tokens=max_new)
    pair = [
        Req(
            rid=f"song{index}{suffix}",
            origin_input_text="",
            origin_input_ids=array("q", [1] * prompt),
            sampling_params=params,
        )
        for suffix in ("", "-cfg")
    ]
    for req in pair:
        req.output_ids = array("q", [2] * generated)
        req._omni_data = SimpleNamespace()
    pair[0]._omni_data.cfg_uncond = pair[1]._omni_data
    return pair


def make_batch(reqs, *, free_tokens=0):
    lengths = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
    allocator = TokenAllocator(sum(lengths) + free_tokens)
    pool = ReqToTokenPool(len(reqs), max(lengths), "cpu", False)
    cache = ChunkCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    rows = pool.alloc(reqs)
    for req, row, length in zip(reqs, rows, lengths, strict=True):
        pool.req_to_token[row, :length] = allocator.alloc(length)
        req.kv.kv_committed_len = length
        req.kv.kv_allocated_len = length
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.req_pool_indices = torch.tensor(rows)
    batch.req_pool_indices_cpu = torch.tensor(rows)
    batch.seq_lens = torch.tensor(lengths)
    batch.seq_lens_cpu = torch.tensor(lengths)
    batch.orig_seq_lens = torch.tensor(lengths)
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, 16)
    return batch


def test_retraction_releases_a_whole_pair_and_filters_real_batch(runtime):
    first = make_pair(0, generated=4)
    second = make_pair(1, generated=8)
    batch = make_batch(first + second)
    retained_rows = batch.req_pool_indices[2:].clone()
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)

    retracted, aborted = scheduler._retract_decode_pairs(batch)

    assert retracted == [tuple(first)]
    assert aborted is None
    assert batch.reqs == second
    assert torch.equal(batch.req_pool_indices, retained_rows)
    assert len(batch.sampling_info) == 2
    assert batch.token_to_kv_pool_allocator.available_size() == 22
    assert batch.req_to_token_pool.available_size() == 2
    for req in first:
        assert req.is_retracted and req.retraction_count == 1
        assert not req.kv.holds_kv
        assert list(req.output_ids) == [2] * 4
    assert batch.check_decode_mem()


def test_last_pair_failure_releases_both_rows(runtime):
    pair = make_pair()
    batch = make_batch(pair)
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)

    retracted, aborted = scheduler._retract_decode_pairs(batch)

    assert retracted == []
    assert aborted == tuple(pair)
    assert batch.is_empty()
    assert batch.token_to_kv_pool_allocator.available_size() == 22
    assert batch.req_to_token_pool.available_size() == 2
    assert all(not req.kv.holds_kv for req in pair)


def make_admission_scheduler(capacity, *, page_size=1, slots=4):
    allocator = TokenAllocator(capacity)
    allocator.page_size = page_size
    cache = ChunkCache(CacheInitParams(True, None, allocator, page_size))
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
    scheduler.page_size = page_size
    scheduler.max_prefill_tokens = 10000
    scheduler.max_total_num_tokens = capacity
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.tree_cache = cache
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    scheduler.get_num_allocatable_reqs = lambda running_size: slots
    return scheduler


def admit_with_upstream(scheduler, reqs, running):
    adder = PrefillAdder(
        scheduler.page_size,
        scheduler.tree_cache,
        scheduler.token_to_kv_pool_allocator,
        running,
        scheduler.new_token_ratio_tracker.current,
        scheduler.max_prefill_tokens,
        None,
    )
    for req in reqs:
        req.init_next_round_input(scheduler.tree_cache)
        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )
        if result != AddReqResult.CONTINUE:
            break
    return adder.can_run_list


@pytest.mark.parametrize(
    ("capacity", "generated", "expected"),
    [(1704, 0, 0), (1705, 0, 2), (2104, 400, 0), (2105, 400, 2), (2200, 400, 2)],
)
def test_replay_pair_matches_both_upstream_admission_steps(
    runtime, capacity, generated, expected
):
    pair = make_pair(prompt=100, generated=generated, max_new=751)
    scheduler = make_admission_scheduler(capacity)
    running = ScheduleBatch(reqs=[])

    limit = scheduler._pair_admission_limit(pair, running)
    admitted = admit_with_upstream(scheduler, pair[:limit], running)

    assert limit == expected
    assert admitted == pair[:expected]


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("generated", [0, 400, 4300])
def test_admitted_prefix_stays_paired_across_paging_and_generation_clip(
    runtime, page_size, generated
):
    queue = make_pair(0, prompt=103, generated=generated, max_new=5000)
    queue += make_pair(1, prompt=71, generated=generated, max_new=5000)
    running = ScheduleBatch(reqs=[])
    for capacity in range(8000, 26000, 257):
        scheduler = make_admission_scheduler(capacity, page_size=page_size)
        scheduler.max_prefill_tokens = 30000
        limit = scheduler._pair_admission_limit(queue, running)
        admitted = admit_with_upstream(scheduler, queue[:limit], running)
        assert len(admitted) == limit
        assert len(admitted) % 2 == 0


def test_unadmittable_pair_errors_without_blocking_the_next_request(runtime):
    impossible = make_pair(0, prompt=100, generated=400, max_new=751)
    feasible = make_pair(1, prompt=100, generated=0, max_new=751)
    scheduler = make_admission_scheduler(2000)
    scheduler.waiting_queue = impossible + feasible
    errors = []

    def abort(request_id, *, defer_running_cleanup):
        scheduler.waiting_queue = [
            req
            for req in scheduler.waiting_queue
            if req.rid not in (request_id, f"{request_id}-cfg")
        ]

    scheduler.abort = abort
    scheduler._emit_request_error = lambda rid, error: errors.append((rid, error))

    scheduler._reject_unadmittable_pairs()
    admitted = admit_with_upstream(
        scheduler, scheduler.waiting_queue, ScheduleBatch(reqs=[])
    )

    assert admitted == feasible
    assert len(errors) == 1
    assert errors[0][0] == impossible[0].rid
    assert "pool_tokens=2000" in str(errors[0][1])
