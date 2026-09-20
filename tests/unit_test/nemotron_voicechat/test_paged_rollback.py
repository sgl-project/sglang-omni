# SPDX-License-Identifier: Apache-2.0
"""Exercise Talker rollback on CPU with simulated decode preparation.

Allocation and release use SGLang's paged allocator, but the test does not
execute its GPU decode allocation kernel.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.models.nemotron_voicechat.talker_scheduler import (
    NemotronTalkerScheduler,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.mark.parametrize(
    "page_size,prompt_lengths",
    [(1, [37]), (64, [37]), (64, [64]), (64, [37, 64])],
    ids=["single-token-pages", "in-page", "page-boundary", "mixed-boundaries"],
)
def test_skipped_decode_preserves_live_pages(page_size, prompt_lengths):
    allocator_args = dict(
        size=page_size * 256,
        dtype=torch.float32,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    allocator = (
        TokenToKVPoolAllocator(**allocator_args)
        if page_size == 1
        else PagedTokenToKVPoolAllocator(page_size=page_size, **allocator_args)
    )
    reqs = []
    pool = torch.zeros(len(prompt_lengths), 128, dtype=torch.long)
    for index, length in enumerate(prompt_lengths):
        slots = allocator.alloc(-(-length // page_size) * page_size)
        pool[index, :length] = slots[:length]
        req = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=[0] * length,
            sampling_params=SamplingParams(max_new_tokens=8),
            vocab_size=16,
        )
        req.kv.kv_committed_len = length
        req.kv.kv_allocated_len = length
        req._omni_data = SimpleNamespace(pending_text_queue=deque())
        reqs.append(req)
    batch = SimpleNamespace(
        reqs=reqs,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        req_pool_indices=torch.arange(len(reqs)),
        req_to_token_pool=SimpleNamespace(req_to_token=pool),
    )
    scheduler = object.__new__(NemotronTalkerScheduler)
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler._model_runner = object.__new__(NemotronVoiceChatTalkerModelRunner)
    initial_free = allocator.available_size()
    committed = pool.clone()

    def assert_competing_allocation_preserves_live_pages():
        live_slots = torch.cat(
            [pool[index, : req.kv.kv_committed_len] for index, req in enumerate(reqs)]
        )
        free_size = allocator.available_size()
        competing_slots = allocator.alloc(free_size)
        assert competing_slots is not None
        assert competing_slots.unique().numel() == free_size
        assert not torch.isin(
            competing_slots // page_size, live_slots // page_size
        ).any()
        allocator.free(competing_slots)
        assert allocator.available_size() == free_size

    def prepare_decode(_scheduler):
        locations = []
        for index, req in enumerate(reqs):
            length = req.kv.kv_committed_len
            location = (
                allocator.alloc(page_size)[0]
                if length % page_size == 0
                else pool[index, length - 1] + 1
            )
            pool[index, length] = location
            locations.append(location)
            req.decode_batch_idx += 1
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
        batch.out_cache_loc = torch.stack(locations)
        batch.seq_lens = torch.tensor([r.kv.kv_committed_len for r in reqs])
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.orig_seq_lens = batch.seq_lens.clone()
        return batch

    with patch.object(OmniScheduler, "get_next_batch_to_run", prepare_decode):
        for _ in range(2):
            assert scheduler.get_next_batch_to_run() is None
            assert allocator.available_size() == initial_free
            assert batch.seq_lens.tolist() == prompt_lengths
            assert batch.seq_lens_cpu.tolist() == prompt_lengths
            assert batch.orig_seq_lens.tolist() == prompt_lengths
            for index, req in enumerate(reqs):
                assert req.decode_batch_idx == 0
                assert req.kv.kv_committed_len == prompt_lengths[index]
                assert req.kv.kv_allocated_len == prompt_lengths[index]
                length = prompt_lengths[index]
                assert torch.equal(pool[index, :length], committed[index, :length])
            assert_competing_allocation_preserves_live_pages()

        for req in reqs:
            req._omni_data.pending_text_queue.append(12)
        assert scheduler.get_next_batch_to_run() is batch
        assert_competing_allocation_preserves_live_pages()
        for index, req in enumerate(reqs):
            assert req.kv.kv_committed_len == prompt_lengths[index] + 1
            live_slots = pool[index, : req.kv.kv_committed_len]
            allocator.free(live_slots)
        assert allocator.available_size() == allocator.size
