from __future__ import annotations

from typing import List, Optional

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class PrefillManager:
    def __init__(
        self,
        page_size,
        chunked_prefill_size,
        max_prefill_tokens,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache,
        model_config,
        enable_overlap,
    ):
        self.page_size = page_size
        self.chunked_prefill_size = chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.model_config = model_config
        self.enable_overlap = enable_overlap

        self.waiting_queue = []

        # unfinished chunked req last round
        self.chunked_req: Optional[Req] = None

    def add_one_request(self, req):
        # TODO(ocs884): whether to use the Req cls from sglang
        self.waiting_queue.append(req)

    def schedule_next_batch(
        self,
        running_batch: Optional[ScheduleBatch],
        num_allocatable_reqs: int,
        new_token_ratio: float = 0.5,
    ):
        # Implement the logic to schedule the next batch of tasks based on the waiting queue
        # Keep scheduling an unfinished chunked request even when no new waiting
        # requests are available or allocatable.
        if self.chunked_req is None and (
            len(self.waiting_queue) == 0 or num_allocatable_reqs <= 0
        ):
            return None

        adder = PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            running_batch=running_batch,
            new_token_ratio=new_token_ratio,
            rem_input_tokens=self.max_prefill_tokens,
            rem_chunk_tokens=self.chunked_prefill_size,
        )

        # if there is ongoing chunked prefill to complete
        if self.chunked_req is not None:
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:

            if len(adder.can_run_list) >= num_allocatable_reqs:
                if running_batch is not None:
                    running_batch.batch_is_full = True
                break

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                # NOTE(ocss8884): not support deterministic infer for now
                truncation_align_size=None,
            )
            # TODO(ocss884): process AddReqResult for mamba cache

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.is_chunked += 1

        # Batch dataclass of req for return
        new_batch = ScheduleBatch.init_new(
            reqs=can_run_list,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            chunked_req=self.chunked_req,
        )
        new_batch.prepare_for_extend()

        return new_batch

    @property
    def runnable(self) -> bool:
        return len(self.waiting_queue) > 0
