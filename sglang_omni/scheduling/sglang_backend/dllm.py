# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import List, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class DllmManager:
    """Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: requests waiting to be scheduled
    - staging_queue: requests allocated resources by PrefillAdder
    """

    def __init__(
        self,
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache,
        model_config,
        dllm_config: DllmConfig,
    ):
        self.server_args = server_args
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.model_config = model_config
        self.dllm_config = dllm_config

        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if len(req.output_ids) == 0]

    def get_decode_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if len(req.output_ids) > 0]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        existing_rids: set[str] = {r.rid for r in self.waiting_queue}
        if any(req.rid in existing_rids for req in reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")
        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def any_staging_reqs(self) -> bool:
        return len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        return len(self.waiting_queue) == 0 and len(self.staging_queue) == 0

    def increment_chunked_count(self) -> None:
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        for req in self.staging_queue:
            req.init_next_round_input()

    def schedule_next_batch(self):
        if self.is_empty():
            return None

        adder = PrefillAdder(
            self.server_args.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,  # running_batch
            0.5,  # new_token_ratio
            self.server_args.max_prefill_tokens,
            self.server_args.chunked_prefill_size,
            dllm_config=self.dllm_config,
        )

        staging_rids = {r.rid for r in self.staging_queue}

        if self.any_staging_reqs():
            self.init_next_round()
            for req in self.staging_queue:
                adder.add_chunked_req(req)
            new_staging = [
                r for r in adder.dllm_staging_reqs if r.rid not in staging_rids
            ]
            if new_staging:
                self.add_staging_reqs(new_staging)
                staging_rids.update(r.rid for r in new_staging)

        prefill_reqs = [
            r for r in self.get_prefill_requests() if r.rid not in staging_rids
        ]
        if prefill_reqs:
            self._process_batch(adder, prefill_reqs)
        else:
            decode_reqs = [
                r for r in self.get_decode_requests() if r.rid not in staging_rids
            ]
            if decode_reqs:
                self._process_batch(adder, decode_reqs)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        for req in adder.dllm_staging_reqs:
            if req.rid not in staging_rids:
                self.add_staging_reqs(req)

        self.increment_chunked_count()

        new_batch = ScheduleBatch.init_new(
            reqs=can_run_list,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()

        return new_batch

    def _process_batch(
        self,
        adder: PrefillAdder,
        batch: List[Req],
    ) -> None:
        """Schedule incoming (non-staging) requests through the PrefillAdder."""
        for req in batch:
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=self.any_staging_reqs(),
                truncation_align_size=None,
            )
            if res != AddReqResult.CONTINUE:
                break
