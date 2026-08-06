# SPDX-License-Identifier: Apache-2.0
"""DllmScheduler — stage-facing scheduler for Diffusion LLM stages.

Provides the same public contract (inbox, outbox, start, stop, abort)
as OmniScheduler so it is interchangeable from the Stage's perspective.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import threading
import time
from array import array
from typing import Any, Callable

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class DllmScheduler:
    """Stage-facing scheduler for Diffusion LLM stages.

    Public contract (used by Stage):
        ``inbox``, ``outbox``, ``start()``, ``stop()``, ``abort(request_id)``
    """

    def __init__(
        self,
        tp_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        dllm_config: Any,
        *,
        request_builder: Callable,
        result_adapter: Callable,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        self._request_builder = request_builder
        self._result_adapter = result_adapter

        self.tp_worker = tp_worker
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.server_args = server_args
        self.model_config = model_config
        self.dllm_config = dllm_config
        self._chunked_prefill_size = (
            getattr(dllm_config, "block_size", None) or server_args.chunked_prefill_size
        )

        self._running = False
        self._abort_lock = threading.Lock()
        self._aborted_request_ids: set[str] = set()
        self._rid_to_req_data: dict[str, Any] = {}
        self._waiting_queue: list[Req] = []
        self._staging_queue: list[Req] = []

    def start(self) -> None:
        self._running = True
        self._event_loop()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted_request_ids.add(request_id)

    def _event_loop(self) -> None:
        while self._running:
            self._drain_and_purge()
            batch = self._schedule_next_batch()

            if batch is None:
                time.sleep(0.001)
                continue

            resolve_deferred_prefill_inputs(batch, self.tp_worker.model_runner.device)
            forward_batch = ForwardBatch.init_new(
                batch,
                self.tp_worker.model_runner,
                return_hidden_states_before_norm=False,
            )
            batch_result = self.tp_worker.forward_batch_generation(
                forward_batch,
                batch=batch,
            )

            self._apply_results(batch, batch_result)
            self._post_step(batch)

    def _drain_and_purge(self) -> None:
        with self._abort_lock:
            aborted = self._aborted_request_ids
            self._aborted_request_ids = set()

        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.request_id in aborted:
                continue

            if msg.type == "new_request":
                req_data = self._request_builder(msg.data)
                req = req_data.req
                self._rid_to_req_data[req.rid] = req_data
                self._waiting_queue.append(req)
            else:
                logger.warning(
                    "DllmScheduler: unhandled message type %r for request %s",
                    msg.type,
                    msg.request_id,
                )

        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in aborted and not r.finished()
        ]
        new_staging = []
        for req in self._staging_queue:
            if req.rid in aborted:
                release_kv_cache(req, self.tree_cache)
            elif not req.finished():
                new_staging.append(req)
        self._staging_queue = new_staging

        for rid in aborted:
            self._rid_to_req_data.pop(rid, None)

    def _schedule_next_batch(self) -> ScheduleBatch | None:
        if not self._waiting_queue and not self._staging_queue:
            return None

        adder = PrefillAdder(
            self.server_args.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,  # running_batch
            0.5,  # new_token_ratio
            self.server_args.max_prefill_tokens,
            self._chunked_prefill_size,
            prefill_max_requests=1,
            dllm_config=self.dllm_config,
        )

        # Re-submit existing staging requests through the dLLM-specific budget
        # path. In FDFO mode an unresolved block must fit in full so its carried
        # algorithm state and resident KV describe the same block next round.
        staging_no_token = False
        for req in self._staging_queue:
            req.init_next_round_input()
            if adder.add_dllm_staging_req(req) == AddReqResult.NO_TOKEN:
                # A staging request that cannot fit stops all admission this
                # round (upstream parity); admitting waiting requests would
                # strand it without a slot.
                staging_no_token = True
                break

        # Add new waiting requests.
        if not staging_no_token:
            for req in self._waiting_queue:
                req.init_next_round_input(self.tree_cache)
                if (
                    adder.add_one_req(
                        req,
                        has_chunked_req=bool(self._staging_queue),
                        truncation_align_size=None,
                    )
                    != AddReqResult.CONTINUE
                ):
                    break

        if not adder.can_run_list:
            return None

        # Diffusion requests need to be rescheduled until they finish. Keep each
        # scheduled request in our stage-local staging queue.
        staging_rids = {r.rid for r in self._staging_queue}
        for req in adder.can_run_list:
            if req.rid not in staging_rids:
                self._staging_queue.append(req)
                staging_rids.add(req.rid)
        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in staging_rids
        ]

        new_batch = ScheduleBatch.init_new(
            reqs=adder.can_run_list,
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

    def _apply_results(self, batch: Any, batch_result: Any) -> None:
        next_token_ids = batch_result.next_token_ids
        if next_token_ids is None:
            return

        token_ids = (
            next_token_ids.tolist()
            if hasattr(next_token_ids, "tolist")
            else next_token_ids
        )
        # This stage runs one request at a time (PrefillAdder is built with
        # prefill_max_requests=1 in _schedule_next_batch), so the model may
        # return a flat list of token ids for the single request rather than a
        # list-per-request. Normalize that flat list into the per-request shape.
        # NOTE: if prefill_max_requests is ever raised above 1, this flat-list
        # branch must be revisited together with the scheduling cap, otherwise
        # the zip() below would pair each Req with a single int.
        if len(batch.reqs) == 1 and (not token_ids or isinstance(token_ids[0], int)):
            token_ids_per_req = [token_ids]
        else:
            token_ids_per_req = token_ids

        fdfo_mode = bool(self.dllm_config.first_done_first_out_mode)
        accept_lengths = batch_result.accept_length_per_req_cpu
        if fdfo_mode and accept_lengths is None:
            raise AssertionError("FDFO dLLM result is missing accept lengths.")
        algo_states = batch_result.dllm_algo_state
        block_size = int(self.dllm_config.block_size)

        if len(token_ids_per_req) != len(batch.reqs):
            raise ValueError(
                "dLLM result/request batch size mismatch: "
                f"{len(token_ids_per_req)} token rows for {len(batch.reqs)} requests"
            )
        if fdfo_mode and len(accept_lengths) != len(batch.reqs):
            raise ValueError(
                "FDFO dLLM accept-length/request batch size mismatch: "
                f"{len(accept_lengths)} accept lengths for {len(batch.reqs)} requests"
            )
        if (
            fdfo_mode
            and algo_states is not None
            and len(algo_states) != len(batch.reqs)
        ):
            raise ValueError(
                "FDFO dLLM algo-state/request batch size mismatch: "
                f"{len(algo_states)} states for {len(batch.reqs)} requests"
            )

        for idx, (req, req_token_ids) in enumerate(zip(batch.reqs, token_ids_per_req)):
            req_token_ids = (
                req_token_ids.tolist()
                if hasattr(req_token_ids, "tolist")
                else list(req_token_ids)
            )
            req_token_ids = [int(token_id) for token_id in req_token_ids]

            if fdfo_mode:
                if len(req_token_ids) != block_size:
                    raise ValueError(
                        "FDFO dLLM result block size mismatch: "
                        f"got {len(req_token_ids)}, expected {block_size}"
                    )
                if accept_lengths[idx] == 0:
                    # The block is only partially denoised. Carry both its token
                    # state and algorithm state, and leave output/finish state
                    # untouched until a later round resolves the whole block.
                    req.dllm_incomplete_ids = array("q", req_token_ids)
                    req.dllm_algo_state = (
                        algo_states[idx] if algo_states is not None else None
                    )
                    continue

                req.dllm_incomplete_ids = array("q")
                req.dllm_algo_state = None

            new_tokens = len(req_token_ids)
            if new_tokens == 0:
                continue

            # Commit real denoised tokens into the fill IDs used by the prefix
            # cache. Without this, the next round keys on the mask block.
            req.full_untruncated_fill_ids[
                req.extend_range.end - new_tokens : req.extend_range.end
            ] = array("q", req_token_ids)

            if fdfo_mode:
                len_input = len(req.origin_input_ids)
                len_fill = req.extend_range.end
                if len_fill <= len_input:
                    continue
                if len_fill - new_tokens < len_input:
                    req_token_ids = req_token_ids[len_input - len_fill :]
                    new_tokens = len(req_token_ids)

            req.output_ids.extend(req_token_ids)
            req.update_finish_state(new_accepted_len=new_tokens)

            if req.finished():
                req_data = self._rid_to_req_data.pop(req.rid, None)
                if req_data is None:
                    continue
                req_data.output_ids = list(req.output_ids_through_stop)
                finished_reason = req.finished_reason
                req_data.finish_reason = (
                    finished_reason.to_json().get("type")
                    if finished_reason is not None
                    else None
                )
                self.outbox.put(
                    OutgoingMessage(
                        request_id=req.rid,
                        type="result",
                        data=self._result_adapter(req_data),
                    )
                )

    def _post_step(self, batch: Any) -> None:
        exclude = set()
        for req in batch.reqs:
            if req.finished():
                release_kv_cache(req, self.tree_cache)
                exclude.add(req)

        new_staging = []
        fdfo_mode = bool(self.dllm_config.first_done_first_out_mode)
        for req in self._staging_queue:
            exclude.add(req)
            if req.finished():
                continue
            if fdfo_mode and req.dllm_incomplete_ids:
                # FDFO reuses the just-written KV and request slot while it
                # continues denoising this block in the next scheduler round.
                new_staging.append(req)
                continue
            self.tree_cache.cache_unfinished_req(req, chunked=True)
            if req.req_pool_idx is not None:
                # Note:(Chenchen Hong) post1 ReqToTokenPool.free takes the Req
                # (reads req.req_pool_idx then resets it to None), not the int.
                self.req_to_token_pool.free(req)
            new_staging.append(req)
        self._staging_queue = new_staging

        batch.filter_batch(chunked_req_to_exclude=list(exclude))
