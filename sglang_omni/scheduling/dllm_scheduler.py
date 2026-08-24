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
from copy import copy
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs
from sglang_omni.scheduling.dllm_group import (
    DllmForwardGroup,
    DllmGroupMember,
    DllmRequestGroupSpec,
    apply_forward_group_padding,
)
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
        self._dllm_group_members: dict[str, tuple[str, ...]] = {}
        self._dllm_rid_to_group: dict[str, str] = {}
        self._dllm_hidden_rids: set[str] = set()
        self._dllm_orphaned_rids: set[str] = set()

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
            self._attach_forward_group(forward_batch, batch)
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
        aborted = self._expand_group_request_ids(aborted)

        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.request_id in aborted:
                continue

            if msg.type == "new_request":
                group_id: str | None = None
                try:
                    req_data = self._request_builder(msg.data)
                    req = req_data.req
                    self._rid_to_req_data[req.rid] = req_data
                    group_spec = getattr(req, "omni_dllm_group_spec", None)
                    if group_spec is None:
                        self._waiting_queue.append(req)
                    else:
                        group_id = req.rid
                        request_group = self._materialize_request_group(req, group_spec)
                        self._waiting_queue.extend(request_group)
                        self._validate_request_group_capacity(request_group)
                except Exception as exc:
                    logger.exception(
                        "DllmScheduler: request setup failed for %s", msg.request_id
                    )
                    if group_id is not None:
                        self._reject_waiting_request_group(group_id, exc)
                    else:
                        self.outbox.put(
                            OutgoingMessage(
                                request_id=msg.request_id,
                                type="error",
                                data=exc,
                            )
                        )
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
        aborted_group_ids = {
            group_id
            for rid in aborted
            if (group_id := getattr(self, "_dllm_rid_to_group", {}).get(rid))
            is not None
        }
        for group_id in aborted_group_ids:
            self._drop_request_group(group_id)

    def _materialize_request_group(
        self,
        primary: Req,
        group_spec: DllmRequestGroupSpec,
    ) -> list[Req]:
        """Create physical CFG companions and register one atomic group."""
        primary_length = len(primary.origin_input_ids)
        group_spec.validate(primary_input_length=primary_length)
        group_id = primary.rid
        if group_id in self._dllm_group_members:
            raise RuntimeError(f"duplicate dLLM request group {group_id}")

        primary.omni_dllm_group_member = DllmGroupMember(
            group_id=group_id,
            role="conditional",
            left_pad_length=group_spec.primary_left_pad_length,
            algorithm_args=group_spec.algorithm_args,
        )
        requests = [primary]
        for companion_index, companion_spec in enumerate(group_spec.companions, 1):
            sampling_params = copy(primary.sampling_params)
            custom_params = getattr(sampling_params, "custom_params", None)
            if isinstance(custom_params, dict):
                sampling_params.custom_params = {
                    key: value
                    for key, value in custom_params.items()
                    if key != "__req__"
                }
            companion = Req(
                rid=f"{group_id}:omni-cfg:{companion_index}",
                origin_input_text="",
                origin_input_ids=array("q", companion_spec.input_ids),
                sampling_params=sampling_params,
                vocab_size=primary.vocab_size,
                eos_token_ids=primary.eos_token_ids,
                dllm_config=(
                    getattr(primary, "dllm_config", None)
                    or getattr(self, "dllm_config", None)
                ),
            )
            companion.tokenizer = getattr(primary, "tokenizer", None)
            companion.omni_model_inputs = None
            companion._omni_consumed = None
            companion.omni_dllm_group_member = DllmGroupMember(
                group_id=group_id,
                role=companion_spec.role,
                left_pad_length=companion_spec.left_pad_length,
                algorithm_args=group_spec.algorithm_args,
            )
            requests.append(companion)

        member_rids = tuple(request.rid for request in requests)
        self._dllm_group_members[group_id] = member_rids
        self._dllm_rid_to_group.update({request.rid: group_id for request in requests})
        self._dllm_hidden_rids.update(member_rids[1:])
        return requests

    def _expand_group_request_ids(self, request_ids: set[str]) -> set[str]:
        expanded = set(request_ids)
        group_members = getattr(self, "_dllm_group_members", {})
        rid_to_group = getattr(self, "_dllm_rid_to_group", {})
        for request_id in request_ids:
            group_id = rid_to_group.get(request_id)
            if group_id is not None:
                expanded.update(group_members[group_id])
        return expanded

    def _drop_request_group(self, group_id: str) -> None:
        member_rids = self._dllm_group_members.pop(group_id, ())
        for member_rid in member_rids:
            self._dllm_rid_to_group.pop(member_rid, None)
            self._dllm_hidden_rids.discard(member_rid)
            self._dllm_orphaned_rids.discard(member_rid)

    def _reject_waiting_request_group(
        self, group_id: str, error: BaseException
    ) -> None:
        """Reject one unadmitted physical group and clear all scheduler state."""
        member_rids = set(self._dllm_group_members.get(group_id, ())) | {group_id}
        self._waiting_queue = [
            request for request in self._waiting_queue if request.rid not in member_rids
        ]
        for member_rid in member_rids:
            self._rid_to_req_data.pop(member_rid, None)
        self._drop_request_group(group_id)
        self.outbox.put(OutgoingMessage(request_id=group_id, type="error", data=error))

    def _request_group_at_head(self, requests: list[Req]) -> list[Req]:
        if not requests:
            return []
        first = requests[0]
        group_id = getattr(self, "_dllm_rid_to_group", {}).get(first.rid)
        if group_id is None:
            return [first]
        expected_rids = self._dllm_group_members[group_id]
        requests_by_rid = {request.rid: request for request in requests}
        missing = [rid for rid in expected_rids if rid not in requests_by_rid]
        if missing:
            raise RuntimeError(
                f"incomplete dLLM request group {group_id}: missing {missing}"
            )
        return [requests_by_rid[rid] for rid in expected_rids]

    def _validate_request_group_capacity(self, requests: list[Req]) -> None:
        """Reject grouped requests that can never fit the configured budgets."""
        if len(requests) <= 1:
            return

        max_running_requests = getattr(self.server_args, "max_running_requests", None)
        if max_running_requests is not None and int(max_running_requests) < len(
            requests
        ):
            raise RuntimeError(
                f"dLLM group requires {len(requests)} running requests, but "
                f"max_running_requests={max_running_requests}"
            )

        block_size = getattr(self.dllm_config, "block_size", None)
        max_prefill_tokens = getattr(self.server_args, "max_prefill_tokens", None)
        if block_size is None or max_prefill_tokens is None:
            return

        page_size = max(int(getattr(self.server_args, "page_size", 1)), 1)
        block_size = int(block_size)
        block_charge = (block_size + page_size - 1) // page_size * page_size
        largest_initial_extend = max(
            len(request.origin_input_ids) + block_size for request in requests
        )
        largest_initial_extend = (
            (largest_initial_extend + page_size - 1) // page_size * page_size
        )
        required_prefill_tokens = (
            largest_initial_extend + (len(requests) - 1) * block_charge + 1
        )
        if int(max_prefill_tokens) < required_prefill_tokens:
            raise RuntimeError(
                "dLLM group requires at least "
                f"{required_prefill_tokens} max_prefill_tokens, but configured "
                f"value is {max_prefill_tokens}"
            )

    def _attach_forward_group(self, forward_batch: Any, batch: Any) -> None:
        if not batch.reqs:
            return
        first_member = getattr(batch.reqs[0], "omni_dllm_group_member", None)
        if first_member is None:
            if any(
                getattr(request, "omni_dllm_group_member", None) is not None
                for request in batch.reqs[1:]
            ):
                raise RuntimeError("cannot mix grouped and ordinary dLLM requests")
            if len(batch.reqs) != 1:
                raise RuntimeError(
                    "ordinary dLLM requests must remain independently scheduled"
                )
            image_token_offset = int(
                getattr(batch.reqs[0], "omni_dllm_image_token_offset", 0)
            )
            if image_token_offset > 0:
                forward_batch.omni_dllm_image_token_offsets = torch.full(
                    (int(forward_batch.batch_size),),
                    image_token_offset,
                    dtype=torch.int64,
                    device=forward_batch.input_ids.device,
                )
            return

        members = tuple(
            getattr(request, "omni_dllm_group_member", None) for request in batch.reqs
        )
        if any(
            member is None or member.group_id != first_member.group_id
            for member in members
        ):
            raise RuntimeError("dLLM forward batch contains multiple request groups")
        expected_rids = self._dllm_group_members[first_member.group_id]
        if tuple(request.rid for request in batch.reqs) != expected_rids:
            raise RuntimeError("dLLM request-group row order changed before forward")

        group = DllmForwardGroup(
            group_id=first_member.group_id,
            roles=tuple(member.role for member in members),
            left_pad_lengths=tuple(member.left_pad_length for member in members),
            algorithm_args=first_member.algorithm_args,
        )
        apply_forward_group_padding(forward_batch, group)
        group_prefill_flags = tuple(request.is_dllm_prefill() for request in batch.reqs)
        if len(set(group_prefill_flags)) != 1:
            raise RuntimeError("dLLM request-group phases diverged before forward")
        forward_batch.omni_dllm_group_is_prefill = group_prefill_flags[0]

    def _rollback_group_admission(
        self,
        admitted_requests: list[Req],
        request_snapshots: list[tuple[Req, dict[str, Any]]],
        *,
        from_staging: bool,
    ) -> None:
        if not from_staging:
            for request in admitted_requests:
                if request.last_node is not None:
                    self.tree_cache.dec_lock_ref(request.last_node)
        for request, snapshot in request_snapshots:
            request.__dict__.clear()
            request.__dict__.update(snapshot)

    @staticmethod
    def _synchronize_request_group_phases(requests: list[Req]) -> None:
        """Keep all physical CFG rows in the conditional row's dLLM phase."""
        if len(requests) < 2:
            return
        primary_phase = getattr(requests[0], "dllm_phase", None)
        for request in requests[1:]:
            request.dllm_phase = primary_phase

    def _schedule_next_batch(self) -> ScheduleBatch | None:
        if not self._waiting_queue and not self._staging_queue:
            return None

        source_queue = (
            self._staging_queue if self._staging_queue else self._waiting_queue
        )
        request_group = self._request_group_at_head(source_queue)
        self._validate_request_group_capacity(request_group)
        snapshots = [(request, request.__dict__.copy()) for request in request_group]

        adder = PrefillAdder(
            self.server_args.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,  # running_batch
            0.5,  # new_token_ratio
            self.server_args.max_prefill_tokens,
            self._chunked_prefill_size,
            prefill_max_requests=len(request_group),
            dllm_config=self.dllm_config,
        )

        # Re-submit existing staging requests through the dLLM-specific budget
        # path. In FDFO mode an unresolved block must fit in full so its carried
        # algorithm state and resident KV describe the same block next round.
        from_staging = source_queue is self._staging_queue
        if from_staging:
            for req in request_group:
                req.init_next_round_input()
                if adder.add_dllm_staging_req(req) == AddReqResult.NO_TOKEN:
                    break
        else:
            for req in request_group:
                req.init_next_round_input(self.tree_cache)
                if (
                    adder.add_one_req(
                        req,
                        has_chunked_req=False,
                        truncation_align_size=None,
                    )
                    != AddReqResult.CONTINUE
                ):
                    break

        expected_rids = [request.rid for request in request_group]
        admitted_rids = [request.rid for request in adder.can_run_list]
        if admitted_rids != expected_rids:
            self._rollback_group_admission(
                adder.can_run_list,
                snapshots,
                from_staging=from_staging,
            )
            return None

        self._synchronize_request_group_phases(adder.can_run_list)

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
        # Ordinary one-row requests can return a flat token list; grouped CFG
        # always returns one row per physical branch.
        if len(batch.reqs) == 1 and (not token_ids or isinstance(token_ids[0], int)):
            token_ids_per_req = [token_ids]
        else:
            token_ids_per_req = token_ids

        fdfo_mode = bool(self.dllm_config.first_done_first_out_mode)
        accept_lengths = batch_result.accept_length_per_req_cpu
        group_is_prefill = bool(batch.reqs) and all(
            request.is_dllm_prefill() for request in batch.reqs
        )
        if group_is_prefill and not any(token_ids_per_req):
            return
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

        is_grouped = bool(
            batch.reqs
            and getattr(batch.reqs[0], "omni_dllm_group_member", None) is not None
        )
        if is_grouped:
            normalized_rows = [
                tuple(
                    int(token_id)
                    for token_id in (row.tolist() if hasattr(row, "tolist") else row)
                )
                for row in token_ids_per_req
            ]
            if any(row != normalized_rows[0] for row in normalized_rows[1:]):
                raise RuntimeError("grouped dLLM result rows diverged")
            if fdfo_mode and any(
                length != accept_lengths[0] for length in accept_lengths[1:]
            ):
                raise RuntimeError("grouped dLLM FDFO accept lengths diverged")

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
                hidden_rids = getattr(self, "_dllm_hidden_rids", set())
                if req.rid not in hidden_rids:
                    group_id = getattr(self, "_dllm_rid_to_group", {}).get(req.rid)
                    if group_id is not None:
                        self._dllm_orphaned_rids.update(
                            rid
                            for rid in self._dllm_group_members[group_id]
                            if rid in hidden_rids
                        )
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
                try:
                    result = self._result_adapter(req_data)
                except Exception as exc:
                    logger.exception(
                        "DllmScheduler: result adapter failed for %s", req.rid
                    )
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=req.rid,
                            type="error",
                            data=exc,
                        )
                    )
                else:
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=req.rid,
                            type="result",
                            data=result,
                        )
                    )

    def _post_step(self, batch: Any) -> None:
        exclude = set()
        retired_rids = {req.rid for req in batch.reqs if req.finished()} | set(
            getattr(self, "_dllm_orphaned_rids", set())
        )
        group_members = getattr(self, "_dllm_group_members", {})
        rid_to_group = getattr(self, "_dllm_rid_to_group", {})
        retired_group_ids = {
            group_id
            for rid in retired_rids
            if (group_id := rid_to_group.get(rid)) is not None
        }
        for group_id in retired_group_ids:
            retired_rids.update(group_members[group_id])

        self._waiting_queue = [
            req for req in self._waiting_queue if req.rid not in retired_rids
        ]
        released_rids: set[str] = set()

        def release_once(req: Req) -> None:
            if req.rid in released_rids:
                return
            release_kv_cache(req, self.tree_cache)
            released_rids.add(req.rid)

        new_staging = []
        fdfo_mode = bool(
            getattr(
                getattr(self, "dllm_config", None),
                "first_done_first_out_mode",
                False,
            )
        )
        for req in self._staging_queue:
            exclude.add(req)
            if req.rid in retired_rids:
                release_once(req)
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

        for req in batch.reqs:
            if req.rid in retired_rids:
                exclude.add(req)
                release_once(req)

        for group_id in retired_group_ids:
            self._drop_request_group(group_id)
        if hasattr(self, "_dllm_orphaned_rids"):
            self._dllm_orphaned_rids.difference_update(retired_rids)

        batch.filter_batch(chunked_req_to_exclude=list(exclude))
