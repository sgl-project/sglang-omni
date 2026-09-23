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
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import broadcast_pyobj

from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


@dataclass
class DllmForwardBatch(ForwardBatch):
    """Keep DLLM metadata through the eager runner's dataclass batch rebuilds."""

    reqs: list[Req] = field(default_factory=list)
    dllm_left_pad_lens_cpu: list[int] = field(default_factory=list)


def release_kv_once(req: Req, tree_cache: Any) -> None:
    if req.kv.holds_kv:
        release_kv_cache(req, tree_cache)


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
        request_builder: Callable[[Any], Any],
        result_adapter: Callable[[Any], Any],
    ) -> None:
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
            dllm_config.block_size or get_schedule().chunked_prefill_size
        )

        self._running = False
        self._abort_lock = threading.Lock()
        self._aborted_request_ids: set[str] = set()
        self._rid_to_req_data: dict[str, Any] = {}
        self._waiting_queue: list[Req] = []
        self._staging_queue: list[Req] = []

        self.tp_rank = tp_worker.tp_rank
        self.tp_size = get_parallel().tp_size
        self.requires_tp_work_fanout = False

        # CFG tracking: cond_rid <-> uncond_rid(s)
        self._cond_to_unconds: dict[str, list[str]] = {}
        self._uncond_to_cond: dict[str, str] = {}
        self._uncond_rids: set[str] = set()
        self._orphaned_uncond_rids: set[str] = set()

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
        while self.drain_and_purge():
            batch = self.schedule_next_batch()

            if batch is None:
                time.sleep(0.001)
                continue

            resolve_deferred_prefill_inputs(batch, self.tp_worker.model_runner.device)
            forward_batch = DllmForwardBatch.init_new(
                batch,
                self.tp_worker.model_runner,
                return_hidden_states_before_norm=False,
            )
            forward_batch.reqs = batch.reqs
            self.apply_cfg_padding_metadata(forward_batch, batch)
            batch_result = self.tp_worker.forward_batch_generation(
                forward_batch,
                batch=batch,
            )

            self.apply_results(batch, batch_result)
            self.post_step(batch)

    def drain_and_purge(self) -> bool:
        with self._abort_lock:
            aborted = self._aborted_request_ids
            self._aborted_request_ids = set()
        messages: list[IncomingMessage] = []
        while True:
            try:
                messages.append(self.inbox.get_nowait())
            except _queue_mod.Empty:
                break
        running = self._running
        if self.tp_size > 1:
            # note (Anmuliar): All ranks must cancel and stop at the same forward boundary.
            group = self.tp_worker.model_runner.tp_group
            running, aborted, messages = broadcast_pyobj(
                [running, aborted, messages] if self.tp_rank == 0 else [],
                group.rank,
                group.cpu_group,
                src=group.ranks[0],
            )
        if not running:
            return False
        # Aborting any member of a CFG group must purge the whole group.
        aborted_groups: set[str] = set()
        for rid in aborted:
            cond_rid = self._uncond_to_cond.get(rid, rid)
            aborted_groups.add(cond_rid)
            aborted_groups.update(self._cond_to_unconds.get(cond_rid, ()))
        aborted = aborted_groups

        for msg in messages:
            if msg.request_id in aborted:
                continue

            if msg.type == "new_request":
                req_data = self._request_builder(msg.data)
                req = req_data.req
                self._rid_to_req_data[req.rid] = req_data
                self._waiting_queue.append(req)

                # CFG: create companion uncond Reqs (text guidance always; image
                # guidance only when cfg_image_scale > 0).
                uncond_ids = getattr(req, "_uncond_input_ids", None)
                if uncond_ids is not None:
                    self.create_uncond_companion(
                        req,
                        uncond_ids,
                        getattr(req, "_uncond_left_pad_len", 0),
                        "-uncond",
                        mark_img=False,
                    )
                    uncond_img_ids = getattr(req, "_uncond_img_input_ids", None)
                    if uncond_img_ids is not None:
                        self.create_uncond_companion(
                            req,
                            uncond_img_ids,
                            getattr(req, "_uncond_img_left_pad_len", 0),
                            "-uncond-img",
                            mark_img=True,
                        )

                companion_rids = self._cond_to_unconds.get(req.rid, ())
                reqs_by_rid = {queued.rid: queued for queued in self._waiting_queue}
                request_group = [reqs_by_rid[rid] for rid in (req.rid, *companion_rids)]
                try:
                    self.validate_request_group_capacity(request_group)
                except RuntimeError as exc:
                    logger.warning(
                        "DllmScheduler: rejecting request %s: %s",
                        req.rid,
                        exc,
                    )
                    self.reject_waiting_request_group(req.rid, str(exc))
            else:
                logger.warning(
                    "DllmScheduler: unhandled message type %r for request %s",
                    msg.type,
                    msg.request_id,
                )

        # Purge aborted requests
        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in aborted and not r.finished()
        ]
        new_staging = []
        for req in self._staging_queue:
            if req.rid in aborted:
                release_kv_once(req, self.tree_cache)
            elif not req.finished():
                new_staging.append(req)
        self._staging_queue = new_staging

        for rid in aborted:
            self._rid_to_req_data.pop(rid, None)
            for uncond_rid in self._cond_to_unconds.pop(rid, []):
                self._uncond_to_cond.pop(uncond_rid, None)
            cond_rid = self._uncond_to_cond.pop(rid, None)
            if cond_rid is not None and cond_rid in self._cond_to_unconds:
                companions = self._cond_to_unconds[cond_rid]
                if rid in companions:
                    companions.remove(rid)
            self._uncond_rids.discard(rid)
            self._orphaned_uncond_rids.discard(rid)

        return True

    def create_uncond_companion(
        self,
        cond_req: Req,
        uncond_input_ids: list[int],
        left_pad_len: int,
        rid_suffix: str,
        mark_img: bool,
    ) -> None:
        """Create a companion uncond Req for CFG and add to waiting queue."""
        from sglang.srt.sampling.sampling_params import SamplingParams

        uncond_rid = f"{cond_req.rid}{rid_suffix}"
        uncond_input_ids = list(uncond_input_ids)
        if len(uncond_input_ids) != len(cond_req.origin_input_ids):
            raise ValueError(
                "CFG companion input must be physically aligned with the "
                f"conditional input: cond={len(cond_req.origin_input_ids)}, "
                f"companion={len(uncond_input_ids)}"
            )
        if not 0 <= int(left_pad_len) <= len(uncond_input_ids):
            raise ValueError("CFG left-pad length is outside the companion prompt")
        uncond_sampling_params = SamplingParams(
            max_new_tokens=cond_req.sampling_params.max_new_tokens,
            temperature=0.0,
        )
        # The companion still needs normalized SGLang sampling metadata.
        uncond_sampling_params.normalize(None)
        uncond_sampling_params.verify(cond_req.vocab_size)
        uncond_req = Req(
            rid=uncond_rid,
            origin_input_text="",
            origin_input_ids=array("q", uncond_input_ids),
            sampling_params=uncond_sampling_params,
            vocab_size=cond_req.vocab_size,
            eos_token_ids=cond_req.eos_token_ids,
            dllm_config=cond_req.dllm_config,
        )
        uncond_req.tokenizer = cond_req.tokenizer
        uncond_req._is_uncond = True
        uncond_req._dllm_left_pad_len = int(left_pad_len)
        uncond_req._cfg_scale = getattr(cond_req, "_cfg_scale", 4.0)
        uncond_req._cfg_rescale = getattr(cond_req, "_cfg_rescale", 0.7)
        cond_req._cfg_group_rid = cond_req.rid
        uncond_req._cfg_group_rid = cond_req.rid
        if mark_img:
            uncond_req._is_uncond_img = True

        self._waiting_queue.append(uncond_req)
        self._cond_to_unconds.setdefault(cond_req.rid, []).append(uncond_rid)
        self._uncond_to_cond[uncond_rid] = cond_req.rid
        self._uncond_rids.add(uncond_rid)

    def reject_waiting_request_group(
        self,
        cond_rid: str,
        error: str,
    ) -> None:
        companion_rids = self._cond_to_unconds.pop(cond_rid, [])
        group_rids = {cond_rid, *companion_rids}
        self._waiting_queue = [
            req for req in self._waiting_queue if req.rid not in group_rids
        ]
        self._rid_to_req_data.pop(cond_rid, None)
        for companion_rid in companion_rids:
            self._uncond_to_cond.pop(companion_rid, None)
            self._uncond_rids.discard(companion_rid)
            self._orphaned_uncond_rids.discard(companion_rid)
        self.outbox.put(
            OutgoingMessage(
                request_id=cond_rid,
                type="error",
                data=error,
            )
        )

    def apply_cfg_padding_metadata(
        self,
        forward_batch: DllmForwardBatch,
        batch: ScheduleBatch,
    ) -> None:
        """Apply mask and position metadata for mask-padded CFG branches."""
        left_pad_lengths = [
            int(getattr(req, "_dllm_left_pad_len", 0)) for req in batch.reqs
        ]
        forward_batch.dllm_left_pad_lens_cpu = left_pad_lengths
        if not any(left_pad_lengths):
            return
        if any(left_pad_length < 0 for left_pad_length in left_pad_lengths):
            raise RuntimeError("CFG left-pad lengths must be non-negative")
        if not forward_batch.forward_mode.is_extend():
            raise RuntimeError("CFG left-pad metadata requires an extend batch")

        extend_sequence_lengths = list(forward_batch.extend_seq_lens_cpu)
        if len(extend_sequence_lengths) != len(left_pad_lengths):
            raise RuntimeError(
                f"CFG pad metadata batch mismatch: {len(left_pad_lengths)} vs "
                f"{len(extend_sequence_lengths)}"
            )

        position_start = 0
        for left_pad_length, extend_sequence_length in zip(
            left_pad_lengths, extend_sequence_lengths
        ):
            position_end = position_start + int(extend_sequence_length)
            if left_pad_length:
                request_positions = forward_batch.positions[position_start:position_end]
                request_positions.sub_(left_pad_length).clamp_min_(0)
            position_start = position_end
        if position_start != forward_batch.positions.numel():
            raise RuntimeError(
                f"CFG position span {position_start} != "
                f"{forward_batch.positions.numel()}"
            )

    def synchronize_cfg_phases(self, reqs: list[Req]) -> None:
        """Keep CFG companions in the conditional request's DLLM phase."""
        if len(reqs) < 2:
            return
        cond_req = next(
            (req for req in reqs if not getattr(req, "_is_uncond", False)), None
        )
        if cond_req is None:
            return
        for req in reqs:
            if getattr(req, "_is_uncond", False):
                req.dllm_phase = cond_req.dllm_phase

    def get_request_group(self, queue: list[Req]) -> list[Req]:
        """Return the complete logical request group at the head of a queue."""
        if not queue:
            return []

        first_req = queue[0]
        cond_rid = self._uncond_to_cond.get(first_req.rid, first_req.rid)
        expected_rids = [
            cond_rid,
            *self._cond_to_unconds.get(cond_rid, ()),
        ]
        reqs_by_rid = {req.rid: req for req in queue}
        missing_rids = [rid for rid in expected_rids if rid not in reqs_by_rid]
        if missing_rids:
            raise RuntimeError(
                f"Incomplete CFG request group {cond_rid}: missing {missing_rids}"
            )
        return [reqs_by_rid[rid] for rid in expected_rids]

    def validate_request_group_capacity(self, reqs: list[Req]) -> None:
        if len(reqs) <= 1:
            return

        max_running_requests = getattr(
            self.dllm_config,
            "max_running_requests",
            None,
        )
        if max_running_requests is not None and max_running_requests < len(reqs):
            raise RuntimeError(
                "CFG request group requires "
                f"{len(reqs)} running requests, but max_running_requests="
                f"{max_running_requests}"
            )

        if self.dllm_config.first_done_first_out_mode:
            raise RuntimeError(
                "DLLM CFG requires synchronous execution; FDFO is unsupported"
            )
        block_size = self.dllm_config.block_size
        max_prefill_tokens = get_schedule().max_prefill_tokens
        page_size = get_schedule().page_size
        # Chunked DLLM admission charges one block per branch, not its full prompt.
        block_charge = (block_size + page_size - 1) // page_size * page_size
        required_prefill_tokens = len(reqs) * block_charge
        if (
            max_prefill_tokens is not None
            and max_prefill_tokens < required_prefill_tokens
        ):
            raise RuntimeError(
                "CFG request group requires at least "
                f"{required_prefill_tokens} max_prefill_tokens, but configured "
                f"value is {max_prefill_tokens}"
            )

    def rollback_partial_admission(
        self,
        admitted_reqs: list[Req],
        *,
        from_staging: bool,
        request_snapshots: list[tuple[Req, dict[str, Any]]],
    ) -> None:
        """Undo request and cache mutations from an incomplete group probe."""
        if not from_staging:
            for req in admitted_reqs:
                self.tree_cache.dec_lock_ref(req.last_node)
        for req, state in request_snapshots:
            req.__dict__.clear()
            req.__dict__.update(state)

    def schedule_next_batch(self) -> ScheduleBatch | None:
        if not self._waiting_queue and not self._staging_queue:
            return None

        source_queue = (
            self._staging_queue if self._staging_queue else self._waiting_queue
        )
        request_group = self.get_request_group(source_queue)
        self.validate_request_group_capacity(request_group)
        # Admission mutates Req.kv as well as top-level phase/range fields.
        request_snapshots = [
            (req, {**req.__dict__, "kv": copy(req.kv)}) for req in request_group
        ]

        adder = PrefillAdder(
            get_schedule().page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,  # running_batch
            0.5,  # new_token_ratio
            get_schedule().max_prefill_tokens,
            self._chunked_prefill_size,
            prefill_max_requests=len(request_group),
            dllm_config=self.dllm_config,
        )

        from_staging = source_queue is self._staging_queue
        if from_staging:
            for req in request_group:
                req.init_next_round_input()
                result = adder.add_dllm_staging_req(req)
                if result == AddReqResult.NO_TOKEN:
                    break
        else:
            for req in request_group:
                req.init_next_round_input(self.tree_cache)
                result = adder.add_one_req(
                    req,
                    has_chunked_req=False,
                    truncation_align_size=None,
                )
                if result != AddReqResult.CONTINUE:
                    break

        expected_rids = [req.rid for req in request_group]
        scheduled_rids = [req.rid for req in adder.can_run_list]
        ready = scheduled_rids == expected_rids
        if ready and len(request_group) > 1:
            self.synchronize_cfg_phases(adder.can_run_list)
            spans = {
                (req.extend_range.start, req.extend_range.end)
                for req in adder.can_run_list
            }
            if len(spans) != 1 or any(
                req.extend_range.length != self.dllm_config.block_size
                for req in adder.can_run_list
            ):
                ready = False

        if not ready:
            self.rollback_partial_admission(
                adder.can_run_list,
                from_staging=from_staging,
                request_snapshots=request_snapshots,
            )
            return None

        # Reschedule the same logical request until all of its blocks finish.
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

    def apply_results(self, batch: Any, batch_result: Any) -> None:
        # Mask-token left padding is prompt data, never generated output.
        if len(batch.reqs) > 1 and all(req.is_dllm_prefill() for req in batch.reqs):
            return

        next_token_ids = batch_result.next_token_ids
        if next_token_ids is None:
            return

        token_ids = (
            next_token_ids.tolist()
            if hasattr(next_token_ids, "tolist")
            else next_token_ids
        )
        # CFG returns one token row per physical branch.
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

        if not token_ids_per_req:
            return
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
            # Companions share generated tokens but only the conditional
            # request owns stop conditions and a user-visible result.
            if req.rid in self._uncond_rids:
                continue
            req.update_finish_state(new_accepted_len=new_tokens)

            if req.finished():
                for rid in self._cond_to_unconds.pop(req.rid, []):
                    self._uncond_to_cond.pop(rid, None)
                    self._orphaned_uncond_rids.add(rid)
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
                    message_type = "result"
                except Exception as exc:
                    logger.exception("DLLM result adapter failed for %s", req.rid)
                    result = str(exc)
                    message_type = "error"
                self.outbox.put(
                    OutgoingMessage(
                        request_id=req.rid,
                        type=message_type,
                        data=result,
                    )
                )

    def post_step(self, batch: Any) -> None:
        orphaned = self._orphaned_uncond_rids
        exclude = set()
        for req in batch.reqs:
            if req.finished() or req.rid in orphaned:
                release_kv_once(req, self.tree_cache)
                exclude.add(req)

        new_staging = []
        fdfo_mode = bool(self.dllm_config.first_done_first_out_mode)
        for req in self._staging_queue:
            exclude.add(req)
            if req.rid in orphaned:
                release_kv_once(req, self.tree_cache)
                continue
            if req.finished():
                continue
            if fdfo_mode and req.dllm_incomplete_ids:
                # FDFO reuses the just-written KV and request slot while it
                # continues denoising this block in the next scheduler round.
                new_staging.append(req)
                continue
            self.tree_cache.cache_unfinished_req(req, chunked=True)
            # Keep the row until finish/abort so release_kv_cache owns both
            # cached tokens and the request slot throughout staging.
            new_staging.append(req)
        self._staging_queue = new_staging

        self._waiting_queue = [r for r in self._waiting_queue if r.rid not in orphaned]
        self._uncond_rids.difference_update(orphaned)
        self._orphaned_uncond_rids.clear()
        batch.filter_batch(chunked_req_to_exclude=list(exclude))
