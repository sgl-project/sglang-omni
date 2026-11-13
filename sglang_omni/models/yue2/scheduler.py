# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler that chains the YuE2 score and audio phases and pairs CFG rows."""

import threading
from collections.abc import Iterable

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .model_runner import rewind_sampling
from .request import ABC_END
from .sglang_request_builder import (
    Yue2ARRequestData,
    Yue2RequestBuilder,
    guidance_request_id,
    public_request_id,
    score_request_id,
)


def request_data(req: Req) -> Yue2ARRequestData | None:
    return req.omni_data


def admission_width(queue: list[Req], index: int) -> int:
    """Rows the queue entry at index occupies: two when its CFG twin follows it."""
    data = request_data(queue[index])
    if (
        data is not None
        and data.cfg_uncond is not None
        and index + 1 < len(queue)
        and request_data(queue[index + 1]) is data.cfg_uncond
    ):
        return 2
    else:
        return 1


def paired_queue(queue: list[Req]) -> list[Req]:
    """The queue with each unconditional row moved right behind its request."""
    queued = {id(req) for req in queue}
    twins = {
        id(data.cfg_uncond.req)
        for req in queue
        if (data := request_data(req)) is not None
        and data.cfg_uncond is not None
        and id(data.cfg_uncond.req) in queued
    }
    ordered: list[Req] = []
    for req in queue:
        data = request_data(req)
        if id(req) in twins:
            pass
        elif (
            data is not None
            and data.cfg_uncond is not None
            and id(data.cfg_uncond.req) in twins
        ):
            ordered.extend([req, data.cfg_uncond.req])
        else:
            ordered.append(req)
    return ordered


def prefill_token_count(req: Req) -> int:
    """Tokens a prefill of req computes; a retracted request replays its output."""
    return len(req.origin_input_ids) + len(req.output_ids)


class Yue2PrefillAdder(PrefillAdder):
    """Admit a CFG request together with its unconditional row, or neither."""

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: int | None
    ) -> AddReqResult:
        result = super().add_one_req(req, has_chunked_req, truncation_align_size)
        data = request_data(req)
        admitted = bool(self.can_run_list) and self.can_run_list[-1] is req
        previous = (
            request_data(self.can_run_list[-1])
            if self.can_run_list and not admitted
            else None
        )
        if data is None:
            pass
        elif (
            data.cfg_uncond is not None and admitted and result != AddReqResult.CONTINUE
        ):
            # note (ruiling): admission stops before the twin, so the request waits with it.
            self.withdraw_last()
        elif (
            data.is_cfg_uncond and previous is not None and previous.cfg_uncond is data
        ):
            self.withdraw_last()
        else:
            pass
        return result

    def withdraw_last(self) -> None:
        req = self.can_run_list.pop()
        self.tree_cache.unlock(req.lock)
        req.lock = None


class Yue2Scheduler(OmniScheduler[Yue2ARRequestData]):
    request_builder: Yue2RequestBuilder
    decoding_batch: ScheduleBatch | None = None

    def enqueue_built_request(
        self,
        payload: StagePayload,
        pending_stream_done: bool,
        req_data: Yue2ARRequestData,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        capacity_error = self.kv_capacity_error(req_data)
        if capacity_error is not None:
            self.emit_request_error(payload.request_id, ValueError(capacity_error))
            self.abort(payload.request_id)
            return
        else:
            pass
        super().enqueue_built_request(
            payload,
            pending_stream_done,
            req_data,
            request_admission_lock_held=request_admission_lock_held,
        )
        unconditional = req_data.cfg_uncond
        if unconditional is None:
            return
        elif request_admission_lock_held:
            self.enqueue_unconditional(req_data, unconditional)
        else:
            with self.request_admission_lock:
                self.enqueue_unconditional(req_data, unconditional)

    def kv_capacity_error(self, req_data: Yue2ARRequestData) -> str | None:
        """Reject a request whose rows can never hold the KV cache together."""
        guided = req_data.cfg_uncond is not None
        rows = [req_data, req_data.cfg_uncond] if guided else [req_data]
        required = sum(
            len(row.req.origin_input_ids) + row.req.sampling_params.max_new_tokens
            for row in rows
        )
        # note (ruiling): SGLang admits a row only with one spare page, and keeps one token free.
        capacity = (
            self.tp_worker.model_runner.effective_max_total_num_tokens
            - 1
            - len(rows) * self.page_size
        )
        if required <= capacity:
            return None
        else:
            # note (ruiling): SGLang checks each row alone; a CFG pair that never fits would wait forever and block the queue.
            rows_note = " for its two classifier-free guidance rows" if guided else ""
            budget = "abc_max_tokens" if req_data.phase == "abc" else "max_new_tokens"
            return (
                f"YuE2 request: needs {required} KV cache tokens{rows_note}, above "
                f"the {capacity}-token limit. Lower {budget} or raise "
                "--yue2_ar.engine.max_total_tokens."
            )

    def enqueue_unconditional(
        self, req_data: Yue2ARRequestData, unconditional: Yue2ARRequestData
    ) -> None:
        conditional = req_data.req
        if not self.waiting_queue or self.waiting_queue[-1] is not conditional:
            return
        else:
            pass
        req = unconditional.req
        self.normalize_req_token_arrays(req)
        req._coalesce_enqueue_t = (
            conditional._coalesce_enqueue_t
        )  # noqa: leading-underscore  # upstream spelling
        req._omni_terminal_claimed = (
            False  # noqa: leading-underscore  # upstream spelling
        )
        req.omni_data = unconditional
        self.waiting_queue.append(req)

    def get_prefill_adder_class(self) -> type[PrefillAdder]:
        return Yue2PrefillAdder

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        queue = self.waiting_queue
        limit = self.admission_limit(queue, running_batch)
        token_budget = self.max_prefill_tokens
        if limit >= 2 and admission_width(queue, 0) == 2:
            leading_pair_tokens = prefill_token_count(queue[0]) + prefill_token_count(
                queue[1]
            )
        else:
            leading_pair_tokens = 0
        deferred = queue[limit:]
        del queue[limit:]
        if leading_pair_tokens >= token_budget:
            # note (ruiling): SGLang admits only one request above the prefill budget; a CFG pair needs both.
            self.max_prefill_tokens = token_budget + leading_pair_tokens
        else:
            pass
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.max_prefill_tokens = token_budget
            self.waiting_queue.extend(deferred)

    def admission_limit(self, queue: list[Req], running_batch: ScheduleBatch) -> int:
        """How many leading queue entries the adder may see, never splitting a CFG pair."""
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        budget = int(self.max_prefill_tokens)
        index = tokens = 0
        while index < len(queue):
            width = admission_width(queue, index)
            group_tokens = sum(
                prefill_token_count(queue[row]) for row in range(index, index + width)
            )
            if index + width > allocatable or (
                index and tokens + group_tokens > budget
            ):
                break
            else:
                tokens += group_tokens
                index += width
        return index

    def stream_output(
        self,
        reqs: Iterable[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        emitted = []
        for req in reqs:
            data = request_data(req)
            if (
                data is None
                or not (data.is_cfg_uncond or data.phase == "abc")
                or isinstance(req.finished_reason, FINISH_ABORT)
                or req.rid in self.aborted_request_ids
            ):
                emitted.append(req)
            elif not req.finished():
                continue
            elif data.is_cfg_uncond:
                self.close_completed_request(req)
            else:
                self.close_completed_request(req)
                self.start_audio_phase(req, data)
        super().stream_output(emitted, return_logprob, skip_req)

    def start_audio_phase(self, req: Req, data: Yue2ARRequestData) -> None:
        payload = data.stage_payload
        assert payload is not None and data.request is not None
        request_id = payload.request_id
        if request_id in self.aborted_request_ids:
            return
        else:
            pass
        tokens = list(req.output_ids)
        finished = bool(tokens) and tokens[-1] == ABC_END
        try:
            semantic = self.request_builder.semantic(
                payload,
                data.request,
                tokens[:-1] if finished else tokens,
                abc_truncated=not finished,
            )
        except ValueError as error:
            self.emit_request_error(request_id, error)
            self.abort(request_id)
            return
        self.enqueue_built_request(payload, False, semantic)

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        self.decoding_batch = batch
        try:
            return super().update_running_batch(batch)
        finally:
            self.decoding_batch = None

    def _add_request_to_queue(  # noqa: leading-underscore  # upstream spelling
        self, req: Req, is_retracted: bool = False
    ) -> None:
        data = request_data(req)
        batch = self.decoding_batch
        if (
            is_retracted
            and batch is not None
            and data is not None
            and (data.cfg_uncond is not None or data.is_cfg_uncond)
        ):
            self.requeue_cfg_pair(req, data, batch)
        else:
            self.requeue(req, is_retracted)

    def requeue(self, req: Req, is_retracted: bool) -> None:
        data = request_data(req)
        if req.is_retracted and data is not None:
            # note (ruiling): SGLang drops the token a retracted row sampled one step ahead.
            rewind_sampling(data, req.output_ids)
        else:
            pass
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    def retract_running_requests(self) -> int:
        retracted = super().retract_running_requests()
        # note (ruiling): this path queues rows one by one in batch order, which SGLang reorders when it retracts for memory.
        self.waiting_queue[:] = paired_queue(self.waiting_queue)
        return retracted

    def requeue_cfg_pair(
        self, req: Req, data: Yue2ARRequestData, batch: ScheduleBatch
    ) -> None:
        """Retract both rows of a CFG pair when SGLang retracts one, then queue them in order."""
        if any(queued is req for queued in self.waiting_queue):
            return
        elif data.is_cfg_uncond:
            guided = next(
                (
                    running
                    for running in batch.reqs
                    if (running_data := request_data(running)) is not None
                    and running_data.cfg_uncond is data
                ),
                None,
            )
            # note (ruiling): a request that left the batch was retracted in this step and queues both rows, or was aborted.
            pair = [] if guided is None else [guided, req]
        else:
            assert data.cfg_uncond is not None
            twin = data.cfg_uncond.req
            if twin.to_finish is not None and req.to_finish is None:
                # note (ruiling): SGLang aborts the last row when even it does not fit the KV cache.
                self.emit_request_error(
                    req.rid, RuntimeError("YuE2 CFG rows do not fit the KV cache")
                )
                pair = []
            else:
                # note (ruiling): an aborted pair is requeued whole and finishes on its next step.
                pair = [req, twin]
        for row in pair:
            index = next(
                (
                    position
                    for position, running in enumerate(batch.reqs)
                    if running is row
                ),
                None,
            )
            if index is None:
                pass
            else:
                batch.release_req(index, len(batch.reqs) - 1)
                batch.filter_batch(
                    keep_indices=[
                        position
                        for position in range(len(batch.reqs))
                        if position != index
                    ]
                )
        for row in pair:
            self.requeue(row, True)

    def emit_request_error(self, request_id: str, error: Exception) -> None:
        super().emit_request_error(public_request_id(request_id), error)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        public_id = public_request_id(request_id)
        if (
            self.scheduler_thread_id is not None
            and self.scheduler_thread_id != threading.get_ident()
        ):
            # note (ruiling): between two steps all rows of a song stop together; aborting them
            # one by one from another thread can leave a CFG row without its request.
            self.inbox.put(IncomingMessage(request_id=public_id, type="abort"))
            return
        else:
            pass
        for internal_id in (
            public_id,
            score_request_id(public_id),
            guidance_request_id(public_id),
        ):
            super().abort(internal_id, defer_running_cleanup=defer_running_cleanup)
