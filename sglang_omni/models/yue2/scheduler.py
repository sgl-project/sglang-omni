# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler that chains the YuE2 score and audio phases and pairs CFG rows."""

from collections.abc import Iterable

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .request import ABC_END
from .sglang_request_builder import (
    GUIDANCE_SUFFIX,
    SCORE_SUFFIX,
    Yue2ARRequestData,
    Yue2RequestBuilder,
    guidance_request_id,
    score_request_id,
)


def request_data(req: Req) -> Yue2ARRequestData | None:
    return getattr(req, "omni_data", None)


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


class Yue2Scheduler(OmniScheduler[Yue2ARRequestData]):
    request_builder: Yue2RequestBuilder

    def enqueue_built_request(
        self,
        payload: StagePayload,
        pending_stream_done: bool,
        req_data: Yue2ARRequestData,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
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

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        queue = self.waiting_queue
        limit = self.admission_limit(queue, running_batch)
        if limit >= len(queue):
            return super().get_new_batch_prefill(running_batch)
        else:
            pass
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.waiting_queue.extend(deferred)

    def admission_limit(self, queue: list[Req], running_batch: ScheduleBatch) -> int:
        """How many leading queue entries the adder may see, never splitting a CFG pair."""
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        budget = int(self.max_prefill_tokens)
        index = tokens = 0
        while index < len(queue):
            width = admission_width(queue, index)
            group_tokens = sum(
                len(queue[row].origin_input_ids) for row in range(index, index + width)
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

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if request_id.endswith((SCORE_SUFFIX, GUIDANCE_SUFFIX)):
            return
        else:
            pass
        for internal_id in (
            score_request_id(request_id),
            guidance_request_id(request_id),
        ):
            super().abort(internal_id, defer_running_cleanup=defer_running_cleanup)
