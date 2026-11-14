"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import replace

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import async_h2d
from typing_extensions import Unpack

from sglang_omni.models.qwen3_omni.config import TALKER_START_MIN_CHUNKS
from sglang_omni.models.qwen3_omni.payload_types import NO_SPEECH
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, OmniSchedulerArguments
from sglang_omni.scheduling.sglang_backend.ar_session import SessionUnit
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)

# note (ratish): a reply with no spoken token has nothing to speak; a reply of one
# token terminates.
MIN_SPOKEN_TOKENS = 1


def configure_talker_server_args(
    server_args: ServerArgs, *, feedback_enabled: bool = True
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were requested so the caller can capture them
    after the model worker is constructed.
    """
    from sglang.srt.arg_groups.model_override_base import resolved_view

    cfg = resolved_view(server_args)
    want_cuda_graph = not bool(cfg.disable_cuda_graph)
    overrides = {"disable_radix_cache": True, "chunked_prefill_size": 0}
    if feedback_enabled:
        overrides["disable_overlap_schedule"] = True
    else:
        pass
    override_server_args(server_args, "qwen3_omni.talker", **overrides)
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler[SGLangARRequestData]):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    excluded_row_steps: int = 0
    excluded_batch: ScheduleBatch | None = None

    def __init__(
        self,
        *args: object,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = TALKER_START_MIN_CHUNKS,
        im_end_token_id: int | None = None,
        **kwargs: Unpack[OmniSchedulerArguments[SGLangARRequestData]],
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < TALKER_START_MIN_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {TALKER_START_MIN_CHUNKS}, got {partial_start_min_chunks}"
            )
        else:
            pass
        self.enable_partial_start = bool(enable_partial_start)
        self.partial_start_min_chunks = int(partial_start_min_chunks)
        self.im_end_token_id = im_end_token_id
        self.excluded_row_steps = 0
        self.excluded_batch = None

    def count_usable_prefetched_chunks(self, prefetched: list[StreamItem]) -> int:
        im_end = self.im_end_token_id
        if im_end is None or not prefetched:
            return len(prefetched)
        else:
            pass
        metadata = getattr(prefetched[-1], "metadata", None) or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == int(im_end):
            return len(prefetched) - 1
        else:
            pass
        return len(prefetched)

    def is_request_build_ready(
        self, payload: StagePayload, *, pending_stream_done: bool
    ) -> bool:
        if pending_stream_done:
            return True
        else:
            pass
        if not self.enable_partial_start:
            return False
        else:
            pass
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return (
            self.count_usable_prefetched_chunks(prefetched)
            >= self.partial_start_min_chunks
        )

    def complete_without_generation(
        self, payload: StagePayload, session_unit: SessionUnit | None
    ) -> StagePayload | None:
        """A finished reply too short to speak passes on without audio."""
        if not payload.prefetched_stream_done or (
            self.count_usable_prefetched_chunks(payload.prefetched_chunks)
            >= MIN_SPOKEN_TOKENS
        ):
            return None
        elif session_unit is None:
            return replace(payload, data={**payload.data, NO_SPEECH: True})
        else:
            return self.session_bridge.adapter.complete_without_generation(
                session_unit.session_identity, session_unit.chunk, payload
            )

    def initialize_request_stream_state(
        self, req_data: SGLangARRequestData, payload: StagePayload
    ) -> None:
        del req_data, payload
        return None

    def should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> bool:
        del request_id, chunk
        return self.enable_partial_start

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        """Decode the rows whose next input has arrived; the others are excluded from the step and keep their KV."""
        ready_rows = self.model_runner.ready_decode_rows(batch)
        if len(ready_rows) == batch.batch_size():
            return super().update_running_batch(batch)
        else:
            pass
        ready = set(ready_rows)
        excluded_rows: list[int] = []
        for row_index, request in enumerate(batch.reqs):
            if row_index in ready:
                pass
            elif request.to_finish is None:
                excluded_rows.append(row_index)
            else:
                # note (ratish): an abort's cleanup runs at the row's next step, which a row without input never takes.
                self.abort(request.rid, defer_running_cleanup=False)
        if len(ready_rows) + len(excluded_rows) < batch.batch_size():
            batch.batch_is_full = False
        else:
            pass
        if not ready_rows:
            batch.filter_batch(keep_indices=excluded_rows)
            return None
        elif excluded_rows and not batch.check_decode_mem(selected_indices=ready_rows):
            # note (ratish): as for rows finishing in flight, the ready rows wait rather than be retracted around rows holding KV outside the step; once every row has its text, retraction runs over the whole batch.
            batch.filter_batch(keep_indices=sorted(ready_rows + excluded_rows))
            batch.batch_is_full = False
            return None
        else:
            pass
        if excluded_rows:
            self.excluded_batch = self.copy_rows(batch, excluded_rows)
        else:
            pass
        batch.filter_batch(keep_indices=ready_rows)
        decode_batch = super().update_running_batch(batch)
        if decode_batch is None:
            self.merge_excluded_rows(batch)
        else:
            pass
        return decode_batch

    def copy_rows(self, batch: ScheduleBatch, row_indices: list[int]) -> ScheduleBatch:
        """A batch of these rows that keeps their KV slot, length and next input id."""
        rows_batch = ScheduleBatch.init_new(
            reqs=[batch.reqs[row_index] for row_index in row_indices],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        host_indices = torch.tensor(row_indices, dtype=torch.int64)
        device_indices = async_h2d(row_indices, dtype=torch.int64, device=batch.device)
        rows_batch.forward_mode = ForwardMode.DECODE
        rows_batch.req_pool_indices = batch.req_pool_indices[device_indices]
        rows_batch.req_pool_indices_cpu = batch.req_pool_indices_cpu[host_indices]
        rows_batch.seq_lens = batch.seq_lens[device_indices]
        rows_batch.seq_lens_cpu = batch.seq_lens_cpu[host_indices]
        rows_batch.orig_seq_lens = batch.orig_seq_lens[device_indices]
        rows_batch.input_ids = (
            None if batch.input_ids is None else batch.input_ids[device_indices]
        )
        rows_batch.multimodal_inputs = (
            None
            if batch.multimodal_inputs is None
            else [batch.multimodal_inputs[row_index] for row_index in row_indices]
        )
        # note (ratish): the talker samples from its own buffers, so these rows get fresh sampling state, as a batch built for rows already holding KV does.
        rows_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            rows_batch, self.model_config.vocab_size
        )
        return rows_batch

    def pending_batches(self) -> Iterator[ScheduleBatch]:
        """Rows excluded from the current step stay visible to request lookup and cancellation."""
        yield from super().pending_batches()
        if self.excluded_batch is not None:
            yield self.excluded_batch
        else:
            pass

    def merge_excluded_rows(self, batch: ScheduleBatch) -> int:
        """Merge the rows excluded from the step back into the batch; returns how many there were."""
        excluded_batch = self.excluded_batch
        if excluded_batch is None:
            return 0
        else:
            pass
        self.excluded_batch = None
        excluded_row_count = excluded_batch.batch_size()
        batch.merge_batch(excluded_batch)
        return excluded_row_count

    def process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        super().process_batch_result(batch, result)
        self.excluded_row_steps += self.merge_excluded_rows(batch)
        metrics_reporter = self.metrics_reporter
        if (
            self.excluded_row_steps > 0
            and metrics_reporter.forward_ct_decode
            % metrics_reporter.decode_log_interval
            == 0
        ):
            logger.info(
                f"talker rows excluded from a decode step for thinker text: {self.excluded_row_steps} row steps in the last {metrics_reporter.decode_log_interval} decode steps"
            )
            self.excluded_row_steps = 0
        else:
            pass

    def handle_batch_failure(self, batch: ScheduleBatch, error: Exception) -> None:
        super().handle_batch_failure(batch, error)
        self.merge_excluded_rows(batch)

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and (not self.running_batch.is_empty()):
            return
        else:
            pass
        if self.waiting_queue:
            return
        else:
            pass
        super().self_check_during_idle()

    @staticmethod
    def append_stream_chunk_default(
        req_data: SGLangARRequestData, chunk: StreamItem
    ) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        else:
            pass
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def mark_stream_done(self, req_data: SGLangARRequestData) -> None:
        if self.stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        else:
            pass
        self.stream_done_handler(req_data)
