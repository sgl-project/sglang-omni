"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Protocol, TypeVar

from sglang_omni.models.qwen3_omni.config import (
    ENABLE_TALKER_START_TOPOLOGY,
    MIN_PARTIAL_START_CHUNKS,
    TALKER_START_MIN_CHUNKS,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass

logger = logging.getLogger(__name__)

ChunkT = TypeVar("ChunkT")

_CHUNK_WAIT_LOG_INTERVAL_S = 10.0


class DecodeMode(Protocol):
    def is_decode(self) -> object: ...


class DecodeBatch(Protocol):
    @property
    def forward_mode(self) -> DecodeMode: ...


def configure_talker_server_args(
    server_args: "ServerArgs", *, feedback_enabled: bool = True
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


class QwenTalkerScheduler(OmniScheduler["SGLangARRequestData"]):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    talker_start_topology: bool = ENABLE_TALKER_START_TOPOLOGY
    chunk_wait_steps: int = 0
    chunk_wait_last_log_s: float = 0.0

    def __init__(
        self,
        *args: object,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        enable_talker_start_topology: bool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, got {partial_start_min_chunks}"
            )
        else:
            pass
        self.enable_partial_start = bool(enable_partial_start)
        self.partial_start_min_chunks = int(partial_start_min_chunks)
        self.im_end_token_id = im_end_token_id
        self.talker_start_topology = self.enable_partial_start and (
            ENABLE_TALKER_START_TOPOLOGY
            if enable_talker_start_topology is None
            else bool(enable_talker_start_topology)
        )
        self.chunk_wait_steps = 0
        self.chunk_wait_last_log_s = 0.0
        if self.talker_start_topology:
            logger.info(
                "talker-start topology on: building at %d thinker chunk(s); later chunks gate decode per step (partial_start_min_chunks=%d applies to the legacy path only)",
                TALKER_START_MIN_CHUNKS,
                self.partial_start_min_chunks,
            )
        else:
            pass

    def count_usable_prefetched_chunks(self, prefetched: list[ChunkT]) -> int:
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
        self, payload: object, *, pending_stream_done: bool
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
        usable = self.count_usable_prefetched_chunks(prefetched)
        if self.talker_start_topology:
            return usable >= TALKER_START_MIN_CHUNKS
        else:
            pass
        return usable >= self.partial_start_min_chunks

    def initialize_request_stream_state(
        self, req_data: object, payload: object
    ) -> None:
        del req_data, payload
        return None

    def should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: object
    ) -> bool:
        del request_id, chunk
        return self.enable_partial_start

    def is_batch_ready_to_run(self, batch: DecodeBatch | None) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and (self.model_runner is not None)
            and hasattr(self.model_runner, "is_decode_batch_ready")
            and (not self.model_runner.is_decode_batch_ready(batch))
        ):
            self.note_chunk_wait(batch)
            return False
        else:
            pass
        return True

    def note_chunk_wait(self, batch: object) -> None:
        self.chunk_wait_steps += 1
        logger.debug("Deferring decode batch until talker feedback/text input is ready")
        now = time.monotonic()
        if now - self.chunk_wait_last_log_s < _CHUNK_WAIT_LOG_INTERVAL_S:
            return
        else:
            pass
        self.chunk_wait_last_log_s = now
        logger.info(
            "talker chunk gate: %d decode steps deferred so far (current batch rows=%d)",
            self.chunk_wait_steps,
            len(getattr(batch, "reqs", ()) or ()),
        )

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        batch = super().get_next_batch_to_run()
        if batch is not None and (not self.is_batch_ready_to_run(batch)):
            self.rollback_decode_prep_after_skip(batch)
            return None
        else:
            pass
        return batch

    def rollback_decode_prep_after_skip(self, batch: "ScheduleBatch") -> None:
        if not batch.forward_mode.is_decode():
            return
        else:
            pass
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        else:
            pass
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv.kv_committed_len -= 1
            req.kv.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0

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
        req_data: "SGLangARRequestData", chunk: object
    ) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        else:
            pass
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def mark_stream_done(self, req_data: "SGLangARRequestData") -> None:
        if self.stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        else:
            pass
        self.stream_done_handler(req_data)
