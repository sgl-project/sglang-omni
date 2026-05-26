# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)


def configure_talker_server_args(
    server_args: Any,
    *,
    feedback_enabled: bool = True,
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were originally requested so the caller can
    re-enable graph capture after the model worker is constructed.
    """

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if feedback_enabled:
        server_args.disable_overlap_schedule = True
        if want_cuda_graph:
            server_args.disable_cuda_graph = True
    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = 0
    return want_cuda_graph


# Floor for the partial-start gate; pinned by
# ``test_partial_prompt_prefill_layout_invariants``.
MIN_PARTIAL_START_CHUNKS = 3


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness.

    The decode-stall handling (``get_next_batch_to_run`` override +
    ``_rollback_decode_prep_after_skip``) is scoped here, not on the shared
    ``OmniScheduler``, because the rollback is **not a full inverse** of
    upstream ``ScheduleBatch.prepare_for_decode``. Upstream writes/clears
    (depending on server_args): ``out_cache_loc``, ``input_ids`` /
    ``output_ids``, ``seq_lens`` / ``seq_lens_cpu`` / ``orig_seq_lens`` /
    ``seq_lens_sum``, per-req ``decode_batch_idx`` / ``kv_committed_len`` /
    ``kv_allocated_len``, ``forward_mode``, ``input_embeds``,
    ``attn_cp_metadata``, ``sampling_info.penalizer_orchestrator``
    cumulate-output-tokens state, ``hisparse_coordinator``, Mamba buffers,
    and ``req_to_token_pool`` writes.

    This rollback only undoes the first set explicitly. The remainder is
    safe **only** because the talker config disables them or makes them
    idempotent: ``configure_talker_server_args`` sets
    ``disable_overlap_schedule=True`` (so ``enable_overlap`` is False and
    no overlap tensor swap fires), Qwen3-Omni has no Mamba state (so
    ``mamba_track_indices`` / ``mamba_track_mask`` are unused), the
    repetition-penalty scatter inside
    ``sampling_info.penalizer_orchestrator`` is idempotent on the same
    output_id within one step, ``is_spec_v2`` is False (no spec-decode
    fast path), and hisparse is unused. Other schedulers MUST NOT inherit
    this rollback without verifying their server_args produce the same
    effective subset; see
    ``test_prepare_for_decode_side_effect_contract_with_upstream``.
    """

    # Class-level defaults so object.__new__ test helpers see a disabled state.
    _enable_partial_start: bool = False
    _partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS
    _im_end_token_id: int | None = None

    def __init__(
        self,
        *args: Any,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
                f"got {partial_start_min_chunks}"
            )
        self._enable_partial_start = bool(enable_partial_start)
        self._partial_start_min_chunks = int(partial_start_min_chunks)
        self._im_end_token_id = im_end_token_id

    def _count_usable_prefetched_chunks(self, prefetched: list[Any]) -> int:
        """Count chunks that survive prefill's <|im_end|> stripping."""
        im_end = self._im_end_token_id
        if im_end is None:
            return len(prefetched)
        usable = 0
        for chunk in prefetched:
            metadata = getattr(chunk, "metadata", None) or {}
            token_id = metadata.get("token_id")
            if token_id is not None and int(token_id) == int(im_end):
                continue
            usable += 1
        return usable

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        if pending_stream_done:
            return True
        if not self._enable_partial_start:
            return False
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return (
            self._count_usable_prefetched_chunks(prefetched)
            >= self._partial_start_min_chunks
        )

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        # No-op: request_builder seeds pending_text_queue itself.
        del req_data, payload
        return None

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and self._model_runner is not None
            and hasattr(self._model_runner, "is_decode_batch_ready")
            and not self._model_runner.is_decode_batch_ready(batch)
        ):
            logger.debug(
                "Deferring decode batch until talker feedback/text input is ready"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Talker-scoped decode-stall handling. Shared OmniScheduler stays
    # pure; the side-effect rollback is safe only under the talker
    # server_args (disable_overlap_schedule=True, no Mamba).
    # ------------------------------------------------------------------

    def get_next_batch_to_run(self):
        batch = _Upstream.get_next_batch_to_run(self)
        if batch is not None and not self._is_batch_ready_to_run(batch):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def _rollback_decode_prep_after_skip(self, batch: Any) -> None:
        if not batch.forward_mode.is_decode():
            return
        assert isinstance(batch.seq_lens_sum, int), (
            f"seq_lens_sum is {type(batch.seq_lens_sum).__name__}, expected int "
            "— sglang upstream prepare_for_decode changed; update rollback."
        )
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        if batch.output_ids is None:
            batch.output_ids = batch.input_ids
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.seq_lens_sum -= len(batch.reqs)

    def self_check_during_idle(self) -> None:
        # Partial-start stalled reqs hold live KV slots; not a leak.
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        _Upstream.self_check_during_idle(self)

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        self._stream_done_handler(req_data)
