from __future__ import annotations

import logging

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)


class NemotronTalkerScheduler(OmniScheduler):
    @staticmethod
    def _append_stream_chunk_default(req_data, chunk) -> None:
        # The thinker ships each text token as a one-element tensor, because
        # that is what crosses the relay between stage processes.
        req_data.pending_text_queue.append(int(chunk.data.reshape(-1)[0]))

    @staticmethod
    def _mark_stream_done(req_data) -> None:
        req_data.thinker_chunks_done = True

    def get_next_batch_to_run(self):
        batch = super().get_next_batch_to_run()
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and not self._model_runner.is_decode_batch_ready(batch)
        ):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def _rollback_decode_prep_after_skip(self, batch) -> None:
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        super().self_check_during_idle()
