"""MOSS-TTS Local (v1.5) model runner for OmniScheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts_local.radix_hash import build_rows_and_radix_token_ids
from sglang_omni.models.moss_tts_local.request_builders import (
    MOSS_STREAM_TRANSPORT_BATCH_FRAMES,
)
from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeJournal
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput

if TYPE_CHECKING:
    from queue import Queue

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.moss_tts_local.request_builders import (
        MossTTSLocalSGLangRequestData,
    )
    from sglang_omni.models.moss_tts_local.sglang_model import MossTTSLocalSGLangModel
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.scheduling.types import (
        ARRequestData,
        SchedulerOutput,
        SchedulerRequest,
    )
else:
    pass


class MossTTSLocalModelRunner(ModelRunner):
    """Drives the per-frame local-transformer decode and feedback embeddings.

    Per step: the backbone (radix-cached, CUDA-graphed) produces one hidden
    state per request; :meth:`collect_frame` then runs the batched local
    micro-decode — a binary continue/stop decision and 12 sequentially
    sampled RVQ codes — and stages the next frame's summed embedding through
    ``model._decode_input_embedding`` so the next decode step stays
    CUDA-graph-replayable (decode input_ids are row indices).
    """

    model: MossTTSLocalSGLangModel

    outbox: Queue[OutgoingMessage] | None = None
    vocoder_target = "vocoder"

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.outbox: Queue[OutgoingMessage] | None = None
        self.vocoder_target = "vocoder"

    def set_stream_outbox(self, outbox: Queue[OutgoingMessage]) -> None:
        self.outbox = outbox

    def flush_stream_rows(
        self, request_id: str, data: MossTTSLocalSGLangRequestData, *, force: bool
    ) -> None:
        metadata = data.stream_metadata
        if metadata is None or self.outbox is None:
            return
        else:
            pass
        pending = data.stream_pending_rows
        if not pending:
            return
        else:
            pass
        first_sent = data.stream_first_batch_sent
        threshold = 1 if not first_sent else MOSS_STREAM_TRANSPORT_BATCH_FRAMES
        if not force and len(pending) < threshold:
            return
        else:
            pass
        rows = pending[0] if len(pending) == 1 else torch.stack(pending)
        pending.clear()
        data.stream_first_batch_sent = True
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=self.vocoder_target,
                data=rows,
                metadata=metadata,
            )
        )

    def on_request_finished(self, request_id: str, req_data: ARRequestData) -> None:
        self.flush_stream_rows(request_id, req_data, force=True)

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self.build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self.write_decode_input_embedding(forward_batch, requests)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        try:
            is_prefill_only = schedule_batch.is_prefill_only
        except AttributeError:
            is_prefill_only = False
        if bool(is_prefill_only):
            return
        else:
            pass
        self.collect_frame(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_frame(result, forward_batch, schedule_batch, requests)

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """Route to sync when the batch cannot take the graphed frame-decode
        path: any request with ``audio_repetition_penalty != 1`` (its eager
        rep-history gather lags one frame under lookahead and would diverge from
        sync) or ``bs > frame_graph_max_bs``.
        """
        try:
            reqs = batch.reqs
        except AttributeError:
            reqs = None
        reqs = reqs or []
        try:
            frame_graph_max_bs = self.model.frame_graph_max_bs
        except AttributeError:
            frame_graph_max_bs = 0
        if len(reqs) > int(frame_graph_max_bs):
            return False
        else:
            pass
        for req in reqs:
            try:
                data = req.omni_data  # noqa: leading-underscore
            except AttributeError:
                data = None
            if data is None:
                continue
            else:
                pass
            try:
                audio_repetition_penalty = data.audio_repetition_penalty
            except AttributeError:
                audio_repetition_penalty = 1.0
            if float(audio_repetition_penalty) != 1.0:
                return False
            else:
                pass
        return True

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data: MossTTSLocalSGLangRequestData = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS Local prefill requires prompt_rows")
            else:
                pass
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            pool = self.model.state_pool
            if data.output_rows:
                generated = torch.stack(data.output_rows, dim=0)
                rows = torch.cat([rows.to(generated.device), generated], dim=0)
            else:
                pass
            generation_steps = int(data.generation_steps)
            data.sampling_steps = generation_steps
            pool.reset_for_refill(sched_req.request_id, generation_steps)
            if data.output_rows:
                pool.rebuild_audio_history(sched_req.request_id, data.output_rows)
            else:
                pass
            current_rows = rows[prefix_len : prefix_len + req_len]
            if int(current_rows.shape[0]) != req_len:
                raise RuntimeError(
                    f"MOSS-TTS Local prefill row mismatch for {req.rid}: have {int(current_rows.shape[0])} rows, need {req_len} (prefix={prefix_len}, prompt={int(data.prompt_rows.shape[0])}, generated={len(data.output_rows)})"
                )
            else:
                pass
            embeds = self.model.prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        else:
            pass
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device, dtype=self.model.dtype
        )

    def write_decode_input_embedding(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        pool = self.model.state_pool
        weight = self.model.decode_input_embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-TTS Local decode input_ids must contain one row id per request"
            )
        else:
            pass
        if batch_size > pool.padding_row:
            raise RuntimeError(
                f"MOSS-TTS Local decode batch exceeds the staged decode-embedding rows ({batch_size} > {pool.padding_row})"
            )
        else:
            pass
        row_tensor, pool_rows, has_audio_repetition_penalty = pool.prepare_active_rows(
            requests
        )
        with torch.no_grad():
            weight[:batch_size].copy_(pool.feedback_embeds[row_tensor])
        forward_batch.moss_pool_row_t = row_tensor
        forward_batch.moss_pool_rows = pool_rows
        forward_batch.moss_has_audio_repetition_penalty = has_audio_repetition_penalty
        row_ids = torch.arange(
            batch_size, dtype=torch.long, device=forward_batch.input_ids.device
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def collect_frame(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        if not requests:
            return
        else:
            pass
        rows, end_id, next_token_ids = self.run_frame_decode(
            result, forward_batch, requests
        )
        result.next_token_ids = next_token_ids
        self.stage_token_ids(result, next_token_ids)

    def run_frame_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """GPU half shared by sync ``collect_frame`` and async
        ``post_decode_launch``. Returns ``(rows, end_id, next_token_ids)`` and
        does NOT publish the ids; the caller does, because the async path keeps
        a private device snapshot of the published ids for resolve to restore.
        """
        try:
            hidden_states = result.logits_output.hidden_states
        except AttributeError as exc:
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            ) from exc
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            )
        else:
            pass
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        else:
            pass
        cfg = self.model.config
        device = hidden_states.device
        pool = self.model.state_pool
        batch_size = len(requests)
        num_channels = int(cfg.n_vq) + 1
        try:
            row_t = forward_batch.moss_pool_row_t
            pool_rows = forward_batch.moss_pool_rows
        except AttributeError:
            row_t, pool_rows, has_audio_repetition_penalty = pool.prepare_active_rows(
                requests
            )
        else:
            try:
                has_audio_repetition_penalty = (
                    forward_batch.moss_has_audio_repetition_penalty
                )
            except AttributeError:
                has_audio_repetition_penalty = pool.rows_have_audio_repetition_penalty(
                    pool_rows
                )
        params = {
            "text_temp": pool.text_temp[row_t],
            "text_top_p": pool.text_top_p[row_t],
            "text_top_k": pool.text_top_k[row_t],
            "audio_temp": pool.audio_temp[row_t],
            "audio_top_p": pool.audio_top_p[row_t],
            "audio_top_k": pool.audio_top_k[row_t],
            "seeds": pool.seeds[row_t],
        }
        text_temp = params["text_temp"]
        text_top_p = params["text_top_p"]
        text_top_k = params["text_top_k"]
        audio_temp = params["audio_temp"]
        audio_top_p = params["audio_top_p"]
        audio_top_k = params["audio_top_k"]
        sampling_seeds = params["seeds"]
        emit_indices = [
            i
            for i, sched_req in enumerate(requests)
            if not self.is_chunked_request(sched_req)
        ]
        if self.async_enabled:
            gen_steps = torch.maximum(
                pool.sampling_steps[row_t].to(device=device),
                pool.generation_steps[row_t].to(device=device),
            )
        else:
            gen_steps = pool.generation_steps[row_t].to(device=device)
        rep_penalties = pool.audio_repetition_penalty[row_t].to(
            device=device, dtype=torch.float32
        )

        def sample_text(logits: torch.Tensor) -> torch.Tensor:
            return MossTTSModelRunner.sample_tokens(
                logits,
                temperature=text_temp,
                top_p=text_top_p,
                top_k=text_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels,
            )

        def sample_audio(logits: torch.Tensor, channel: int) -> torch.Tensor:
            if has_audio_repetition_penalty:
                presence = pool.audio_token_presence[row_t, channel].to(
                    device=logits.device
                )
                if int(presence.shape[-1]) != int(logits.shape[-1]):
                    presence = presence[:, : logits.shape[-1]]
                else:
                    pass
                self.apply_audio_repetition_penalty_mask(
                    logits,
                    presence,
                    rep_penalties.to(device=logits.device, dtype=logits.dtype),
                )
            else:
                pass
            return MossTTSModelRunner.sample_tokens(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels + channel + 1,
            )

        try:
            frame_graph_max_bs = int(self.model.frame_graph_max_bs)
        except AttributeError:
            frame_graph_max_bs = 0
        use_graph = (
            not has_audio_repetition_penalty and batch_size <= frame_graph_max_bs
        )
        if use_graph:
            stop_choice, codes, feedback = self.model.decode_frame_graphed(
                hidden_states,
                text_temperature=text_temp,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temp,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                seeds=sampling_seeds,
                base_positions=gen_steps * num_channels,
            )
            codes = codes.clone()
            embeds = feedback.clone()
        else:
            stop_choice, codes = self.model.decode_frame(
                hidden_states, sample_text=sample_text, sample_audio=sample_audio
            )
            embeds = None
        slot_id = int(cfg.audio_assistant_slot_token_id)
        end_id = int(cfg.audio_end_token_id)
        rows, next_token_ids = build_rows_and_radix_token_ids(
            stop_choice, codes, slot_id, end_id
        )
        next_text = rows[:, 0]
        if embeds is None:
            embeds = self.model.prepare_multi_modal_inputs(
                rows.to(device=self.model.device)
            )
        else:
            pass
        if emit_indices:
            emit_pool_rows = [pool_rows[i] for i in emit_indices]
            all_emit = len(emit_indices) == batch_size
            if all_emit:
                emit_row_t = row_t
                emit_rows = rows
                emit_steps = gen_steps
                emit_embeds = embeds
                emit_next_text = next_text
            else:
                emit_index_t = torch.tensor(
                    emit_indices, dtype=torch.long, device=rows.device
                )
                emit_row_t = row_t[emit_index_t.to(device=row_t.device)]
                emit_rows = rows.index_select(0, emit_index_t)
                emit_steps = gen_steps.index_select(
                    0, emit_index_t.to(device=gen_steps.device)
                )
                emit_embeds = embeds.index_select(
                    0, emit_index_t.to(device=embeds.device)
                )
                if has_audio_repetition_penalty:
                    emit_next_text = emit_rows[:, 0]
                else:
                    pass
            pool.sampling_steps[emit_row_t] = (emit_steps + 1).to(
                device=pool.sampling_steps.device, dtype=torch.int64
            )
            if has_audio_repetition_penalty:
                keep_history = emit_next_text != end_id
                emit_penalty_active = (
                    pool.audio_repetition_penalty[emit_row_t]
                    .to(device=keep_history.device)
                    .ne(1.0)
                )
                keep_history = keep_history & emit_penalty_active
                pool.update_audio_history(
                    emit_row_t[keep_history.to(device=emit_row_t.device)],
                    emit_rows[keep_history.to(device=emit_rows.device)],
                )
            else:
                pass
            pool.feedback_embeds[emit_row_t] = emit_embeds.detach().to(
                device=pool.feedback_embeds.device, dtype=pool.feedback_embeds.dtype
            )
            result.moss_journal = MossTTSLocalDecodeJournal(
                rids=[requests[i].request_id for i in emit_indices],
                pool_rows=emit_pool_rows,
                rows=emit_rows,
            )
        else:
            pass
        return (rows, end_id, next_token_ids)

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        """Async-decode GPU half of ``post_decode``: run the frame micro-decode
        (``run_frame_decode``) and publish the device-computed radix ids, no
        host sync. Returns a private device snapshot of those ids for resolve:
        the base aliases ``next_token_ids`` onto ``output_ids``, which the next
        step overwrites in place before this step's lagged resolve, clobbering
        the stop id and silently dropping a bs=1 eos finish (4096-frame runaway).
        The clone preserves it; resolve swaps it back.
        """
        if not requests:
            return None
        else:
            pass
        rows, end_id, next_token_ids = self.run_frame_decode(
            result, forward_batch, requests
        )
        result.next_token_ids = next_token_ids
        return next_token_ids.clone()

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor | None,
        result: GenerationBatchResult | None,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Async-decode host half: restore the launch-time ``next_token_ids``
        snapshot (a pointer swap) so the shared ``finalize`` tail reads the real
        stop id, which the next step's in-place write clobbered from the aliased
        tensor before this lagged resolve.
        """
        del forward_batch, schedule_batch, requests
        if launch_buf is not None and result is not None:
            result.next_token_ids = launch_buf
        else:
            pass

    @staticmethod
    def advance_sampling_position(data: MossTTSLocalSGLangRequestData) -> int:
        """RNG position for this collect, advancing the launch-side counter in
        floor mode: ``max(sampling_steps or 0, generation_steps)``. On the sync
        path the two stay equal (generation_steps increments after every collect)
        so the floor is a no-op and the position is bit-identical to before;
        under lookahead generation_steps lags, so the floor lifts launch(N+1) off
        the stale N.
        """
        try:
            sampling_steps = data.sampling_steps
        except AttributeError:
            sampling_steps = None
        s = max(int(sampling_steps or 0), int(data.generation_steps))
        data.sampling_steps = s + 1
        return s

    @staticmethod
    def apply_audio_repetition_penalty_mask(
        logits: torch.Tensor, token_presence: torch.Tensor, penalties: torch.Tensor
    ) -> None:
        """In-place penalty on fp32 logits, matching upstream order (before
        temperature scaling)."""
        penalties = penalties.to(device=logits.device, dtype=logits.dtype)
        active = token_presence.to(device=logits.device, dtype=torch.bool) & (
            penalties != 1.0
        ).unsqueeze(-1)
        penalties = penalties.unsqueeze(-1)
        penalized = torch.where(logits < 0, logits * penalties, logits / penalties)
        logits.copy_(torch.where(active, penalized, logits))

    @staticmethod
    def is_chunked_request(sched_req: SchedulerRequest) -> bool:
        try:
            req = sched_req.data.req
        except AttributeError:
            return False
        if req is None:
            return False
        else:
            pass
        try:
            middle_chunks = req.inflight_middle_chunks
        except AttributeError:
            return False
        return int(middle_chunks) > 0

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        """Non-final chunked-prefill rows must not advance ``generation_steps``.

        Their micro-decode still runs (as today), but the spurious step would
        shift the final chunk's sampling position off the no-chunk path; the
        sampling is positional (``position = generation_steps * num_channels +
        channel``), so suppressing the advance keeps the chunked path
        bit-identical to the single-shot prefill path.
        """
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if self.is_chunked_request(sched_req)
        }

    def on_generation_step_advanced(
        self, sched_req: SchedulerRequest, generation_steps: int
    ) -> None:
        try:
            pool = self.model.state_pool
        except AttributeError:
            return
        if pool is not None:
            pool.commit_generation_step(sched_req.request_id, generation_steps)
        else:
            pass

    def on_generation_steps_advanced(
        self,
        advanced_steps: list[tuple[SchedulerRequest, int]],
        forward_batch: ForwardBatch | None,
    ) -> None:
        try:
            pool = self.model.state_pool
        except AttributeError:
            return
        if pool is None or not advanced_steps:
            return
        else:
            pass
        steps = [int(generation_steps) for _, generation_steps in advanced_steps]
        try:
            row_t = forward_batch.moss_pool_row_t
        except AttributeError:
            row_t = None
        if row_t is not None and int(row_t.numel()) == len(steps):
            step_t = torch.tensor(steps, dtype=torch.long, device=row_t.device)
            pool.commit_generation_steps(row_t, step_t)
            return
        else:
            pass
        rows = []
        row_steps = []
        for sched_req, generation_steps in advanced_steps:
            row = pool.row_for(sched_req.request_id)
            if row is None:
                continue
            else:
                pass
            rows.append(row)
            row_steps.append(int(generation_steps))
        if not rows:
            return
        else:
            pass
        device = pool.generation_steps.device
        row_t = torch.tensor(rows, dtype=torch.long, device=device)
        step_t = torch.tensor(row_steps, dtype=torch.long, device=device)
        pool.commit_generation_steps(row_t, step_t)

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        try:
            journal = result.moss_journal
        except AttributeError:
            return
        if journal is None:
            return
        else:
            pass
        end_id = int(self.model.config.audio_end_token_id)
        expected_reqs = [
            sched_req
            for sched_req in scheduler_output.requests
            if not self.is_chunked_request(sched_req)
        ]
        expected_rids = [sched_req.request_id for sched_req in expected_reqs]
        rows_len = int(journal.rows.shape[0])
        if len(journal.rids) != rows_len or len(journal.pool_rows) != rows_len:
            raise RuntimeError(
                f"MOSS-TTS Local journal length mismatch: rids={len(journal.rids)} pool_rows={len(journal.pool_rows)} rows={rows_len}"
            )
        else:
            pass
        if journal.rids != expected_rids:
            raise RuntimeError(
                f"MOSS-TTS Local journal/batch alignment broken: {journal.rids} != {expected_rids}"
            )
        else:
            pass
        for i, sched_req in enumerate(expected_reqs):
            req = sched_req.data.req
            if req is not None:
                try:
                    finished_fn = req.finished
                except AttributeError:
                    finished_fn = None
                try:
                    is_retracted = req.is_retracted
                except AttributeError:
                    is_retracted = False
                if callable(finished_fn) and finished_fn() or bool(is_retracted):
                    continue
                else:
                    pass
            else:
                pass
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == end_id:
                self.flush_stream_rows(sched_req.request_id, sched_req.data, force=True)
                continue
            else:
                pass
            sched_req.data.output_rows.append(journal.rows[i])
            stream_metadata = getattr(sched_req.data, "stream_metadata", None)
            if stream_metadata is None or self.outbox is None:
                continue
            else:
                pass
            pending = getattr(sched_req.data, "stream_pending_rows", None)
            if pending is None:
                pending = []
                sched_req.data.stream_pending_rows = pending
            else:
                pass
            pending.append(journal.rows[i])
            self.flush_stream_rows(sched_req.request_id, sched_req.data, force=False)
