# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict

import torch
from sglang.srt.utils.common import async_h2d

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.utils.device import has_device_streams

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import (
        PredictorGraphStats,
        Qwen3TTSTalker,
    )
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
else:
    pass

# note (ratish): official qwen-tts generates the talker with min_new_tokens=2; an EOS on the first frame ended a request with no audio.
QWEN3_TTS_MIN_NEW_FRAMES = 2


class Qwen3TTSModelInfo(TypedDict):
    predictor_cuda_graph: PredictorGraphStats


@dataclass(kw_only=True, slots=True)
class Qwen3TTSLaunchedStep:
    """A launched decode step's sampled ids, codec frame and inputs, held until resolve."""

    ids: torch.Tensor
    codes: torch.Tensor
    feedback_rows: list[torch.Tensor]
    input_rows: list[torch.Tensor]
    ready: torch.Event | None


def is_lookahead_safe_request(req: Req) -> bool:
    params = req.sampling_params
    return (
        params.frequency_penalty == 0.0
        and params.presence_penalty == 0.0
        and params.min_new_tokens == 0
        and req.custom_logit_processor is None
    )


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    model: Qwen3TTSTalker
    tp_worker: ModelWorker
    # Resolved in __init__: post_process_outputs records a readiness event per
    # decode step, and the device cannot change under the runner.
    codes_event_module: ModuleType | None = None
    resolved_step: Qwen3TTSLaunchedStep | None = None
    launch_input_rows: list[torch.Tensor] | None = None
    penalty_rows_key: tuple[object, ...] | None = None
    penalty_rows: tuple[torch.Tensor, torch.Tensor] | None = None

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        leading_silence_mask_frames: int,
        silence_codec_ids: torch.Tensor,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.has_pending_code_step = False
        self.row_ids_cache: torch.Tensor | None = None
        self.leading_silence_mask_frames = leading_silence_mask_frames
        self.silence_codec_ids = silence_codec_ids
        self.codes_event_module = (
            torch.get_device_module(self.device)
            if has_device_streams(self.device)
            else None
        )

    def model_info(self) -> Qwen3TTSModelInfo:
        return {"predictor_cuda_graph": self.model.predictor_graph_stats()}

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self.build_prefill_input_embeds(
                    forward_batch,
                    requests,
                ),
            ),
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        if is_lookahead:
            self.launch_input_rows = self.write_lookahead_feedback_buffers(
                forward_batch, requests
            )
            self.add_inflight_repetition_penalty(forward_batch, requests)
        else:
            self.write_feedback_buffers(forward_batch, requests)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_codes(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def lookahead_eligible(self, batch: ScheduleBatch | None) -> bool:
        if (
            batch is None
            or batch.forward_mode is None
            or not batch.forward_mode.is_decode()
        ):
            return False
        else:
            return bool(batch.reqs) and all(
                is_lookahead_safe_request(req) for req in batch.reqs
            )

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> Qwen3TTSLaunchedStep | None:
        if not requests or result.next_token_ids is None:
            return None
        else:
            pass
        layer0_codes = result.next_token_ids
        self.run_code_predictor(result, forward_batch, layer0_codes)
        batch_size = len(requests)
        codes = self.model.output_codes[:batch_size].detach().clone()
        embeds = self.model.output_embeds[:batch_size].detach().clone()
        ready = self.record_codes_ready()
        feedback_rows = [embeds[row_idx] for row_idx in range(batch_size)]
        codec_ids = layer0_codes.reshape(-1)[:batch_size].to(torch.int64)
        codec_id_rows = [codec_ids[row_idx] for row_idx in range(batch_size)]
        for row_idx, sched_req in enumerate(requests):
            sched_req.data.inflight_feedback = feedback_rows[row_idx]
            sched_req.data.inflight_codec_id = codec_id_rows[row_idx]
        input_rows = self.launch_input_rows or []
        self.launch_input_rows = None
        ids = layer0_codes.reshape(-1)[:batch_size]
        host_ids = self.next_host_staging((batch_size,), ids.dtype)
        host_ids[:batch_size].copy_(ids, non_blocking=True)
        return Qwen3TTSLaunchedStep(
            ids=host_ids,
            codes=codes,
            feedback_rows=feedback_rows,
            input_rows=input_rows,
            ready=ready,
        )

    def post_decode_resolve(
        self,
        launch_buf: Qwen3TTSLaunchedStep | None,
        result: GenerationBatchResult | None,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        if launch_buf is None or result is None or not requests:
            return
        else:
            pass
        result.next_token_ids = launch_buf.ids[: len(requests)]
        self.resolved_step = launch_buf

    def record_codes_ready(self) -> torch.Event | None:
        if self.codes_event_module is None:
            return None
        else:
            ready = self.codes_event_module.Event()
            ready.record()
            return ready

    def emitted_codec_frames(self, data: Qwen3TTSSGLangRequestData) -> int:
        inflight = 0 if data.inflight_feedback is None else 1
        return len(data.output_codes) + inflight

    # ------------------------------------------------------------------
    # Qwen3-TTS logit shaping
    # ------------------------------------------------------------------

    def apply_codec_suppress_tokens(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return
        else:
            pass

        # Qwen3-TTS reserves the final 1024 configured token IDs for codec
        # control and suppresses that range except for codec EOS.
        configured_vocab = int(self.model.config.vocab_size)
        suppress_start = max(0, configured_vocab - 1024)
        suppress_stop = min(configured_vocab, logits.shape[1])
        if suppress_start >= suppress_stop:
            return
        else:
            pass

        active_logits = logits[: len(requests)]
        codec_eos = int(self.model.config.codec_eos_token_id)
        if suppress_start <= codec_eos < suppress_stop:
            active_logits[:, suppress_start:codec_eos] = float("-inf")
            active_logits[:, codec_eos + 1 : suppress_stop] = float("-inf")
        else:
            active_logits[:, suppress_start:suppress_stop] = float("-inf")
        for row_index, scheduled_request in enumerate(requests):
            if (
                self.emitted_codec_frames(scheduled_request.data)
                < QWEN3_TTS_MIN_NEW_FRAMES
            ):
                # note (ratish): a slice fills on the device; assigning one element copies a host scalar and waits for the stream.
                active_logits[row_index, codec_eos : codec_eos + 1] = float("-inf")
            else:
                pass

        # note (luojiaxuan): a cold-start clone that samples a silence id first
        # tends to stay silent for several frames, so silence ids are excluded
        # from its opening frames.
        if self.leading_silence_mask_frames > 0:
            for row_index, scheduled_request in enumerate(requests):
                if (
                    scheduled_request.data.mask_leading_silence
                    and self.emitted_codec_frames(scheduled_request.data)
                    < self.leading_silence_mask_frames
                ):
                    active_logits[row_index].index_fill_(
                        0, self.silence_codec_ids, float("-inf")
                    )
                else:
                    pass
        else:
            pass

    def install_sampling_seeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Install the staged semantic seeds when a request in the batch carries a
        seed, or deterministic inference has already seeded every row."""
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_seed is None and all(
            scheduled_request.data.req.sampling_params.sampling_seed is None
            for scheduled_request in requests
        ):
            return
        else:
            pass
        self.validate_seeded_sampling_supported(sampling_info)
        sampling_info.sampling_seed = self.model.semantic_sampling_seed_tensor[
            : len(requests)
        ]

    def collect_codes(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.has_pending_code_step = False
        if result.next_token_ids is None:
            return
        else:
            pass
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        # note(ratish): the layer 0 id is the step's only host read, staged ahead
        # of the predictor so the finalize wait covers the sample, not the predictor.
        self.stage_token_ids(result, result.next_token_ids)
        self.run_code_predictor(result, forward_batch, layer0_codes)
        self.has_pending_code_step = True

    def run_code_predictor(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        layer0_codes: torch.Tensor,
    ) -> None:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        else:
            pass
        semantic_positions = self.sample_positions(forward_batch, layer0_codes.device)
        self.model.code_predictor_forward(
            layer0_codes,
            hidden,
            semantic_positions=semantic_positions,
        )

    def post_process_outputs(
        self,
        result: GenerationBatchResult | None,
        scheduler_output: SchedulerOutput | None,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        resolved = self.resolved_step
        if resolved is not None:
            self.resolved_step = None
            self.apply_resolved_step(resolved, scheduler_output, outputs)
            return
        else:
            pass
        if not self.has_pending_code_step:
            return
        else:
            pass
        self.has_pending_code_step = False
        eos_id = int(self.model.config.codec_eos_token_id)
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        batch_size = len(scheduler_output.requests)
        codes_snap = self.model.output_codes[:batch_size].detach().clone()
        embeds_snap = self.model.output_embeds[:batch_size].detach().clone()
        codes_ready = None
        if self.codes_event_module is not None:
            codes_ready = self.codes_event_module.Event()
            codes_ready.record()
        else:
            pass
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            # note (ratish): a middle prefill chunk samples at a prompt row; SGLang drops
            # that token, so it gives no frame, feedback or stream chunk either.
            if sched_req.data.req.inflight_middle_chunks > 0 or req_output.data is None:
                continue
            else:
                pass
            sched_req.data.speech_output_tokens += 1
            if int(req_output.data) == eos_id:
                continue
            else:
                pass
            code_chunk = codes_snap[row_idx]
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.latest_stream_code_chunk = code_chunk
            sched_req.data.codes_ready_event = codes_ready
            sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])

    def apply_resolved_step(
        self,
        resolved: Qwen3TTSLaunchedStep,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        eos_id = int(self.model.config.codec_eos_token_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            data = sched_req.data
            if data.inflight_feedback is resolved.feedback_rows[row_idx]:
                data.inflight_feedback = None
                data.inflight_codec_id = None
            else:
                pass
            req = data.req
            req_output = outputs[sched_req.request_id]
            if (
                req.finished()
                or self.req_is_retracted(req)
                or req.inflight_middle_chunks > 0
                or req_output.data is None
            ):
                continue
            else:
                pass
            QwenTalkerModelRunner.pop_next_decode_inputs(data)
            QwenTalkerModelRunner.append_decode_input_history(
                data, resolved.input_rows[row_idx]
            )
            data.speech_output_tokens += 1
            if int(req_output.data) == eos_id:
                continue
            else:
                pass
            code_chunk = resolved.codes[row_idx]
            data.output_codes.append(code_chunk)
            data.latest_stream_code_chunk = code_chunk
            data.codes_ready_event = resolved.ready
            data.pending_feedback_queue.append(resolved.feedback_rows[row_idx])

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if sched_req.data.req.inflight_middle_chunks > 0
        }

    def sample_positions(
        self, forward_batch: ForwardBatch, device: torch.device
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            return forward_batch.positions.to(device=device, dtype=torch.long)
        else:
            seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)
            return (seq_lens - 1).clamp_min(0)

    def write_feedback_buffers(
        self, forward_batch: ForwardBatch | None, requests: list[SchedulerRequest]
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        decode_feedback_embedding = self.model.decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        else:
            pass
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        else:
            pass
        row_ids = self.decode_row_ids(batch_size, input_ids)
        weight = decode_feedback_embedding.weight
        device = weight.device
        dtype = weight.dtype
        feedback_rows: list[torch.Tensor] = []
        text_rows: list[torch.Tensor] = []
        for sched_req in requests:
            data: Qwen3TTSSGLangRequestData = sched_req.data
            inputs = QwenTalkerModelRunner.peek_next_decode_inputs(data)
            if inputs is None:
                raise RuntimeError(
                    f"Qwen3-TTS decode for {sched_req.request_id} has no queued "
                    "feedback embedding"
                )
            else:
                pass
            feedback, text = inputs
            feedback_rows.append(
                QwenTalkerModelRunner.decode_row(feedback, device=device, dtype=dtype)
            )
            text_rows.append(
                QwenTalkerModelRunner.decode_row(text, device=device, dtype=dtype)
            )
            QwenTalkerModelRunner.pop_next_decode_inputs(data)

        with torch.no_grad():
            target = weight[:batch_size]
            torch.stack(feedback_rows, dim=0, out=target)
            target.add_(torch.stack(text_rows, dim=0))
            # note(ratish): the history outlives the buffer, a retracted request
            # replays it in its re-prefill.
            history = target.detach().clone()
        for row_idx, sched_req in enumerate(requests):
            QwenTalkerModelRunner.append_decode_input_history(
                sched_req.data, history[row_idx]
            )
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)

    def add_inflight_repetition_penalty(
        self, forward_batch: ForwardBatch, requests: list[SchedulerRequest]
    ) -> None:
        scaling = forward_batch.sampling_info.acc_scaling_penalties
        if scaling is None:
            return
        else:
            pass
        rows = [
            row_idx
            for row_idx, sched_req in enumerate(requests)
            if sched_req.data.inflight_codec_id is not None
        ]
        if not rows:
            return
        else:
            pass
        codec_ids = torch.stack(
            [requests[row_idx].data.inflight_codec_id for row_idx in rows]
        )
        row_index, penalties = self.staged_penalty_rows(
            requests, rows, scaling.device, scaling.dtype
        )
        scaling.index_put_((row_index, codec_ids), penalties)

    def staged_penalty_rows(
        self,
        requests: list[SchedulerRequest],
        rows: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            tuple(
                (sched_req.request_id, sched_req.data.decode_prep_epoch)
                for sched_req in requests
            ),
            tuple(rows),
        )
        if self.penalty_rows_key == key and self.penalty_rows is not None:
            return self.penalty_rows
        else:
            pass
        row_index = async_h2d(rows, dtype=torch.int64, device=device)
        penalties = async_h2d(
            [
                float(requests[row_idx].data.req.sampling_params.repetition_penalty)
                for row_idx in rows
            ],
            dtype=dtype,
            device=device,
        )
        self.penalty_rows_key = key
        self.penalty_rows = (row_index, penalties)
        return self.penalty_rows

    def write_lookahead_feedback_buffers(
        self, forward_batch: ForwardBatch | None, requests: list[SchedulerRequest]
    ) -> list[torch.Tensor]:
        batch_size = len(requests)
        if batch_size == 0:
            return []
        else:
            pass
        decode_feedback_embedding = self.model.decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        else:
            pass
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        else:
            pass
        row_ids = self.decode_row_ids(batch_size, input_ids)
        weight = decode_feedback_embedding.weight
        device = weight.device
        dtype = weight.dtype
        feedback_rows: list[torch.Tensor] = []
        text_rows: list[torch.Tensor] = []
        for sched_req in requests:
            feedback, text = self.peek_lookahead_decode_inputs(sched_req)
            feedback_rows.append(
                QwenTalkerModelRunner.decode_row(feedback, device=device, dtype=dtype)
            )
            text_rows.append(
                QwenTalkerModelRunner.decode_row(text, device=device, dtype=dtype)
            )
        with torch.no_grad():
            target = weight[:batch_size]
            torch.stack(feedback_rows, dim=0, out=target)
            target.add_(torch.stack(text_rows, dim=0))
            history = target.detach().clone()
        input_ids[:batch_size].copy_(row_ids)
        return [history[row_idx] for row_idx in range(batch_size)]

    @staticmethod
    def peek_lookahead_decode_inputs(
        sched_req: SchedulerRequest,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data: Qwen3TTSSGLangRequestData = sched_req.data
        if data.inflight_feedback is None:
            feedback = QwenTalkerModelRunner.peek_left(data.pending_feedback_queue)
            text_offset = 0
        else:
            feedback = data.inflight_feedback
            text_offset = 1
        if feedback is None:
            raise RuntimeError(
                f"Qwen3-TTS decode for {sched_req.request_id} has no queued "
                "feedback embedding"
            )
        else:
            pass
        text_queue = data.pending_text_queue
        if len(text_queue) > text_offset:
            return feedback, text_queue[text_offset]
        elif data.thinker_chunks_done:
            return feedback, data.tts_pad_embed
        else:
            raise RuntimeError(
                f"Qwen3-TTS decode for {sched_req.request_id} has no queued text "
                "embedding"
            )

    def decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = self.row_ids_cache
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self.row_ids_cache = cached
        else:
            pass
        return cached[:batch_size]

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for sched_req in requests:
            data: Qwen3TTSSGLangRequestData = sched_req.data
            req = data.req
            req_len = req.extend_len
            prefix_len = req.prefix_len
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            else:
                pass
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            else:
                pass
            piece = QwenTalkerModelRunner.projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=req_len,
                device=forward_batch.input_ids.device,
            )
            if piece is None or int(piece.shape[0]) != req_len:
                have = 0 if piece is None else int(piece.shape[0])
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {have} rows, need {req_len}"
                )
            else:
                pass
            pieces.append(piece)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
