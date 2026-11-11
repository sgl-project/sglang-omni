# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS model runner for OmniScheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.models.voxtral_tts.request_builders import VoxtralSGLangRequestData
from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


@dataclass(kw_only=True)
class VoxtralDecodeStep:
    """Device codes and pinned host rows owned by one unresolved decode step."""

    audio_codes: torch.Tensor
    host_audio_codes: torch.Tensor


class VoxtralTTSModelRunner(ModelRunner):
    model: VoxtralSGLangTTSModel
    pending_host_audio_codes: torch.Tensor | None = None
    pending_skip_request_ids: frozenset[str] = frozenset()

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.pending_audio_codes: torch.Tensor | None = None
        self.pending_audio_embeds: torch.Tensor | None = None

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self.build_prefill_input_embeds(
            forward_batch, requests
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del forward_batch, schedule_batch
        self.write_decode_input_embed_buffer(requests)

    def write_decode_input_embed_buffer(self, requests: list[SchedulerRequest]) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        buffer = self.model.decode_input_embed_buffer
        rows: list[torch.Tensor] = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size, device=buffer.device))
                continue
            else:
                pass
            rows.append(queue.popleft())
        stacked = torch.stack(rows, dim=0).to(
            device=buffer.device,
            dtype=buffer.dtype,
        )
        buffer[:batch_size].copy_(stacked)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch
        self.collect_audio_step(result, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch
        self.collect_audio_step(result, schedule_batch, requests)

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> VoxtralDecodeStep:
        """Publish feedback before host collection so the next forward can launch."""
        self.collect_audio_step(result, None, requests)
        audio_codes = self.pending_audio_codes
        audio_embeddings = self.pending_audio_embeds
        assert audio_codes is not None and audio_embeddings is not None
        self.pending_audio_codes = None
        self.pending_audio_embeds = None
        for row_index, request in enumerate(requests):
            request.data.pending_feedback_queue.append(
                audio_embeddings[row_index, 0].detach().clone()
            )
        host_audio_codes = self.next_host_staging(audio_codes.shape, audio_codes.dtype)
        host_audio_codes = host_audio_codes[: len(requests)]
        host_audio_codes.copy_(audio_codes, non_blocking=True)
        return VoxtralDecodeStep(
            audio_codes=audio_codes, host_audio_codes=host_audio_codes
        )

    def post_decode_resolve(
        self,
        launch_buf: VoxtralDecodeStep,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.pending_audio_codes = launch_buf.audio_codes
        self.pending_audio_embeds = None
        # note (Codex): streaming rows must outlive the reused pinned host buffer.
        self.pending_host_audio_codes = launch_buf.host_audio_codes.clone()
        result.next_token_ids = self.pending_host_audio_codes[:, 0]
        self.pending_skip_request_ids = frozenset(
            request.request_id
            for request in requests
            if request.data.req.finished() or self.req_is_retracted(request.data.req)
        )
        for request in requests:
            if request.request_id in self.pending_skip_request_ids:
                request.data.pending_feedback_queue.clear()
            else:
                pass

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        input_embeds = self.model.get_input_embeddings()(input_ids)
        offset = 0
        for sched_req in requests:
            data: VoxtralSGLangRequestData = sched_req.data
            req = data.req
            req_len = req.extend_len
            prefix_len = req.prefix_len
            full_ids = data.input_ids
            current_ids = full_ids[prefix_len : prefix_len + req_len]
            audio_positions = (current_ids == int(data.audio_token_id)).nonzero(
                as_tuple=True
            )[0]
            if audio_positions.numel() == 0 or data.voice_embedding is None:
                offset += req_len
                continue
            else:
                pass
            previous_audio = int(
                (full_ids[:prefix_len] == int(data.audio_token_id)).sum()
            )
            voice = data.voice_embedding.to(
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            )
            n_frames = min(
                int(audio_positions.numel()), voice.shape[0] - previous_audio
            )
            if n_frames > 0:
                rows = (
                    audio_positions[:n_frames].to(device=input_embeds.device) + offset
                )
                input_embeds[rows] = voice[previous_audio : previous_audio + n_frames]
            else:
                pass
            offset += req_len
        return input_embeds

    def collect_audio_step(
        self,
        result: GenerationBatchResult,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del requests
        hidden = result.logits_output.hidden_states
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        else:
            pass
        codes = self.model.run_acoustic_transformer(hidden)
        semantic_ids = codes[:, 0].to(dtype=torch.long)
        result.next_token_ids = semantic_ids

        self.pending_audio_codes = codes
        self.pending_audio_embeds = self.model.audio_token_embedding(
            codes.unsqueeze(2)
        ).sum(dim=1)

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        codes = self.pending_audio_codes
        embeds = self.pending_audio_embeds
        host_codes = self.pending_host_audio_codes
        skip_request_ids = self.pending_skip_request_ids
        self.pending_audio_codes = None
        self.pending_audio_embeds = None
        self.pending_host_audio_codes = None
        self.pending_skip_request_ids = frozenset()
        if codes is None:
            return
        else:
            pass

        eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        if host_codes is None and any(
            request.data.stream for request in scheduler_output.requests
        ):
            host_codes = codes.to("cpu")
        else:
            pass
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if (
                sched_req.request_id in skip_request_ids
                or req_output.data is None
                or int(req_output.data) == eos_id
            ):
                continue
            else:
                pass
            sched_req.data.output_codes.append(codes[row_idx].detach().clone())
            if host_codes is not None and sched_req.data.stream:
                sched_req.data.stream_pending_rows.append(host_codes[row_idx])
            else:
                pass
            if embeds is not None:
                sched_req.data.pending_feedback_queue.append(
                    embeds[row_idx, 0].detach().clone()
                )
            else:
                pass
