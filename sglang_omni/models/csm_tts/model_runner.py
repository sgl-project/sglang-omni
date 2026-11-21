# SPDX-License-Identifier: Apache-2.0
"""Frame-step ModelRunner for CSM TTS.

Prefill embed overlay, decode-buffer collect, and stream emission.
"""

from __future__ import annotations

from queue import Queue

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN, Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.csm_tts.decode_buffers import (
    pack_decode_staging,
    populate_decode_buffers,
)
from sglang_omni.models.csm_tts.generation_params import depth_sampling_params
from sglang_omni.models.csm_tts.model import CsmTTSModel
from sglang_omni.models.csm_tts.prefill_embeds import build_prefill_input_embeds
from sglang_omni.models.csm_tts.utils import CODEBOOK_EOS
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest

__all__ = ["CsmTTSModelRunner"]


class CsmTTSModelRunner(ModelRunner):
    """ModelRunner for CsmTTSModel."""

    model: CsmTTSModel

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.outbox: Queue[OutgoingMessage] | None = None
        self.vocoder_target = "vocoder"

    def set_stream_outbox(self, outbox: Queue[OutgoingMessage]) -> None:
        """Wire the OmniScheduler outbox used by emit_code_chunk."""
        self.outbox = outbox

    @staticmethod
    def is_chunked(request: Req) -> bool:
        """Req.is_chunked if present, else inflight_middle_chunks."""
        # only one of the two names exists; which depends on SGLang version.
        chunk_flag = getattr(request, "is_chunked", None)
        if chunk_flag is not None:
            return int(chunk_flag) > 0
        else:
            return int(getattr(request, "inflight_middle_chunks", 0) or 0) > 0

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        forward_batch.request_ids = [request.request_id for request in requests]
        depth_temperatures, depth_top_ks = depth_sampling_params(requests)
        forward_batch.csm_depth_params = list(zip(depth_temperatures, depth_top_ks))
        # prefill CUDA-graph admission rejects ForwardBatch.input_embeds.
        input_embeds = self.build_prefill_input_embeds(forward_batch, requests)
        if input_embeds is not None:
            attach_omni_prefill_inputs(
                forward_batch, OmniPrefillInputs(input_embeds=input_embeds)
            )
        else:
            pass

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_step_outputs(result, requests)

    def build_prefill_input_embeds(
        self, forward_batch: ForwardBatch, requests: list[SchedulerRequest]
    ) -> torch.Tensor | None:
        return build_prefill_input_embeds(self.model, forward_batch, requests)

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        forward_batch.request_ids = [request.request_id for request in requests]
        populate_decode_buffers(
            self.model,
            forward_batch,
            requests,
            is_lookahead=is_lookahead,
            async_enabled=self.async_enabled,
        )

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_step_outputs_cg(result, forward_batch, requests)

    def collect_step_outputs_cg(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Pack GPU staging, copy to host, then collect codes."""
        num_requests = len(requests)
        if num_requests == 0:
            return
        else:
            pass
        self.ensure_batch_covers_requests(forward_batch, num_requests)
        staging = pack_decode_staging(self.model, num_requests)
        # sync path; one blocking D2H for the whole batch.
        combined_cpu = staging[:num_requests].cpu()
        self.decode_collect_host(
            combined_cpu,
            result,
            requests,
            next_token_device=result.logits_output.next_token_logits.device,
        )

    @staticmethod
    def ensure_batch_covers_requests(
        forward_batch: ForwardBatch | None, num_requests: int
    ) -> None:
        batch_size = int(forward_batch.batch_size)
        if batch_size < num_requests:
            raise ValueError(
                f"forward_batch.batch_size ({batch_size}) < len(requests) "
                f"({num_requests})"
            )
        else:
            pass

    def decode_collect_host(
        self,
        staging_host: torch.Tensor,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
        *,
        next_token_device: torch.device | None = None,
    ) -> None:
        """Collect one decode step from the staged host snapshot."""
        num_codebooks = self.model.cg_codes_BN.shape[1]
        codes_BN_cpu = staging_host[:, :num_codebooks]
        was_done_cpu = staging_host[:, num_codebooks].bool().tolist()
        generation_done_cpu = staging_host[:, num_codebooks + 1].bool().tolist()
        codebook0_per_row: list[int] = []
        for row, scheduler_request in enumerate(requests):
            request_data = scheduler_request.data
            request = request_data.req
            # was_done latches EOS only; overrun / length-finished skip too.
            if self.is_chunked(request) or request.finished() or was_done_cpu[row]:
                codebook0_per_row.append(0)
                continue
            else:
                pass
            codes_32 = codes_BN_cpu[row].to(torch.long).clone()
            request_data.output_codes.append(codes_32)
            request_data.generation_done = bool(generation_done_cpu[row])
            self.mark_sampler_finished(request, request_data.generation_done)
            if not request_data.generation_done:
                self.emit_code_chunk(scheduler_request, codes_32)
            else:
                pass
            codebook0_per_row.append(int(codes_32[0].item()))

        if next_token_device is None:
            result.next_token_ids = torch.tensor(codebook0_per_row, dtype=torch.long)
        else:
            result.next_token_ids = torch.tensor(
                codebook0_per_row, dtype=torch.long, device=next_token_device
            )

    def collect_step_outputs(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        """Prefill collect for frame 0. next_token_ids is codebook-0."""
        if len(requests) == 0:
            return
        else:
            pass

        model = self.model
        codebook0_per_row: list[int] = []
        for scheduler_request in requests:
            request_data = scheduler_request.data
            request = request_data.req
            request_id = scheduler_request.request_id
            sampler_slot = model.request_id_to_sampler_slot.get(request_id)
            emitted_codes = model.output_codes.get(request_id)
            if (
                self.is_chunked(request)
                or sampler_slot is None
                or not emitted_codes
                or request.finished()
            ):
                codebook0_per_row.append(0)
                continue
            else:
                pass
            codes_32 = emitted_codes[-1].detach().cpu().clone()
            request_data.output_codes.append(codes_32)
            request_data.generation_done = bool(
                model.sampler_pool.generation_done[sampler_slot].item()
            )
            self.mark_sampler_finished(request, request_data.generation_done)
            if not request_data.generation_done:
                self.emit_code_chunk(scheduler_request, codes_32)
            else:
                pass
            codebook0_per_row.append(int(codes_32[0].item()))

        result.next_token_ids = torch.tensor(
            codebook0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        """Async GPU half: pack staging and copy to a pinned host buffer."""
        num_requests = len(requests)
        if num_requests == 0:
            return None
        else:
            pass
        self.ensure_batch_covers_requests(forward_batch, num_requests)
        staging = pack_decode_staging(self.model, num_requests)
        host_buffer = self.next_host_staging(self.model.cg_collect_staging)
        host_buffer[:num_requests].copy_(staging[:num_requests], non_blocking=True)
        result.next_token_ids = (
            self.model.cg_codes_BN[:num_requests, 0].clamp_min(0).to(torch.long).clone()
        )
        return host_buffer

    def post_decode_resolve(
        self,
        host_buffer: torch.Tensor | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Async host half: collect from the pinned snapshot."""
        num_requests = len(requests)
        if num_requests == 0:
            return
        else:
            pass
        self.decode_collect_host(
            host_buffer[:num_requests],
            result,
            requests,
            next_token_device=None,
        )

    def emit_code_chunk(
        self, scheduler_request: SchedulerRequest, codes_32: torch.Tensor
    ) -> None:
        """Stream one non-EOS frame to the vocoder."""
        if self.outbox is None:
            return
        else:
            pass
        metadata = scheduler_request.data.stream_metadata
        if metadata is None:
            return
        else:
            pass
        self.outbox.put(
            OutgoingMessage(
                request_id=scheduler_request.request_id,
                type="stream",
                target=self.vocoder_target,
                data=codes_32,
                metadata=metadata,
            )
        )

    @staticmethod
    def mark_sampler_finished(request: Req, generation_done: bool) -> None:
        """Mark the request finished so upstream can release KV."""
        if generation_done and request.finished_reason is None:
            request.finished_reason = FINISH_MATCHED_TOKEN(CODEBOOK_EOS)
        else:
            pass
