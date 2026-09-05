# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.fun_cosyvoice3.streaming import (
    AR_FOLLOWUP_FLUSH_TOKENS,
    first_ar_flush_tokens,
    prompt_token_len,
)
from sglang_omni.scheduling.messages import OutgoingMessage

from .sglang_model import VOCAB_SIZE


class FunCosyVoice3ModelRunner(ModelRunner):
    """Runs Fun-CosyVoice3 AR steps and collects generated speech tokens."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        self._flush_code_chunks(request_id, req_data, force=True)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_tokens(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_tokens(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> bool:
        """Sample the first speech token before collecting prefill output."""
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> bool:
        """Sample each speech token before collecting decode output."""
        del forward_batch, schedule_batch, requests
        return True

    def _collect_tokens(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if result.next_token_ids is None:
            return
        token_ids = result.next_token_ids
        if token_ids.ndim != 1:
            token_ids = token_ids.reshape(-1)
        # note (guozhihao-224): one batched D2H instead of per-request .item() syncs.
        token_ids_cpu = token_ids.tolist()
        for idx, sched_req in enumerate(requests):
            token_id = int(token_ids_cpu[idx])
            if token_id >= VOCAB_SIZE:
                continue
            token = torch.tensor([token_id], dtype=torch.long)
            sched_req.data.output_codes.append(token)
            self._queue_or_emit_code_chunk(sched_req, token)

    def _queue_or_emit_code_chunk(
        self,
        sched_req: Any,
        token: torch.Tensor,
    ) -> None:
        data = sched_req.data
        if self._outbox is None or data.stream_metadata is None:
            return
        data.stream_code_buffer.append(token)
        data.stream_code_seen += 1
        if int(data.stream_code_next_flush) <= 0:
            # note (guozhihao-224): first flush is 28 + prompt_pad so the
            # vocoder can decode the first causal hop without waiting for 53.
            data.stream_code_next_flush = first_ar_flush_tokens(
                prompt_token_len(data.flow_prompt_speech_token)
            )
        if data.stream_code_seen >= data.stream_code_next_flush:
            self._flush_code_chunks(sched_req.request_id, data, force=False)

    def _flush_code_chunks(
        self,
        request_id: str,
        data: Any,
        *,
        force: bool,
    ) -> None:
        pending = data.stream_code_buffer
        if not pending:
            return
        payload = pending[0] if len(pending) == 1 else torch.cat(pending, dim=0)
        pending.clear()
        if not force:
            data.stream_code_next_flush = (
                int(data.stream_code_seen) + AR_FOLLOWUP_FLUSH_TOKENS
            )
        self._emit_code_chunk(request_id, data, payload)

    def _emit_code_chunk(
        self,
        request_id: str,
        data: Any,
        codes: torch.Tensor,
    ) -> None:
        if self._outbox is None:
            return
        metadata = data.stream_metadata
        if metadata is None:
            return
        chunk_metadata = dict(metadata)
        # note (guozhihao-224): first chunk carries Flow prompt tensors so
        # vocoder can start before the AR payload arrives.
        if not data.stream_prompt_sent:
            chunk_metadata["flow_prompt_speech_token"] = data.flow_prompt_speech_token
            chunk_metadata["flow_prompt_speech_feat"] = data.flow_prompt_speech_feat
            chunk_metadata["flow_embedding"] = data.flow_embedding
            data.stream_prompt_sent = True
        self._outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=self._vocoder_target,
                data=codes,
                metadata=chunk_metadata,
            )
        )

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prompt_input_embeds
            if prompt_embeds is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 prefill requires prompt_input_embeds"
                )
            pieces.append(prompt_embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
