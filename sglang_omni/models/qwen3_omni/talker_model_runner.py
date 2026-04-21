# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with buffer-backed feedback + batched code predictor."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.messages import OutgoingMessage


class QwenTalkerModelRunner(ModelRunner):

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        code2wav_target: str = "code2wav",
        feedback_enabled: bool = True,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox = outbox
        self._code2wav_target = code2wav_target
        self._feedback_enabled = bool(feedback_enabled)

    def execute(self, scheduler_output: Any):
        return super().execute(scheduler_output)

    def prepare_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        return self._run_projected_prefill_forward(
            forward_batch, schedule_batch, requests
        )

    def prepare_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del forward_batch
        del schedule_batch
        if not self._feedback_enabled:
            return None

        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(requests)
        return None

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not self._feedback_enabled:
            return

        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        talker_hidden = result.logits_output.hidden_states
        if isinstance(talker_hidden, torch.Tensor) and talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        self.model.code_predictor_forward(layer0_codes, talker_hidden)
        schedule_batch.output_ids = result.next_token_ids
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not self._feedback_enabled:
            return

        batch_size = len(requests)
        result.next_token_ids = self.model._sampled_token_ids[:batch_size].clone()
        schedule_batch.output_ids = result.next_token_ids
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def _emit_code_chunks_and_feedback(
        self,
        *,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        for idx, sched_req in enumerate(requests):
            req = schedule_batch.reqs[idx]
            code_chunk = self.model._output_codes[idx].detach().clone()
            feedback_row = self.model._output_embeds[idx].detach().clone()
            self._outbox.put(
                OutgoingMessage(
                    request_id=req.rid,
                    type="stream",
                    data=code_chunk,
                    target=self._code2wav_target,
                )
            )
            sched_req.data.feedback_embeds = feedback_row

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def _run_projected_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        has_projected = forward_batch.input_embeds is not None or any(
            bool(req.data.input_embeds_are_projected) for req in requests
        )
        if not has_projected:
            return None

        input_embeds = forward_batch.input_embeds
        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        input_embeds_are_projected = bool(projected_flags) and all(projected_flags)
        if input_embeds is None:
            rows = []
            for sched_req in requests:
                req = sched_req.data.req
                embeds = req.input_embeds
                if embeds:
                    prefix_len = len(req.prefix_indices)
                    rows.extend(embeds[prefix_len:])
            if not rows:
                return None
            input_embeds = torch.as_tensor(
                rows,
                device=forward_batch.input_ids.device,
                dtype=torch.float32,
            )

        result = self._forward_with_input_embeds(
            forward_batch,
            input_embeds=input_embeds,
            input_embeds_are_projected=input_embeds_are_projected,
        )
        return result

    def _write_feedback_buffers(self, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        feedback_mask[:batch_size] = False

        for row_idx, sched_req in enumerate(requests):
            combined = self._combine_feedback_embed(
                sched_req=sched_req,
                device=feedback_buffer.device,
                dtype=feedback_buffer.dtype,
            )
            if combined is None:
                continue
            feedback_buffer[row_idx].copy_(combined)
            feedback_mask[row_idx] = True
            sched_req.data.feedback_embeds = None

    @staticmethod
    def _combine_feedback_embed(
        *,
        sched_req: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        data = sched_req.data
        feedback = data.feedback_embeds
        if feedback is None:
            return None

        combined = feedback.to(device=device, dtype=dtype).reshape(-1)
        projected = bool(data.input_embeds_are_projected)
        step_offset = 2 if projected else 1
        step_index = max(int(data.generation_steps) - step_offset, 0)
        trailing = data.trailing_text_hidden
        tts_pad_embed = data.tts_pad_embed
        thinker_chunks_done = bool(data.thinker_chunks_done)

        trailing_value = None
        if isinstance(trailing, list) and step_index < len(trailing):
            trailing_value = trailing[step_index]
        elif isinstance(trailing, torch.Tensor) and step_index < trailing.shape[0]:
            trailing_value = trailing[step_index]

        if trailing_value is not None:
            combined = combined + trailing_value.to(
                device=device,
                dtype=dtype,
            ).reshape(-1)
        elif thinker_chunks_done and tts_pad_embed is not None:
            combined = combined + tts_pad_embed.to(
                device=device,
                dtype=dtype,
            ).reshape(-1)
        return combined

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor,
        input_deepstack_embeds: torch.Tensor | None = None,
        input_deepstack_mask: torch.Tensor | None = None,
        input_embeds_are_projected: bool = False,
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
        if input_deepstack_embeds is not None:
            input_deepstack_embeds = input_deepstack_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=model_dtype,
            )

        logits_output = self.model(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_deepstack_embeds=input_deepstack_embeds,
            input_deepstack_mask=input_deepstack_mask,
            input_embeds_are_projected=input_embeds_are_projected,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
