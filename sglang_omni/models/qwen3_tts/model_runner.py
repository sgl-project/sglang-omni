# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    def prepare_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(
            forward_batch,
            input_embeds,
        )

    def prepare_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del forward_batch, schedule_batch
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
        del forward_batch
        self._collect_codes(result, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        self._collect_codes(result, schedule_batch, requests)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def _collect_codes(self, result: Any, schedule_batch: Any, requests: list) -> None:
        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        self.model.code_predictor_forward(layer0_codes, hidden)
        schedule_batch.output_ids = result.next_token_ids

        eos_id = int(self.model.config.codec_eos_token_id)
        for row_idx, sched_req in enumerate(requests):
            semantic_token = int(result.next_token_ids[row_idx].item())
            if semantic_token == eos_id:
                continue
            code_chunk = self.model._output_codes[row_idx].detach().clone()
            feedback = self.model._output_embeds[row_idx].detach().clone()
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.pending_feedback_queue.append(feedback)

    def _write_feedback_buffers(self, requests: list) -> None:
        batch_size = len(requests)
        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        feedback_mask[:batch_size] = False

        for row_idx, sched_req in enumerate(requests):
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=feedback_buffer.device,
                dtype=feedback_buffer.dtype,
            )
            if combined is None:
                continue
            feedback_buffer[row_idx].copy_(combined)
            feedback_mask[row_idx] = True

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prompt_input_embeds
            if prompt_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
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
        logits_output = self.model(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_embeds_are_projected=True,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
