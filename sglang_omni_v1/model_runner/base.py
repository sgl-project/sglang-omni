# SPDX-License-Identifier: Apache-2.0
"""Base model runner — shared execute() pipeline for all AR models.

Handles: ForwardBatch construction, phase-aware pre/post hooks, forward
pass, sampling, logit post-processing, and output extraction.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni_v1.scheduling.types import ModelRunnerOutput

logger = logging.getLogger(__name__)


class ModelRunner:
    """Base AR model runner.

    Subclasses provide phase-specific behavior:
      - prefill hooks for extend/prompt processing
      - decode hooks for single-step autoregressive decode processing
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        self.tp_worker = tp_worker
        self.output_processor = output_processor
        self.device = torch.device(f"cuda:{tp_worker.gpu_id}")
        self.model = tp_worker.model_runner.model

    def execute(self, scheduler_output: Any) -> ModelRunnerOutput:
        """Full pipeline: build batch → prepare → forward → post → sample → output."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()
        is_prefill = bool(schedule_batch.forward_mode.is_extend())

        capture_hidden_mode = (
            self.requested_capture_hidden_mode_prefill(
                schedule_batch, scheduler_output.requests
            )
            if is_prefill
            else self.requested_capture_hidden_mode_decode(
                schedule_batch, scheduler_output.requests
            )
        )
        if capture_hidden_mode is not None:
            model_worker_batch.capture_hidden_mode = capture_hidden_mode
        elif self.output_processor._capture_hidden:
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.tp_worker.model_runner
        )

        # Hook: model-specific preparation. Returns batch_result if it ran
        # a custom forward path, or None for standard forward.
        batch_result = (
            self.prepare_prefill(
                forward_batch, schedule_batch, scheduler_output.requests
            )
            if is_prefill
            else self.prepare_decode(
                forward_batch, schedule_batch, scheduler_output.requests
            )
        )

        if batch_result is None:
            # Standard forward path
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)

        if (
            not schedule_batch.is_prefill_only
            and batch_result.next_token_ids is None
            and (
                self.sample_before_post_prefill(
                    forward_batch, schedule_batch, scheduler_output.requests
                )
                if is_prefill
                else self.sample_before_post_decode(
                    forward_batch, schedule_batch, scheduler_output.requests
                )
            )
        ):
            batch_result.next_token_ids = self._sample_next_token_ids(
                batch_result.logits_output,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
            schedule_batch.output_ids = batch_result.next_token_ids

        # Hook: model-specific post-processing
        if is_prefill:
            self.post_prefill(
                batch_result,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
        else:
            self.post_decode(
                batch_result,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )

        # Sampling + logit processing
        if schedule_batch.is_prefill_only:
            if batch_result.next_token_ids is None:
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
        elif batch_result.next_token_ids is None:
            batch_result.next_token_ids = self._sample_next_token_ids(
                batch_result.logits_output,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
        schedule_batch.output_ids = batch_result.next_token_ids

        # Output extraction
        outputs = self.output_processor.process(batch_result, scheduler_output)
        for sched_req in scheduler_output.requests:
            data = sched_req.data
            data.generation_steps = int(data.generation_steps) + 1
            req_output = outputs[sched_req.request_id]
            extra = req_output.extra
            if isinstance(extra, dict) and extra:
                data.extra_model_outputs.update(extra)
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def prepare_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Called before prefill forward.

        Return a batch result if the subclass handled the forward itself,
        or None to use the standard tp_worker forward path.
        """
        return None

    def prepare_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Called before decode forward."""
        return None

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after prefill forward."""

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after decode forward."""

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return False

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return False

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any | None:
        return None

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any | None:
        return None

    # ------------------------------------------------------------------
    # Shared logit processing
    # ------------------------------------------------------------------

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._apply_repetition_penalty(logits_output, requests)
        self._apply_codec_suppress_tokens(logits_output, requests)
        return self.tp_worker.model_runner.sample(logits_output, forward_batch)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            penalty = req.sampling_params.repetition_penalty
            if penalty == 1.0:
                continue
            output_ids = req.output_ids
            if not output_ids:
                continue
            token_ids = list(set(output_ids))
            valid = [t for t in token_ids if 0 <= t < logits.shape[1]]
            if not valid:
                continue
            idx = torch.tensor(valid, dtype=torch.long, device=logits.device)
            scores = logits[row_idx, idx]
            scores = torch.where(scores > 0, scores / penalty, scores * penalty)
            logits[row_idx, idx] = scores

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            suppress_tokens = data.suppress_tokens
            if not suppress_tokens:
                req = data.req
                suppress_tokens = req._codec_suppress_tokens
            if not suppress_tokens:
                continue
            for token_id in suppress_tokens:
                if 0 <= int(token_id) < logits.shape[1]:
                    logits[row_idx, int(token_id)] = float("-inf")
