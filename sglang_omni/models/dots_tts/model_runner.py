# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for dots.tts continuous latent feedback."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.dots_tts.request_builders import DotsFlowResume


class DotsTTSModelRunner(ModelRunner):
    """Use the shared SGLang forward path and own only latent recurrence."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._request_data: dict[str, Any] = {}

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        if not requests:
            return
        self._release_retracted_flow_states()
        rows = []
        materialized = []
        try:
            for request in requests:
                data = request.data
                schedule = data.generation_schedule
                if schedule is None or data.span_positions is None:
                    raise RuntimeError(
                        "dots.tts request is missing its generation schedule"
                    )
                if data.flow_state is not None and not isinstance(
                    data.flow_state, DotsFlowResume
                ):
                    self._suspend_request_data(data)
                self._request_data.pop(request.request_id, None)
                resume = data.flow_state
                flow_state, prompt_embeddings = self.model.flow.new_request(
                    max_audio_patch_count=int(data.span_positions.numel()),
                    prompt_latents=data.state.prompt_latents,
                    speaker_embedding=data.state.speaker_embedding,
                    speaker_scale=data.state.speaker_scale,
                    rng=(
                        resume.rng_state
                        if isinstance(resume, DotsFlowResume)
                        else data.state.seed
                    ),
                )
                data.flow_state = flow_state
                self._request_data[request.request_id] = data
                materialized.append((request.request_id, data))
                device = forward_batch.input_ids.device
                prefill_ids = schedule[:, : data.prefill_end].to(device=device)
                embeddings = self.model.get_input_embeddings()(prefill_ids).clone()
                prompt_positions = data.prompt_span_positions
                if prompt_positions is not None and prompt_positions.numel():
                    if prompt_embeddings is None:
                        raise RuntimeError(
                            "dots.tts prompt spans require prompt embeddings"
                        )
                    embeddings[:, prompt_positions.to(device=device), :] = (
                        prompt_embeddings.to(device=device, dtype=embeddings.dtype)
                    )
                prefix_len = len(data.req.prefix_indices)
                req_len = int(data.req.extend_range.length)
                assert prefix_len == 0, "dots.tts radix prefix reuse is disabled"
                prompt_len = embeddings.size(1)
                generated_count = req_len - prompt_len
                assert (
                    generated_count
                    == len(data.req.output_ids)
                    == len(data.decoded_latent_patches)
                ), "dots.tts re-prefill history must match generated tokens"
                request_rows = [embeddings[0]]
                if generated_count:
                    request_rows.append(
                        self.model.flow.replay_feedback(
                            flow_state,
                            data.decoded_latent_patches,
                        ).to(device=device, dtype=embeddings.dtype)
                    )
                rows.append(torch.cat(request_rows, dim=0))
            forward_batch.input_embeds = torch.cat(rows, dim=0)
        except BaseException:
            for request_id, data in materialized:
                self._request_data.pop(request_id, None)
                self._clear_request_data(data)
            raise

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        if not requests:
            return
        rows = []
        for request in requests:
            queue = request.data.pending_feedback_queue
            if not queue:
                raise RuntimeError("dots.tts decode is missing its latent feedback")
            feedback = queue.popleft()
            if feedback.ndim > 1:
                feedback = feedback.reshape(-1, feedback.shape[-1])[-1]
            rows.append(feedback)
        forward_batch.input_embeds = torch.stack(rows).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.LAST

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        if not requests:
            return
        hidden = self._hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        elif hidden.ndim != 2:
            raise RuntimeError(
                f"dots.tts expected rank-2/3 prefill hidden, got {hidden.ndim}"
            )
        offset = 0
        last_hidden = []
        for request in requests:
            data = request.data
            length = int(data.req.extend_range.length)
            request_hidden = hidden[offset : offset + length].unsqueeze(0)
            if request_hidden.size(1) != length:
                raise RuntimeError("dots.tts prefill hidden rows are incomplete")
            self.model.flow.initialize_history(
                data.flow_state,
                hidden_states=request_hidden,
                prompt_span_positions=data.prompt_span_positions,
                audio_span_token_ids=set(data.state.audio_span_token_ids),
                generation_schedule=data.generation_schedule.to(hidden.device),
                prefill_end=data.prefill_end,
                decoded_latent_patches=data.decoded_latent_patches,
            )
            last_hidden.append(request_hidden[0, -1])
            offset += length
        if offset != hidden.size(0):
            raise RuntimeError("dots.tts prefill hidden rows do not match requests")
        self._run_flow_batch(
            result,
            requests,
            torch.stack(last_hidden),
            append_hidden=False,
        )

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        if not requests:
            return
        hidden = self._hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        elif hidden.ndim != 2:
            raise RuntimeError(
                f"dots.tts expected rank-2/3 decode hidden, got {hidden.ndim}"
            )
        self._run_flow_batch(result, requests, hidden, append_hidden=True)

    def _run_flow_batch(
        self,
        result: Any,
        requests: list,
        hidden: torch.Tensor,
        *,
        append_hidden: bool,
    ) -> None:
        data_rows = [request.data for request in requests]
        steps = self.model.flow.decode_batch(
            [data.flow_state for data in data_rows],
            hidden_states=hidden,
            num_steps=[data.state.num_steps for data in data_rows],
            ode_methods=[data.state.ode_method for data in data_rows],
            guidance_scales=[data.state.guidance_scale for data in data_rows],
            eos_thresholds=[data.state.eos_threshold for data in data_rows],
            append_hidden=append_hidden,
        )
        next_token_ids = []
        for data, step in zip(data_rows, steps, strict=True):
            data.pending_feedback_queue.append(step.feedback_embedding.detach())
            decoded_latent = step.latent_patch.detach()
            data.decoded_latent_patches.append(decoded_latent)
            if step.emit:
                data.latest_latent_patch = decoded_latent
                data.latent_patches.append(decoded_latent)
            next_token_ids.append(data.control_token_id)
            if step.finished:
                data.req.finished_reason = FINISH_MATCHED_TOKEN(data.control_token_id)
        result.next_token_ids = torch.tensor(
            next_token_ids,
            dtype=torch.long,
            device=hidden.device,
        )

    @staticmethod
    def _hidden_states(result: Any) -> torch.Tensor:
        logits_output = getattr(result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            hidden = getattr(result, "hidden_states", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("dots.tts SGLang forward did not return hidden states")
        return hidden

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        self._request_data.pop(request_id, None)
        self._clear_request_data(req_data)

    def reset_request(self, request_id: str) -> None:
        req_data = self._request_data.pop(request_id, None)
        if req_data is not None:
            self._clear_request_data(req_data)

    def _release_retracted_flow_states(self) -> None:
        for req_data in self._request_data.values():
            if (
                req_data.flow_state is not None
                and not isinstance(req_data.flow_state, DotsFlowResume)
                and req_data.req.is_retracted
            ):
                self._suspend_request_data(req_data)

    def _suspend_request_data(self, req_data: Any) -> None:
        flow_state = req_data.flow_state
        assert flow_state is not None and not isinstance(flow_state, DotsFlowResume)
        req_data.flow_state = DotsFlowResume(
            self.model.flow.suspend_request(flow_state)
        )
        req_data.pending_feedback_queue.clear()

    def _clear_request_data(self, req_data: Any) -> None:
        flow_state = req_data.flow_state
        if flow_state is not None and not isinstance(flow_state, DotsFlowResume):
            self.model.flow.release_request(flow_state)
        req_data.pending_feedback_queue.clear()
        req_data.flow_state = None


__all__ = ["DotsTTSModelRunner"]
