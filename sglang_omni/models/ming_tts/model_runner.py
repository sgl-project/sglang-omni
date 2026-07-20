# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.ming_tts.payload_types import (
    decode_prompt_latent,
    decode_speaker_embedding,
)
from sglang_omni.models.ming_tts.sglang_model import MingTTSTailInputs


@dataclass
class MingTTSTPStepUpdate:
    """Rank-synchronized output of one Ming AR recurrence step.

    Only the entry rank owns generated acoustic latents for serialization.
    Follower ranks consume the synchronized token and feedback fields only to
    keep their next backbone decode input aligned.
    """

    control_tensor: torch.Tensor
    feedback_embeddings: torch.Tensor

    @classmethod
    def empty_for_broadcast(
        cls,
        *,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        feedback_dtype: torch.dtype,
    ) -> "MingTTSTPStepUpdate":
        return cls(
            control_tensor=torch.zeros(
                3,
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            feedback_embeddings=torch.zeros(
                int(batch_size),
                int(hidden_size),
                dtype=feedback_dtype,
                device=device,
            ),
        )

    @property
    def next_token_ids(self) -> torch.Tensor:
        return self.control_tensor[0]

    @property
    def feedback_mask(self) -> torch.Tensor:
        return self.control_tensor[1]

    @property
    def tail_failed(self) -> torch.Tensor:
        return self.control_tensor[2]


@dataclass
class _MingTTSRequestState:
    prefill_input_embeds: torch.Tensor | None = None
    feedback_embeddings: list[torch.Tensor] = field(default_factory=list)
    latent_history: torch.Tensor | None = None
    generated_latents: list[torch.Tensor] = field(default_factory=list)
    generated_last_chunk: list[bool] = field(default_factory=list)
    stop_step: int | None = None


class MingTTSModelRunner(ModelRunner):
    """Runs Ming-Omni-TTS AR steps and samples continuous acoustic latents."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        server_args = getattr(tp_worker, "server_args", None)
        self._tp_rank = int(getattr(tp_worker, "tp_rank", 0) or 0)
        self._tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        self._request_states: dict[str, _MingTTSRequestState] = {}

    def reset_request(self, request_id: str) -> None:
        self._request_states.pop(request_id, None)

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        for sched_req in requests:
            self._materialize_request_state(sched_req)

    def _materialize_request_state(self, sched_req: Any) -> None:
        request_id = sched_req.request_id
        if request_id in self._request_states:
            return

        data = sched_req.data
        state = data.state
        weight = self.model._decode_input_embedding.weight
        device = weight.device
        dtype = weight.dtype
        speaker_embedding = decode_speaker_embedding(
            state,
            device=device,
            dtype=dtype,
        )
        prompt_latent = decode_prompt_latent(
            state,
            device=device,
            dtype=torch.float32,
        )
        if prompt_latent is not None and prompt_latent.ndim == 2:
            prompt_latent = prompt_latent.unsqueeze(0)

        prefill_input_embeds = None
        if speaker_embedding is not None or prompt_latent is not None:
            with torch.no_grad():
                prefill_input_embeds = self.model.get_input_embeddings()(
                    data.input_ids.to(device=device)
                ).to(dtype=dtype)

                if speaker_embedding is not None:
                    positions = state.spk_injection_positions
                    if positions is None:
                        positions = [
                            int(position) + 1
                            for position in (state.spk_token_positions or [])
                        ]
                    projected_speaker = self.model.spk_head(speaker_embedding)
                    for row, position in enumerate(positions):
                        prefill_input_embeds[int(position)] = projected_speaker[row].to(
                            dtype=prefill_input_embeds.dtype
                        )

                if prompt_latent is not None:
                    start = state.prompt_latent_start_position
                    if start is None:
                        start = int(state.audio_token_position) + 1
                    token_count = int(state.prompt_latent_token_count)
                    projected_prompt = self.model.linear_proj_audio(
                        prompt_latent.to(dtype=dtype).reshape(
                            -1,
                            int(self.model.patch_size),
                            int(self.model.latent_dim),
                        )
                    )
                    projected_prompt = projected_prompt.reshape(
                        -1,
                        int(projected_prompt.shape[-1]),
                    )
                    prefill_input_embeds[int(start) : int(start) + token_count] = (
                        projected_prompt.to(dtype=prefill_input_embeds.dtype)
                    )

                prefill_input_embeds = prefill_input_embeds.detach()

        latent_history = None
        if self._is_entry_rank:
            latent_history = torch.zeros(
                1,
                int(self.model.history_patch_size),
                int(self.model.latent_dim),
                device=device,
                dtype=torch.float32,
            )
            if prompt_latent is not None:
                history_len = int(latent_history.shape[1])
                prompt_len = int(prompt_latent.shape[1])
                if prompt_len >= history_len:
                    latent_history.copy_(prompt_latent[:, -history_len:, :])
                else:
                    latent_history[:, -prompt_len:, :].copy_(prompt_latent)

        self._request_states[request_id] = _MingTTSRequestState(
            prefill_input_embeds=prefill_input_embeds,
            latent_history=latent_history,
        )

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        batch_parts = []
        dtype = self.model._decode_input_embedding.weight.dtype
        device = forward_batch.input_ids.device
        embedding = self.model.get_input_embeddings()
        for sched_req in requests:
            data = sched_req.data
            request_state = self._request_states[sched_req.request_id]
            req = data.req
            prefix_len = len(req.prefix_indices)
            extend_len = int(req.extend_input_len)
            end = prefix_len + extend_len
            prompt_ids = data.input_ids
            prompt_len = int(prompt_ids.shape[0])
            req_parts = []

            prompt_start = min(prefix_len, prompt_len)
            prompt_stop = min(end, prompt_len)
            if prompt_stop > prompt_start:
                if request_state.prefill_input_embeds is None:
                    prompt_rows = embedding(
                        prompt_ids[prompt_start:prompt_stop].to(device=device)
                    ).to(dtype=dtype)
                else:
                    prompt_rows = request_state.prefill_input_embeds[
                        prompt_start:prompt_stop
                    ].to(device=device, dtype=dtype)
                req_parts.append(prompt_rows)

            # Note (yzxiao): Retraction may re-prefill generated audio tokens,
            # whose rows live in feedback embeddings rather than token embeds.
            gen_start = max(prefix_len, prompt_len) - prompt_len
            gen_end = max(end - prompt_len, 0)
            if gen_end > gen_start:
                feedback_rows = [
                    feedback.to(device=device, dtype=dtype)
                    for feedback in request_state.feedback_embeddings[gen_start:gen_end]
                ]
                req_parts.append(torch.stack(feedback_rows, dim=0))

            req_embeds = torch.cat(req_parts, dim=0)
            batch_parts.append(req_embeds)
        return torch.cat(batch_parts, dim=0)

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=self.model._decode_input_embedding.weight.dtype,
        )

        model_runner = self.tp_worker.model_runner
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
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

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch
        if is_lookahead:
            raise RuntimeError("Ming TTS async lookahead is currently unsupported")
        batch_size = len(requests)
        if batch_size == 0:
            return

        rows = []
        weight = self.model._decode_input_embedding.weight
        for sched_req in requests:
            rows.append(
                self._request_states[sched_req.request_id]
                .feedback_embeddings[-1]
                .to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
            )

        row_ids = self.model.stage_decode_feedback(torch.stack(rows, dim=0))
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_ming_tts_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_ming_tts_step(result, forward_batch, schedule_batch, requests)

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        batch = getattr(scheduler_output, "batch_data", None)
        if bool(getattr(batch, "is_prefill_only", False)):
            return {sched_req.request_id for sched_req in scheduler_output.requests}
        return set()

    def _collect_ming_tts_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return

        hidden = result.logits_output.hidden_states
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        hidden_states = hidden

        weight = self.model._decode_input_embedding.weight
        step_update = MingTTSTPStepUpdate.empty_for_broadcast(
            batch_size=len(requests),
            hidden_size=int(weight.shape[1]),
            device=hidden_states.device,
            feedback_dtype=weight.dtype,
        )
        if self._is_entry_rank:
            try:
                self._run_entry_tail_step(hidden_states, requests, step_update)
            except Exception:
                step_update.control_tensor.zero_()
                step_update.tail_failed.fill_(1)
                step_update.feedback_embeddings.zero_()
                raise
            finally:
                self._broadcast_tp_step_update(step_update)
        else:
            self._broadcast_tp_step_update(step_update)
            self._apply_follower_step_update(step_update, requests)

        next_token_ids = step_update.next_token_ids
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

    def _run_entry_tail_step(
        self,
        hidden_states: torch.Tensor,
        requests: list[Any],
        step_update: MingTTSTPStepUpdate,
    ) -> None:
        weight = self.model._decode_input_embedding.weight
        device = hidden_states.device
        next_ids = []

        if device.type == "cuda":
            dtype = weight.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                dtype = torch.bfloat16
            context = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            context = nullcontext()

        with context:
            request_states = [self._request_states[req.request_id] for req in requests]
            steps = [int(req.data.generation_steps) for req in requests]
            max_steps = [int(req.data.max_new_tokens) for req in requests]
            histories = [state.latent_history for state in request_states]
            history_batch = torch.cat(histories, dim=0)
            steps_tensor = torch.tensor(steps, dtype=torch.long, device=device)
            max_steps_tensor = torch.tensor(
                max_steps,
                dtype=torch.long,
                device=device,
            )
            cfg_tensor = torch.tensor(
                [float(req.data.state.cfg) for req in requests],
                dtype=torch.float32,
                device=device,
            )
            sigma_tensor = torch.tensor(
                [float(req.data.state.sigma) for req in requests],
                dtype=torch.float32,
                device=device,
            )
            temperature_tensor = torch.tensor(
                [float(req.data.state.temperature) for req in requests],
                dtype=torch.float32,
                device=device,
            )

            tail_outputs = self.model.run_tail_step(
                MingTTSTailInputs(
                    hidden_states=hidden_states,
                    latent_history=history_batch,
                    cfg=cfg_tensor,
                    sigma=sigma_tensor,
                    temperature=temperature_tensor,
                )
            )
            sampled = tail_outputs.sampled
            stop_prob = tail_outputs.stop_prob
            feedback_embeddings = tail_outputs.feedback_embeddings
            stop_flags = (stop_prob > 0.5) & (steps_tensor > 3)
            length_flags = steps_tensor + 1 >= max_steps_tensor
            feedback_mask = ~(stop_flags | length_flags)
            step_update.feedback_mask.copy_(feedback_mask)
            decision_rows = torch.stack((stop_flags, length_flags)).cpu().tolist()
            stop_list, length_list = decision_rows
            for row_idx, request_state in enumerate(request_states):
                data = requests[row_idx].data
                step = steps[row_idx]
                sampled_row = sampled[row_idx : row_idx + 1]
                sampled_chunk = sampled_row.squeeze(0).detach()
                request_state.generated_latents.append(sampled_chunk)

                stop = stop_list[row_idx]
                length = length_list[row_idx]
                request_state.generated_last_chunk.append(stop or length)
                if stop:
                    request_state.stop_step = step
                    next_ids.append(int(data.audio_eos_token_id))
                    continue

                self._advance_latent_history(
                    request_state.latent_history,
                    sampled_row,
                )
                next_ids.append(int(data.audio_patch_token_id))
                if not length:
                    feedback = feedback_embeddings[row_idx].detach()
                    request_state.feedback_embeddings.append(feedback)
                    step_update.feedback_embeddings[row_idx].copy_(
                        feedback.to(
                            device=step_update.feedback_embeddings.device,
                            dtype=step_update.feedback_embeddings.dtype,
                        )
                    )

            for row_idx, (stop, length) in enumerate(zip(stop_list, length_list)):
                if not (stop or length):
                    continue
                request_state = request_states[row_idx]
                data = requests[row_idx].data
                data.generated_latents = torch.stack(
                    request_state.generated_latents,
                    dim=0,
                ).to(device="cpu", dtype=torch.float32)
                data.generated_last_chunk = list(request_state.generated_last_chunk)
                data.stop_step = request_state.stop_step

        step_update.next_token_ids.copy_(
            torch.tensor(next_ids, dtype=torch.long, device=device)
        )

    @staticmethod
    def _advance_latent_history(
        latent_history: torch.Tensor,
        sampled_row: torch.Tensor,
    ) -> None:
        patch = int(sampled_row.shape[1])
        history_len = int(latent_history.shape[1])
        sampled_row = sampled_row.to(
            device=latent_history.device,
            dtype=latent_history.dtype,
        )
        if patch >= history_len:
            latent_history.copy_(sampled_row[:, -history_len:, :])
            return
        latent_history[:, :-patch, :].copy_(latent_history[:, patch:, :].clone())
        latent_history[:, -patch:, :].copy_(sampled_row)

    def _apply_follower_step_update(
        self,
        step_update: MingTTSTPStepUpdate,
        requests: list[Any],
    ) -> None:
        feedback_list, tail_failure_list = step_update.control_tensor[1:].cpu().tolist()
        if tail_failure_list[0]:
            raise RuntimeError("Ming TTS acoustic tail failed on the entry rank")
        for row_idx, sched_req in enumerate(requests):
            request_state = self._request_states[sched_req.request_id]
            if feedback_list[row_idx]:
                feedback = step_update.feedback_embeddings[row_idx].detach().clone()
                request_state.feedback_embeddings.append(feedback)

    @property
    def _is_entry_rank(self) -> bool:
        # Note (yzxiao): FlowLoss is not tensor-parallel, so rank 0 owns
        # acoustic sampling while followers only mirror the next AR input.
        return self._tp_rank == 0

    def _broadcast_tp_step_update(self, step_update: MingTTSTPStepUpdate) -> None:
        if self._tp_size <= 1:
            return
        for tensor in (
            step_update.control_tensor,
            step_update.feedback_embeddings,
        ):
            self._broadcast_tensor_from_entry(tensor)

    def _broadcast_tensor_from_entry(self, tensor: torch.Tensor) -> None:
        import torch.distributed as dist

        tp_group = self._get_tp_group()
        if tp_group is None:
            raise RuntimeError("Ming TTS TP broadcast requires a TP group")
        ranks = getattr(tp_group, "ranks", None)
        src_rank = int(ranks[0]) if ranks else int(getattr(tp_group, "first_rank", 0))
        dist_group = getattr(tp_group, "device_group", None)
        if dist_group is None:
            dist_group = getattr(tp_group, "group", None)
        dist.broadcast(tensor, src=src_rank, group=dist_group)

    def _get_tp_group(self) -> Any:
        getter = getattr(self.tp_worker, "get_tp_group", None)
        if callable(getter):
            return getter()
        model_runner = getattr(self.tp_worker, "model_runner", None)
        return getattr(model_runner, "tp_group", None)
