# SPDX-License-Identifier: Apache-2.0
"""dots.tts model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context


@dataclass
class _DotsTtsRequestState:
    """Per-request tail bookkeeping; the GPU state lives in the tail pools."""

    slot: int
    prompt_span_start: int
    prompt_patch_count: int
    prompt_latents: torch.Tensor
    all_mods: torch.Tensor
    prefill_input_embeds: torch.Tensor
    feedback: torch.Tensor | None = None
    generated: list[torch.Tensor] = field(default_factory=list)
    stop_step: int | None = None


class DotsTTSModelRunner(ModelRunner):
    """Runs dots.tts AR steps and samples continuous acoustic latents."""

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        nfe: int,
        eos_threshold: float,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._nfe = int(nfe)
        self._eos_threshold = float(eos_threshold)
        self._request_states: dict[str, _DotsTtsRequestState] = {}

    def reset_request(self, request_id: str) -> None:
        state = self._request_states.pop(request_id, None)
        if state is not None:
            self.model.tail.release_slot(state.slot)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        for sched_req in requests:
            self._materialize_request_state(sched_req)

    @torch.no_grad()
    def _materialize_request_state(self, sched_req: Any) -> None:
        request_id = sched_req.request_id
        if request_id in self._request_states:
            return

        model = self.model
        data = sched_req.data
        payload_state = data.state
        weight = model._decode_input_embedding.weight
        device = weight.device
        dtype = weight.dtype

        if payload_state.spk_emb is None or payload_state.prompt_latent is None:
            raise RuntimeError(
                "dots.tts tts_engine requires the speaker embedding and prompt "
                "latents from reference_encode"
            )
        speaker_embedding = payload_state.spk_emb.to(
            device=device, dtype=dtype
        ).reshape(1, -1)
        prompt_latents = payload_state.prompt_latent.to(
            device=device, dtype=dtype
        ).reshape(-1, int(model.latent_dim))

        patch_count = int(data.prompt_patch_count)
        if prompt_latents.size(0) != patch_count * int(model.latent_patch_size):
            raise RuntimeError(
                "dots.tts prompt latents do not match the prompt patch count: "
                f"frames={prompt_latents.size(0)} patches={patch_count}"
            )

        # note (chenyang): the slot is only reachable through _request_states, so
        # anything that raises before the state is registered would leak it for
        # the lifetime of the process and eventually exhaust the pool.
        slot = model.tail.acquire_slot()
        try:
            if payload_state.seed is not None:
                model.tail.set_slot_seed(slot, int(payload_state.seed))

            with self._tail_autocast():
                g_cond = model.speaker_condition(
                    speaker_embedding, float(payload_state.speaker_scale)
                )
                all_mods = model.meanflow_modulations(g_cond, nfe=self._nfe)
                prompt_patch_embeds = model.tail.encode_prompt_patches(
                    slot,
                    model.patch_encoder_input(
                        prompt_latents, normalized=False
                    ).unsqueeze(0),
                )

            span_start = int(data.prompt_span_start)
            prefill_input_embeds = model.get_input_embeddings()(
                data.input_ids.to(device=device)
            ).to(dtype=dtype)
            prefill_input_embeds[span_start : span_start + patch_count] = (
                prompt_patch_embeds.to(dtype=dtype)
            )

            self._request_states[request_id] = _DotsTtsRequestState(
                slot=slot,
                prompt_span_start=span_start,
                prompt_patch_count=patch_count,
                prompt_latents=prompt_latents,
                all_mods=all_mods,
                prefill_input_embeds=prefill_input_embeds.detach(),
            )
        except BaseException:
            model.tail.release_slot(slot)
            raise

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = torch.cat(
            [
                self._require_full_prefill(sched_req).prefill_input_embeds
                for sched_req in requests
            ],
            dim=0,
        )
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def _require_full_prefill(self, sched_req: Any) -> _DotsTtsRequestState:
        state = self._request_states[sched_req.request_id]
        req = sched_req.data.req
        prompt_len = int(state.prefill_input_embeds.size(0))
        if len(req.prefix_indices) != 0 or int(req.extend_range.length) != prompt_len:
            raise RuntimeError(
                "dots.tts prefill must cover the whole schedule prefix in one "
                "extend; prefix reuse and retraction are not supported "
                f"(prefix={len(req.prefix_indices)} "
                f"extend={int(req.extend_range.length)} prompt={prompt_len})"
            )
        return state

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
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=forward_batch.positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

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
            raise RuntimeError("dots.tts async lookahead is currently unsupported")
        if not requests:
            return

        rows = []
        for sched_req in requests:
            feedback = self._request_states[sched_req.request_id].feedback
            if feedback is None:
                raise RuntimeError(
                    "dots.tts decode step is missing the previous patch feedback "
                    f"for request {sched_req.request_id}"
                )
            rows.append(feedback)
        row_ids = self.model.stage_decode_feedback(torch.stack(rows, dim=0))
        forward_batch.input_ids[: len(requests)].copy_(row_ids)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if bool(schedule_batch.is_prefill_only):
            return
        hidden_states = result.logits_output.hidden_states
        last_hidden = self._seed_flow_history(hidden_states, requests)
        self._run_tail_step(result, requests, last_hidden, is_first_step=True)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        hidden_states = result.logits_output.hidden_states
        self._run_tail_step(result, requests, hidden_states, is_first_step=False)

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        batch = scheduler_output.batch_data
        if batch is not None and bool(batch.is_prefill_only):
            return {sched_req.request_id for sched_req in scheduler_output.requests}
        return set()

    @torch.no_grad()
    def _seed_flow_history(
        self, hidden_states: torch.Tensor, requests: list
    ) -> torch.Tensor:
        """Install every prefilling request's prompt flow history.

        Returns the backbone hidden of each request's last prompt audio span,
        which is the row the first acoustic tail step samples from.
        """
        model = self.model
        offset = 0
        last_rows = []
        with self._tail_autocast():
            for sched_req in requests:
                state = self._request_states[sched_req.request_id]
                prompt_len = int(state.prefill_input_embeds.size(0))
                span_start = state.prompt_span_start
                patches = state.prompt_patch_count
                request_hidden = hidden_states[offset : offset + prompt_len]
                fm_rows = model.prompt_flow_rows(
                    hidden_states=request_hidden[
                        span_start - 1 : span_start + patches - 1
                    ],
                    prompt_latents=state.prompt_latents,
                )
                model.tail.seed_fm_history(
                    state.slot, fm_rows=fm_rows, all_mods=state.all_mods
                )
                last_rows.append(request_hidden[prompt_len - 1])
                offset += prompt_len
        return torch.stack(last_rows, dim=0)

    def _tail_autocast(self) -> AbstractContextManager[None]:
        if self.device.type != "cuda":
            return nullcontext()
        dtype = self.model._decode_input_embedding.weight.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def _run_tail_step(
        self,
        result: Any,
        requests: list,
        hidden_states: torch.Tensor,
        *,
        is_first_step: bool,
    ) -> None:
        if not requests:
            return

        model = self.model
        states = [self._request_states[req.request_id] for req in requests]
        slots = [state.slot for state in states]

        with self._tail_autocast():
            stop_flags = (
                [False] * len(requests)
                if is_first_step
                else (model.stop_probability(hidden_states) > self._eos_threshold)
                .cpu()
                .tolist()
            )
            latents = model.tail.sample_patches(
                slots,
                fm_hidden_rows=model.project_hidden(hidden_states),
                latent_proj=model.latent_proj,
            )
            feedback = model.tail.encode_feedback(
                slots,
                model.patch_encoder_input(latents, normalized=True),
            )

        next_token_ids = []
        for row, (sched_req, state) in enumerate(zip(requests, states)):
            data = sched_req.data
            state.feedback = feedback[row].detach()
            # note (chenyang): the first tail step regenerates the final prompt
            # patch, which the reference audio already covers, so it is dropped.
            if not is_first_step:
                state.generated.append(model.denormalize_latents(latents[row]).detach())

            stop = bool(stop_flags[row])
            length = int(data.generation_steps) + 1 >= int(data.max_new_tokens)
            if stop:
                state.stop_step = int(data.generation_steps)
                next_token_ids.append(int(data.audio_eos_token_id))
            else:
                next_token_ids.append(int(data.audio_span_token_id))
            if stop or length:
                data.stop_step = state.stop_step
                data.generated_latents = (
                    torch.stack(state.generated, dim=0).to(
                        device="cpu", dtype=torch.float32
                    )
                    if state.generated
                    else None
                )

        result.next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.long, device=hidden_states.device
        )


__all__ = ["DotsTTSModelRunner"]
