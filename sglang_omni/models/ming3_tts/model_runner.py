# SPDX-License-Identifier: Apache-2.0
"""Autoregressive Ming 3 TTS steps that sample one continuous latent per token."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.ming3_tts.sglang_model import (
    FLOW_STEPS,
    AcousticInputs,
    Ming3TTSModel,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class Ming3TTSRequestData(SGLangARRequestData):
    """Per-request speech state; latent_history holds the last generated latent."""

    speaker_embedding: torch.Tensor
    speaker_position: int
    sampling_temperature: float
    noise_generator: torch.Generator
    latent_history: torch.Tensor
    feedback_embeddings: list[torch.Tensor] = field(default_factory=list)
    latents: list[torch.Tensor] = field(default_factory=list)
    zero_speaker: bool = False
    prompt_latents: torch.Tensor | None = None
    latent_position: int = -1


@dataclass(frozen=True, kw_only=True)
class StopPolicy:
    stop_probability_threshold: float
    minimum_decode_steps: int
    audio_patch_token_id: int
    audio_end_token_id: int


class Ming3TTSModelRunner(ModelRunner):
    model: Ming3TTSModel

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        stop_policy: StopPolicy,
        acoustic_graph_max_batch_size: int | None,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.stop_policy: StopPolicy = stop_policy
        if acoustic_graph_max_batch_size is not None:
            self.model.capture_acoustic_graphs(acoustic_graph_max_batch_size)
        else:
            pass

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        token_embedding = self.model.get_input_embeddings()
        window_embeddings: list[torch.Tensor] = []
        for request, prefix_length, extend_length in zip(
            requests,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            strict=True,
        ):
            request_state: Ming3TTSRequestData = request.data
            prompt_embeddings = token_embedding(request_state.input_ids.to(self.device))
            if request_state.speaker_position < 0:
                pass
            elif request_state.zero_speaker:
                prompt_embeddings[request_state.speaker_position] = 0
            else:
                prompt_embeddings[request_state.speaker_position] = self.model.spk_head(
                    request_state.speaker_embedding.to(
                        self.device, dtype=prompt_embeddings.dtype
                    )
                )
            if request_state.prompt_latents is not None:
                projected = self.model.linear_proj_audio(
                    request_state.prompt_latents.to(
                        self.device, dtype=prompt_embeddings.dtype
                    ).reshape(-1, 1, request_state.prompt_latents.shape[-1])
                ).reshape(-1, prompt_embeddings.shape[-1])
                start = request_state.latent_position
                prompt_embeddings[start : start + projected.shape[0]] = projected
            else:
                pass
            # note (chenyang): retraction re-prefills generated tokens, whose
            # inputs are the feedback embeddings rather than token embeddings.
            sequence_embeddings = torch.cat(
                [
                    prompt_embeddings,
                    *[row[None] for row in request_state.feedback_embeddings],
                ]
            )
            window_embeddings.append(
                sequence_embeddings[prefix_length : prefix_length + extend_length]
            )
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=torch.cat(window_embeddings)),
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        assert not is_lookahead, "Ming 3 TTS decodes synchronously"
        row_ids = self.model.stage_decode_feedback(
            torch.stack([request.data.feedback_embeddings[-1] for request in requests])
        )
        forward_batch.input_ids[: len(requests)].copy_(row_ids)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.append_acoustic_step(result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.append_acoustic_step(result, requests)

    def append_acoustic_step(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        request_states: list[Ming3TTSRequestData] = [
            request.data for request in requests
        ]
        latent_dim = request_states[0].latent_history.shape[-1]
        initial_noise = torch.cat(
            [
                torch.randn(
                    (1, latent_dim, 1),
                    generator=request_state.noise_generator,
                    device=self.device,
                )
                for request_state in request_states
            ]
        )
        solver_noise = torch.cat(
            [
                torch.randn(
                    (FLOW_STEPS - 1, 1, 1, latent_dim),
                    generator=request_state.noise_generator,
                    device=self.device,
                )
                for request_state in request_states
            ],
            dim=1,
        )
        acoustic_step = self.model.sample_acoustic_step(
            AcousticInputs(
                hidden_states=result.logits_output.hidden_states,
                latent_history=torch.stack(
                    [request_state.latent_history for request_state in request_states]
                ),
                initial_noise=initial_noise,
                solver_noise=solver_noise,
                temperatures=torch.tensor(
                    [
                        request_state.sampling_temperature
                        for request_state in request_states
                    ],
                    device=self.device,
                ),
            )
        )
        generated_latent_counts = torch.tensor(
            [request_state.generation_steps + 1 for request_state in request_states],
            device=self.device,
        )
        stop_policy = self.stop_policy
        is_stopped = (
            acoustic_step.stop_probabilities > stop_policy.stop_probability_threshold
        ) & (generated_latent_counts >= stop_policy.minimum_decode_steps)
        result.next_token_ids = torch.where(
            is_stopped, stop_policy.audio_end_token_id, stop_policy.audio_patch_token_id
        )
        for row, request_state in enumerate(request_states):
            request_state.latent_history = acoustic_step.latents[row]
            request_state.latents.append(acoustic_step.latents[row])
            request_state.feedback_embeddings.append(
                acoustic_step.feedback_embeddings[row]
            )
