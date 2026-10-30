# SPDX-License-Identifier: Apache-2.0
"""Autoregressive Ming 3 TTS steps that sample one continuous latent per token."""

from __future__ import annotations

from collections.abc import Iterable
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
from sglang_omni.models.ming3_tts.acoustic import FLOW_STEPS, AcousticInputs
from sglang_omni.models.ming3_tts.sglang_model import Ming3DenseTTSConfig, Ming3TTSModel
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


@dataclass(kw_only=True)
class LaunchState:
    """The inputs a request's next acoustic step reads, as of one launch."""

    latent_history: torch.Tensor
    next_feedback: torch.Tensor | None
    launched_latents: int
    noise_state: torch.Tensor


@dataclass(kw_only=True)
class Ming3TTSRequestData(SGLangARRequestData):
    """Per-request speech state.

    latents and feedback_embeddings hold committed steps. latent_history,
    next_feedback and launched_latents follow the newest launched step, which a
    lookahead decode launches from before the previous step commits.
    """

    speaker_embedding: torch.Tensor
    speaker_position: int
    sampling_temperature: float
    noise_generator: torch.Generator
    latent_history: torch.Tensor
    feedback_embeddings: list[torch.Tensor] = field(default_factory=list)
    latents: list[torch.Tensor] = field(default_factory=list)
    next_feedback: torch.Tensor | None = None
    launched_latents: int = 0
    emitted_latent_steps: int = 0
    zero_speaker: bool = False
    prompt_rows: torch.Tensor | None = None
    latent_position: int = -1
    initial_noise_buffer: torch.Tensor | None = None
    solver_noise_buffer: torch.Tensor | None = None


def build_stream_output(
    request_id: str,
    request: Ming3TTSRequestData,
    output: RequestOutput,
) -> list[OutgoingMessage]:
    """Transfer each generated latent once, independently of the final payload."""
    if not request.stage_payload.request.params.get("stream", False):
        return []
    else:
        pass
    messages = [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            target="vocoder",
            data=latent.detach().to(device="cpu", dtype=torch.float32).clone(),
            metadata={"modality": "audio_latents", "stream": True},
        )
        for latent in request.latents[request.emitted_latent_steps :]
    ]
    request.emitted_latent_steps = len(request.latents)
    return messages


@dataclass(kw_only=True)
class LaunchedAcousticStep:
    """One launched step's rows; host_token_ids is the lookahead's pinned copy."""

    latents: torch.Tensor
    feedback_embeddings: torch.Tensor
    is_middle_chunk: list[bool]
    host_token_ids: torch.Tensor | None
    before_launch: list[LaunchState] = field(default_factory=list)


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
        *,
        enable_fused_flow_head: bool = False,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.model.enable_fused_flow_head = enable_fused_flow_head
        self.stop_policy: StopPolicy = stop_policy
        self.step_input_host_bufs: list[torch.Tensor] = []
        self.step_input_host_slot: int = 0
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
        token_embedding_weight = self.model.get_input_embeddings().weight
        window_embeddings: list[torch.Tensor] = []
        for request, prefix_length, extend_length in zip(
            requests,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            strict=True,
        ):
            request_state: Ming3TTSRequestData = request.data
            prompt_length = len(request_state.input_ids)
            window_end = prefix_length + extend_length
            if prefix_length < prompt_length:
                request_state.latent_history = request_state.latent_history.to(
                    self.device
                )
                prompt_embeddings = torch.nn.functional.embedding(
                    request_state.input_ids.to(self.device), token_embedding_weight
                )
                if request_state.speaker_position < 0:
                    pass
                elif request_state.zero_speaker:
                    prompt_embeddings[request_state.speaker_position] = 0
                else:
                    prompt_embeddings[request_state.speaker_position] = (
                        self.model.spk_head(
                            request_state.speaker_embedding.to(
                                self.device, dtype=prompt_embeddings.dtype
                            )
                        )
                    )
                if request_state.prompt_rows is not None:
                    start = request_state.latent_position
                    prompt_embeddings[
                        start : start + request_state.prompt_rows.shape[0]
                    ] = request_state.prompt_rows.to(self.device)
                else:
                    pass
                window_embeddings.append(prompt_embeddings[prefix_length:window_end])
            else:
                pass
            # note (chenyang): retraction re-prefills generated tokens, whose
            # inputs are the feedback embeddings rather than token embeddings.
            # A decode row of a mixed batch takes its newest one the same way.
            window_embeddings.extend(
                row[None]
                for row in request_state.feedback_embeddings[
                    max(prefix_length - prompt_length, 0) : max(
                        window_end - prompt_length, 0
                    )
                ]
            )
        input_embeds = torch.cat(window_embeddings)
        token_count = input_embeds.shape[0]
        if isinstance(self.model.config, Ming3DenseTTSConfig):
            self.model.prefill_inputs.weight[:token_count].copy_(input_embeds)
            forward_batch.input_ids.copy_(self.model.prefill_input_rows[:token_count])
        else:
            pass
        attach_omni_prefill_inputs(
            forward_batch, OmniPrefillInputs(input_embeds=input_embeds)
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        row_ids = self.model.stage_decode_feedback(
            torch.stack([request.data.next_feedback for request in requests])
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

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> LaunchedAcousticStep | None:
        """Launch the step without committing it; resolve commits the live rows."""
        if not requests:
            return None
        else:
            pass
        before_launch = [
            LaunchState(
                latent_history=request.data.latent_history,
                next_feedback=request.data.next_feedback,
                launched_latents=request.data.launched_latents,
                noise_state=request.data.noise_generator.get_state(),
            )
            for request in requests
        ]
        launched = self.launch_acoustic_step(result, requests)
        launched.before_launch = before_launch
        host_token_ids = self.next_host_staging(
            (len(requests),), result.next_token_ids.dtype
        )
        host_token_ids[: len(requests)].copy_(result.next_token_ids, non_blocking=True)
        launched.host_token_ids = host_token_ids
        return launched

    def post_decode_resolve(
        self,
        launch_buf: LaunchedAcousticStep | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Commit rows of running requests; a retracted one rolls back its launch."""
        if launch_buf is None or not requests:
            return
        else:
            pass
        committed_rows: list[int] = []
        for row, request in enumerate(requests):
            request_state: Ming3TTSRequestData = request.data
            if self.req_is_retracted(request_state.req):
                before_launch = launch_buf.before_launch[row]
                request_state.latent_history = before_launch.latent_history
                request_state.next_feedback = before_launch.next_feedback
                request_state.launched_latents = before_launch.launched_latents
                request_state.noise_generator.set_state(before_launch.noise_state)
            elif request_state.req.finished():
                pass
            else:
                committed_rows.append(row)
        self.commit_acoustic_step(launch_buf, requests, committed_rows)
        result.next_token_ids = launch_buf.host_token_ids[: len(requests)]

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        """A prompt chunk before the last one samples no latent, so it is no step."""
        return {
            request.request_id
            for request in scheduler_output.requests
            if request.data.req.inflight_middle_chunks > 0
        }

    def append_acoustic_step(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        launched = self.launch_acoustic_step(result, requests)
        self.commit_acoustic_step(launched, requests, range(len(requests)))

    def launch_acoustic_step(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> LaunchedAcousticStep:
        """Sample the step and advance each request's next-step inputs."""
        request_states: list[Ming3TTSRequestData] = [
            request.data for request in requests
        ]
        is_middle_chunk = [
            request_state.req.inflight_middle_chunks > 0
            for request_state in request_states
        ]
        latent_dim = request_states[0].latent_history.shape[-1]
        initial_noise = torch.cat(
            [
                (
                    torch.zeros((1, latent_dim, 1), device=self.device)
                    if middle_chunk
                    else torch.randn(
                        (1, latent_dim, 1),
                        generator=request_state.noise_generator,
                        device=self.device,
                        out=request_state.initial_noise_buffer,
                    )
                )
                for request_state, middle_chunk in zip(
                    request_states, is_middle_chunk, strict=True
                )
            ]
        )
        solver_noise = torch.cat(
            [
                (
                    torch.zeros((FLOW_STEPS - 1, 1, 1, latent_dim), device=self.device)
                    if middle_chunk
                    else torch.randn(
                        (FLOW_STEPS - 1, 1, 1, latent_dim),
                        generator=request_state.noise_generator,
                        device=self.device,
                        out=request_state.solver_noise_buffer,
                    )
                )
                for request_state, middle_chunk in zip(
                    request_states, is_middle_chunk, strict=True
                )
            ],
            dim=1,
        )
        # note (rsi): one pinned non-blocking copy carries the per-row latent
        # count and temperature, so the step never stalls on a pageable copy.
        step_inputs = self.pinned_pingpong(
            "step_input_host_bufs",
            "step_input_host_slot",
            (len(request_states), 2),
            torch.float32,
            realloc_on_grow=True,
        )[: len(request_states)]
        step_inputs.copy_(
            torch.tensor(
                [
                    [
                        request_state.launched_latents + 1,
                        request_state.sampling_temperature,
                    ]
                    for request_state in request_states
                ],
                dtype=torch.float32,
            )
        )
        step_inputs = step_inputs.to(self.device, non_blocking=True)
        acoustic_step = self.model.sample_acoustic_step(
            AcousticInputs(
                hidden_states=result.logits_output.hidden_states,
                latent_history=torch.stack(
                    [request_state.latent_history for request_state in request_states]
                ),
                initial_noise=initial_noise,
                solver_noise=solver_noise,
                temperatures=step_inputs[:, 1],
            )
        )
        stop_policy = self.stop_policy
        is_stopped = (
            acoustic_step.stop_probabilities > stop_policy.stop_probability_threshold
        ) & (step_inputs[:, 0] >= stop_policy.minimum_decode_steps)
        result.next_token_ids = torch.where(
            is_stopped, stop_policy.audio_end_token_id, stop_policy.audio_patch_token_id
        )
        for row, (request_state, middle_chunk) in enumerate(
            zip(request_states, is_middle_chunk, strict=True)
        ):
            if middle_chunk:
                continue
            else:
                pass
            request_state.latent_history = acoustic_step.latents[row]
            request_state.next_feedback = acoustic_step.feedback_embeddings[row]
            request_state.launched_latents += 1
        return LaunchedAcousticStep(
            latents=acoustic_step.latents,
            feedback_embeddings=acoustic_step.feedback_embeddings,
            is_middle_chunk=is_middle_chunk,
            host_token_ids=None,
        )

    def commit_acoustic_step(
        self,
        launched: LaunchedAcousticStep,
        requests: list[SchedulerRequest],
        rows: Iterable[int],
    ) -> None:
        """Append the launched latents and feedback rows of the given requests."""
        for row in rows:
            if launched.is_middle_chunk[row]:
                continue
            else:
                pass
            request_state: Ming3TTSRequestData = requests[row].data
            request_state.latents.append(launched.latents[row])
            request_state.feedback_embeddings.append(launched.feedback_embeddings[row])
