# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 autoregressive hooks and per-request flow execution."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.firered_tts3.flow_graph import (
    FlowGraphRunner,
    euler_flow_loop,
    flow_t_span,
)
from sglang_omni.models.firered_tts3.payload_types import TEXT_TOKEN_BUDGET
from sglang_omni.models.firered_tts3.prefill import (
    audio_hidden_row_offset,
    build_base_prefill_embeds,
    build_chatml_prefill_embeds,
    build_replay_rows,
    dual_stream_conditioning_rows,
    text_output_token_count,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    DS_STAGE_PATCH,
    DS_STAGE_TEXT,
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_FINALIZE,
    TEXT_MODE_SOSP,
    TEXT_MODE_TEXT,
    FireRedFlowResume,
    FireRedTTS3SGLangRequestData,
    FlowState,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest

# note (kermit): Base/Instruct share these ids; Speaker checkpoints resolve tokens by name.
TEXT_EOT_ID = 151677
AUDIO_SOS_ID = 151669


@dataclass(kw_only=True)
class FlowRow:
    """One request's inputs to a batched Euler integration."""

    request_data: FireRedTTS3SGLangRequestData
    history_latents: torch.Tensor
    initial_noise: torch.Tensor
    conditioning: torch.Tensor


# note (kermit): Unused hook arguments are required by the shared ModelRunner interface.
class FireRedTTS3ModelRunner(ModelRunner[FireRedTTS3SGLangRequestData]):
    """Own flow execution and request lifecycle through the shared runner hooks."""

    model: FireRedTTS3SGLangModel
    tp_worker: ModelWorker
    output_processor: SGLangOutputProcessor

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        flow_graphs: FlowGraphRunner | None = None,
        flow_batching: bool,
        control_token_fastpath: bool,
        skip_min_patch_stop: bool,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.active_requests: dict[str, FireRedTTS3SGLangRequestData] = {}
        self.t_span_cache: dict[int, torch.Tensor] = {}
        self.prefill_launch_at: float | None = None
        self.flow_graphs: FlowGraphRunner | None = flow_graphs
        self.flow_batching: bool = flow_batching
        self.control_token_fastpath: bool = control_token_fastpath
        self.skip_min_patch_stop: bool = skip_min_patch_stop

    def t_span(self, n_timesteps: int, device: torch.device) -> torch.Tensor:
        """Cache the fp32 Euler timestep schedule on the execution device."""
        timestep_schedule = self.t_span_cache.get(n_timesteps)
        if timestep_schedule is None or timestep_schedule.device != device:
            timestep_schedule = flow_t_span(n_timesteps, device)
            self.t_span_cache[n_timesteps] = timestep_schedule
        else:
            pass
        return timestep_schedule

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Materialize prompt and replay embeddings with per-request flow state."""
        if not requests:
            return
        else:
            pass
        assert forward_batch is not None
        self.release_retracted_flow_states()
        model = self.model
        device = forward_batch.input_ids.device
        embedding_dtype = next(model.parameters()).dtype
        model.text_logits_requested = any(
            (request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)
        )
        embedding_rows = []
        materialized_requests = []
        try:
            for request in requests:
                request_data = request.data
                if request_data.flow_state is not None and (
                    not isinstance(request_data.flow_state, FireRedFlowResume)
                ):
                    self.suspend_request_data(request_data)
                else:
                    pass
                self.active_requests.pop(request.request_id, None)
                resume = request_data.flow_state
                state = request_data.state
                if state.variant == "speaker":
                    if (
                        state.prompt_latents is not None
                        or state.speaker_embedding is not None
                    ):
                        raise RuntimeError(
                            "FireRedTTS3 speaker-variant requests carry no reference-conditioning tensors (prompt_latents / speaker_embedding)"
                        )
                    else:
                        pass
                elif state.variant != "instruct" and (
                    state.prompt_latents is None or state.speaker_embedding is None
                ):
                    raise RuntimeError(
                        "FireRedTTS3 request is missing its reference-conditioning tensors (prompt_latents / speaker_embedding)"
                    )
                else:
                    pass
                prompt_latents = (
                    state.prompt_latents.to(device=device, dtype=torch.float32)
                    if state.prompt_latents is not None
                    else None
                )
                speaker_embedding = (
                    state.speaker_embedding.to(device=device, dtype=torch.float32)
                    if state.speaker_embedding is not None
                    else None
                )
                generator = torch.Generator(device=device)
                if (
                    isinstance(resume, FireRedFlowResume)
                    and resume.rng_state is not None
                ):
                    generator.set_state(resume.rng_state)
                elif state.seed is not None:
                    generator.manual_seed(int(state.seed))
                else:
                    generator.manual_seed(
                        derive_sampling_seed("fireredtts3-flow", request.request_id)
                    )
                if prompt_latents is not None:
                    latent_history = F.pad(
                        prompt_latents, (0, 0, model.history_length, 0)
                    )
                else:
                    latent_history = torch.zeros(
                        1, model.history_length, model.redae_dim, device=device
                    )
                if request_data.latent_patches:
                    latent_history = torch.cat(
                        [latent_history, *request_data.latent_patches], dim=1
                    )
                else:
                    pass
                request_data.flow_state = FlowState(
                    generator=generator,
                    prompt_patch_count=request_data.prompt_patch_count,
                    speaker_conditioning=(
                        model.spk_proj_dit(speaker_embedding)
                        if model.spk_proj_dit is not None
                        else None
                    ),
                    backbone_conditioning=torch.zeros(
                        1, model.history_patches, model.llm_hidden_size, device=device
                    ),
                    latent_history=latent_history,
                    patch_count=len(request_data.latent_patches),
                )
                request_data.probe = ProfilingProbe.create()
                self.active_requests[request.request_id] = request_data
                materialized_requests.append((request.request_id, request_data))
                with (
                    torch.inference_mode(),
                    request_data.probe.segment("prefill_embeds"),
                ):
                    if state.variant == "base":
                        prompt_embeddings = build_base_prefill_embeds(
                            model, state, prompt_latents, speaker_embedding, device
                        )
                    else:
                        prompt_embeddings = build_chatml_prefill_embeds(
                            model, state, device
                        )
                    if prompt_embeddings.shape[1] != request_data.prefill_len:
                        raise RuntimeError(
                            f"FireRedTTS3 prefill row is {prompt_embeddings.shape[1]} positions, expected {request_data.prefill_len}"
                        )
                    else:
                        pass
                    request_embeddings = [prompt_embeddings[0].to(embedding_dtype)]
                    replay_embeddings = build_replay_rows(
                        model, request_data, device, embedding_dtype
                    )
                    if replay_embeddings is not None:
                        request_embeddings.append(replay_embeddings)
                    else:
                        pass
                prefix_token_count = request_data.req.prefix_len
                request_token_count = request_data.req.extend_len
                generated_token_count = request_token_count - request_data.prefill_len
                assert (
                    prefix_token_count == 0
                ), "FireRedTTS3 radix prefix reuse is disabled"
                if request_data.dual_stream:
                    patch_feeds = sum(
                        (
                            1
                            for token_id in request_data.req.output_ids
                            if int(token_id) == CONTROL_TOKEN_ID
                        )
                    )
                    excess = patch_feeds - len(request_data.latent_patches)
                    assert (
                        generated_token_count == len(request_data.req.output_ids)
                        and excess >= 0
                        and (excess == 0 or request_data.stop_pending)
                    ), "FireRedTTS3 dual-stream re-prefill history is inconsistent"
                else:
                    excess = (
                        len(request_data.req.output_ids)
                        - text_output_token_count(request_data)
                        - len(request_data.latent_patches)
                    )
                    assert generated_token_count == len(
                        request_data.req.output_ids
                    ) and (
                        excess >= 0 and (excess == 0 or request_data.stop_pending)
                    ), "FireRedTTS3 re-prefill history must match generated tokens"
                embedding_rows.append(torch.cat(request_embeddings, dim=0))
            forward_batch.input_embeds = torch.cat(embedding_rows, dim=0)
            if any(
                (
                    request_data.probe.enabled
                    for request_id, request_data in materialized_requests
                )
            ):
                self.prefill_launch_at = time.perf_counter()
            else:
                pass
        except BaseException:
            for request_id, request_data in materialized_requests:
                self.active_requests.pop(request_id, None)
                self.clear_request_data(request_data)
            raise

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Stage feedback and text embeddings in the upcoming batch row order."""
        if not requests:
            return
        else:
            pass
        assert forward_batch is not None
        model = self.model
        model.text_logits_requested = any(
            (request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)
        )
        embedding_rows = []
        for request_row_index, request in enumerate(requests):
            request_data = request.data
            feed_is_patch = (
                not request_data.dual_stream
                and request_data.text_ar_mode == TEXT_MODE_AUDIO
                or (
                    request_data.dual_stream and request_data.ds_stage == DS_STAGE_PATCH
                )
            )
            if feed_is_patch:
                feedback_queue = request_data.pending_feedback_queue
                if feedback_queue:
                    embedding_rows.append(feedback_queue.popleft())
                elif request_data.stop_pending:
                    # note (kermit): Deferred stop discards this feed; decode still owns KV cleanup.
                    embedding_rows.append(
                        torch.zeros(
                            model.llm_hidden_size,
                            device=forward_batch.input_ids.device,
                            dtype=next(model.parameters()).dtype,
                        )
                    )
                else:
                    raise RuntimeError(
                        "FireRedTTS3 decode is missing its latent feedback"
                    )
            else:
                token_id = int(forward_batch.input_ids[request_row_index])
                embedding_rows.append(
                    model.get_input_embeddings()(
                        torch.tensor(
                            [token_id],
                            dtype=torch.long,
                            device=forward_batch.input_ids.device,
                        )
                    )[0]
                )
        feedback_buffer = model.graph_feedback_buffer
        if feedback_buffer is not None:
            # note (kermit): Decode graphs read this persistent buffer rather than input_embeds.
            torch.stack(embedding_rows, out=feedback_buffer[: len(embedding_rows)])
        else:
            forward_batch.input_embeds = torch.stack(embedding_rows).to(
                device=forward_batch.input_ids.device,
                dtype=next(model.parameters()).dtype,
            )

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch | None, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Preserve all prompt hidden rows for flow history reconstruction."""
        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch | None, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Match the FULL mode baked into a feedback-buffer decode graph."""
        if self.model.graph_feedback_buffer is not None:
            return CaptureHiddenMode.FULL
        else:
            pass
        return CaptureHiddenMode.LAST

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        return any(
            (request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)
        )

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        return any(
            (request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)
        )

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Resume prefill tails while deferring finish until decode owns KV cleanup."""
        if schedule_batch is not None and schedule_batch.is_prefill_only:
            return
        else:
            pass
        if not requests:
            return
        else:
            pass
        if self.prefill_launch_at is not None:
            engine_prefill = time.perf_counter() - self.prefill_launch_at
            self.prefill_launch_at = None
            for request in requests:
                probe = request.data.probe
                if probe is not None and probe.enabled:
                    probe.segments["engine_prefill"] = engine_prefill
                else:
                    pass
        else:
            pass
        hidden_states = self.hidden_states(result)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        elif hidden_states.ndim != 2:
            raise RuntimeError(
                f"FireRedTTS3 expected rank-2/3 prefill hidden, got {hidden_states.ndim}"
            )
        else:
            pass
        hidden_row_offset = 0
        next_token_ids = []
        flow_rows: list[FlowRow] | None = [] if self.flow_batching else None
        for request_row_index, request in enumerate(requests):
            request_data = request.data
            request_token_count = request_data.req.extend_len
            request_hidden = hidden_states[
                hidden_row_offset : hidden_row_offset + request_token_count
            ]
            if request_hidden.size(0) != request_token_count:
                raise RuntimeError("FireRedTTS3 prefill hidden rows are incomplete")
            else:
                pass
            hidden_row_offset += request_token_count
            hidden_states_float32 = request_hidden.to(torch.float32)
            if request_data.stop_pending:
                next_token_ids.append(CONTROL_TOKEN_ID)
                continue
            else:
                pass
            flow = request_data.flow_state
            assert isinstance(flow, FlowState)
            text_feed_token_id = self.advance_text_phase(
                request_data, result, request_row_index
            )
            if text_feed_token_id is not None:
                next_token_ids.append(text_feed_token_id)
                continue
            else:
                pass
            if request_data.dual_stream:
                if not request_data.req.output_ids:
                    next_token_ids.append(self.next_text_feed(request_data))
                    continue
                else:
                    pass
                if flow.patch_count > 0:
                    flow.backbone_conditioning = torch.cat(
                        [
                            flow.backbone_conditioning,
                            dual_stream_conditioning_rows(
                                request_data, hidden_states_float32
                            ),
                        ],
                        dim=1,
                    )
                else:
                    pass
                next_token_ids.append(
                    self.dual_stream_step(
                        request_data,
                        hidden_states_float32[-1:],
                        flow_rows,
                        defer_stop=True,
                    )
                )
                continue
            else:
                pass
            if flow.patch_count > 0:
                prompt_hidden_start = request_data.prefill_len - flow.prompt_patch_count
                audio_hidden_start = request_data.prefill_len + audio_hidden_row_offset(
                    request_data
                )
                conditioning_parts = [
                    flow.backbone_conditioning,
                    hidden_states_float32[
                        prompt_hidden_start : request_data.prefill_len
                    ].unsqueeze(0),
                    hidden_states_float32[audio_hidden_start:-1].unsqueeze(0),
                ]
                flow.backbone_conditioning = torch.cat(conditioning_parts, dim=1)
            else:
                pass
            self.flow_step(
                request_data, hidden_states_float32, flow_rows, defer_stop=True
            )
            next_token_ids.append(CONTROL_TOKEN_ID)
        self.finish_step(
            result,
            next_token_ids,
            hidden_states,
            requests,
            flow_rows,
            consumed_prefill_rows=hidden_row_offset,
        )

    def is_sync_speaker_audio(self, request_data: FireRedTTS3SGLangRequestData) -> bool:
        return (
            not self.async_enabled
            and request_data.state.variant == "speaker"
            and (not request_data.dual_stream)
            and (request_data.text_ar_mode == TEXT_MODE_AUDIO)
        )

    def make_next_token_ids(
        self,
        next_token_ids: list[int],
        device: torch.device,
        requests: list[SchedulerRequest],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Separate known reporting tokens from device control-token inputs."""
        if (
            self.control_token_fastpath
            and all((self.is_sync_speaker_audio(request.data) for request in requests))
            and all((token_id == CONTROL_TOKEN_ID for token_id in next_token_ids))
        ):
            return (
                torch.full(
                    (len(next_token_ids),),
                    CONTROL_TOKEN_ID,
                    dtype=torch.long,
                    device=device,
                ),
                torch.tensor(next_token_ids, dtype=torch.long, device="cpu"),
            )
        else:
            return (torch.tensor(next_token_ids, dtype=torch.long, device=device), None)

    def record_text_sample(
        self, request_data: FireRedTTS3SGLangRequestData, sampled: torch.Tensor
    ) -> int:
        """Retain sampled text, excluding EOT, before the audio transition feeds."""
        token_id = int(sampled)
        request_data.text_fed += 1
        if token_id == TEXT_EOT_ID:
            request_data.eot_sampled = True
            request_data.text_ar_mode = TEXT_MODE_FINALIZE
            return token_id
        else:
            pass
        request_data.text_token_ids.append(token_id)
        if request_data.text_fed >= TEXT_TOKEN_BUDGET:
            request_data.text_ar_mode = TEXT_MODE_FINALIZE
        else:
            pass
        return token_id

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.run_post_decode(result, requests)

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Run flow at launch and defer finish until the scheduler resolves the step."""
        self.run_post_decode(result, requests, defer_stop=True)

    def post_decode_resolve(
        self,
        launch_buf: None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        """Apply deferred stops only to requests still owned by this batch."""
        for request in requests:
            self.resolve_pending_stop(request.data)

    def run_post_decode(
        self,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
        *,
        defer_stop: bool = False,
    ) -> None:
        """Advance one text or audio feed and collect per-request flow work."""
        if not requests:
            return
        else:
            pass
        hidden_states = self.hidden_states(result)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1]
        elif hidden_states.ndim != 2:
            raise RuntimeError(
                f"FireRedTTS3 expected rank-2/3 decode hidden, got {hidden_states.ndim}"
            )
        else:
            pass
        next_token_ids = []
        flow_rows: list[FlowRow] | None = [] if self.flow_batching else None
        for request_row_index, request in enumerate(requests):
            request_data = request.data
            if request_data.stop_pending:
                next_token_ids.append(CONTROL_TOKEN_ID)
                if not defer_stop:
                    self.resolve_pending_stop(request_data)
                else:
                    pass
                continue
            else:
                pass
            if request_data.dual_stream:
                next_token_ids.append(
                    self.dual_stream_step(
                        request_data,
                        hidden_states[request_row_index : request_row_index + 1].to(
                            torch.float32
                        ),
                        flow_rows,
                        defer_stop=defer_stop,
                    )
                )
                continue
            else:
                pass
            text_feed_token_id = self.advance_text_phase(
                request_data, result, request_row_index
            )
            if text_feed_token_id is not None:
                next_token_ids.append(text_feed_token_id)
            else:
                self.flow_step(
                    request_data,
                    hidden_states[request_row_index : request_row_index + 1].to(
                        torch.float32
                    ),
                    flow_rows,
                    defer_stop=defer_stop,
                )
                next_token_ids.append(CONTROL_TOKEN_ID)
        self.finish_step(result, next_token_ids, hidden_states, requests, flow_rows)

    def next_text_feed(self, request_data: FireRedTTS3SGLangRequestData) -> int:
        """Feed the next text token, extending exhausted text with its padding token."""
        token_ids = request_data.state.text_stream_ids
        text_token_index = request_data.text_pos
        request_data.text_pos += 1
        return (
            int(token_ids[text_token_index])
            if text_token_index < len(token_ids)
            else int(request_data.state.text_pad_id)
        )

    def dual_stream_step(
        self,
        request_data: FireRedTTS3SGLangRequestData,
        hidden_states_float32: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
        *,
        defer_stop: bool = False,
    ) -> int:
        """Interleave text and patch feeds, checking stop after each final patch feed."""
        assert request_data.interleave_ratio is not None
        text_steps, patch_steps = request_data.interleave_ratio
        if request_data.ds_stage == DS_STAGE_TEXT:
            if request_data.ds_index >= text_steps - 1:
                self.run_flow(request_data, hidden_states_float32, flow_rows)
                request_data.ds_stage = DS_STAGE_PATCH
                request_data.ds_index = 0
                return CONTROL_TOKEN_ID
            else:
                pass
            request_data.ds_index += 1
            return self.next_text_feed(request_data)
        else:
            pass
        if request_data.ds_index >= patch_steps - 1:
            request_data.ds_stage = DS_STAGE_TEXT
            request_data.ds_index = 0
            if self.stop_predicted(request_data, hidden_states_float32):
                self.mark_stop(request_data, defer=defer_stop)
                return CONTROL_TOKEN_ID
            else:
                pass
            return self.next_text_feed(request_data)
        else:
            pass
        self.run_flow(request_data, hidden_states_float32, flow_rows)
        request_data.ds_index += 1
        return CONTROL_TOKEN_ID

    def flow_step(
        self,
        request_data: FireRedTTS3SGLangRequestData,
        hidden_states_float32: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
        *,
        defer_stop: bool = False,
    ) -> None:
        """Check stop before generating the next patch in a non-interleaved request."""
        if self.stop_predicted(request_data, hidden_states_float32):
            self.mark_stop(request_data, defer=defer_stop)
            return
        else:
            pass
        self.run_flow(request_data, hidden_states_float32, flow_rows)

    @staticmethod
    def mark_stop(request_data: FireRedTTS3SGLangRequestData, *, defer: bool) -> None:
        """Defer prefill and launch stops; decode may finish immediately."""
        if defer:
            request_data.stop_pending = True
        else:
            request_data.req.finished_reason = FINISH_MATCHED_TOKEN(CONTROL_TOKEN_ID)

    def stop_predicted(
        self,
        request_data: FireRedTTS3SGLangRequestData,
        hidden_states_float32: torch.Tensor,
    ) -> bool:
        """Evaluate the fp32 stop head with the minimum patch floor."""
        model = self.model
        flow = request_data.flow_state
        assert isinstance(flow, FlowState)
        state = request_data.state
        if (
            self.skip_min_patch_stop
            and self.is_sync_speaker_audio(request_data)
            and (flow.patch_count < state.min_patches)
        ):
            return False
        else:
            pass
        assert request_data.probe is not None
        with torch.inference_mode(), request_data.probe.segment("stop_head"):
            stop_logits = model.stop_head(hidden_states_float32[-1:].unsqueeze(0))
            stop_score = float(torch.sigmoid(stop_logits).reshape(-1)[-1].item())
        return (
            stop_score >= state.stop_threshold and flow.patch_count >= state.min_patches
        )

    def run_flow(
        self,
        request_data: FireRedTTS3SGLangRequestData,
        hidden_states_float32: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
    ) -> None:
        """Collect a flow row or run it immediately through the serial path."""
        flow_row = self.prepare_flow_row(request_data, hidden_states_float32)
        if flow_rows is not None:
            flow_rows.append(flow_row)
            return
        else:
            pass
        self.run_flow_rows([flow_row])

    def prepare_flow_row(
        self,
        request_data: FireRedTTS3SGLangRequestData,
        hidden_states_float32: torch.Tensor,
    ) -> FlowRow:
        """Append conditioning and draw noise from this request generator."""
        model = self.model
        flow = request_data.flow_state
        assert isinstance(flow, FlowState)
        assert request_data.probe is not None
        with torch.inference_mode(), request_data.probe.segment("cond"):
            if flow.patch_count == 0 and flow.prompt_patch_count > 0:
                conditioning_rows = hidden_states_float32[-flow.prompt_patch_count :]
            else:
                conditioning_rows = hidden_states_float32[-1:]
            flow.backbone_conditioning = torch.cat(
                [flow.backbone_conditioning, conditioning_rows.unsqueeze(0)], dim=1
            )
            conditioning_window = flow.backbone_conditioning[
                :, -(model.history_patches + 1) :
            ]
            backbone_conditioning = model.dit_head(conditioning_window)
            backbone_conditioning = backbone_conditioning.repeat_interleave(
                model.patch_size, dim=1
            )
            if flow.speaker_conditioning is not None:
                speaker_conditioning = flow.speaker_conditioning.unsqueeze(1).expand(
                    -1, backbone_conditioning.shape[1], -1
                )
                conditioning = torch.cat(
                    [backbone_conditioning, speaker_conditioning], dim=-1
                )
            else:
                conditioning = backbone_conditioning
            history_latents = flow.latent_history[:, -model.history_length :]
            initial_noise = torch.empty(
                1,
                model.patch_size,
                model.redae_dim,
                device=conditioning.device,
                dtype=torch.float32,
            )
            initial_noise.normal_(generator=flow.generator)
        return FlowRow(
            request_data=request_data,
            history_latents=history_latents,
            initial_noise=initial_noise,
            conditioning=conditioning,
        )

    def run_flow_rows(self, flow_rows: list[FlowRow]) -> None:
        """Group flow rows by Euler step count and write feedback in request order."""
        model = self.model
        groups: dict[int, list[FlowRow]] = {}
        for flow_row in flow_rows:
            groups.setdefault(int(flow_row.request_data.state.n_timesteps), []).append(
                flow_row
            )
        feedback_dtype = next(model.parameters()).dtype
        for n_timesteps, flow_group in groups.items():
            with torch.inference_mode():
                history_latents = torch.cat(
                    [flow_row.history_latents for flow_row in flow_group]
                )
                initial_noise = torch.cat(
                    [flow_row.initial_noise for flow_row in flow_group]
                )
                conditioning = torch.cat(
                    [flow_row.conditioning for flow_row in flow_group]
                )
                guidance_scales = torch.tensor(
                    [
                        float(flow_row.request_data.state.inference_cfg)
                        for flow_row in flow_group
                    ],
                    dtype=torch.float32,
                    device=conditioning.device,
                ).view(-1, 1, 1)
                probes = [flow_row.request_data.probe for flow_row in flow_group]
                with ProfilingProbe.shared_segment("dit_flow", probes):
                    graphs = self.flow_graphs
                    patches = (
                        graphs.run(
                            history_latents,
                            initial_noise,
                            conditioning,
                            guidance_scales,
                            n_timesteps,
                        )
                        if graphs is not None
                        else None
                    )
                    if patches is None:
                        patches = euler_flow_loop(
                            model.dit,
                            torch.cat([history_latents, initial_noise], dim=1),
                            conditioning,
                            self.t_span(n_timesteps, conditioning.device),
                            guidance_scales,
                            n_timesteps,
                            model.patch_size,
                        )
                    else:
                        pass
                with ProfilingProbe.shared_segment("patch_encoder", probes):
                    feedbacks = model.patch_encoder(patches)
                for request_row_index, flow_row in enumerate(flow_group):
                    request_data = flow_row.request_data
                    flow = request_data.flow_state
                    patch = patches[request_row_index : request_row_index + 1]
                    flow.latent_history = torch.cat([flow.latent_history, patch], dim=1)
                    flow.patch_count += 1
                    request_data.latent_patches.append(patch)
                    request_data.pending_feedback_queue.append(
                        feedbacks[0, request_row_index].to(dtype=feedback_dtype)
                    )

    @staticmethod
    def hidden_states(result: GenerationBatchResult) -> torch.Tensor:
        """Read the hidden tensor guaranteed by the SGLang forward result contract."""
        logits_output = result.logits_output
        if logits_output is None or not isinstance(
            logits_output.hidden_states, torch.Tensor
        ):
            raise RuntimeError(
                "FireRedTTS3 SGLang forward did not return hidden states"
            )
        else:
            return logits_output.hidden_states

    def on_request_finished(
        self, request_id: str, request_data: FireRedTTS3SGLangRequestData
    ) -> None:
        active_request_data = self.active_requests.pop(request_id, None)
        if active_request_data is not None and active_request_data.probe is not None:
            active_request_data.probe.log(request_id)
        else:
            pass
        self.clear_request_data(request_data)

    def reset_request(self, request_id: str) -> None:
        request_data = self.active_requests.pop(request_id, None)
        if request_data is not None:
            self.clear_request_data(request_data)
        else:
            pass

    def release_retracted_flow_states(self) -> None:
        for request_data in self.active_requests.values():
            if (
                request_data.flow_state is not None
                and (not isinstance(request_data.flow_state, FireRedFlowResume))
                and request_data.req.is_retracted
            ):
                self.suspend_request_data(request_data)
            else:
                pass

    def suspend_request_data(self, request_data: FireRedTTS3SGLangRequestData) -> None:
        """Release temporary flow buffers while retaining generator state."""
        flow = request_data.flow_state
        assert flow is not None and (not isinstance(flow, FireRedFlowResume))
        request_data.flow_state = FireRedFlowResume(
            rng_state=flow.generator.get_state()
        )
        request_data.pending_feedback_queue.clear()

    def clear_request_data(self, request_data: FireRedTTS3SGLangRequestData) -> None:
        request_data.pending_feedback_queue.clear()
        request_data.flow_state = None
        request_data.stop_pending = False

    def advance_text_phase(
        self,
        request_data: FireRedTTS3SGLangRequestData,
        result: GenerationBatchResult,
        request_row_index: int,
    ) -> int | None:
        """Advance a text feed; None hands the current hidden row to audio."""
        if request_data.text_ar_mode == TEXT_MODE_TEXT:
            assert isinstance(result.next_token_ids, torch.Tensor)
            return self.record_text_sample(
                request_data, result.next_token_ids[request_row_index]
            )
        elif request_data.text_ar_mode == TEXT_MODE_FINALIZE:
            request_data.text_ar_mode = TEXT_MODE_SOSP
            return AUDIO_SOS_ID
        # note (kermit): The audio loop consumes a second audio-start feed before patch zero.
        elif request_data.text_ar_mode == TEXT_MODE_SOSP:
            request_data.text_ar_mode = TEXT_MODE_AUDIO_ZERO
            return AUDIO_SOS_ID
        else:
            if request_data.text_ar_mode == TEXT_MODE_AUDIO_ZERO:
                request_data.text_ar_mode = TEXT_MODE_AUDIO
            else:
                pass
            return None

    @staticmethod
    def resolve_pending_stop(request_data: FireRedTTS3SGLangRequestData) -> None:
        """Finish an owned request at decode or resolve, after the stop was deferred."""
        if request_data.stop_pending:
            request_data.stop_pending = False
            if not request_data.req.is_retracted and (not request_data.req.finished()):
                request_data.req.finished_reason = FINISH_MATCHED_TOKEN(
                    CONTROL_TOKEN_ID
                )
            else:
                pass
        else:
            pass

    def finish_step(
        self,
        result: GenerationBatchResult,
        next_token_ids: list[int],
        hidden_states: torch.Tensor,
        requests: list[SchedulerRequest],
        flow_rows: list[FlowRow] | None,
        *,
        consumed_prefill_rows: int | None = None,
    ) -> None:
        """Execute collected flow rows and publish the next token inputs together."""
        if flow_rows:
            self.run_flow_rows(flow_rows)
        else:
            pass
        if (
            consumed_prefill_rows is not None
            and consumed_prefill_rows != hidden_states.size(0)
        ):
            raise RuntimeError("FireRedTTS3 prefill hidden rows do not match requests")
        else:
            pass
        result.next_token_ids, host_token_ids = self.make_next_token_ids(
            next_token_ids, hidden_states.device, requests
        )
        if host_token_ids is not None:
            self.stage_token_ids(result, host_token_ids)
        else:
            pass


__all__ = ["FireRedTTS3ModelRunner", "FlowRow"]
