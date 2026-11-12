# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 autoregressive hooks and per-request flow execution."""

from __future__ import annotations

import time

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.firered_tts3.embeddings import materialize_decode_embeddings
from sglang_omni.models.firered_tts3.flow_batch import (
    FlowRow,
    build_flow_row,
    execute_flow_rows,
)
from sglang_omni.models.firered_tts3.flow_execution import flow_t_span
from sglang_omni.models.firered_tts3.flow_graph import FlowGraphRunner
from sglang_omni.models.firered_tts3.payload_types import TEXT_TOKEN_BUDGET
from sglang_omni.models.firered_tts3.prefill import (
    materialize_prefill_batch,
    restore_prefill_conditioning,
)
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
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest

# note (kermit): Base/Instruct share these ids; Speaker checkpoints resolve tokens by name.
TEXT_EOT_ID = 151677
AUDIO_SOS_ID = 151669


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
        prefill_launch_at = materialize_prefill_batch(
            self.model,
            forward_batch,
            requests,
            self.active_requests,
            self.suspend_request_data,
            self.clear_request_data,
        )
        if prefill_launch_at is not None:
            self.prefill_launch_at = prefill_launch_at
        else:
            pass

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
        self.model.text_logits_requested = any(
            (request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)
        )
        materialize_decode_embeddings(self.model, forward_batch, requests)

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
        next_token_ids: list[int] = []
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
            flow_state = request_data.flow_state
            assert isinstance(flow_state, FlowState)
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
                restore_prefill_conditioning(request_data, hidden_states_float32)
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
            restore_prefill_conditioning(request_data, hidden_states_float32)
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
        next_token_ids: list[int] = []
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
        text_token_index = request_data.text_token_index
        request_data.text_token_index += 1
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
        if request_data.dual_stream_stage == DS_STAGE_TEXT:
            if request_data.dual_stream_step_index >= text_steps - 1:
                self.run_flow(request_data, hidden_states_float32, flow_rows)
                request_data.dual_stream_stage = DS_STAGE_PATCH
                request_data.dual_stream_step_index = 0
                return CONTROL_TOKEN_ID
            else:
                pass
            request_data.dual_stream_step_index += 1
            return self.next_text_feed(request_data)
        else:
            pass
        if request_data.dual_stream_step_index >= patch_steps - 1:
            request_data.dual_stream_stage = DS_STAGE_TEXT
            request_data.dual_stream_step_index = 0
            if self.stop_predicted(request_data, hidden_states_float32):
                self.mark_stop(request_data, defer=defer_stop)
                return CONTROL_TOKEN_ID
            else:
                pass
            return self.next_text_feed(request_data)
        else:
            pass
        self.run_flow(request_data, hidden_states_float32, flow_rows)
        request_data.dual_stream_step_index += 1
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
        flow_state = request_data.flow_state
        assert isinstance(flow_state, FlowState)
        state = request_data.state
        if (
            self.skip_min_patch_stop
            and self.is_sync_speaker_audio(request_data)
            and (flow_state.patch_count < state.min_patches)
        ):
            return False
        else:
            pass
        assert request_data.probe is not None
        with torch.inference_mode(), request_data.probe.segment("stop_head"):
            stop_logits = model.stop_head(hidden_states_float32[-1:].unsqueeze(0))
            stop_score = float(torch.sigmoid(stop_logits).reshape(-1)[-1].item())
        return (
            stop_score >= state.stop_threshold
            and flow_state.patch_count >= state.min_patches
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
        return build_flow_row(self.model, request_data, hidden_states_float32)

    def run_flow_rows(self, flow_rows: list[FlowRow]) -> None:
        """Group flow rows by Euler step count and write feedback in request order."""
        execute_flow_rows(self.model, self.flow_graphs, flow_rows, self.t_span)

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
        flow_state = request_data.flow_state
        assert flow_state is not None and (
            not isinstance(flow_state, FireRedFlowResume)
        )
        request_data.flow_state = FireRedFlowResume(
            rng_state=flow_state.generator.get_state()
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
