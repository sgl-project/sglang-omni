# SPDX-License-Identifier: Apache-2.0
"""The graph test's Drama 3 runner: stages the test's controls and records every
step after it ran, the same way with or without a CUDA graph."""

from __future__ import annotations

import hashlib
import queue
from collections import Counter
from dataclasses import replace

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker, PrefillCudaGraphUsage
from sglang_omni.models.fishaudio_drama3.model_runner import (
    Drama3ModelRunner,
    request_data_of,
)
from sglang_omni.models.fishaudio_drama3.sampling import EMPTY_RAS_SLOT
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3StepInputs
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import GenerationBatchResult, ScheduleBatch
from tests.unit_test.fishaudio_drama3.graph_fixtures import (
    GraphForwardRecord,
    GraphRequestSpec,
    GraphRequestStep,
    SamplingMode,
    bits,
)
from tests.unit_test.fishaudio_drama3.graph_recording_model import (
    FusedLayerCalls,
    GraphRecordingModel,
    LeanChoiceRecorder,
)

# Far above any logit the synthetic heads produce, so greedy sampling takes eos.
FORCED_END_OF_SPEECH_LOGIT = 1.0e4


def full_attention_kv_buffers(kv_pool: HybridLinearKVPool) -> list[torch.Tensor]:
    return [
        buffer
        for layer_id in sorted(kv_pool.full_attention_layer_id_mapping)
        for buffer in (
            kv_pool.get_key_buffer(layer_id),
            kv_pool.get_value_buffer(layer_id),
        )
    ]


def changed_slots(
    snapshot: list[torch.Tensor], buffers: list[torch.Tensor], slot_dim: int
) -> set[int]:
    changed = torch.zeros(
        buffers[0].shape[slot_dim], dtype=torch.bool, device=buffers[0].device
    )
    for before, after in zip(snapshot, buffers, strict=True):
        differs = bits(before) != bits(after)
        changed |= differs.movedim(slot_dim, 0).flatten(1).any(dim=1)
    return set(changed.nonzero().flatten().tolist())


class GraphRecordingRunner(Drama3ModelRunner):
    """Drama3ModelRunner that stages eos logits, greedy windows and aborts, and
    reads each step's records from the pools and buffers after the step."""

    model: GraphRecordingModel

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.model_worker = tp_worker
        sglang_runner = tp_worker.model_runner
        kv_pool = sglang_runner.token_to_kv_pool
        request_pool = sglang_runner.req_to_token_pool
        assert isinstance(kv_pool, HybridLinearKVPool)
        assert isinstance(request_pool, HybridReqToTokenPool)
        self.kv_pool = kv_pool
        self.request_pool = request_pool
        self.full_attention_layer_ids = sorted(kv_pool.full_attention_layer_id_mapping)
        attention_backend = sglang_runner.attn_backend
        assert isinstance(attention_backend, HybridLinearAttnBackend)
        full_attention_backend = attention_backend.full_attn_backend
        assert isinstance(full_attention_backend, TritonAttnBackend)
        self.full_attention_backend = full_attention_backend
        self.decode_graph_runner = sglang_runner.decode_cuda_graph_runner
        self.request_specs: dict[str, GraphRequestSpec] = {}
        self.sampling_mode: SamplingMode = "greedy"
        self.scenario_name = ""
        self.abort_inbox: queue.Queue[IncomingMessage] | None = None
        self.aborted_after_frames: dict[str, int] = {}
        self.forwards: list[GraphForwardRecord] = []
        self.staged_ras_windows: list[list[int]] = []
        self.staged_prefill_inputs: Drama3StepInputs | None = None
        self.kv_snapshot: list[torch.Tensor] = []
        self.state_snapshot: list[torch.Tensor] = []
        self.usage_snapshot = PrefillCudaGraphUsage()
        self.tower_prefill_runs_snapshot = 0
        self.fused_layer_calls_snapshot = FusedLayerCalls()
        self.lean_choice_recorder: LeanChoiceRecorder | None = None
        self.first_frame_replays_snapshot: Counter[int] = Counter()
        self.first_frame_sampler_calls_snapshot = 0
        self.lean_choice_count_snapshot = 0
        self.unresolved_forward_indices: list[int] = []

    def state_buffers(self) -> list[torch.Tensor]:
        """Every linear-attention state tensor, [layers, slots, ...]."""
        cache = self.request_pool.mamba_pool.mamba_cache
        return [*cache.conv, cache.temporal]

    def sampled_count(self, request: SchedulerRequest) -> int:
        request_data = request_data_of(request)
        return len(request_data.frame_rows) - request_data.prompt_row_count

    def stage_end_of_speech(
        self, forward_batch: ForwardBatch | None, requests: list[SchedulerRequest]
    ) -> None:
        assert forward_batch is not None
        end_of_speech_logits = [
            (
                FORCED_END_OF_SPEECH_LOGIT
                if self.request_specs[request.request_id].end_at_frame
                == sequence_length - request_data_of(request).prompt_row_count
                else float("-inf")
            )
            for sequence_length, request in zip(
                forward_batch.seq_lens_cpu.tolist(), requests, strict=True
            )
        ]
        recording_buffers = self.model.recording_buffers
        assert recording_buffers is not None
        recording_buffers.end_of_speech_logits[: len(requests)].copy_(
            torch.tensor(end_of_speech_logits)
        )

    def take_snapshots(self) -> None:
        self.kv_snapshot = [
            buffer.clone() for buffer in full_attention_kv_buffers(self.kv_pool)
        ]
        self.state_snapshot = [buffer.clone() for buffer in self.state_buffers()]
        usage = self.model_worker.prefill_cuda_graph_usage
        self.usage_snapshot = replace(
            usage, replay_buckets=Counter(usage.replay_buckets)
        )
        tower_prefill_runs = self.model.tower_prefill_runs
        self.tower_prefill_runs_snapshot = (
            tower_prefill_runs.extend_mode + tower_prefill_runs.other_modes
        )
        self.fused_layer_calls_snapshot = replace(self.model.fused_layer_calls)
        self.first_frame_replays_snapshot = Counter(self.model.first_frame_replays)
        self.first_frame_sampler_calls_snapshot = self.model.first_frame_sampler_calls
        assert self.lean_choice_recorder is not None
        self.lean_choice_count_snapshot = len(self.lean_choice_recorder.choices)

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        super().before_prefill(forward_batch, schedule_batch, requests)
        staged_inputs = self.model.step_exchange.staged_inputs
        assert staged_inputs is not None
        self.staged_ras_windows = staged_inputs.ras_windows.tolist()
        if self.sampling_mode == "greedy":
            staged_inputs = replace(
                staged_inputs,
                ras_windows=torch.full_like(staged_inputs.ras_windows, EMPTY_RAS_SLOT),
            )
            self.model.step_exchange.stage(staged_inputs)
        else:
            pass
        self.staged_prefill_inputs = staged_inputs
        self.stage_end_of_speech(forward_batch, requests)
        self.take_snapshots()

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        super().before_decode(
            forward_batch, schedule_batch, requests, is_lookahead=is_lookahead
        )
        row_count = len(requests)
        self.staged_ras_windows = self.decode_buffers.ras_windows[:row_count].tolist()
        if self.sampling_mode == "greedy":
            self.decode_buffers.ras_windows[:row_count].fill_(EMPTY_RAS_SLOT)
        else:
            pass
        self.stage_end_of_speech(forward_batch, requests)
        self.take_snapshots()

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        super().post_prefill(result, forward_batch, schedule_batch, requests)
        assert forward_batch is not None
        self.record_forward(
            result,
            forward_batch,
            requests,
            [request_data_of(request).frame_rows[-1] for request in requests],
            is_extend=True,
            is_lookahead=False,
        )
        self.abort_requests_due(requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        super().post_decode(result, forward_batch, schedule_batch, requests)
        assert forward_batch is not None
        self.record_forward(
            result,
            forward_batch,
            requests,
            [request_data_of(request).frame_rows[-1] for request in requests],
            is_extend=False,
            is_lookahead=False,
        )
        self.abort_requests_due(requests)

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        launch_buffer = super().post_decode_launch(result, forward_batch, requests)
        assert forward_batch is not None
        self.record_forward(
            result,
            forward_batch,
            requests,
            self.decode_buffers.sampled_frames[: len(requests)].tolist(),
            is_extend=False,
            is_lookahead=True,
        )
        self.unresolved_forward_indices.append(len(self.forwards) - 1)
        return launch_buffer

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        forward_index = self.unresolved_forward_indices[0]
        kept_requests = []
        for request in requests:
            sglang_request = request_data_of(request).req
            assert sglang_request is not None
            if sglang_request.finished() or self.req_is_retracted(sglang_request):
                pass
            else:
                kept_requests.append(request)
        kept_request_ids = {request.request_id for request in kept_requests}
        record = self.forwards[forward_index]
        self.forwards[forward_index] = replace(
            record,
            steps=[
                replace(step, is_discarded=step.request_id not in kept_request_ids)
                for step in record.steps
            ],
        )
        super().post_decode_resolve(
            launch_buf, result, forward_batch, schedule_batch, requests
        )
        self.abort_requests_due(kept_requests)
        self.unresolved_forward_indices.pop(0)

    def record_forward(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        sampled_frames: list[list[int]],
        *,
        is_extend: bool,
        is_lookahead: bool,
    ) -> None:
        recording_buffers = self.model.recording_buffers
        assert recording_buffers is not None
        request_count = len(requests)
        if is_extend:
            row_starts = forward_batch.extend_start_loc.tolist()
            row_counts = forward_batch.extend_seq_lens.tolist()
        else:
            row_starts = list(range(request_count))
            row_counts = [1] * request_count
        positions = forward_batch.positions.tolist()
        sequence_lengths = forward_batch.seq_lens_cpu.tolist()
        written_slots = forward_batch.out_cache_loc.tolist()
        request_pool_rows = forward_batch.req_pool_indices.tolist()
        mamba_slots = self.request_pool.translate_mamba_indices(
            self.request_pool.get_mamba_indices(forward_batch.req_pool_indices)
        ).tolist()
        draws = recording_buffers.draws[:request_count].tolist()
        if is_extend:
            staged_inputs = self.staged_prefill_inputs
            assert staged_inputs is not None
            input_frames = staged_inputs.frames.tolist()
            model_ras_windows = staged_inputs.ras_windows.tolist()
            temperature = staged_inputs.temperature.tolist()
            top_k = staged_inputs.top_k.tolist()
            top_p = staged_inputs.top_p.tolist()
            stray_kv_slots = sorted(
                changed_slots(
                    self.kv_snapshot, full_attention_kv_buffers(self.kv_pool), 0
                )
                - set(written_slots)
            )
        else:
            input_frames = recording_buffers.frames[:request_count].tolist()
            model_ras_windows = recording_buffers.ras_windows[:request_count].tolist()
            temperature = recording_buffers.temperature[:request_count].tolist()
            top_k = recording_buffers.top_k[:request_count].tolist()
            top_p = recording_buffers.top_p[:request_count].tolist()
            # The KV pool's slot 0 is where padded graph rows write.
            stray_kv_slots = sorted(
                changed_slots(
                    self.kv_snapshot, full_attention_kv_buffers(self.kv_pool), 0
                )
                - set(written_slots)
                - {0}
            )
        stray_mamba_slots = sorted(
            changed_slots(self.state_snapshot, self.state_buffers(), 1)
            - set(mamba_slots)
        )
        usage = self.model_worker.prefill_cuda_graph_usage
        is_graph_replay = bool(result.can_run_cuda_graph)
        first_frame_replays = dict(
            self.model.first_frame_replays - self.first_frame_replays_snapshot
        )
        first_frame_kv_splits: list[int]
        if not is_extend:
            first_frame_kv_splits = []
        elif first_frame_replays:
            first_frame_kv_splits = (
                self.full_attention_backend.cuda_graph_num_kv_splits[
                    :request_count
                ].tolist()
            )
        else:
            first_frame_kv_splits = (
                self.full_attention_backend.forward_metadata.num_kv_splits.tolist()
            )
        lean_choice_recorder = self.lean_choice_recorder
        assert lean_choice_recorder is not None
        steps = []
        for index, request in enumerate(requests):
            request_data = request_data_of(request)
            forward_rows = slice(
                row_starts[index], row_starts[index] + row_counts[index]
            )
            state = [
                tensor[:, mamba_slots[index]].cpu() for tensor in self.state_buffers()
            ]
            state_digest = hashlib.sha256()
            for tensor in state:
                state_digest.update(bits(tensor.contiguous()).numpy().tobytes())
            steps.append(
                GraphRequestStep(
                    request_id=request.request_id,
                    request_row=request_pool_rows[index],
                    mamba_slot=mamba_slots[index],
                    positions=positions[forward_rows],
                    written_slots=written_slots[forward_rows],
                    input_frames=input_frames[forward_rows],
                    staged_ras_window=self.staged_ras_windows[index],
                    model_ras_window=model_ras_windows[index],
                    temperature=temperature[index],
                    top_k=top_k[index],
                    top_p=top_p[index],
                    first_draw=draws[index][0],
                    ras_draw=draws[index][1],
                    sampled_frame=sampled_frames[index],
                    written_cache=self.written_cache(written_slots[forward_rows]),
                    state_digest=state_digest.hexdigest(),
                    is_state_finite=all(
                        bool(tensor.isfinite().all()) for tensor in state
                    ),
                    final_state=(
                        state
                        if self.is_last_step(
                            sampled_frames[index],
                            sequence_lengths[index] - request_data.prompt_row_count + 1,
                            request.request_id,
                        )
                        else None
                    ),
                    semantic_logits=recording_buffers.semantic_logits[index].cpu(),
                    head_logits=recording_buffers.head_logits[index].cpu(),
                )
            )
        self.forwards.append(
            GraphForwardRecord(
                forward_index=len(self.forwards),
                scenario_name=self.scenario_name,
                is_extend=is_extend,
                is_lookahead=is_lookahead,
                is_graph_replay=is_graph_replay,
                graph_batch_size=(
                    self.decode_graph_runner.bs
                    if is_graph_replay
                    and not is_extend
                    and self.decode_graph_runner is not None
                    else None
                ),
                token_count=len(written_slots),
                prefill_replay_count=usage.replay_count
                - self.usage_snapshot.replay_count,
                prefill_eager_count=usage.standard_eager_count
                - self.usage_snapshot.standard_eager_count,
                prefill_replay_buckets=dict(
                    usage.replay_buckets - self.usage_snapshot.replay_buckets
                ),
                tower_prefill_runs=self.model.tower_prefill_runs.extend_mode
                + self.model.tower_prefill_runs.other_modes
                - self.tower_prefill_runs_snapshot,
                fused_layer_calls=self.model.fused_layer_calls.total
                - self.fused_layer_calls_snapshot.total,
                tower_prefill_fused_layer_calls=(
                    self.model.fused_layer_calls.during_tower_prefill
                    - self.fused_layer_calls_snapshot.during_tower_prefill
                ),
                first_frame_replays=first_frame_replays,
                first_frame_eager_count=self.model.first_frame_sampler_calls
                - self.first_frame_sampler_calls_snapshot,
                first_frame_sampler_calls=self.model.first_frame_sampler_calls
                - self.first_frame_sampler_calls_snapshot,
                first_frame_kv_splits=first_frame_kv_splits,
                generator_offset=torch.cuda.default_generators[
                    forward_batch.out_cache_loc.device.index
                ].get_offset(),
                sampling_step=int(self.model.audio_decoder.sampling_step),
                lean_choices=lean_choice_recorder.choices[
                    self.lean_choice_count_snapshot :
                ],
                steps=steps,
                stray_kv_slots=stray_kv_slots,
                stray_mamba_slots=stray_mamba_slots,
            )
        )

    def written_cache(self, slots: list[int]) -> torch.Tensor:
        """[full-attention layers, slots, 2, kv width] as the pool stores them."""
        slot_indices = torch.tensor(slots, device=self.device)
        return torch.stack(
            [
                torch.stack(
                    (
                        self.kv_pool.get_key_buffer(layer_id)[slot_indices].flatten(1),
                        self.kv_pool.get_value_buffer(layer_id)[slot_indices].flatten(
                            1
                        ),
                    ),
                    dim=1,
                )
                for layer_id in self.full_attention_layer_ids
            ]
        ).cpu()

    def is_last_step(
        self, sampled_frame: list[int], sampled_count: int, request_id: str
    ) -> bool:
        return (
            sampled_frame[0] == self.model.config.eos_token_id
            or sampled_count == self.request_specs[request_id].max_new_tokens
        )

    def abort_requests_due(self, requests: list[SchedulerRequest]) -> None:
        """Abort through the inbox, so the scheduler acts on it at its next step."""
        for request in requests:
            request_id = request.request_id
            sampled_count = self.sampled_count(request)
            if (
                self.request_specs[request_id].abort_after_frames == sampled_count
                and request_id not in self.aborted_after_frames
            ):
                assert self.abort_inbox is not None
                self.aborted_after_frames[request_id] = sampled_count
                self.abort_inbox.put(
                    IncomingMessage(request_id=request_id, type="abort")
                )
            else:
                pass
