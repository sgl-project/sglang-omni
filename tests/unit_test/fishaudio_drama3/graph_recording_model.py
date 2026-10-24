# SPDX-License-Identifier: Apache-2.0
"""The graph test's Drama 3 model: copies what each forward read and sampled into
fixed buffers, inside any captured graph, and takes its eos logits from one."""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch import nn

from sglang_omni.models.fishaudio_drama3.first_frame_graphs import (
    Drama3FirstFrameGraphs,
    FirstFrameInputs,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.sampling import RAS_WINDOW_FRAMES
from sglang_omni.models.fishaudio_drama3.sampling_kernels import CodebookSampler
from sglang_omni.models.fishaudio_drama3.sglang_model import (
    Drama3ForCausalLM,
    Drama3TowerPrefill,
)
from sglang_omni.models.fishaudio_drama3.tower_kernels import (
    FusedTowerLayers,
    QueryAttention,
    TowerLayerShape,
    TowerLayerWeights,
)
from tests.unit_test.fishaudio_drama3.graph_fixtures import LeanChoice


@dataclass(frozen=True, kw_only=True)
class RecordingBuffers:
    """One row per request: the eos logit the runner stages, then copies of what
    the forward read and sampled."""

    end_of_speech_logits: torch.Tensor
    frames: torch.Tensor
    ras_windows: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    # [rows, 2]: the first semantic draw, then the RAS draw.
    draws: torch.Tensor
    semantic_logits: torch.Tensor
    head_logits: torch.Tensor


@dataclass(kw_only=True)
class TowerPrefillRuns:
    extend_mode: int = 0
    other_modes: int = 0


@dataclass(kw_only=True)
class FusedLayerCalls:
    total: int = 0
    during_tower_prefill: int = 0


@dataclass(frozen=True, kw_only=True)
class CountingFirstFrameGraphs(Drama3FirstFrameGraphs):
    replays_by_request_count: Counter[int]

    def replay(self, inputs: FirstFrameInputs) -> torch.Tensor:
        self.replays_by_request_count[inputs.semantic_logits.shape[0]] += 1
        return super().replay(inputs)


@dataclass(frozen=True, kw_only=True)
class CountingTowerPrefill(Drama3TowerPrefill):
    runs: TowerPrefillRuns
    fused_layer_calls: FusedLayerCalls

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            self.runs.extend_mode += 1
        else:
            self.runs.other_modes += 1
        calls_before = self.fused_layer_calls.total
        super().run(hidden_states, positions, forward_batch)
        self.fused_layer_calls.during_tower_prefill += (
            self.fused_layer_calls.total - calls_before
        )


class CountingFusedTowerLayers(FusedTowerLayers):
    def __init__(
        self,
        *,
        shape: TowerLayerShape,
        rotary_cos_sin: torch.Tensor,
        device: torch.device,
        calls: FusedLayerCalls,
    ) -> None:
        super().__init__(shape=shape, rotary_cos_sin=rotary_cos_sin, device=device)
        self.calls = calls

    def forward_layer(
        self,
        *,
        weights: TowerLayerWeights,
        layer_index: int,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention: QueryAttention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.total += 1
        return super().forward_layer(
            weights=weights,
            layer_index=layer_index,
            hidden_states=hidden_states,
            code_ids=code_ids,
            label_is_semantic=label_is_semantic,
            shared_input=shared_input,
            positions=positions,
            cache_slots=cache_slots,
            key_cache=key_cache,
            value_cache=value_cache,
            attention=attention,
        )


class AttentionBreak(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        save_kv_cache: bool,
        layer_id: int,
        *,
        use_mha_companion: bool,
        key_value_num_tokens: int | None,
    ) -> None: ...


class CaptureBreakRecorder:
    def __init__(self, attention_break: AttentionBreak) -> None:
        self.attention_break = attention_break
        self.layer_ids_by_token_count: dict[int, list[int]] = {}

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        save_kv_cache: bool,
        layer_id: int,
        *,
        use_mha_companion: bool,
        key_value_num_tokens: int | None,
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            self.layer_ids_by_token_count.setdefault(query.shape[0], []).append(
                layer_id
            )
        else:
            pass
        self.attention_break(
            query,
            key,
            value,
            output,
            save_kv_cache,
            layer_id,
            use_mha_companion=use_mha_companion,
            key_value_num_tokens=key_value_num_tokens,
        )


class EndOfSpeechControlledLogits(nn.Module):
    """Wraps the logits processor: each row's eos logit comes from a fixed buffer."""

    def __init__(self, logits_processor: LogitsProcessor, eos_token_id: int) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.eos_token_id = eos_token_id
        self.recording_buffers: RecordingBuffers | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        output = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        recording_buffers = self.recording_buffers
        assert recording_buffers is not None
        logits = output.next_token_logits
        row_count = logits.shape[0]
        logits[:, self.eos_token_id] = recording_buffers.end_of_speech_logits[
            :row_count
        ]
        recording_buffers.semantic_logits[:row_count].copy_(logits)
        return output


class GraphRecordingModel(Drama3ForCausalLM):
    """Drama3ForCausalLM that copies its reads and draws into RecordingBuffers."""

    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.end_of_speech_logits = EndOfSpeechControlledLogits(
            self.logits_processor, config.eos_token_id
        )
        self.logits_processor = self.end_of_speech_logits
        self.recording_buffers: RecordingBuffers | None = None
        self.draw_column = 0
        self.first_frame_sampler_calls = 0
        self.first_frame_replays: Counter[int] = Counter()
        self.tower_prefill_runs = TowerPrefillRuns()
        self.fused_layer_calls = FusedLayerCalls()
        for head_index, head in enumerate(self.audio_decoder.codebook_heads):
            head.register_forward_hook(
                functools.partial(self.keep_head_logits, head_index)
            )

    def allocate_recording_buffers(self, row_count: int, device: torch.device) -> None:
        decoder_config = self.config.audio_decoder_config
        recording_buffers = RecordingBuffers(
            end_of_speech_logits=torch.zeros(row_count, device=device),
            frames=torch.zeros(
                (row_count, 1 + decoder_config.num_codebooks),
                dtype=torch.int64,
                device=device,
            ),
            ras_windows=torch.zeros(
                (row_count, RAS_WINDOW_FRAMES), dtype=torch.int64, device=device
            ),
            temperature=torch.zeros(row_count, device=device),
            top_k=torch.zeros(row_count, dtype=torch.int64, device=device),
            top_p=torch.zeros(row_count, device=device),
            draws=torch.zeros((row_count, 2), dtype=torch.int64, device=device),
            semantic_logits=torch.zeros(
                (row_count, self.config.text_config.vocab_size), device=device
            ),
            head_logits=torch.zeros(
                (
                    row_count,
                    decoder_config.num_codebooks - 1,
                    decoder_config.vocab_size,
                ),
                device=device,
            ),
        )
        self.recording_buffers = recording_buffers
        self.end_of_speech_logits.recording_buffers = recording_buffers

    def keep_head_logits(
        self,
        head_index: int,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        logits: torch.Tensor,
    ) -> None:
        """Forward-hook contract; the decode step calls the heads in order."""
        recording_buffers = self.recording_buffers
        assert recording_buffers is not None
        recording_buffers.head_logits[: logits.shape[0], head_index].copy_(logits)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        self.draw_column = 0
        output = super().forward(input_ids, positions, forward_batch)
        recording_buffers = self.recording_buffers
        decode_buffers = self.decode_buffers
        assert recording_buffers is not None and decode_buffers is not None
        if forward_batch.forward_mode == ForwardMode.DECODE:
            row_count = input_ids.shape[0]
            for staged, recorded in (
                (decode_buffers.frames, recording_buffers.frames),
                (decode_buffers.ras_windows, recording_buffers.ras_windows),
                (decode_buffers.temperature, recording_buffers.temperature),
                (decode_buffers.top_k, recording_buffers.top_k),
                (decode_buffers.top_p, recording_buffers.top_p),
            ):
                recorded[:row_count].copy_(staged[:row_count])
        else:
            pass
        return output

    def sample_first_frames(
        self,
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        self.draw_column = 0
        self.first_frame_sampler_calls += 1
        return super().sample_first_frames(
            semantic_logits,
            hidden_states,
            positions,
            input_frames,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            forward_batch,
        )


class TopKTopPSampler(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        sampler: CodebookSampler | None,
        stream_id: int,
    ) -> torch.Tensor: ...


class DrawRecorder:
    """Stands in for sampling.sample_top_k_top_p, which a decode step calls twice
    for the main token: samples, then keeps each draw in the model's buffers."""

    def __init__(self, model: GraphRecordingModel, sample: TopKTopPSampler) -> None:
        self.model = model
        self.sample = sample

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        sampler: CodebookSampler | None,
        stream_id: int,
    ) -> torch.Tensor:
        candidate_columns = self.sample(
            logits, temperature, top_k, top_p, sampler, stream_id
        )
        recording_buffers = self.model.recording_buffers
        column = self.model.draw_column
        assert recording_buffers is not None and column < 2
        recording_buffers.draws[: candidate_columns.shape[0], column].copy_(
            self.model.audio_decoder.main_token_candidate_ids[candidate_columns]
        )
        self.model.draw_column = column + 1
        return candidate_columns


class LeanCapturePolicy(Protocol):
    def __call__(
        self, num_q_heads: int, kv_group_num: int, batch: int, is_mla: bool = False
    ) -> bool: ...


class LeanSequenceLengthGate(Protocol):
    def __call__(
        self,
        num_q_heads: int,
        kv_group_num: int,
        batch: int,
        seq_lens_sum: int | None,
        is_mla: bool = False,
    ) -> bool: ...


class LeanChoiceRecorder:
    def __init__(
        self,
        capture_policy: LeanCapturePolicy,
        sequence_length_gate: LeanSequenceLengthGate,
    ) -> None:
        self.capture_policy = capture_policy
        self.sequence_length_gate = sequence_length_gate
        self.choices: list[LeanChoice] = []

    def record_capture_policy(
        self, num_q_heads: int, kv_group_num: int, batch: int, is_mla: bool = False
    ) -> bool:
        is_lean = self.capture_policy(num_q_heads, kv_group_num, batch, is_mla)
        self.choices.append(
            LeanChoice(policy="capture", batch_size=batch, is_lean=is_lean)
        )
        return is_lean

    def record_sequence_length_gate(
        self,
        num_q_heads: int,
        kv_group_num: int,
        batch: int,
        seq_lens_sum: int | None,
        is_mla: bool = False,
    ) -> bool:
        is_lean = self.sequence_length_gate(
            num_q_heads, kv_group_num, batch, seq_lens_sum, is_mla
        )
        self.choices.append(
            LeanChoice(policy="sequence_length", batch_size=batch, is_lean=is_lean)
        )
        return is_lean
