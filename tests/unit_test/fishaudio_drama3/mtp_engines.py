# SPDX-License-Identifier: Apache-2.0
"""Recorded Drama 3 engine for test_mtp_engine, one per process.

Records the runner's RAS windows but samples with empty windows and a pinned
end-of-speech logit, so every sampled token is a deterministic argmax.
"""

from __future__ import annotations

import argparse
import functools
import logging
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import torch
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.registry import ModelRegistry
from torch import nn

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_drama3.engine_builder import (
    FISH_OMNI_ARCHITECTURE,
    Drama3EngineBuilder,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.model_runner import (
    Drama3ModelRunner,
    request_data_of,
)
from sglang_omni.models.fishaudio_drama3.request_builders import (
    Drama3GenerationInput,
    Drama3GenerationResult,
    Drama3RequestData,
)
from sglang_omni.models.fishaudio_drama3.sampling import EMPTY_RAS_SLOT
from sglang_omni.models.fishaudio_drama3.sglang_model import (
    Drama3ForCausalLM,
    Drama3StepInputs,
)
from sglang_omni.proto.admin import ADMIN_CONTINUE_GENERATION, ADMIN_PAUSE_GENERATION
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import GenerationDefaults
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import GenerationBatchResult, ScheduleBatch
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    EngineFacts,
    EngineRun,
    EngineSpec,
    ExpertLayerFacts,
    ForwardRecord,
    FreePoolCounts,
    LayerQuantMethod,
    LoadFacts,
    MoeCall,
    QuantizableLayerKind,
    RequestOutcome,
    RequestSpec,
    RequestStep,
    Scenario,
    ScenarioOutcome,
)

SCENARIO_TIMEOUT_SECONDS = 300.0
IDLE_TIMEOUT_SECONDS = 30.0
OUTBOX_POLL_SECONDS = 0.05
# Far above any logit the synthetic heads produce, so greedy sampling takes eos.
FORCED_EOS_LOGIT = 1.0e4
SAMPLING_TEMPERATURE = 1.0

logger = logging.getLogger(__name__)


class EosControlledLogits(nn.Module):
    """Wraps the logits processor: eos only at each request's chosen frame."""

    def __init__(
        self,
        logits_processor: LogitsProcessor,
        *,
        eos_token_id: int,
        request_specs: dict[str, RequestSpec],
        prompt_row_counts: dict[str, int],
    ) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.eos_token_id = eos_token_id
        self.request_specs = request_specs
        self.prompt_row_counts = prompt_row_counts
        self.last_logits: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        output = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        logits = output.next_token_logits
        sequence_lengths = forward_batch.seq_lens_cpu.tolist()
        for row, (request_id, sequence_length) in enumerate(
            zip(forward_batch.rids, sequence_lengths, strict=True)
        ):
            frame_index = sequence_length - self.prompt_row_counts[request_id]
            if frame_index == self.request_specs[request_id].end_at_frame:
                logits[row, self.eos_token_id] = FORCED_EOS_LOGIT
            else:
                logits[row, self.eos_token_id] = float("-inf")
        self.last_logits = logits.float().cpu()
        return output


class RecordingDrama3Model(Drama3ForCausalLM):
    """Drama3ForCausalLM that records every forward for the dense reference."""

    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.request_specs: dict[str, RequestSpec] = {}
        self.prompt_row_counts: dict[str, int] = {}
        self.sampled_frame_counts: dict[str, int] = {}
        self.forwards: list[ForwardRecord] = []
        self.eos_controlled_logits = EosControlledLogits(
            self.logits_processor,
            eos_token_id=config.eos_token_id,
            request_specs=self.request_specs,
            prompt_row_counts=self.prompt_row_counts,
        )
        self.logits_processor = self.eos_controlled_logits
        self.backbone_hidden_states: torch.Tensor | None = None
        self.model.register_forward_hook(self.keep_backbone_hidden_states)
        self.head_logits: list[torch.Tensor] = []
        for head in self.audio_decoder.codebook_heads:
            head.register_forward_hook(self.keep_head_logits)
        self.moe_calls: list[MoeCall] = []
        self.unloaded_parameters: list[str] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """NaN-fill every parameter first, so what the checkpoint misses stays NaN."""
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(float("nan"))
        super().load_weights(weights)
        self.unloaded_parameters = [
            name
            for name, parameter in self.named_parameters()
            if bool(parameter.detach().float().isnan().any())
        ]

    def record_moe_calls(self) -> None:
        for module in self.model.modules():
            if isinstance(module, FusedMoE):
                module.register_forward_hook(
                    functools.partial(self.keep_moe_call, module.layer_id),
                    with_kwargs=True,
                )
            else:
                pass

    def keep_moe_call(
        self,
        layer_index: int,
        module: nn.Module,
        args: tuple[torch.Tensor | StandardTopKOutput, ...],
        kwargs: dict[str, torch.Tensor | StandardTopKOutput],
        output: torch.Tensor,
    ) -> None:
        """Forward-hook contract; copies before the shared expert is added in place."""
        hidden_states = (
            kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        )
        topk_output = kwargs["topk_output"] if "topk_output" in kwargs else args[1]
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(topk_output, StandardTopKOutput)
        self.moe_calls.append(
            MoeCall(
                layer_index=layer_index,
                hidden_states=hidden_states.cpu(),
                topk_ids=topk_output.topk_ids.cpu(),
                topk_weights=topk_output.topk_weights.cpu(),
                output=output.cpu(),
            )
        )

    def keep_backbone_hidden_states(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        hidden_states: torch.Tensor,
    ) -> None:
        """Forward-hook contract; only the backbone output is kept."""
        self.backbone_hidden_states = hidden_states

    def keep_head_logits(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        logits: torch.Tensor,
    ) -> None:
        """Forward-hook contract; the decode step calls the heads in order."""
        self.head_logits.append(logits.float().cpu())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        self.head_logits.clear()
        self.moe_calls.clear()
        row_count = input_ids.shape[0]
        decode_buffers = self.decode_buffers
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            staged_inputs = self.step_exchange.staged_inputs
            assert staged_inputs is not None
            self.step_exchange.stage(
                replace(
                    staged_inputs,
                    ras_windows=torch.full_like(
                        staged_inputs.ras_windows, EMPTY_RAS_SLOT
                    ),
                )
            )
        else:
            assert decode_buffers is not None
            staged_inputs = Drama3StepInputs(
                request_ids=forward_batch.rids,
                frames=decode_buffers.frames[:row_count].clone(),
                ras_windows=decode_buffers.ras_windows[:row_count].clone(),
                temperature=decode_buffers.temperature[:row_count].clone(),
                top_k=decode_buffers.top_k[:row_count].clone(),
                top_p=decode_buffers.top_p[:row_count].clone(),
            )
            decode_buffers.ras_windows[:row_count].fill_(EMPTY_RAS_SLOT)
        # The staged frames carry the batch's tokens as their main tokens.
        assert torch.equal(staged_inputs.frames[:, 0], input_ids)
        output = super().forward(input_ids, positions, forward_batch)
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            sampled_frames = self.step_exchange.sampled_frames
            assert sampled_frames is not None
        else:
            assert decode_buffers is not None
            sampled_frames = decode_buffers.sampled_frames[:row_count]
        self.forwards.append(
            self.forward_record(forward_batch, positions, staged_inputs, sampled_frames)
        )
        return output

    def forward_record(
        self,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        staged_inputs: Drama3StepInputs,
        sampled_frames: torch.Tensor,
    ) -> ForwardRecord:
        attention_backend = get_attn_backend()
        kv_pool = attention_backend.token_to_kv_pool
        assert isinstance(kv_pool, HybridLinearKVPool)
        request_slots = attention_backend.req_to_token_pool.req_to_token
        tower_layer_ids = [layer.layer_id for layer in self.tower_attention]
        hidden_states = self.backbone_hidden_states
        logits = self.eos_controlled_logits.last_logits
        assert hidden_states is not None
        assert logits is not None
        is_extend = forward_batch.forward_mode == ForwardMode.EXTEND
        if is_extend:
            row_starts = forward_batch.extend_start_loc.tolist()
            row_counts = forward_batch.extend_seq_lens.tolist()
        else:
            row_starts = list(range(forward_batch.batch_size))
            row_counts = [1] * forward_batch.batch_size
        sequence_lengths = forward_batch.seq_lens_cpu.tolist()
        pool_indices = forward_batch.req_pool_indices.tolist()
        sampled_rows = sampled_frames.tolist()
        # [requests, heads, codebook size]
        head_logits = torch.stack(self.head_logits, dim=1)
        steps = []
        for request_index, request_id in enumerate(forward_batch.rids):
            rows = slice(
                row_starts[request_index],
                row_starts[request_index] + row_counts[request_index],
            )
            written_slots = forward_batch.out_cache_loc[rows]
            sequence_length = sequence_lengths[request_index]
            frame_index = sequence_length - self.prompt_row_counts[request_id]
            self.sampled_frame_counts[request_id] = frame_index + 1
            sampled_frame = sampled_rows[request_index]
            is_last_step = (
                sampled_frame[0] == self.config.eos_token_id
                or frame_index + 1 == self.request_specs[request_id].max_new_tokens
            )
            sequence_slots = request_slots[
                pool_indices[request_index], :sequence_length
            ]
            steps.append(
                RequestStep(
                    request_id=request_id,
                    is_extend=is_extend,
                    positions=positions[rows].tolist(),
                    hidden_states=hidden_states[rows].cpu(),
                    input_frames=staged_inputs.frames[rows].cpu(),
                    semantic_logits=logits[request_index],
                    head_logits=head_logits[request_index],
                    ras_window=staged_inputs.ras_windows[request_index].tolist(),
                    sampled_frame=sampled_frame,
                    request_row=pool_indices[request_index],
                    written_slots=written_slots.tolist(),
                    sequence_slots=sequence_slots.tolist(),
                    written_cache=tower_cache(kv_pool, tower_layer_ids, written_slots),
                    sequence_cache=(
                        tower_cache(kv_pool, tower_layer_ids, sequence_slots)
                        if is_last_step
                        else None
                    ),
                )
            )
        return ForwardRecord(
            forward_index=len(self.forwards),
            is_extend=is_extend,
            steps=steps,
            moe_calls=list(self.moe_calls),
        )


def tower_cache(
    kv_pool: HybridLinearKVPool, tower_layer_ids: list[int], slots: torch.Tensor
) -> torch.Tensor:
    """[tower layers, slots, 2 (key, value), kv heads * head_dim] copied off the pool."""
    return torch.stack(
        [
            torch.stack(
                (
                    kv_pool.get_key_buffer(layer_id)[slots].flatten(1),
                    kv_pool.get_value_buffer(layer_id)[slots].flatten(1),
                ),
                dim=1,
            )
            for layer_id in tower_layer_ids
        ]
    ).cpu()


def load_facts(model: RecordingDrama3Model) -> LoadFacts:
    """How the loaded model quantizes and runs each layer, and what it left unloaded."""
    quant_methods = {}
    expert_layers = {}
    for name, module in model.named_modules():
        layer_kind: QuantizableLayerKind
        if isinstance(module, LinearBase):
            layer_kind = "linear"
        elif isinstance(module, VocabParallelEmbedding):
            layer_kind = "embedding"
        elif isinstance(module, FusedMoE):
            layer_kind = "routed_experts"
        else:
            continue
        quant_methods[name] = LayerQuantMethod(
            layer_kind=layer_kind, quant_method=type(module.quant_method).__name__
        )
        if isinstance(module, FusedMoE):
            method = module.quant_method
            if isinstance(method, Fp8MoEMethod):
                quant_info = method.get_triton_quant_info(module)
                uses_fp8_w8a8 = quant_info.use_fp8_w8a8
                block_shape = quant_info.block_shape
            else:
                uses_fp8_w8a8 = False
                block_shape = None
            expert_layers[module.layer_id] = ExpertLayerFacts(
                quant_method=type(method).__name__,
                runner_backend=method.runner.runner_backend.name,
                uses_fp8_w8a8=uses_fp8_w8a8,
                block_shape=block_shape,
                fused_shared_expert_count=module.num_fused_shared_experts,
                tensors={
                    parameter_name: parameter.detach().cpu()
                    for parameter_name, parameter in module.named_parameters()
                },
            )
        else:
            pass
    return LoadFacts(
        unloaded_parameters=model.unloaded_parameters,
        quant_methods=quant_methods,
        parameter_dtypes={
            name: parameter.dtype for name, parameter in model.named_parameters()
        },
        expert_layers=expert_layers,
        is_deep_gemm_enabled=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM,
        device_capability=torch.cuda.get_device_capability(),
    )


class LookaheadRecordingRunner(Drama3ModelRunner):
    model: RecordingDrama3Model

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.lookahead_forward_indices: list[int] = []
        self.unresolved_forward_indices: list[int] = []
        self.discarded_steps: list[tuple[int, str]] = []

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        launch_buffer = super().post_decode_launch(result, forward_batch, requests)
        forward_index = len(self.model.forwards) - 1
        self.lookahead_forward_indices.append(forward_index)
        self.unresolved_forward_indices.append(forward_index)
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
        for request in requests:
            sglang_request = request_data_of(request).req
            assert sglang_request is not None
            if sglang_request.finished() or self.req_is_retracted(sglang_request):
                self.discarded_steps.append((forward_index, request.request_id))
            else:
                pass
        super().post_decode_resolve(
            launch_buf, result, forward_batch, schedule_batch, requests
        )
        self.unresolved_forward_indices.pop(0)


class RecordingEngineBuilder(Drama3EngineBuilder):
    """The production builder serving RecordingDrama3Model with the spec's knobs."""

    def __init__(self, *, spec: EngineSpec) -> None:
        super().__init__(
            context_length=spec.context_length,
            max_running_requests=spec.max_running_requests,
            enable_async_decode=spec.enable_async_decode,
            async_decode_min_batch_size=spec.async_decode_min_batch_size,
        )
        self.spec = spec

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        super().pre_infra_setup(checkpoint_dir)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = RecordingDrama3Model

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        # The recording model copies to the host inside its forward, so it runs eager.
        return {
            **super().generation_defaults(dtype=dtype),
            "disable_cuda_graph": True,
            "max_prefill_tokens": self.spec.max_prefill_tokens,
            "max_total_tokens": self.spec.max_total_tokens,
        }

    def infra_kwargs(self) -> InfrastructureOptions:
        return {"total_gpu_memory_fraction": self.spec.total_gpu_memory_fraction}

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> LookaheadRecordingRunner:
        return LookaheadRecordingRunner(model_worker, output_proc)


class SchedulerDriver:
    """Feeds OmniScheduler's inbox and files its outbox per request."""

    def __init__(
        self,
        scheduler: OmniScheduler[Drama3RequestData],
        model: RecordingDrama3Model,
        runner: LookaheadRecordingRunner,
        prompts: dict[str, list[list[int]]],
    ) -> None:
        self.scheduler = scheduler
        self.model = model
        self.runner = runner
        self.prompts = prompts
        self.outcomes: dict[str, RequestOutcome] = {}
        self.failures: list[BaseException] = []
        # Daemon so a wedged scheduler cannot keep the engine process alive.
        self.thread = threading.Thread(
            target=self.serve, name="omni-scheduler", daemon=True
        )

    def serve(self) -> None:
        try:
            self.scheduler.start()
        except BaseException as error:
            self.failures.append(error)
            raise

    def start(self) -> None:
        self.thread.start()

    def stop_and_drain(self) -> None:
        """Stop the scheduler, then file every message it left in the outbox."""
        self.scheduler.stop()
        self.thread.join(timeout=SCENARIO_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            raise RuntimeError("OmniScheduler thread did not stop")
        elif self.failures:
            raise RuntimeError("OmniScheduler thread failed") from self.failures[0]
        else:
            pass
        while True:
            try:
                message = self.scheduler.outbox.get_nowait()
            except queue.Empty:
                return
            self.record(message)

    def submit(self, request_id: str, request_spec: RequestSpec) -> str:
        prompt_rows = self.prompts[request_spec.prompt_name]
        self.model.request_specs[request_id] = request_spec
        self.model.prompt_row_counts[request_id] = len(prompt_rows)
        self.outcomes[request_id] = RequestOutcome(
            request_id=request_id, request_spec=request_spec
        )
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs=request_spec.prompt_name),
            data=Drama3GenerationInput(
                prompt_rows=torch.tensor(prompt_rows, dtype=torch.int64),
                max_new_tokens=request_spec.max_new_tokens,
                temperature=SAMPLING_TEMPERATURE,
                top_k=1,
                top_p=1.0,
            ),
        )
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )
        return request_id

    def abort(self, request_id: str) -> None:
        self.outcomes[request_id].aborted = True
        self.scheduler.inbox.put(IncomingMessage(request_id=request_id, type="abort"))

    def record(self, message: OutgoingMessage) -> None:
        outcome = self.outcomes[message.request_id]
        if message.type == "result":
            payload = message.data
            assert isinstance(payload, StagePayload)
            result = payload.data
            assert isinstance(result, Drama3GenerationResult)
            outcome.result_frames.append(result.frames.tolist())
        elif message.type == "error":
            outcome.error_messages.append(repr(message.data))
        elif message.type == "stream":
            outcome.stream_message_count += 1
        else:
            raise RuntimeError(f"unexpected outbox message {message.type!r}")

    def pump(self) -> None:
        """File the next outbox message, if one arrives within a poll interval."""
        if self.failures:
            raise RuntimeError("OmniScheduler thread failed") from self.failures[0]
        elif not self.thread.is_alive():
            raise RuntimeError("OmniScheduler thread exited")
        else:
            pass
        try:
            message = self.scheduler.outbox.get(timeout=OUTBOX_POLL_SECONDS)
        except queue.Empty:
            return
        self.record(message)

    def wait_for_results(self, request_ids: list[str]) -> None:
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        while any(
            not self.outcomes[request_id].result_frames
            and not self.outcomes[request_id].error_messages
            for request_id in request_ids
        ):
            if time.monotonic() > deadline:
                raise TimeoutError(f"requests did not finish: {request_ids}")
            else:
                pass
            self.pump()

    def wait_for_frames(self, request_id: str, frame_count: int) -> None:
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        outcome = self.outcomes[request_id]
        while self.model.sampled_frame_counts.get(request_id, 0) < frame_count:
            if outcome.result_frames or outcome.error_messages:
                raise RuntimeError(f"{request_id} ended before {frame_count} frames")
            elif time.monotonic() > deadline:
                raise TimeoutError(f"{request_id} never reached {frame_count} frames")
            else:
                pass
            self.pump()

    def retract_running_requests(self) -> int:
        """Pause with retract, then resume; returns how many requests were retracted."""
        paused = self.scheduler.admin(ADMIN_PAUSE_GENERATION, {"mode": "retract"})
        resumed = self.scheduler.admin(ADMIN_CONTINUE_GENERATION, {})
        paused_data = paused.get("data")
        if not (paused.get("success") and resumed.get("success")):
            raise RuntimeError(f"admin retract failed: {paused} {resumed}")
        elif not isinstance(paused_data, dict):
            raise RuntimeError(f"admin pause returned no data: {paused}")
        else:
            return int(paused_data["num_paused_requests"])

    def free_pool_counts(self) -> FreePoolCounts:
        pool = self.scheduler.req_to_token_pool
        assert isinstance(pool, HybridReqToTokenPool)
        return FreePoolCounts(
            request_row_count=pool.available_size(),
            mamba_slot_count=pool.mamba_allocator.available_size(),
            kv_token_count=self.scheduler.token_to_kv_pool_allocator.available_size(),
        )

    def is_pipeline_drained(self) -> bool:
        return (
            self.scheduler.async_pending is None
            and self.scheduler.previous_pending_decode is None
            and not self.runner.unresolved_forward_indices
        )

    def wait_for_idle(self, baseline: FreePoolCounts) -> FreePoolCounts:
        """Free counts once every slot is back and no launched step is left to
        resolve, or the last counts on timeout."""
        deadline = time.monotonic() + IDLE_TIMEOUT_SECONDS
        free_counts = self.free_pool_counts()
        while (
            free_counts != baseline or not self.is_pipeline_drained()
        ) and time.monotonic() < deadline:
            self.pump()
            free_counts = self.free_pool_counts()
        return free_counts


def run_scenario(
    driver: SchedulerDriver, scenario: Scenario, baseline: FreePoolCounts
) -> ScenarioOutcome:
    first_forward_index = len(driver.model.forwards)
    request_ids: list[str] = []
    frames_at_abort = None
    results_at_abort = None
    retracted_request_count = None
    if scenario.abort_target is None:
        abort_id = None
    else:
        abort_id = driver.submit(f"{scenario.name}-target", scenario.abort_target)
        request_ids.append(abort_id)
    for wave_index, wave in enumerate(scenario.waves):
        wave_ids = []
        for position, request_spec in enumerate(wave):
            request_id = driver.submit(
                f"{scenario.name}-{wave_index}-{position}-{request_spec.prompt_name}",
                request_spec,
            )
            wave_ids.append(request_id)
            if scenario.is_sequential:
                driver.wait_for_results([request_id])
            else:
                pass
        request_ids.extend(wave_ids)
        if abort_id is not None and wave_index == 0:
            driver.wait_for_frames(abort_id, scenario.abort_after_frames)
            frames_at_abort = driver.model.sampled_frame_counts[abort_id]
            results_at_abort = len(driver.outcomes[abort_id].result_frames)
            driver.abort(abort_id)
        else:
            pass
        if scenario.retract_after_frames is not None and wave_index == 0:
            driver.wait_for_frames(wave_ids[0], scenario.retract_after_frames)
            retracted_request_count = driver.retract_running_requests()
        else:
            pass
        driver.wait_for_results(wave_ids)
    idle_free_counts = driver.wait_for_idle(baseline)
    logger.info(f"scenario {scenario.name}: free pools after drain {idle_free_counts}")
    return ScenarioOutcome(
        name=scenario.name,
        request_ids=request_ids,
        first_forward_index=first_forward_index,
        end_forward_index=len(driver.model.forwards),
        idle_free_counts=idle_free_counts,
        frames_at_abort=frames_at_abort,
        results_at_abort=results_at_abort,
        retracted_request_count=retracted_request_count,
    )


def run_engine(spec: EngineSpec) -> EngineRun:
    builder = RecordingEngineBuilder(spec=spec)
    scheduler = builder.build(spec.model_dir, dtype="bfloat16")
    sglang_runner = scheduler.tp_worker.model_runner
    model = sglang_runner.model
    runner = scheduler.model_runner
    assert isinstance(model, RecordingDrama3Model)
    assert isinstance(runner, LookaheadRecordingRunner)
    if spec.record_moe_calls:
        model.record_moe_calls()
    else:
        pass
    kv_pool = sglang_runner.token_to_kv_pool
    request_pool = sglang_runner.req_to_token_pool
    assert isinstance(kv_pool, HybridLinearKVPool)
    assert isinstance(request_pool, HybridReqToTokenPool)
    facts = EngineFacts(
        full_attention_layer_ids=sorted(kv_pool.full_attention_layer_id_mapping),
        mamba_layer_ids=sorted(request_pool.mamba_map),
        kv_layer_count=kv_pool.full_layer_nums,
        kv_head_count=kv_pool.head_num,
        kv_head_dim=kv_pool.head_dim,
        kv_element_bytes=kv_pool.dtype.itemsize,
        chunked_prefill_size=scheduler.chunked_prefill_size,
        max_prefill_tokens=scheduler.max_prefill_tokens,
        max_running_requests=scheduler.max_running_requests,
        enable_async_decode=scheduler.enable_async_decode,
        async_decode_min_batch_size=scheduler.async_decode_min_batch_size,
        async_decode_resolve_before_prefill=(
            scheduler.async_decode_resolve_before_prefill
        ),
    )
    logger.info(f"Drama 3 test engine: {facts}")
    driver = SchedulerDriver(scheduler, model, runner, spec.prompts)
    baseline = driver.free_pool_counts()
    driver.start()
    try:
        scenarios = [
            run_scenario(driver, scenario, baseline) for scenario in spec.scenarios
        ]
    finally:
        driver.stop_and_drain()
    if runner.unresolved_forward_indices:
        raise RuntimeError(
            f"forwards {runner.unresolved_forward_indices} were launched but never "
            f"resolved"
        )
    else:
        pass
    return EngineRun(
        facts=facts,
        load_facts=load_facts(model),
        baseline_free_counts=baseline,
        scenarios=scenarios,
        outcomes=driver.outcomes,
        forwards=model.forwards,
        lookahead_forward_indices=runner.lookahead_forward_indices,
        discarded_steps=runner.discarded_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    spec = EngineSpec.model_validate_json(args.spec.read_text())
    torch.save(run_engine(spec), args.output)


if __name__ == "__main__":
    main()
