# SPDX-License-Identifier: Apache-2.0
"""Engine for test_graph_engine, one per process: Drama 3 with or without decode
CUDA graphs on OmniScheduler, driven in whole waves and recorded by the runner."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import fields
from pathlib import Path

import torch
from sglang.kernels.ops.attention import decode_attention
from sglang.srt.layers import radix_attention
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_drama3 import sampling
from sglang_omni.models.fishaudio_drama3.engine_builder import (
    FISH_OMNI_ARCHITECTURE,
    Drama3EngineBuilder,
)
from sglang_omni.models.fishaudio_drama3.request_builders import (
    Drama3GenerationInput,
    Drama3GenerationResult,
    Drama3RequestData,
)
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.proto.admin import ADMIN_CONTINUE_GENERATION, ADMIN_PAUSE_GENERATION
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import GenerationDefaults
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from tests.unit_test.fishaudio_drama3.graph_fixtures import (
    CUDA_GRAPH_BATCH_SIZES,
    PREFILL_CUDA_GRAPH_TOKEN_COUNTS,
    ExpertPath,
    GraphEngineFacts,
    GraphEngineRun,
    GraphEngineSpec,
    GraphRequestOutcome,
    GraphRequestSpec,
    GraphScenario,
    GraphScenarioOutcome,
)
from tests.unit_test.fishaudio_drama3.graph_recording_model import (
    CaptureBreakRecorder,
    CountingFirstFrameGraphs,
    CountingFusedTowerLayers,
    CountingTowerPrefill,
    DrawRecorder,
    GraphRecordingModel,
    LeanChoiceRecorder,
)
from tests.unit_test.fishaudio_drama3.graph_recording_runner import (
    GraphRecordingRunner,
    changed_slots,
    full_attention_kv_buffers,
)
from tests.unit_test.fishaudio_drama3.mtp_fixtures import FreePoolCounts

SCENARIO_TIMEOUT_SECONDS = 300.0
KV_SENTINEL = 1.0
IDLE_TIMEOUT_SECONDS = 30.0
OUTBOX_POLL_SECONDS = 0.05

logger = logging.getLogger(__name__)


class GraphRecordingEngineBuilder(Drama3EngineBuilder):
    """The production builder serving GraphRecordingModel with the spec's knobs."""

    def __init__(self, *, spec: GraphEngineSpec) -> None:
        super().__init__(
            context_length=spec.context_length,
            max_running_requests=spec.max_running_requests,
            first_frame_cuda_graphs=spec.first_frame_cuda_graphs,
            first_frame_cuda_graph_max_requests=spec.first_frame_cuda_graph_max_requests,
            triton_sampling=spec.triton_sampling,
            tower_fused_layers=spec.tower_fused_layers,
            enable_async_decode=spec.enable_async_decode,
            async_decode_min_batch_size=spec.async_decode_min_batch_size,
        )
        self.spec = spec
        self.capture_break_recorder = CaptureBreakRecorder(
            radix_attention.breakable_unified_attention_with_output
        )
        self.lean_choice_recorder = LeanChoiceRecorder(
            decode_attention.lean_capture_policy,
            decode_attention.lean_decode_seqlen_gate,
        )
        self.first_frame_capture_written_kv_slots: list[int] = []

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        super().pre_infra_setup(checkpoint_dir)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = GraphRecordingModel
        decode_attention.lean_capture_policy = (
            self.lean_choice_recorder.record_capture_policy
        )
        decode_attention.lean_decode_seqlen_gate = (
            self.lean_choice_recorder.record_sequence_length_gate
        )

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            **super().generation_defaults(dtype=dtype),
            "disable_cuda_graph": self.spec.kind == "eager",
            "max_prefill_tokens": self.spec.max_prefill_tokens,
            "max_total_tokens": self.spec.max_total_tokens,
        }

    def infra_kwargs(self) -> InfrastructureOptions:
        return {"total_gpu_memory_fraction": self.spec.total_gpu_memory_fraction}

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        """Also installs the recording buffers and draw recorder before capture."""
        super().setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        model = model_worker.model_runner.model
        assert isinstance(model, GraphRecordingModel)
        model.allocate_recording_buffers(
            model_worker.model_runner.req_to_token_pool.size, torch.device(device)
        )
        sampling.sample_top_k_top_p = DrawRecorder(model, sampling.sample_top_k_top_p)
        radix_attention.breakable_unified_attention_with_output = (
            self.capture_break_recorder
        )
        tower_prefill = model.model.tower_prefill
        assert tower_prefill is not None
        model.model.tower_prefill = CountingTowerPrefill(
            audio_decoder=tower_prefill.audio_decoder,
            tower_attention=tower_prefill.tower_attention,
            input_frames=tower_prefill.input_frames,
            next_frames=tower_prefill.next_frames,
            runs=model.tower_prefill_runs,
            fused_layer_calls=model.fused_layer_calls,
        )
        fused_layers = model.audio_decoder.fused_layers
        if fused_layers is not None:
            model.audio_decoder.fused_layers = CountingFusedTowerLayers(
                shape=fused_layers.shape,
                rotary_cos_sin=fused_layers.rotary_cos_sin,
                device=torch.device(device),
                calls=model.fused_layer_calls,
            )
        else:
            pass

    def post_cuda_graph_setup(
        self, model: Drama3ForCausalLM, server_args: ServerArgs
    ) -> None:
        sglang_runner = self.sglang_runner
        assert sglang_runner is not None
        kv_pool = sglang_runner.token_to_kv_pool
        assert isinstance(kv_pool, HybridLinearKVPool)
        kv_buffers = full_attention_kv_buffers(kv_pool)
        kv_contents = [buffer.clone() for buffer in kv_buffers]
        for buffer in kv_buffers:
            buffer.fill_(KV_SENTINEL)
        kv_snapshot = [buffer.clone() for buffer in kv_buffers]
        super().post_cuda_graph_setup(model, server_args)
        self.first_frame_capture_written_kv_slots = sorted(
            changed_slots(kv_snapshot, kv_buffers, 0)
        )
        for buffer, contents in zip(kv_buffers, kv_contents, strict=True):
            buffer.copy_(contents)
        first_frame_graphs = model.first_frame_graphs
        if first_frame_graphs is not None:
            assert isinstance(model, GraphRecordingModel)
            model.first_frame_graphs = CountingFirstFrameGraphs(
                **{
                    field.name: getattr(first_frame_graphs, field.name)
                    for field in fields(first_frame_graphs)
                },
                replays_by_request_count=model.first_frame_replays,
            )
        else:
            pass

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> GraphRecordingRunner:
        return GraphRecordingRunner(model_worker, output_proc)


class GraphSchedulerDriver:
    """Admits whole waves while the scheduler is paused and files its outbox."""

    def __init__(
        self,
        scheduler: OmniScheduler[Drama3RequestData],
        runner: GraphRecordingRunner,
        prompts: dict[str, list[list[int]]],
    ) -> None:
        self.scheduler = scheduler
        self.runner = runner
        self.prompts = prompts
        self.outcomes: dict[str, GraphRequestOutcome] = {}
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

    def record(self, message: OutgoingMessage) -> None:
        outcome = self.outcomes[message.request_id]
        if message.type == "result":
            payload = message.data
            assert isinstance(payload, StagePayload)
            result = payload.data
            assert isinstance(result, Drama3GenerationResult)
            outcome.result_frames.append(result.frames.tolist())
        elif message.type == "stream":
            frames = message.data
            assert isinstance(frames, torch.Tensor)
            outcome.streamed_frame_chunks.append(frames.T.tolist())
        elif message.type == "error":
            outcome.error_messages.append(repr(message.data))
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

    def submit(self, request_id: str, request_spec: GraphRequestSpec) -> str:
        self.runner.request_specs[request_id] = request_spec
        self.outcomes[request_id] = GraphRequestOutcome(
            request_id=request_id, request_spec=request_spec
        )
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs=request_spec.prompt_name,
                params={"stream": request_spec.is_streaming},
            ),
            data=Drama3GenerationInput(
                prompt_rows=torch.tensor(
                    self.prompts[request_spec.prompt_name], dtype=torch.int64
                ),
                max_new_tokens=request_spec.max_new_tokens,
                temperature=request_spec.temperature,
                top_k=request_spec.top_k,
                top_p=request_spec.top_p,
            ),
        )
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )
        return request_id

    def admit_wave(
        self, scenario: GraphScenario, wave_index: int, wave: list[GraphRequestSpec]
    ) -> list[str]:
        """Queue the whole wave while paused, so its first prefill takes all of it."""
        # Finished requests hold their batch rows until the scheduler's next idle
        # step drops them, and would otherwise split the wave by timing.
        deadline = time.monotonic() + IDLE_TIMEOUT_SECONDS
        while not (
            self.scheduler.running_batch.is_empty()
            and self.scheduler.last_batch is None
            and self.is_pipeline_drained()
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("the scheduler never went idle before a wave")
            else:
                self.pump()
        paused = self.scheduler.admin(ADMIN_PAUSE_GENERATION, {"mode": "in_place"})
        if not paused.get("success"):
            raise RuntimeError(f"admin pause failed: {paused}")
        else:
            pass
        request_ids = [
            self.submit(
                f"{scenario.name}-{wave_index}-{position}-{request_spec.prompt_name}",
                request_spec,
            )
            for position, request_spec in enumerate(wave)
        ]
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        while len(self.scheduler.waiting_queue) < len(request_ids):
            if time.monotonic() > deadline:
                raise TimeoutError(f"wave {request_ids} never reached the queue")
            else:
                self.pump()
        resumed = self.scheduler.admin(ADMIN_CONTINUE_GENERATION, {})
        if not resumed.get("success"):
            raise RuntimeError(f"admin continue failed: {resumed}")
        else:
            return request_ids

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
                self.pump()

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
        self, scenario: GraphScenario, baseline: FreePoolCounts
    ) -> GraphScenarioOutcome:
        # The scheduler is idle between scenarios, so its thread sees these first.
        self.runner.sampling_mode = scenario.sampling_mode
        self.runner.scenario_name = scenario.name
        request_ids: list[str] = []
        for wave_index, wave in enumerate(scenario.waves):
            wave_ids = self.admit_wave(scenario, wave_index, wave)
            self.wait_for_results(
                [
                    request_id
                    for request_id, request_spec in zip(wave_ids, wave, strict=True)
                    if request_spec.abort_after_frames is None
                ]
            )
            request_ids.extend(wave_ids)
        idle_free_counts = self.wait_for_idle(baseline)
        logger.info(
            f"scenario {scenario.name}: free pools after drain {idle_free_counts}"
        )
        return GraphScenarioOutcome(
            name=scenario.name,
            request_ids=request_ids,
            idle_free_counts=idle_free_counts,
        )


def engine_facts(
    scheduler: OmniScheduler[Drama3RequestData],
    model: GraphRecordingModel,
    builder: GraphRecordingEngineBuilder,
) -> GraphEngineFacts:
    sglang_runner = scheduler.tp_worker.model_runner
    attention_backend = sglang_runner.attn_backend
    assert isinstance(attention_backend, HybridLinearAttnBackend)
    full_attention_backend = attention_backend.full_attn_backend
    assert isinstance(full_attention_backend, TritonAttnBackend)
    decode_buffers = model.decode_buffers
    assert decode_buffers is not None
    tower_prefill = model.model.tower_prefill
    assert tower_prefill is not None
    graph_runner = sglang_runner.decode_cuda_graph_runner
    prefill_graph_runner = sglang_runner.prefill_cuda_graph_runner
    kv_pool = sglang_runner.token_to_kv_pool
    assert isinstance(kv_pool, HybridLinearKVPool)
    first_frame_graphs = model.first_frame_graphs
    expert_paths = []
    for module in model.modules():
        if isinstance(module, FusedMoE):
            moe_quant_method = module.quant_method
            if isinstance(moe_quant_method, Fp8MoEMethod):
                quant_info = moe_quant_method.get_triton_quant_info(module)
                uses_fp8_w8a8 = quant_info.use_fp8_w8a8
                block_shape = quant_info.block_shape
            else:
                uses_fp8_w8a8 = False
                block_shape = None
            expert_paths.append(
                ExpertPath(
                    quant_method=type(moe_quant_method).__name__,
                    runner_backend=moe_quant_method.runner.runner_backend.name,
                    uses_fp8_w8a8=uses_fp8_w8a8,
                    block_shape=block_shape,
                )
            )
        else:
            pass
    return GraphEngineFacts(
        decode_graph_batch_sizes=(
            None if graph_runner is None else list(graph_runner.capture_bs)
        ),
        prefill_runner=type(prefill_graph_runner).__name__,
        prefill_capture_token_counts=(
            list(prefill_graph_runner.capture_num_tokens)
            if isinstance(prefill_graph_runner, PrefillCudaGraphRunner)
            else None
        ),
        recorded_kv_layer_ids=sorted(kv_pool.full_attention_layer_id_mapping),
        tower_layer_ids=[layer.layer_id for layer in model.tower_attention],
        decode_buffer_rows=decode_buffers.frames.shape[0],
        request_pool_size=sglang_runner.req_to_token_pool.size,
        kv_pool_size=sglang_runner.token_to_kv_pool.size,
        max_kv_splits=full_attention_backend.max_kv_splits,
        multiprocessor_count=full_attention_backend.device_core_count,
        device_capability=torch.cuda.get_device_capability(),
        expert_paths=expert_paths,
        tower_prefill_runs_before_serving=model.tower_prefill_runs.extend_mode,
        tower_prefill_runs_outside_extend=model.tower_prefill_runs.other_modes,
        tower_prefill_rows=tower_prefill.input_frames.shape[0],
        prefill_capture_break_layer_ids=(
            builder.capture_break_recorder.layer_ids_by_token_count
        ),
        first_frame_request_counts=(
            None if first_frame_graphs is None else sorted(first_frame_graphs.graphs)
        ),
        first_frame_capture_written_kv_slots=(
            builder.first_frame_capture_written_kv_slots
        ),
        lean_attention_override=full_attention_backend.enable_lean_attention,
        startup_lean_choices=list(builder.lean_choice_recorder.choices),
        uses_triton_sampling=model.audio_decoder.sampler is not None,
        uses_fused_tower_layers=model.audio_decoder.fused_layers is not None,
        fused_max_rows=(
            None
            if model.audio_decoder.fused_layers is None
            else model.audio_decoder.fused_layers.max_rows
        ),
        enable_async_decode=scheduler.enable_async_decode,
        async_decode_min_batch_size=scheduler.async_decode_min_batch_size,
        async_decode_resolve_before_prefill=(
            scheduler.async_decode_resolve_before_prefill
        ),
    )


def run_engine(spec: GraphEngineSpec) -> GraphEngineRun:
    builder = GraphRecordingEngineBuilder(spec=spec)
    if spec.kind == "eager":
        scheduler = builder.build(
            spec.model_dir,
            dtype="bfloat16",
            server_args_overrides={"cuda_graph_backend_prefill": "disabled"},
        )
    elif spec.kind == "graph":
        scheduler = builder.build(
            spec.model_dir,
            dtype="bfloat16",
            server_args_overrides={
                "cuda_graph_bs": CUDA_GRAPH_BATCH_SIZES,
                "cuda_graph_max_bs": max(CUDA_GRAPH_BATCH_SIZES),
                "cuda_graph_backend_prefill": "disabled",
            },
        )
    else:
        scheduler = builder.build(
            spec.model_dir,
            dtype="bfloat16",
            server_args_overrides={
                "cuda_graph_bs": CUDA_GRAPH_BATCH_SIZES,
                "cuda_graph_max_bs": max(CUDA_GRAPH_BATCH_SIZES),
                "cuda_graph_backend_prefill": "breakable",
                "cuda_graph_bs_prefill": PREFILL_CUDA_GRAPH_TOKEN_COUNTS,
                "cuda_graph_max_bs_prefill": max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS),
            },
        )
    model = scheduler.tp_worker.model_runner.model
    runner = scheduler.model_runner
    assert isinstance(model, GraphRecordingModel)
    assert isinstance(runner, GraphRecordingRunner)
    runner.abort_inbox = scheduler.inbox
    runner.lean_choice_recorder = builder.lean_choice_recorder
    # Graph capture drew random numbers after the worker seeded them; both
    # engines start serving from the same generator state.
    torch.manual_seed(spec.random_seed)
    model.audio_decoder.sampling_step.zero_()
    facts = engine_facts(scheduler, model, builder)
    logger.info(f"Drama 3 graph test engine ({spec.kind}): {facts}")
    driver = GraphSchedulerDriver(scheduler, runner, spec.prompts)
    baseline = driver.free_pool_counts()
    driver.thread.start()
    try:
        scenarios = [
            driver.run_scenario(scenario, baseline) for scenario in spec.scenarios
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
    for request_id, sampled_count in runner.aborted_after_frames.items():
        driver.outcomes[request_id].aborted_after_frames = sampled_count
    return GraphEngineRun(
        kind=spec.kind,
        facts=facts,
        baseline_free_counts=baseline,
        scenarios=scenarios,
        outcomes=driver.outcomes,
        forwards=runner.forwards,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed_arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    spec = GraphEngineSpec.model_validate_json(parsed_arguments.spec.read_text())
    torch.save(run_engine(spec), parsed_arguments.output)


if __name__ == "__main__":
    main()
