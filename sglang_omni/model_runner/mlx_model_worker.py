# SPDX-License-Identifier: Apache-2.0
"""Omni adapter around SGLang's native MLX TP worker."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from sglang_omni.model_runner.base import ModelRunner


@dataclass(slots=True)
class _MlxSchedulerPendingStep:
    launch: Any
    reqs: list[Any]
    scheduler_output: Any
    schedule_batch: Any


class MlxSchedulerModelRunner(ModelRunner):
    """Bridge Omni's decode lookahead to SGLang's lazy MLX worker API."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        # note (yexiaodong): The scheduler still owns every pending handle;
        # this reference is only the lazy decode root used to build its successor.
        self._last_mlx_pending: _MlxSchedulerPendingStep | None = None

    def lookahead_eligible(self, batch: Any) -> bool:
        if len(batch.reqs) != 1:
            return False
        previous = self._last_mlx_pending
        if previous is not None:
            previous_ids = [req.rid for req in previous.reqs]
            current_ids = [req.rid for req in batch.reqs]
            if previous.launch.mode != "decode" or previous_ids != current_ids:
                # note (yexiaodong): Returning false makes Omni resolve the
                # in-flight step before it runs a changed batch synchronously.
                return False
        return super().lookahead_eligible(batch)

    def _build_forward_batch(self, scheduler_output: Any):
        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return None
        # note (yexiaodong): SGLang's MLX worker consumes ScheduleBatch
        # directly. Its bookkeeping stub intentionally has no Torch attention
        # backend state from which ForwardBatch could be constructed.
        return None, schedule_batch, bool(schedule_batch.forward_mode.is_extend())

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del requests
        return self.tp_worker.forward_batch_generation(
            batch=schedule_batch,
            forward_batch=forward_batch,
        )

    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del requests
        return self.tp_worker.forward_batch_generation(
            batch=schedule_batch,
            forward_batch=forward_batch,
        )

    def execute_launch(self, scheduler_output: Any):
        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return None
        if not schedule_batch.forward_mode.is_decode():
            raise RuntimeError("MLX lookahead launch requires a decode batch")

        # note (yexiaodong): A batch may carry deferred CPU prefill inputs or a
        # preceding decode token instead of input_ids, so MLX must resolve the
        # same FutureMap contract as SGLang's scheduler.
        if self._execution_bridge is not None:
            from sglang.srt.managers.overlap_utils import resolve_forward_inputs

            resolve_forward_inputs(schedule_batch, self._execution_bridge.future_map)

        reqs = list(schedule_batch.reqs)
        previous = self._last_mlx_pending
        if previous is None:
            launch = self.tp_worker.async_forward_batch_generation_mlx(schedule_batch)
        else:
            previous_ids = [req.rid for req in previous.reqs]
            current_ids = [req.rid for req in reqs]
            if previous.launch.mode != "decode" or previous_ids != current_ids:
                # note (yexiaodong): The scheduler still owns the previous
                # pending step. Keep this reference until resolve so both sides
                # retain the same lazy cache root.
                raise RuntimeError(
                    "MLX chained decode requires an unchanged request batch; "
                    "resolve the outstanding pending step before launching a "
                    "changed batch"
                )
            launch = self.tp_worker.async_chained_decode_mlx(previous.launch.decode)

        schedule_batch_copy = schedule_batch.copy()
        pending = _MlxSchedulerPendingStep(
            launch=launch,
            reqs=reqs,
            scheduler_output=replace(
                scheduler_output,
                batch_data=schedule_batch_copy,
            ),
            schedule_batch=schedule_batch_copy,
        )
        self._last_mlx_pending = pending
        return pending

    def execute_resolve(self, pending: _MlxSchedulerPendingStep | None):
        if pending is None:
            return None

        try:
            batch_result = self.tp_worker.finalize_mlx_result(
                pending.launch,
                pending.reqs,
            )
        except Exception:
            # note (yexiaodong): A predecessor failure invalidates any chained
            # successor that shares its lazily updated cache objects.
            self._last_mlx_pending = None
            raise
        else:
            if self._last_mlx_pending is pending:
                self._last_mlx_pending = None

        if (
            self._execution_bridge is not None
            and batch_result.next_token_ids is not None
        ):
            # note (yexiaodong): The custom MLX worker owns forward execution,
            # so publish its sampled token for a later batch that breaks a chain.
            self._execution_bridge.publish_next_tokens(
                pending.schedule_batch,
                batch_result.next_token_ids,
            )

        skip_rids = {
            request.request_id
            for request in pending.scheduler_output.requests
            if request.data.req.finished() or self._req_is_retracted(request.data.req)
        }
        return self._finalize(
            batch_result,
            None,
            pending.schedule_batch,
            pending.scheduler_output,
            skip_rids=skip_rids,
        )


def create_mlx_model_worker(
    *,
    config: Any,
    server_args: Any,
    gpu_id: int,
    tp_rank: int = 0,
):
    """Construct an MLX worker with the same scheduler-facing contract as Omni."""
    if config.model_arch_override != "Qwen3ASRForConditionalGeneration":
        raise NotImplementedError(
            "Omni's MLX worker currently supports only "
            "Qwen3ASRForConditionalGeneration"
        )

    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import PortArgs

    from sglang_omni.models.qwen3_asr.mlx.runner import make_qwen3_asr_mlx_runner_class

    class OmniQwen3ASRMlxWorker(MlxTpModelWorker):
        @property
        def tp_rank(self) -> int:
            return self.ps.tp_rank

        def _init_model_runner(self):
            MlxModelRunnerStub.validate_startup_weight_load_mode(self.server_args)
            runner_class = make_qwen3_asr_mlx_runner_class()
            init_kwargs = {
                "model_path": self.server_args.model_path,
                "trust_remote_code": self.server_args.trust_remote_code,
                "disable_radix_cache": self.server_args.disable_radix_cache,
                "mem_fraction_static": self.server_args.mem_fraction_static,
                "quantization": self.server_args.quantization,
                "revision": self.server_args.revision,
                "enable_sampling": self.server_args.mlx_enable_sampling,
                "sampling_rng_seed": self.server_args.random_seed,
                "deterministic_seeding": (
                    self.server_args.enable_deterministic_inference
                ),
            }
            if self.server_args.max_total_tokens is not None:
                init_kwargs["pool_size"] = self.server_args.max_total_tokens
            self._mlx_runner = runner_class(**init_kwargs)
            self._model_runner = MlxModelRunnerStub(
                model_config=self.model_config,
                mem_fraction_static=self.server_args.mem_fraction_static,
                gpu_id=self.gpu_id,
                ps=self.ps,
                nccl_port=self.nccl_port,
                server_args=self.server_args,
                is_draft_worker=self.is_draft_worker,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=self.memory_pool_config,
                mlx_pool_size=self._mlx_runner.pool_size,
            )
            self._mlx_active_rids = set()
            self._mlx_pool_initialized = False

        def get_tp_group(self):
            return self.model_runner.tp_group

        def get_attention_tp_group(self):
            return self.model_runner.attention_tp_group

        def get_attention_tp_cpu_group(self):
            return self.model_runner.attention_tp_group.cpu_group

    attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
        compute_dp_attention_world_info(
            server_args.enable_dp_attention,
            tp_rank,
            server_args.tp_size,
            server_args.dp_size,
            server_args.attn_cp_size,
        )
    )
    ps = ParallelState(
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=server_args.dp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_rank=0,
        attn_cp_size=server_args.attn_cp_size,
        attn_dcp_rank=tp_rank % server_args.dcp_size,
        attn_dcp_size=server_args.dcp_size,
        attn_dp_rank=attn_dp_rank,
        attn_dp_size=attn_dp_size,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=server_args.moe_dp_size,
        gpu_id=gpu_id,
    )
    nccl_port = config.nccl_port
    if nccl_port is None:
        nccl_port = PortArgs.init_new(server_args).nccl_port
    # note (yexiaodong): MlxTpModelWorker reads the split runtime configuration
    # while building its model config, before the bookkeeping stub exists.
    publish(server_args, role="scheduler")
    return OmniQwen3ASRMlxWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        ps=ps,
        nccl_port=nccl_port,
    )
