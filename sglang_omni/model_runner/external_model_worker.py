# SPDX-License-Identifier: Apache-2.0
"""Zero-weight SGLang worker for stages whose forward runs outside SGLang.

Some Apple-Silicon stages run their model through an architecture-specific
external runner (Torch MPS or MLX). The scheduler still needs a worker with
real request/KV bookkeeping pools, so this module reuses SGLang's MLX stub
runner purely as that bookkeeping implementation: no weights are loaded and
no model forward is ever executed here.
"""

from __future__ import annotations

from typing import Any

# Architectures whose Apple-Silicon stages hand model execution to an external
# runner instead of SGLang's own model runner.
EXTERNAL_FORWARD_ARCHITECTURES = frozenset(
    {
        "Qwen3OmniThinkerForCausalLM",
        "Qwen3OmniTalker",
    }
)


def _build_parallel_state(server_args: Any, *, gpu_id: int, tp_rank: int):
    """Single-node parallel topology shared by every Omni Apple worker."""
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

    attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
        compute_dp_attention_world_info(
            server_args.enable_dp_attention,
            tp_rank,
            server_args.tp_size,
            server_args.dp_size,
            server_args.attn_cp_size,
        )
    )
    return ParallelState(
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


def _resolve_nccl_port(server_args: Any, nccl_port: int | None) -> int:
    from sglang.srt.server_args import PortArgs

    if nccl_port is not None:
        return nccl_port
    return PortArgs.init_new(server_args).nccl_port


def _publish_scheduler_runtime_context(server_args: Any) -> None:
    # note (yexiaodong): The worker reads the split runtime configuration while
    # building its model config, before the bookkeeping stub exists.
    from sglang.srt.runtime_context import publish

    publish(server_args, role="scheduler")


def _make_external_worker_class():
    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

    class OmniExternalModelWorker(MlxTpModelWorker):
        """Scheduler bookkeeping for a stage whose runner owns the forward."""

        # Read by create_sglang_infrastructure() and the Qwen3-Omni bootstrap:
        # this worker has no real Torch model, so hidden-state capture hooks,
        # Torch graph capture, and sampler wiring must not be installed on it.
        uses_external_forward = True

        def __init__(self, *, external_backend_name: str, **kwargs: Any) -> None:
            self._external_backend_name = external_backend_name
            super().__init__(**kwargs)

        @property
        def tp_rank(self) -> int:
            return self.ps.tp_rank

        def _init_model_runner(self) -> None:
            MlxModelRunnerStub.validate_startup_weight_load_mode(self.server_args)
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
                # The external runner owns the real cache, so the scheduler's
                # token budget is the serving cap or the model context bound.
                mlx_pool_size=(
                    self.server_args.max_total_tokens or self.model_config.context_len
                ),
            )
            self._mlx_active_rids: set[str] = set()
            self._mlx_pool_initialized = False

        def forward_batch_generation(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                f"{self._external_backend_name} runner must own model forward"
            )

        def prepare_for_kv_cache_release(self, req: Any) -> None:
            """No MLX auxiliary state exists to snapshot on this worker.

            The scheduler calls this hook for every finished request. Upstream's
            MLX worker uses it to snapshot ``self._mlx_runner`` state before the
            radix insert; this worker builds no MLX runner at all -- the
            external runner owns the per-request cache and frees it from its own
            completion/abort path -- so the hook must be inert instead of
            reaching for an attribute that was never created.
            """
            del req

        def get_tp_group(self):
            return self.model_runner.tp_group

        def get_attention_tp_group(self):
            return self.model_runner.attention_tp_group

        def get_attention_tp_cpu_group(self):
            return self.model_runner.attention_tp_group.cpu_group

    return OmniExternalModelWorker


def uses_external_forward(model_worker: Any) -> bool:
    """Whether this worker's model forward runs outside SGLang."""
    return bool(getattr(model_worker, "uses_external_forward", False))


def create_external_model_worker(
    *,
    config: Any,
    server_args: Any,
    gpu_id: int,
    tp_rank: int = 0,
    backend_name: str,
):
    """Construct a zero-weight worker with Omni's scheduler-facing contract."""
    if config.model_arch_override not in EXTERNAL_FORWARD_ARCHITECTURES:
        raise NotImplementedError(
            "Omni's external model worker supports only "
            f"{sorted(EXTERNAL_FORWARD_ARCHITECTURES)}; got "
            f"{config.model_arch_override!r}"
        )

    ps = _build_parallel_state(server_args, gpu_id=gpu_id, tp_rank=tp_rank)
    nccl_port = _resolve_nccl_port(server_args, config.nccl_port)
    _publish_scheduler_runtime_context(server_args)
    worker_class = _make_external_worker_class()
    return worker_class(
        external_backend_name=backend_name,
        server_args=server_args,
        gpu_id=gpu_id,
        ps=ps,
        nccl_port=nccl_port,
    )
