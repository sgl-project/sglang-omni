# SPDX-License-Identifier: Apache-2.0
"""Bootstrap helpers for LLaDA2-Uni scheduler."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.dllm.algorithm import algo_name_to_cls
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from sglang_omni.models.llada2_uni.cfg_attention_backend import (
    CFG_ATTENTION_BACKEND,
    register_llada2_cfg_flashinfer_backend,
)
from sglang_omni.models.llada2_uni.low_confidence_cfg import LowConfidenceCFG
from sglang_omni.models.llada2_uni.request_builders import (
    make_dllm_thinker_scheduler_adapters,
)
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
    pin_resolved_device_type,
)
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


def register_llada2_uni_cfg() -> None:
    """Call before build_server_args for the omni LowConfidenceCFG variant.

    The text-only LowConfidence variant does not need this registration.
    """
    algo_name_to_cls["LowConfidenceCFG"] = LowConfidenceCFG
    register_llada2_cfg_flashinfer_backend()


def validate_cfg(server_args: ServerArgs) -> None:
    cfg = resolved_view(server_args)
    if cfg.dllm_algorithm != "LowConfidenceCFG":
        return
    else:
        pass
    register_llada2_uni_cfg()
    if any(backend != CFG_ATTENTION_BACKEND for backend in attention_backends_of(cfg)):
        raise ValueError("LowConfidenceCFG requires llada2_uni_cfg_flashinfer")
    else:
        pass
    if cfg.dllm_fdfo:
        raise ValueError("LowConfidenceCFG requires synchronous DLLM, not FDFO")
    else:
        pass


def create_dllm_thinker_scheduler(
    server_args: ServerArgs,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
) -> DllmScheduler:
    """Create an DllmScheduler for the LLaDA2-Uni thinker.

    Returns a ``DllmScheduler`` with ``dllm_config`` set, ready to be
    driven by a ``Stage``.
    """
    validate_cfg(server_args)
    dllm_config = DllmConfig.from_server_args(server_args)

    # sglang supports radix cache with dLLM, but Omni's dLLM staging
    # path has only been validated without it; keep it disabled deliberately.
    override_server_args(
        server_args,
        "sglang_omni.llada2_uni.disable_radix_cache",
        disable_radix_cache=True,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="LLaDA2MoeModelLM",
    )

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)

    request_builder, result_adapter = make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        dllm_config=dllm_config,
    )

    return DllmScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        dllm_config=dllm_config,
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_sglang_dllm_thinker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 8192,
    dllm_algorithm: str = "LowConfidence",
    dllm_algorithm_config: str | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> DllmScheduler:
    """Create an DllmScheduler for the LLaDA2-Uni thinker."""
    concrete_device = resolve_concrete_device(device, gpu_id)
    resolved_gpu_id = concrete_device.index or 0

    overrides: dict[str, object] = {
        "attention_backend": "flashinfer",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
    }
    if dllm_algorithm == "LowConfidenceCFG":
        register_llada2_uni_cfg()
        overrides["attention_backend"] = CFG_ATTENTION_BACKEND
        overrides["dllm_fdfo"] = False
    else:
        pass
    overrides.update(server_args_overrides or {})
    overrides["tp_size"] = tp_size
    pin_resolved_device_type(overrides, concrete_device.type)

    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        **overrides,
    )
    if dllm_algorithm == "LowConfidenceCFG":
        # note (Anmuliar): DLLM graph defaults replace custom backends with FlashInfer.
        override_server_args(
            server_args,
            "sglang_omni.llada2_uni.cfg_attention",
            attention_backend=overrides["attention_backend"],
        )
    else:
        pass
    cfg = resolved_view(server_args)
    logger.info(
        f"create_sglang_dllm_thinker_executor_from_config: "
        f"dllm_algorithm={cfg.dllm_algorithm}, "
        f"mem_fraction_static={cfg.mem_fraction_static}"
    )
    return create_dllm_thinker_scheduler(
        server_args, resolved_gpu_id, tp_rank=tp_rank, nccl_port=nccl_port
    )
