from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import PortArgs, ServerArgs

from sglang_omni_v1.utils.gpu_memory import (
    calculate_process_scoped_available_bytes,
    format_bytes_gib,
    get_gpu_device_info,
    get_process_gpu_memory_bytes,
)

logger = logging.getLogger(__name__)


def filter_weights_by_prefix(
    weights: Iterator[tuple[str, Any]],
    prefix: str | None,
) -> Iterator[tuple[str, Any]]:
    """Filter weight iterator by prefix, stripping matched prefix from names."""
    if not prefix:
        yield from weights
        return
    for name, tensor in weights:
        if name.startswith(prefix):
            yield name[len(prefix) :], tensor


class SGLModelRunner(ModelRunner):
    """Thin wrapper to bootstrap SGLang ModelRunner from backend args."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        model_arch_override: str | None = None,
        weight_prefix: str | None = None,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        self._weight_prefix = weight_prefix
        self._total_gpu_memory_fraction = total_gpu_memory_fraction
        self._register_omni_model()

        port_args = PortArgs.init_new(server_args)
        tp_size = server_args.tp_size
        self.nccl_port = port_args.nccl_port

        # model_config is already fully configured by ModelWorker._init_model_config()
        # (architecture override, text_config swap, etc. are all done there)

        super().__init__(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def _register_omni_model(self):
        # Register sglang_omni_v1 model classes directly in SGLang's model registry.
        from sglang.srt.models.registry import ModelRegistry

        from sglang_omni_v1.models.fishaudio_s2_pro.sglang_model import (
            S2ProSGLangTextModel,
        )
        from sglang_omni_v1.models.qwen3_omni.components.sglang_thinker import (
            Qwen3OmniThinkerForCausalLM,
        )
        from sglang_omni_v1.models.qwen3_omni.components.talker import Qwen3OmniTalker

        ModelRegistry.models["S2ProSGLangTextModel"] = S2ProSGLangTextModel
        ModelRegistry.models["Qwen3OmniTalker"] = Qwen3OmniTalker
        ModelRegistry.models["Qwen3OmniThinkerForCausalLM"] = (
            Qwen3OmniThinkerForCausalLM
        )

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        """Profile KV-cache headroom for colocated SGLang AR stages.

        Upstream SGLang profiles from global free-memory deltas. That is valid
        for a single AR engine, but colocated Omni stages can load multiple
        SGLang engines in separate processes on the same GPU. In that case
        another process can change global free memory while this process is
        loading weights, making the global delta too small or negative.

        When a stage total-memory budget is provided, compute this process'
        cache headroom as total GPU memory times that budget minus this
        process' current GPU memory. Without a stage budget, keep upstream
        SGLang profiling semantics for ordinary non-colocated AR serving.
        """
        if self._total_gpu_memory_fraction is None:
            return ModelRunnerKVCacheMixin._profile_available_bytes(
                self,
                pre_model_load_memory,
            )

        process_memory = get_process_gpu_memory_bytes(self.gpu_id)
        device_info = get_gpu_device_info(self.gpu_id)
        total_memory = device_info.total_memory_bytes

        if process_memory is None or process_memory <= 0:
            raise RuntimeError(
                "Colocated SGLang AR stage requires NVML process memory "
                f"accounting for gpu_id={self.gpu_id}. Ensure pynvml is installed, "
                "NVML process queries are available, and the current stage process "
                "is visible to NVML after model weights load."
            )
        if total_memory is None:
            raise RuntimeError(
                "Colocated SGLang AR stage requires total GPU memory from NVML "
                f"for gpu_id={self.gpu_id}. Check CUDA_VISIBLE_DEVICES and NVML "
                "permissions."
            )

        available_bytes = calculate_process_scoped_available_bytes(
            total_memory_bytes=total_memory,
            process_memory_bytes=process_memory,
            memory_fraction=self._total_gpu_memory_fraction,
        )
        logger.info(
            f"SGLang AR process-scoped memory profile: gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self._total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"process_used={format_bytes_gib(process_memory)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes
