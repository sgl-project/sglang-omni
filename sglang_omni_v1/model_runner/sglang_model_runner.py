from __future__ import annotations

import logging
from contextlib import contextmanager

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import PortArgs, ServerArgs

from sglang_omni_v1.model_runner.checkpoint_filter import (
    CheckpointFilterConfig,
    install_checkpoint_filter,
)
from sglang_omni_v1.utils.gpu_memory import (
    calculate_process_scoped_available_bytes,
    format_bytes_gib,
    get_gpu_device_info,
    get_process_gpu_memory_bytes,
)

logger = logging.getLogger(__name__)


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
        checkpoint_filter: CheckpointFilterConfig | None = None,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        self._weight_prefix = weight_prefix
        self._checkpoint_filter = checkpoint_filter
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

    def load_model(self):
        """Load model weights with optional stage-aware checkpoint filtering."""
        if self._checkpoint_filter is None:
            return super().load_model()
        with _scoped_checkpoint_filter(self._checkpoint_filter):
            return super().load_model()

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
            return self._profile_upstream_available_bytes(pre_model_load_memory)

        process_memory = get_process_gpu_memory_bytes(self.gpu_id)
        device_info = get_gpu_device_info(self.gpu_id)
        total_memory = device_info.total_memory_bytes

        if process_memory is None or process_memory <= 0:
            raise RuntimeError(
                "Colocated SGLang AR stage requires NVML process memory "
                f"accounting for gpu_id={self.gpu_id} after model weights load. "
                "NVML did not report current-process GPU memory on this device."
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

    def _profile_upstream_available_bytes(self, pre_model_load_memory: int) -> int:
        """Use the SGLang memory profiler exposed by the installed version."""
        upstream_profile = getattr(
            ModelRunnerKVCacheMixin, "_profile_available_bytes", None
        )
        if upstream_profile is not None:
            return upstream_profile(self, pre_model_load_memory)
        return self._profile_available_bytes_from_free_memory_delta(
            pre_model_load_memory
        )

    def _profile_available_bytes_from_free_memory_delta(
        self, pre_model_load_memory: int
    ) -> int:
        """Match SGLang free-memory-delta accounting for non-colocated AR stages."""
        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.utils.common import get_available_gpu_memory

        world_group = get_world_group()
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=world_group.world_size > 1,
            cpu_group=world_group.cpu_group,
        )
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)
        return int(rest_memory * (1 << 30))

    def profile_max_num_token(self, total_gpu_memory: int) -> int:
        """Profile token capacity for SGLang versions that size KV cache by tokens."""
        if self._total_gpu_memory_fraction is None:
            upstream_profile = getattr(
                ModelRunnerKVCacheMixin, "profile_max_num_token", None
            )
            if upstream_profile is None:
                raise AttributeError(
                    "Installed SGLang does not expose profile_max_num_token"
                )
            return upstream_profile(self, total_gpu_memory)

        num_layers = self._num_kv_cache_layers()
        cell_size = self.get_cell_size_per_token(num_layers)
        available_bytes = self._profile_available_bytes(total_gpu_memory)
        if self.mambaish_config is not None:
            available_gib = available_bytes / (1 << 30)
            available_bytes = int(
                self.handle_max_mamba_cache(available_gib) * (1 << 30)
            )
        return available_bytes // cell_size

    def _num_kv_cache_layers(self) -> int:
        """Return the number of layers used by SGLang KV-cache sizing."""
        if self.is_draft_worker:
            return getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        if mambaish := self.mambaish_config:
            return len(
                [
                    layer_id
                    for layer_id in mambaish.full_attention_layer_ids
                    if self.start_layer <= layer_id < self.end_layer
                ]
            )
        return self.num_effective_layers


@contextmanager
def _scoped_checkpoint_filter(profile: CheckpointFilterConfig):
    """Apply a checkpoint filter only while one SGLang model is loading."""
    from sglang.srt.model_executor import model_runner as sglang_model_runner
    from sglang.srt.model_loader.loader import DefaultModelLoader

    original_get_model_loader = sglang_model_runner.get_model_loader

    def _get_model_loader(*args, **kwargs):
        loader = original_get_model_loader(*args, **kwargs)
        if isinstance(loader, DefaultModelLoader):
            install_checkpoint_filter(loader, profile, log=logger)
        return loader

    sglang_model_runner.get_model_loader = _get_model_loader
    try:
        yield
    finally:
        sglang_model_runner.get_model_loader = original_get_model_loader
