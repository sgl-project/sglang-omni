from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs

from sglang_omni.utils.gpu_memory import (
    calculate_stage_budget_available_bytes,
    calculate_stage_load_delta_bytes,
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
        # Register sglang_omni model classes directly in SGLang's model registry.
        import importlib

        from sglang.srt.models.registry import ModelRegistry

        sglang_omni_models = {
            "S2ProSGLangTextModel": "sglang_omni.models.fishaudio_s2_pro.sglang_model:S2ProSGLangTextModel",
            "Qwen3OmniTalker": "sglang_omni.models.qwen3_omni.components.talker:Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM": "sglang_omni.models.qwen3_omni.components.sglang_thinker:Qwen3OmniThinkerForCausalLM",
            "HiggsMultimodalQwen3ForConditionalGeneration": "sglang_omni.models.higgs_tts.model:HiggsTTSModel",
            "Qwen3TTSTalker": "sglang_omni.models.qwen3_tts.sglang_model:Qwen3TTSTalker",
            "MingTTSSGLangModel": "sglang_omni.models.ming_tts.sglang_model:MingTTSSGLangModel",
            "MossTTSDelaySGLangModel": "sglang_omni.models.moss_tts.sglang_model:MossTTSDelaySGLangModel",
            "MossTTSLocalSGLangModel": "sglang_omni.models.moss_tts_local.sglang_model:MossTTSLocalSGLangModel",
            "MossTranscribeDiarizeForConditionalGeneration": "sglang_omni.models.moss_transcribe_diarize.sglang_model:MossTranscribeDiarizeForConditionalGeneration",
            "VoxtralSGLangTTSModel": "sglang_omni.models.voxtral_tts.sglang_model:VoxtralSGLangTTSModel",
            "LLaDA2MoeModelLM": "sglang_omni.models.llada2_uni.components.thinker:LLaDA2MoeModelLM",
            "WhisperForConditionalGeneration": "sglang_omni.models.whisper_asr.sglang_model:WhisperForConditionalGeneration",
            "Qwen3ASRForConditionalGeneration": "sglang_omni.models.qwen3_asr.sglang_model:Qwen3ASRForConditionalGeneration",
            "FunAsrNanoForConditionalGeneration": "sglang_omni.models.fun_asr.sglang_model:FunAsrNanoForConditionalGeneration",
        }
        for arch, path in sglang_omni_models.items():
            module_path, _, attr = path.partition(":")
            try:
                ModelRegistry.models[arch] = getattr(
                    importlib.import_module(module_path), attr
                )
            except Exception as exc:
                logger.warning(f"sglang-omni: skipping model {arch} ({exc})")

        try:
            from sglang_omni.models.ming_omni.registration import (
                register_ming_hf_config,
                register_ming_model_registry,
            )

            register_ming_hf_config()
            register_ming_model_registry()
        except Exception as exc:
            logger.warning(f"sglang-omni: skipping Ming-Omni registration ({exc})")

    def _profile_available_bytes(self, pre_model_load_memory: float) -> int:
        """Profile KV-cache headroom for colocated SGLang AR stages.

        Upstream SGLang profiles from global free-memory deltas. That is valid
        for a single AR engine, but colocated Omni stages can load multiple
        SGLang engines in separate processes on the same GPU. In that case
        another process can change global free memory while this process is
        loading weights, making the global delta too small or negative.

        When a stage total-memory budget is provided, compute cache headroom as
        total GPU memory times that budget minus this stage's measured memory.
        NVML process accounting is preferred. If NVML cannot identify the
        current process, use the stage-local load delta measured inside
        SGLang's serialized initialization window. Without a stage budget, keep
        upstream SGLang profiling semantics for ordinary non-colocated AR
        serving.
        """
        if self._total_gpu_memory_fraction is None:
            return super()._profile_available_bytes(pre_model_load_memory)

        process_memory = get_process_gpu_memory_bytes(self.gpu_id)
        device_info = get_gpu_device_info(self.gpu_id)
        total_memory = device_info.total_memory_bytes

        if total_memory is None:
            raise RuntimeError(
                "Colocated SGLang AR stage requires total GPU memory for "
                f"gpu_id={self.gpu_id}. Check CUDA_VISIBLE_DEVICES and CUDA "
                "device visibility."
            )

        if process_memory is None or process_memory <= 0:
            return self._profile_available_bytes_from_stage_load_delta(
                pre_model_load_memory,
                total_memory,
            )

        return self._profile_available_bytes_from_process_memory(
            total_memory,
            process_memory,
        )

    def _profile_available_bytes_from_stage_load_delta(
        self,
        pre_model_load_memory: float,
        total_memory: int,
    ) -> int:
        """Profile colocated KV headroom from this stage's load-time delta."""
        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.utils.common import get_available_gpu_memory

        world_group = get_world_group()
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=world_group.world_size > 1,
            cpu_group=world_group.cpu_group,
        )
        stage_load_bytes = calculate_stage_load_delta_bytes(
            pre_model_load_memory_gib=pre_model_load_memory,
            post_model_load_memory_gib=post_model_load_memory,
        )
        available_bytes = calculate_stage_budget_available_bytes(
            total_memory_bytes=total_memory,
            accounted_memory_bytes=stage_load_bytes,
            memory_fraction=self._total_gpu_memory_fraction,
            accounted_memory_label="stage_load_used",
        )
        logger.info(
            f"SGLang AR memory profile: gpu_mem_accounting=stage_load_fallback "
            f"gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self._total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"stage_load_used={format_bytes_gib(stage_load_bytes)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes

    def _profile_available_bytes_from_process_memory(
        self,
        total_memory: int,
        process_memory: int,
    ) -> int:
        available_bytes = calculate_stage_budget_available_bytes(
            total_memory_bytes=total_memory,
            accounted_memory_bytes=process_memory,
            memory_fraction=self._total_gpu_memory_fraction,
            accounted_memory_label="process_used",
        )
        logger.info(
            f"SGLang AR memory profile: gpu_mem_accounting=nvml_process "
            f"gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self._total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"process_used={format_bytes_gib(process_memory)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes
