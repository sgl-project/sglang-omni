from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from torch.nn.attention import SDPBackend

    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


class XPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("xpu", local_rank)

    def set_device(self, device: "torch.device | int") -> None:
        index = device.index if isinstance(device, torch.device) else int(device)
        torch.xpu.set_device(0 if index is None else index)

    def enable_code2wav_graph(self):
        return True

    def get_fused_qk_norm_rope_with_cos_sin_cache(self):
        try:
            from sgl_kernel import fused_inplace_qknorm_rope
        except ImportError as exc:
            logger.info(
                f"XPU sgl_kernel has no cos/sin-cache fused QK-norm-RoPE kernel "
                f"({exc}); falling back to the unfused QK-norm and RoPE path"
            )
            return None
        return fused_inplace_qknorm_rope

    def enable_talker_graph(self) -> bool:
        return True

    def enable_thinker_decode_graph(self) -> bool:
        # Capture leaves the scheduler thread's stream recording; host reads fail.
        return False

    def enable_dllm_decode_graph(self) -> bool:
        # A dLLM block arrives as ForwardMode.DLLM_EXTEND, and no attention
        # backend reachable on XPU can put that mode in a graph. flashinfer and
        # flashattention are the only two that handle it -- which is why SGLang's
        # own dLLM pass renames the backend to flashinfer as soon as decode
        # capture is on -- and both are CUDA-only. Of the two XPU candidates,
        # triton raises "Invalid forward mode: DLLM_EXTEND for CUDA Graph" while
        # building its graph metadata, and intel_xpu asserts is_decode_or_idle()
        # with "XPU graph only supports decode mode".
        # Capture is not the blocker the AR thinker hits (the dLLM scheduler owns
        # the thread its forwards run on), so this can be flipped once an XPU
        # backend grows the mode.
        return False

    def get_dllm_attention_backend(self) -> str:
        # triton honors AttentionType.ENCODER_ONLY, keeping a dLLM block
        # bidirectional. intel_xpu derives causal from is_cross_attention alone,
        # so it would mask each block's later positions and quietly return the
        # wrong tokens. SGLang's own dLLM pass has no XPU branch, so naming the
        # backend here also keeps it from reaching for CUDA-only flashinfer.
        return "triton"

    def dllm_max_requests_per_round(self) -> int | None:
        # A batched dLLM round returns fluent nonsense for some of its requests
        # here. Measured on LLaDA2.0-Uni under TP=2, eight concurrent requests
        # per trial: 3 of 6 trials had at least one garbled reply (one trial 4 of
        # 8), and the dirty trials were also the slow ones (119s, 154s against
        # 70s clean). Capping the round cleared 13 of 13 trials at 63-93s, so the
        # cap costs no measurable throughput at this concurrency.
        #
        # What is left after eliminating the obvious causes -- the paged radix
        # path (disable_radix_cache, page_size 1, still garbled), sampling RNG
        # (the requests are temperature 0), and the algorithm's own softmax /
        # argmax / gather (bit-identical to CPU on XPU at every batch size 1-8)
        # -- is the multi-request forward itself: either triton's bidirectional
        # extend mixing sequences, or the round sequence diverging across ranks,
        # since with several requests in flight how many rounds a block takes
        # depends on who it shares the batch with. One request per round removes
        # both: the k-th forward is then the same request's same denoising round
        # on every rank, whenever the work arrived.
        return 1

    def _get_device_graph_backend(self) -> DeviceGraphBackend:
        from sglang_omni.platforms.device_graph import XpuDeviceGraphBackend

        return XpuDeviceGraphBackend()

    def get_decode_cuda_graph_backend(self) -> str | None:
        # SGLang leaves XPU decode capture opt-in and accepts only full.
        from sglang.srt.model_executor.cuda_graph_config import Backend

        return Backend.FULL

    def get_graph_capture_sdpa_backends(self) -> tuple["SDPBackend", ...]:
        """Efficient attention is left out: XPU reaches math before its
        unsupported efficient branch, so naming it changes nothing."""
        from torch.nn.attention import SDPBackend

        return (SDPBackend.FLASH_ATTENTION, SDPBackend.MATH)

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )

        cfg = resolved_view(server_args)
        moe_runner_backend = cfg.moe_runner_backend
        if model_arch_override in (
            "Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM",
        ) and moe_runner_backend in ("flashinfer_cutlass", "cutlass"):
            raise ValueError(
                f"Qwen3-Omni on Intel XPU cannot use "
                f"moe_runner_backend={moe_runner_backend!r}; the CUTLASS "
                "MoE runners are CUDA-only. Leave the backend as 'auto' or pass "
                "'triton'."
            )
        else:
            pass

        return effective_quantization

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Keep every card visible, preserving a group-wide ZE_AFFINITY_MASK."""
        if spec.tp_size <= 1:
            return {}
        else:
            pass
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        else:
            pass

        updates = {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"}
        source_env = env if env is not None else os.environ
        mask = (source_env.get("ZE_AFFINITY_MASK") or "").strip()
        if not mask:
            return updates
        else:
            pass

        visible = [item.strip() for item in mask.split(",") if item.strip()]
        if len(visible) < spec.tp_size:
            raise ValueError(
                f"tp stage {spec.stage_name!r} needs tp_size={spec.tp_size} cards, but "
                f"ZE_AFFINITY_MASK={mask!r} exposes {len(visible)}. Widen the mask to "
                "cover the whole TP group: every rank must see its peers for XCCL "
                "discovery, and dropping the mask instead would relocate the stage "
                "onto different physical cards."
            )
        else:
            pass
        if spec.gpu_id >= len(visible):
            raise ValueError(
                f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, but "
                f"ZE_AFFINITY_MASK={mask!r} exposes only {len(visible)} cards "
                f"({', '.join(visible)}). gpu_id indexes into the mask, not the host."
            )
        else:
            pass
        return updates
