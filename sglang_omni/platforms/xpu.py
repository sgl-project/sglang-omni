from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.platforms.xpu import XpuDeviceMixin

from sglang_omni.platforms.interface import JointRopeInplaceKernel, OmniPlatform
from sglang_omni.quantization import is_pack_quantized, resolve_quant_config
from sglang_omni.utils.misc import model_config_has_moe
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from torch.nn.attention import SDPBackend

    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


def target_matches(value: str, target: str, check_contains: bool) -> bool:
    """SGLang's compressed-tensors target test: a re: regex, else substring or equality."""
    if target.startswith("re:"):
        return re.match(target[3:], value) is not None
    elif check_contains:
        return target.lower() in value.lower()
    else:
        return target == value


def expert_config_group(
    quant_dict: dict[str, object], expert_name: str
) -> dict[str, object] | None:
    """The config group SGLang's MoE lookup applies to an expert, or None if none matches."""
    group_by_target: dict[str, dict[str, object]] = {}
    for group in (quant_dict.get("config_groups") or {}).values():
        for target in group.get("targets") or []:
            group_by_target[target] = group
    # note (siju-samuel): Linear stands in for FusedMoE only when no group targets FusedMoE.
    if "Linear" in group_by_target and "FusedMoE" not in group_by_target:
        group_by_target["FusedMoE"] = group_by_target["Linear"]
    else:
        pass
    for value, check_contains in ((expert_name, False), ("FusedMoE", True)):
        for target, group in group_by_target.items():
            if target_matches(value, target, check_contains):
                return group
            else:
                pass
    return None


def is_weight_only_int_group(group: dict[str, object]) -> bool:
    """SGLang's WNA16 test: static int weights by channel or group, activations unquantized."""
    weights = group.get("weights") or {}
    return (
        group.get("input_activations") is None
        and weights.get("type") == "int"
        and weights.get("strategy") in ("channel", "group")
        and not weights.get("dynamic")
    )


def stage_has_weight_only_packed_experts(
    quant_dict: dict[str, object], num_hidden_layers: int
) -> bool:
    """True when some MoE layer's experts take SGLang's weight-only WNA16 scheme."""
    ignored = quant_dict.get("ignore") or []
    for index in range(num_hidden_layers):
        expert = f"model.layers.{index}.mlp.experts.0.gate_proj"
        if should_ignore_layer(expert, ignore=ignored):
            continue
        else:
            pass
        group = expert_config_group(quant_dict, expert)
        if group is not None and is_weight_only_int_group(group):
            return True
        else:
            pass
    return False


class XPUOmniPlatform(XpuDeviceMixin, OmniPlatform):
    def set_device(self, device: "torch.device | int") -> None:
        index = device.index if isinstance(device, torch.device) else int(device)
        torch.xpu.set_device(0 if index is None else index)

    def enable_code2wav_graph(self):
        return True

    def get_encoder_decoder_attention_backend(self) -> str | None:
        # note (jianan): intel_xpu requires graphs disabled for encoder-decoder
        # models; torch_native makes SGLang disable both graph phases.
        return "torch_native"

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

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        from sgl_kernel.jit.rope import apply_rope_inplace

        return apply_rope_inplace

    def enable_talker_graph(self) -> bool:
        return True

    def enable_tts_vocoder_fast_path(self) -> bool:
        return False

    def enable_thinker_decode_graph(self) -> bool:
        # Capture leaves the scheduler thread's stream recording; host reads fail.
        return False

    def enable_zonos2_torch_compile(self) -> bool:
        return False

    def supports_fp8_moe(self) -> bool:
        return False

    def zonos2_bf16_mem_fraction_static(self, device: torch.device) -> float | None:
        if device.type != self.device_type:
            return None
        else:
            # Measured on an Arc Pro B60: 14.34 GiB of bf16 experts plus a
            # 5.98 GiB KV pool.
            return 0.85

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

    def supports_graph_captured_fft(self) -> bool:
        return False

    def supports_graph_captured_host_read(self) -> bool:
        # Note (siju): a host read inside a capture raises on XPU, which is what
        # the Mimi encoder's mask helper does.
        return False

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
        is_qwen3_omni_arch = model_arch_override in (
            "Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM",
        )
        if is_qwen3_omni_arch and moe_runner_backend in (
            "flashinfer_cutlass",
            "cutlass",
        ):
            raise ValueError(
                f"Qwen3-Omni on Intel XPU cannot use "
                f"moe_runner_backend={moe_runner_backend!r}; the CUTLASS "
                "MoE runners are CUDA-only. Leave the backend as 'auto' or pass "
                "'triton'."
            )
        else:
            pass

        # note (siju-samuel): Marlin has no XPU kernel, so only a stage with weight-only packed experts needs Triton.
        quant_dict = resolve_quant_config(model_config.hf_config)
        if (
            is_qwen3_omni_arch
            and moe_runner_backend == "auto"
            and model_config_has_moe(model_config)
            and is_pack_quantized(quant_dict)
            and stage_has_weight_only_packed_experts(
                quant_dict, model_config.num_hidden_layers
            )
        ):
            override_server_args(
                server_args,
                "sglang-omni-xpu-backend-policy",
                moe_runner_backend="triton",
            )
            logger.info(
                "Intel XPU: selected moe_runner_backend=triton for the "
                "group-packed weight-only checkpoint; the default Marlin "
                "runner for this scheme is CUDA-only"
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
