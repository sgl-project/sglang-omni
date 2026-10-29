from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.platforms.xpu import XpuDeviceMixin

from sglang_omni.platforms.interface import JointRopeInplaceKernel, OmniPlatform

logger = logging.getLogger(__name__)

PROBE_ROWS = 64


def replayed_collectives_match(
    reduced: torch.Tensor, gathered: torch.Tensor, world_size: int
) -> bool:
    """Whether a replayed all-reduce and all-gather both produced their expected values.

    Every rank stages 10 * (its rank + 1) before the replay, so a recorded all-reduce leaves the
    sum over ranks in every element and a recorded all-gather leaves each rank's value in its own
    slice. A dropped collective leaves the staged fill, which neither expectation matches.
    """
    expected_reduced = 10 * world_size * (world_size + 1) / 2
    expected_gathered = 10 * torch.arange(
        1, world_size + 1, device=gathered.device, dtype=gathered.dtype
    ).repeat_interleave(PROBE_ROWS)
    return bool(torch.all(reduced == expected_reduced).item()) and torch.equal(
        gathered, expected_gathered
    )


if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from torch.nn.attention import SDPBackend

    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


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

    def supports_async_device_to_host_copy(self) -> bool:
        # Note (siju): on XPU a non_blocking copy into pinned memory holds the host until the
        # stream's queued work drains, so a lookahead launch would wait for its own forward.
        return False

    def graph_capture_records_collectives(self, group: GroupCoordinator) -> bool:
        if group.world_size == 1:
            return True
        else:
            pass
        # Note (siju): oneCCL 2022.1.1, which torch 2.14.0+xpu pins, drops an all-reduce or
        # all-gather issued during XPU graph capture without an error, so a TP decode graph
        # would replay stale activations. Capture both, replay, and compare on every rank.
        device = torch.device("xpu", torch.xpu.current_device())
        rank_value = float(group.rank_in_group + 1)
        reduced = torch.full((PROBE_ROWS,), rank_value, device=device)
        local = torch.full((PROBE_ROWS,), rank_value, device=device)
        gathered = torch.zeros(group.world_size * PROBE_ROWS, device=device)
        dist.all_reduce(reduced, group=group.device_group)
        dist.all_gather_into_tensor(gathered, local, group=group.device_group)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        stream = torch.xpu.Stream()
        stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.graph(graph, stream=stream):
            dist.all_reduce(reduced, group=group.device_group)
            dist.all_gather_into_tensor(gathered, local, group=group.device_group)
        reduced.fill_(10 * rank_value)
        local.fill_(10 * rank_value)
        gathered.zero_()
        torch.xpu.synchronize()
        dist.barrier(group=group.cpu_group)
        graph.replay()
        torch.xpu.synchronize()
        is_recorded = torch.tensor(
            int(replayed_collectives_match(reduced, gathered, group.world_size))
        )
        dist.all_reduce(is_recorded, op=dist.ReduceOp.MIN, group=group.cpu_group)
        return bool(is_recorded.item())

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
