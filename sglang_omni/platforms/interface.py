"""SGLang Omni hardware platform hooks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.platforms.device_mixin import DeviceMixin

from sglang_omni.utils.gpu_memory import narrow_visible_devices
from sglang_omni.utils.misc import normalize_quantization

if TYPE_CHECKING:
    import torch
    from torch.nn.attention import SDPBackend

    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
    from sglang_omni.profiler.torch_profiler import TorchProfiler
else:
    pass


# Note(yzxiao): Joint RoPE rotates all supplied Q/K heads in place. Same-dtype
# Q/K have shapes [T, Hq, D] and [T, Hk, D], a contiguous last dimension,
# and matching head strides.
# The contiguous FP32 [P, D] cache stores cos then sin; contiguous int32/int64
# positions [T] index its rows. All tensors share a device. Cache and positions
# are read-only; this operation does not apply Q/K norm or write KV caches.
# is_neox selects half-split (True) or interleaved (False) rotation. Providers
# use the caller's stream and support graph capture after their real warmup.
class JointRopeInplaceKernel(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        is_neox: bool,
    ) -> None: ...


def build_cuda_compatible_stage_process_env(
    spec: StageLaunchConfig,
    env: Mapping[str, str] | None,
    native_visibility_variable: str,
) -> dict[str, str]:
    """Build a CUDA-compatible worker environment from a native device mask."""
    source_env = env if env is not None else os.environ
    native_visible = source_env.get(native_visibility_variable)
    visibility_variable = (
        native_visibility_variable if native_visible else "CUDA_VISIBLE_DEVICES"
    )
    original_visible_devices = source_env.get(visibility_variable)
    # note (zijiecode): Native masks win; CUDA stays aligned for locks and physical-device helpers.
    if spec.allow_child_processes and spec.owned_gpu_ids:
        # Native children inherit this mask; narrowing keeps unpinned host threads on owned GPUs.
        owned_visible_devices = narrow_visible_devices(
            original_visible_devices,
            spec.owned_gpu_ids,
            visibility_variable,
        )
        if native_visible:
            return {
                "CUDA_VISIBLE_DEVICES": owned_visible_devices,
                native_visibility_variable: owned_visible_devices,
            }
        else:
            return {"CUDA_VISIBLE_DEVICES": owned_visible_devices}
    elif spec.tp_size <= 1:
        return {}
    else:
        pass

    if spec.gpu_id is None:
        raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
    else:
        pass
    if original_visible_devices:
        visible_devices = [item.strip() for item in original_visible_devices.split(",")]
        if spec.gpu_id >= len(visible_devices):
            raise ValueError(
                f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                f"but {visibility_variable} only exposes {visible_devices}"
            )
        else:
            pass
        mapped_gpu = visible_devices[spec.gpu_id]
    else:
        mapped_gpu = str(spec.gpu_id)

    env_updates = {
        "CUDA_VISIBLE_DEVICES": mapped_gpu,
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    }
    if native_visible:
        env_updates[native_visibility_variable] = mapped_gpu
    else:
        pass
    return env_updates


class OmniPlatform(DeviceMixin):
    _omni_platform_qualname: str | None = None

    @classmethod
    def is_float64_supported(cls) -> bool:
        """Whether device kernels support native float64 tensors."""
        return True

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return per-process environment overrides needed before child startup."""
        return {}

    def get_intra_node_transport(self) -> TransportKind:
        """Get TransportKind between devices on the same node"""
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.SHM

    def get_fused_qk_norm_rope(self):
        """Get the fused QK norm RoPE kernel if available, else return None."""
        return None

    def get_fused_qk_norm_rope_with_cos_sin_cache(self):
        """Get the cos/sin-cache fused QK norm RoPE kernel, else return None.

        Separate from get_fused_qk_norm_rope: this ABI takes q and k as their own
        tensors plus a cos/sin table, not the packed QKV and rotary parameters.
        """
        return None

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel | None:
        # Note(yzxiao): None means this platform has no implementation. The
        # model decides whether this capability is required or optional.
        return None

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        """Apply Omni backend policy after checkpoint quantization is known."""

        cfg = resolved_view(server_args)
        effective_quantization = normalize_quantization(model_config.quantization)
        server_quantization = normalize_quantization(cfg.quantization)
        if server_quantization is not None:
            effective_quantization = server_quantization
        else:
            pass
        return effective_quantization

    def get_device_graph_backend(
        self, device: torch.device
    ) -> DeviceGraphBackend | None:
        """The backend that records model-owned graphs on this device, or None.

        None is also the answer for a device that is not this platform's own, so
        a caller holding a tensor's device does not have to check that first.
        """
        if device.type != self.device_type:
            return None
        else:
            pass
        return self._get_device_graph_backend()

    def _get_device_graph_backend(self) -> DeviceGraphBackend | None:
        return None

    def enable_code2wav_graph(self):
        """Check if current platform support Graph for code2wav in Qwen3-Omni"""
        return True

    def enable_codec_decode_graph(self) -> bool:
        """Check if current platform captures codec decode graphs (Higgs TTS, Mimi)"""
        return False

    def enable_talker_graph(self) -> bool:
        return True

    def enable_tts_predictor_graph(self) -> bool:
        return True

    def enable_tts_vocoder_fast_path(self) -> bool:
        """Whether the Qwen3-TTS vocoder takes its device fast path by default:
        asynchronous decode on a priority stream, and the codec decode graphs its
        warmup captures."""
        return True

    def enable_thinker_decode_graph(self) -> bool:
        return True

    def enable_zonos2_torch_compile(self) -> bool:
        return True

    def zonos2_bf16_mem_fraction_static(self, device: torch.device) -> float | None:
        return None

    def get_decode_cuda_graph_backend(self) -> str | None:
        return None

    def get_encoder_decoder_attention_backend(self) -> str | None:
        """Default backend for all attention in encoder-decoder models, if needed."""
        return None

    def supports_fp8_moe(self) -> bool:
        return True

    def supports_torchaudio_resample(self) -> bool:
        """Check if current platform support torchaudio.functional.resample"""
        return True

    def supports_graph_captured_fft(self) -> bool:
        return True

    def supports_graph_captured_host_read(self) -> bool:
        """Whether a capture tolerates a body that reads a tensor on the host."""
        return True

    def get_graph_capture_sdpa_backends(self) -> tuple["SDPBackend", ...]:
        """Empty leaves dispatch alone."""
        return ()

    def graph_capture_attention(self) -> AbstractContextManager[object]:
        backends = self.get_graph_capture_sdpa_backends()
        if not backends:
            return nullcontext()
        else:
            pass

        from torch.nn.attention import sdpa_kernel

        return sdpa_kernel(list(backends))

    def get_torch_profiler(self) -> TorchProfiler:
        from sglang_omni.profiler.torch_profiler import TorchProfiler

        return TorchProfiler
