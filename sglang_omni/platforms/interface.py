"""SGLang Omni hardware platform hooks."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.platforms.device_mixin import DeviceMixin

from sglang_omni.utils.misc import normalize_quantization

if TYPE_CHECKING:
    import torch
    from torch.nn.attention import SDPBackend

    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
    from sglang_omni.profiler.torch_profiler import TorchProfiler


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


class OmniPlatform(DeviceMixin):
    _omni_platform_qualname: str | None = None

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
        return self._get_device_graph_backend()

    def _get_device_graph_backend(self) -> DeviceGraphBackend | None:
        return None

    def enable_code2wav_graph(self):
        """Check if current platform support Graph for code2wav in Qwen3-Omni"""
        return True

    def enable_talker_graph(self) -> bool:
        return True

    def enable_thinker_decode_graph(self) -> bool:
        return True

    def get_decode_cuda_graph_backend(self) -> str | None:
        return None

    def supports_torchaudio_resample(self) -> bool:
        """Check if current platform support torchaudio.functional.resample"""
        return True

    def get_graph_capture_sdpa_backends(self) -> tuple["SDPBackend", ...]:
        """Empty leaves dispatch alone."""
        return ()

    def graph_capture_attention(self) -> AbstractContextManager[object]:
        backends = self.get_graph_capture_sdpa_backends()
        if not backends:
            return nullcontext()

        from torch.nn.attention import sdpa_kernel

        return sdpa_kernel(list(backends))

    def get_torch_profiler(self) -> TorchProfiler:
        from sglang_omni.profiler.torch_profiler import TorchProfiler

        return TorchProfiler
