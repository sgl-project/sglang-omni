"""SGLang Omni hardware platform hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.device_mixin import DeviceMixin

from sglang_omni.utils.misc import normalize_quantization

if TYPE_CHECKING:
    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


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

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        """Apply Omni backend policy after checkpoint quantization is known."""

        effective_quantization = normalize_quantization(model_config.quantization)
        server_quantization = normalize_quantization(server_args.quantization)
        if server_quantization is not None:
            effective_quantization = server_quantization
        return effective_quantization

    def enable_code2wav_graph(self):
        """Check if current platform support Graph for code2wav in Qwen3-Omni"""
        return True

    def enable_sglang_cuda_graph(self):
        """Check if current platform supports SGLang generation CUDA graph
        capture (e.g. CUDA). Ascend NPU ATB operators are incompatible with
        SGLang decode graph capture, so the NPU platform disables it.
        """
        return True

    def get_fused_topk_topp_renorm(self):
        """Return the fused ``(top_k_renorm_prob, top_p_renorm_prob)``
        callables from ``sgl_kernel`` when this platform provides them
        (e.g. CUDA), else ``None``.

        The Higgs TTS sampler uses these fused kernels on CUDA to replace a
        full-vocab ``torch.sort`` in decode; Ascend NPU has no fused renorm
        kernels, so it returns ``None`` and the sampler falls back to an
        equivalent ``torch`` implementation.
        """
        return None

    def enable_torch_compile(self):
        """Check if current platform supports ``torch.compile`` (e.g. CUDA).

        Ascend NPU has no native ``torch.compile`` backend, so models that
        optionally wrap sub-modules with ``torch.compile`` must skip it there.
        """
        return True
