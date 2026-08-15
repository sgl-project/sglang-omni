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
