# SPDX-License-Identifier: Apache-2.0
"""Per-platform qualification policy for MiniMax Music 3.

Keep hardware checks and supported/unverified/unsupported reasons centralized as
required by the multi-hardware RFC:
https://github.com/sgl-project/sglang-omni/issues/1310
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sglang_omni.platforms import current_platform

if TYPE_CHECKING:
    from sglang_omni.platforms.interface import OmniPlatform


class Qualification(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class MiniMaxMusic3PlatformPolicy:
    qualification: Qualification
    reason: str
    device_type: str
    generation_cuda_graph: bool
    rvq_cuda_graph: bool
    acoustic_compile: bool
    breakable_cuda_graph: bool
    configure_cuda_backends: bool

    def require_runnable(self) -> None:
        if self.qualification is Qualification.UNSUPPORTED:
            raise RuntimeError(
                f"MiniMax Music 3 is unsupported on {self.device_type}: {self.reason}"
            )


def get_minimax_music3_platform_policy(
    platform: OmniPlatform | None = None,
) -> MiniMaxMusic3PlatformPolicy:
    """Return qualified features without spreading vendor checks through the model."""
    platform = current_platform if platform is None else platform
    device_type = platform.device_type

    if platform.is_cuda():
        return MiniMaxMusic3PlatformPolicy(
            qualification=Qualification.SUPPORTED,
            reason="the existing CUDA path is covered by accelerator tests",
            device_type=device_type,
            generation_cuda_graph=True,
            rvq_cuda_graph=True,
            acoustic_compile=True,
            breakable_cuda_graph=True,
            configure_cuda_backends=True,
        )
    if platform.is_musa():
        # Preserve the pre-existing MUSA behavior. Its qualification is not
        # changed as part of the Intel XPU enablement.
        return MiniMaxMusic3PlatformPolicy(
            qualification=Qualification.UNVERIFIED,
            reason=(
                "the pre-existing MUSA execution policy is preserved, but its "
                "qualification evidence is outside this enablement"
            ),
            device_type=device_type,
            generation_cuda_graph=True,
            rvq_cuda_graph=True,
            acoustic_compile=True,
            breakable_cuda_graph=True,
            configure_cuda_backends=True,
        )
    if platform.is_xpu():
        return MiniMaxMusic3PlatformPolicy(
            qualification=Qualification.UNVERIFIED,
            reason=(
                "the eager torch_sdpa path requires real-XPU operator, numeric, "
                "and memory validation"
            ),
            device_type=device_type,
            generation_cuda_graph=False,
            rvq_cuda_graph=False,
            acoustic_compile=False,
            breakable_cuda_graph=False,
            configure_cuda_backends=False,
        )
    return MiniMaxMusic3PlatformPolicy(
        qualification=Qualification.UNSUPPORTED,
        reason="no eager fallback and numeric validation are available",
        device_type=device_type,
        generation_cuda_graph=False,
        rvq_cuda_graph=False,
        acoustic_compile=False,
        breakable_cuda_graph=False,
        configure_cuda_backends=False,
    )


__all__ = [
    "MiniMaxMusic3PlatformPolicy",
    "Qualification",
    "get_minimax_music3_platform_policy",
]
