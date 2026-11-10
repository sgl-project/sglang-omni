# SPDX-License-Identifier: Apache-2.0
"""Platform-capability routing for the Fun-CosyVoice3 device adapters."""

from __future__ import annotations

from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform


def test_only_the_npu_drops_float64() -> None:
    assert NPUOmniPlatform().is_float64_supported() is False
    for platform_type in (OmniPlatform, CUDAOmniPlatform, CPUOmniPlatform):
        assert platform_type().is_float64_supported() is True
