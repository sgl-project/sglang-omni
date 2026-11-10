# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for Fun-CosyVoice3."""

from __future__ import annotations

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform


class FakePlatform:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type


def test_the_npu_keeps_the_dit_eager() -> None:
    assert stages.dit_torch_compile_allowed(True, FakePlatform("npu")) is False


def test_other_devices_keep_the_requested_flag() -> None:
    assert stages.dit_torch_compile_allowed(True, FakePlatform("cuda")) is True
    assert stages.dit_torch_compile_allowed(True, FakePlatform("cpu")) is True


def test_an_explicit_opt_out_never_compiles() -> None:
    """A caller that turned compile off keeps it off, whatever the device says."""
    assert stages.dit_torch_compile_allowed(False, FakePlatform("cuda")) is False


def test_only_the_npu_drops_float64() -> None:
    assert NPUOmniPlatform().is_float64_supported() is False
    for platform_type in (OmniPlatform, CUDAOmniPlatform, CPUOmniPlatform):
        assert platform_type().is_float64_supported() is True
