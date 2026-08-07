# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform

import sglang_omni.platforms as platforms
import sglang_omni.platforms.xpu as xpu_platform
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform


class _VendorDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "vendor"
    device_type = "vendor"

    def get_device(self, device_id: int = 0) -> str:
        return f"vendor:{device_id}"

    def set_device(self, device: torch.device) -> None:
        pass


class _VendorSRTPlatform(SRTPlatform, _VendorDeviceMixin):
    pass


def test_npu_probe_handles_torch_without_npu(monkeypatch) -> None:
    monkeypatch.delattr(torch, "npu", raising=False)

    assert platforms._is_npu_available() is False


def test_cpu_platform_needs_no_stage_process_env() -> None:
    spec = SimpleNamespace(stage_name="cpu", tp_size=2, gpu_id=None)

    assert CPUOmniPlatform().get_stage_process_env(spec, {}) == {}


def test_rocm_platform_keeps_cuda_compatible_tp_mapping() -> None:
    platform = platforms._as_omni_platform(RocmSRTPlatform())
    spec = SimpleNamespace(stage_name="thinker", tp_size=2, gpu_id=1)

    assert platform.is_rocm()
    assert (
        platform.get_stage_process_env(spec, {"CUDA_VISIBLE_DEVICES": "3,4"})[
            "CUDA_VISIBLE_DEVICES"
        ]
        == "4"
    )


def test_srt_plugin_identity_round_trips_to_spawned_process() -> None:
    qualname = f"{__name__}._VendorSRTPlatform"
    platform = platforms._load_platform_class(qualname)()

    restored = platforms._load_platform_class(platforms.get_platform_spec(platform))()

    assert isinstance(restored, OmniPlatform)
    assert restored.get_device(2) == "vendor:2"
    assert restored.get_stage_process_env(SimpleNamespace(), {}) == {}


@pytest.mark.parametrize(
    ("argument", "expected_index"),
    [
        # stage_workers passes a bare int; the rest come from get_device().
        (2, 2),
        (torch.device("xpu", 3), 3),
        (torch.device("xpu", 0), 0),  # index 0 must not read as "no index"
        (torch.device("xpu"), 0),
    ],
)
def test_xpu_set_device_accepts_an_index_or_a_device(
    monkeypatch, argument, expected_index
) -> None:
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    seen: list[int] = []
    monkeypatch.setattr(
        xpu_platform.torch,
        "xpu",
        SimpleNamespace(set_device=seen.append),
        raising=False,
    )

    XPUOmniPlatform().set_device(argument)

    assert seen == [expected_index]


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(None, id="torch_has_no_xpu_attribute"),
        pytest.param(RuntimeError("Level Zero init failed"), id="driver_probe_raises"),
    ],
)
def test_xpu_probe_never_breaks_platform_resolution(monkeypatch, broken) -> None:
    """A misconfigured runtime must resolve to "not available", not propagate.

    This probe runs at import time, so anything it raises would stop
    sglang_omni.platforms from importing at all.
    """
    if broken is None:
        monkeypatch.delattr(platforms.torch, "xpu", raising=False)
    else:

        def raise_it() -> bool:
            raise broken

        monkeypatch.setattr(
            platforms.torch, "xpu", SimpleNamespace(is_available=raise_it)
        )

    assert platforms._is_xpu_available() is False
