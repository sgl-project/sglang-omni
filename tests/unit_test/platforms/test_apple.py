# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.platforms.interface import SRTPlatform

from sglang_omni import platforms
from sglang_omni.platforms.apple import AppleOmniPlatform


def test_generic_sglang_platform_resolves_to_apple_when_mps_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(platforms, "_is_apple_silicon_mps_available", lambda: True)

    resolved = platforms._as_omni_platform(SRTPlatform())

    assert isinstance(resolved, AppleOmniPlatform)
    assert resolved.is_mps()
    assert resolved.device_type == "mps"


def test_apple_device_binding_is_single_device() -> None:
    apple = AppleOmniPlatform()

    assert apple.get_device(0) == torch.device("mps")
    apple.set_device(torch.device("mps"))
    with pytest.raises(ValueError, match="one Metal device"):
        apple.get_device(1)
    with pytest.raises(ValueError, match="Expected an MPS device"):
        apple.set_device(torch.device("cpu"))


def test_apple_device_total_memory_uses_torch_without_mlx(monkeypatch) -> None:
    import sglang.srt.utils.tensor_bridge as tensor_bridge

    expected = 12_713_115_648
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: expected)

    assert AppleOmniPlatform().get_device_total_memory(0) == expected


def test_apple_stage_rejects_tensor_parallelism() -> None:
    apple = AppleOmniPlatform()
    spec = SimpleNamespace(stage_name="asr", tp_size=2, gpu_id=0)

    with pytest.raises(ValueError, match="requires tp_size=1"):
        apple.get_stage_process_env(spec)


def test_apple_stage_accepts_device_zero() -> None:
    apple = AppleOmniPlatform()
    spec = SimpleNamespace(stage_name="asr", tp_size=1, gpu_id=0)

    assert apple.get_stage_process_env(spec) == {}
