# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.srt.platforms.interface import SRTPlatform

from sglang_omni import platforms
from sglang_omni.platforms.apple import AppleOmniPlatform


def test_generic_sglang_platform_resolves_to_apple_when_mps_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(platforms, "is_apple_silicon_mps_available", lambda: True)

    resolved = platforms.as_omni_platform(SRTPlatform())

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


def test_apple_platform_does_not_claim_float64_support() -> None:
    assert AppleOmniPlatform.is_float64_supported() is False


def test_apple_device_total_memory_uses_torch_without_mlx(monkeypatch) -> None:
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    expected = 12_713_115_648
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
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


@pytest.fixture
def mlx_core(monkeypatch) -> ModuleType:
    """Provide a stub MLX package and force MLX dispatch."""
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    core = ModuleType("mlx.core")
    package = ModuleType("mlx")
    package.core = core

    # Register the parent package too so this works when MLX is not installed.
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    return core


def test_apple_device_total_memory_uses_mlx_when_enabled(mlx_core, monkeypatch) -> None:
    mlx_core.device_info = lambda: {"max_recommended_working_set_size": 111}
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 222)

    assert AppleOmniPlatform().get_device_total_memory(0) == 111


def test_apple_current_memory_usage_uses_mlx_when_enabled(
    mlx_core, monkeypatch
) -> None:
    mlx_core.get_active_memory = lambda: 111
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 222)

    assert AppleOmniPlatform().get_current_memory_usage() == 111.0


def test_apple_current_memory_usage_uses_torch_without_mlx(monkeypatch) -> None:
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 222)

    assert AppleOmniPlatform().get_current_memory_usage() == 222.0


def test_apple_current_memory_usage_rejects_non_mps_device() -> None:
    with pytest.raises(ValueError, match="Expected an MPS device"):
        AppleOmniPlatform().get_current_memory_usage(torch.device("cpu"))


def test_apple_empty_cache_uses_mlx_when_enabled(mlx_core, monkeypatch) -> None:
    calls: list[str] = []
    mlx_core.clear_cache = lambda: calls.append("mlx")
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("torch"))

    AppleOmniPlatform().empty_cache()

    assert calls == ["mlx"]


def test_apple_empty_cache_uses_torch_without_mlx(monkeypatch) -> None:
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    calls: list[str] = []
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("torch"))

    AppleOmniPlatform().empty_cache()

    assert calls == ["torch"]


def test_apple_synchronize_uses_mlx_when_enabled(mlx_core, monkeypatch) -> None:
    calls: list[str] = []
    mlx_core.synchronize = lambda: calls.append("mlx")
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("torch"))

    AppleOmniPlatform().synchronize()

    assert calls == ["mlx"]


def test_apple_synchronize_uses_torch_without_mlx(monkeypatch) -> None:
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    calls: list[str] = []
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("torch"))

    AppleOmniPlatform().synchronize()

    assert calls == ["torch"]


def test_apple_device_name_is_apple_metal() -> None:
    assert AppleOmniPlatform().get_device_name(0) == "Apple Metal"


def test_apple_intra_node_transport_is_shm() -> None:
    from sglang_omni.comm.data_ref import TransportKind

    assert AppleOmniPlatform().get_intra_node_transport() is TransportKind.SHM


def test_apple_stage_rejects_nonzero_gpu_id() -> None:
    apple = AppleOmniPlatform()
    spec = SimpleNamespace(stage_name="asr", tp_size=1, gpu_id=1)

    with pytest.raises(ValueError, match="one Metal device"):
        apple.get_stage_process_env(spec)
