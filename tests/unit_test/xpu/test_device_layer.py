# SPDX-License-Identifier: Apache-2.0
"""Unit tests for device-spec retargeting (no accelerator required)."""

from __future__ import annotations

import importlib

import sglang_omni.utils.device as dev
from sglang_omni.platforms import current_platform


def test_module_exports_match():
    mod = importlib.import_module("sglang_omni.utils.device")
    for name in mod.__all__:
        assert hasattr(mod, name), f"exported {name} missing"


def test_none_is_the_only_automatic_selection():
    """Auto-selection is opt-in: only None consults current_platform, so a supplied
    device can never be silently retargeted."""
    live = current_platform.device_type

    assert dev.resolve_device_spec(None) == live
    assert dev.resolve_device_spec(None, 3) == ("cpu" if live == "cpu" else f"{live}:3")


def test_a_supplied_device_is_honored_with_its_placement_index():
    """The caller's explicit device type survives, and the index argument
    decides its number, including cpu and any live accelerator."""
    live = current_platform.device_type
    assert dev.resolve_device_spec("cpu") == "cpu"
    assert dev.resolve_device_spec("cpu", 5) == "cpu"
    if live == "cpu":
        return
    assert dev.resolve_device_spec(live) == live
    assert dev.resolve_device_spec(live, 5) == f"{live}:5"


def test_a_device_string_naming_its_own_index_is_rejected():
    """device names only the type; a caller who wants a card passes gpu_id, not
    'cpu:N'.
    """
    import pytest

    with pytest.raises(ValueError, match="names an index"):
        dev.resolve_device_spec("cpu:1")
    with pytest.raises(ValueError, match="names an index"):
        dev.resolve_device_spec("cpu:0", 5)


def test_a_device_this_host_cannot_serve_is_rejected_not_retargeted():
    """Silently remapping could not tell a legacy 'cuda' default from an operator
    genuinely asking for CUDA, so a mismatch is an error at the boundary now.
    """
    import pytest

    live = current_platform.device_type
    absent = "xpu" if live == "cuda" else "cuda"

    with pytest.raises(ValueError, match="this host resolved to"):
        dev.resolve_device_spec(f"{absent}:1")
    with pytest.raises(ValueError, match="this host resolved to"):
        dev.resolve_device_spec("npu:0")


def test_the_availability_probe_is_gone():
    """Validating against the already-resolved platform removes any need to re-probe
    torch, so the broad-exception availability probe must not come back."""
    assert not hasattr(dev, "_accel_available")
    assert not hasattr(dev, "remap_accelerator_spec")


def test_sglang_caps_xpu_free_memory_against_the_allocator(monkeypatch) -> None:
    """Omni sizes the KV pool from this number. SGLang derives it from the
    allocator, total device memory minus what this process has allocated, not
    from a driver free-memory query that can over-report.
    """
    from types import SimpleNamespace

    import torch
    from sglang.srt.utils.common import get_available_gpu_memory

    total_bytes = 64 << 30
    allocated_bytes = 24 << 30
    fake_xpu = SimpleNamespace(
        device_count=lambda: 1,
        current_device=lambda: 0,
        memory_allocated=lambda gpu_id: allocated_bytes,
        get_device_properties=lambda gpu_id: SimpleNamespace(total_memory=total_bytes),
    )
    monkeypatch.setattr(torch, "xpu", fake_xpu)

    free_gb = get_available_gpu_memory("xpu", 0, distributed=False, empty_cache=False)

    assert free_gb == (total_bytes - allocated_bytes) / (1 << 30)


def test_the_xpu_platform_implements_the_device_operations_it_advertises():
    """Every device operation omni calls through current_platform must reach
    torch.xpu. The base DeviceMixin answers them with a silent no-op or
    NotImplementedError, so inheriting any of them means stage memory reclaim,
    KV budget sizing and seeding quietly stop working on XPU.
    """
    from sglang.srt.platforms.device_mixin import DeviceMixin

    from sglang_omni.platforms.xpu import XPUOmniPlatform

    for name in (
        "empty_cache",
        "synchronize",
        "get_available_memory",
        "get_device_total_memory",
        "get_current_memory_usage",
        "get_device_name",
        "is_pin_memory_available",
        "seed_everything",
    ):
        assert getattr(XPUOmniPlatform, name) is not getattr(
            DeviceMixin, name
        ), f"XPUOmniPlatform inherits the unimplemented {name}"


def test_free_device_memory_is_reported_in_gib_from_the_platform_query(
    monkeypatch,
) -> None:
    """Stage startup logs and the colocated KV budget read this number, so it
    must come from the live platform rather than a CUDA-only query.
    """
    from sglang_omni.utils.misc import avail_gpu_mem

    total_bytes = 24 << 30
    free_bytes = 20 << 30
    monkeypatch.setattr(
        type(current_platform),
        "get_available_memory",
        lambda self, device_id=0: (free_bytes, total_bytes),
    )

    assert avail_gpu_mem(0) == free_bytes / (1024**3)


def test_an_unimplemented_query_reports_no_free_memory_instead_of_raising(
    monkeypatch,
) -> None:
    """Startup logging must not be the thing that fails a stage launch."""
    from sglang_omni.utils.misc import avail_gpu_mem

    def raise_not_implemented(self, device_id: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    monkeypatch.setattr(
        type(current_platform), "get_available_memory", raise_not_implemented
    )

    assert avail_gpu_mem(0) is None


def test_an_explicit_device_is_validated_on_the_concrete_path_too():
    """An explicit device must fail at this boundary, not deep inside the engine."""
    import pytest

    live = current_platform.device_type
    absent = "xpu" if live == "cuda" else "cuda"
    with pytest.raises(ValueError, match="this host resolved to"):
        dev.resolve_concrete_device(absent, 1)
