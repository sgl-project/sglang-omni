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
    """The caller's explicit device survives, including cpu and any :N index."""
    live = current_platform.device_type
    assert dev.resolve_device_spec("cpu") == "cpu"
    assert dev.resolve_device_spec("cpu", 5) == "cpu"
    if live == "cpu":
        return
    assert dev.resolve_device_spec(f"{live}:2") == f"{live}:2"
    assert dev.resolve_device_spec(f"{live}:2", 5) == f"{live}:5"
    assert dev.resolve_device_spec(live) == live


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


def test_sglang_caps_xpu_free_memory_against_the_allocator() -> None:
    """Omni relies on SGLang owning this (upstream #32044) instead of correcting it.

    Sizing the KV pool from the driver's free memory alone over-allocates, so if this
    cap ever disappears upstream the pool silently OOMs again.
    """
    import inspect

    import torch
    from sglang.srt.utils.common import get_available_gpu_memory

    source = inspect.getsource(get_available_gpu_memory)
    xpu_branch = source.split('elif device == "xpu":', 1)
    assert len(xpu_branch) == 2, "SGLang no longer has a dedicated XPU branch"
    xpu_body = xpu_branch[1].split("elif device ==", 1)[0]
    assert "mem_get_info" in xpu_body
    assert "memory_allocated" in xpu_body, "the allocator cross-check is gone"

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return
    free_gb = get_available_gpu_memory("xpu", 0, empty_cache=False)
    total_gb = torch.xpu.get_device_properties(0).total_memory / (1 << 30)
    assert 0 <= free_gb <= total_gb


def test_an_explicit_device_keeps_its_platform_and_takes_the_placement_index():
    """The shared builder applies a stage's gpu_id to whatever device the caller
    named. Rewriting the type would send an explicit xpu or cpu stage to CUDA.
    """
    assert dev.place_device_spec("xpu", 1) == "xpu:1"
    assert dev.place_device_spec("cuda:0", 1) == "cuda:1"
    assert dev.place_device_spec("cpu", 1) == "cpu"
    # No placement supplied: the spec's own index survives.
    assert dev.place_device_spec("cuda:3") == "cuda:3"
    assert dev.place_device_spec("xpu:2") == "xpu:2"


def test_placing_an_explicit_device_does_not_validate_against_the_platform():
    """A CUDA-only stage must keep CUDA on a non-CUDA host, where resolving would
    raise. Only device=None consults current_platform.
    """
    live = current_platform.device_type
    absent = "xpu" if live == "cuda" else "cuda"

    assert dev.place_device_spec(f"{absent}:1") == f"{absent}:1"
