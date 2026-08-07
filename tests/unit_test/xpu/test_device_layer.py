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


def test_remap_accelerator_spec_honors_the_caller():
    """Only an unavailable accelerator type is remapped; the caller's explicit
    device -- including cpu and any ``:N`` -- survives."""
    live = current_platform.device_type
    # An explicit cpu request is never rerouted onto the accelerator, and cpu
    # carries no index (a stage on cpu is not per-card).
    assert dev.remap_accelerator_spec("cpu") == "cpu"
    assert dev.remap_accelerator_spec("cpu", 5) == "cpu"
    if live == "cpu":
        # No accelerator on this host: every spec collapses to plain cpu.
        assert dev.remap_accelerator_spec("cuda:2") == "cpu"
        return
    # The index is carried over, not reset to 0 (the bug this replaced).
    assert dev.remap_accelerator_spec(f"{live}:2") == f"{live}:2"
    # An explicit gpu_id wins over the spec's own index.
    assert dev.remap_accelerator_spec(f"{live}:2", 5) == f"{live}:5"
    # A spec for an accelerator this host lacks is retargeted at the live one.
    absent = "xpu" if live == "cuda" else "cuda"
    assert dev.remap_accelerator_spec(f"{absent}:1") == f"{live}:1"
    # Also a type this torch build does not know at all: torch.device() rejects
    # "npu:0" outright, so the spec must be parsed by hand to retarget it.
    assert dev.remap_accelerator_spec("npu:0") == f"{live}:0"
