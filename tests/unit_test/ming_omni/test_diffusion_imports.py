# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for Ming image-generation diffusion modules."""

from __future__ import annotations

import importlib


def test_ming_diffusion_backend_contract_imports() -> None:
    module = importlib.import_module("sglang_omni.models.ming_omni.diffusion.backend")

    assert hasattr(module, "DiffusionBackend")
    assert hasattr(module, "ImageGenParams")


def test_ming_diffusion_lightweight_modules_import() -> None:
    modules = [
        "sglang_omni.models.ming_omni.diffusion",
        "sglang_omni.models.ming_omni.diffusion.bailing_moe_config",
        "sglang_omni.models.ming_omni.diffusion.bailing_moe_model",
        "sglang_omni.models.ming_omni.diffusion.byt5_encoder",
        "sglang_omni.models.ming_omni.diffusion.query_info",
        "sglang_omni.models.ming_omni.diffusion.semantic_conditioner",
        "sglang_omni.models.ming_omni.diffusion.semantic_encoder",
        "sglang_omni.models.ming_omni.diffusion.zimage_backend",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name
