# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ming model architecture resolution (issue #273).

Covers:
- Registry discovery finds Ming architecture even without __init__.py re-export
- ConfigManager.from_model_path resolves Ming via raw config.json fallback
- Unsupported models still raise clear errors
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Test 1 – Registry discovers Ming architecture
# ---------------------------------------------------------------------------


def test_registry_discovers_ming_architecture():
    """BailingMM2NativeForConditionalGeneration must appear in the registry."""
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    supported = PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    assert (
        "BailingMM2NativeForConditionalGeneration" in supported
    ), f"Ming architecture not found in registry. Registered: {sorted(supported)}"


def test_registry_returns_ming_config_class():
    """Registry should map Ming architecture to MingOmniPipelineConfig."""
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config(
        "BailingMM2NativeForConditionalGeneration"
    )
    assert config_cls.__name__ == "MingOmniPipelineConfig"


# ---------------------------------------------------------------------------
# Test 2 – Raw config.json fallback resolves architecture
# ---------------------------------------------------------------------------


def test_raw_config_json_resolves_architecture():
    """try_resolve_arch_from_raw_config should parse architectures from JSON."""
    from sglang_omni.utils.hf import try_resolve_arch_from_raw_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "architectures": ["BailingMM2NativeForConditionalGeneration"],
                    "model_type": "bailingmm_moe_v2_lite",
                },
                f,
            )
        arch = try_resolve_arch_from_raw_config(tmpdir)
    assert arch == "BailingMM2NativeForConditionalGeneration"


def test_raw_config_json_falls_back_to_model_type():
    """When architectures list is empty, try model_type mapping."""
    from sglang_omni.utils.hf import try_resolve_arch_from_raw_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"model_type": "voxtral_tts"}, f)
        arch = try_resolve_arch_from_raw_config(tmpdir)
    assert arch == "VoxtralTTSForConditionalGeneration"


def test_raw_config_json_returns_none_for_unknown():
    """Unknown model_type and no architectures → None."""
    from sglang_omni.utils.hf import try_resolve_arch_from_raw_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"model_type": "unknown_model"}, f)
        arch = try_resolve_arch_from_raw_config(tmpdir)
    assert arch is None


def test_raw_config_json_returns_none_for_missing_file():
    """No config.json at all → None."""
    from sglang_omni.utils.hf import try_resolve_arch_from_raw_config

    with tempfile.TemporaryDirectory() as tmpdir:
        arch = try_resolve_arch_from_raw_config(tmpdir)
    assert arch is None


# ---------------------------------------------------------------------------
# Test 3 – ConfigManager.from_model_path resolves Ming from local config.json
# ---------------------------------------------------------------------------


def test_config_manager_resolves_ming_from_local_config():
    """from_model_path should resolve Ming via raw config.json when AutoConfig fails."""
    from sglang_omni.config.manager import ConfigManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "architectures": ["BailingMM2NativeForConditionalGeneration"],
                    "model_type": "bailingmm_moe_v2_lite",
                },
                f,
            )

        mgr = ConfigManager.from_model_path(tmpdir)
    assert mgr.config is not None
    assert type(mgr.config).__name__ == "MingOmniPipelineConfig"


# ---------------------------------------------------------------------------
# Test 4 – Unsupported models still raise clear error
# ---------------------------------------------------------------------------


def test_unsupported_model_raises_clear_error():
    """Completely unknown architecture should raise ValueError with architecture list."""
    from sglang_omni.config.manager import ConfigManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {"architectures": ["TotallyUnknownArchitecture"]},
                f,
            )

        with pytest.raises(ValueError, match="TotallyUnknownArchitecture"):
            ConfigManager.from_model_path(tmpdir)


def test_no_config_at_all_raises_clear_error():
    """Empty directory with no config files should raise ValueError."""
    from sglang_omni.config.manager import ConfigManager

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Could not resolve model architecture"):
            ConfigManager.from_model_path(tmpdir)
