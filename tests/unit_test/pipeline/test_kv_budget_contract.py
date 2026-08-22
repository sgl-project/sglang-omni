# SPDX-License-Identifier: Apache-2.0
"""Registry-wide contract: every registered model accepts a KV byte budget.

The budget is delivered through the stage worker's ambient scope, so no model
factory signature can miss it. What can still fail per model is the config
resolution layer: a stage whose model config pins a conflicting memory
setting, or a validation path that trips over the new typed field. Walking the
registry makes such a regression fail here instead of on a user's deployment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sglang_omni.models
from sglang_omni.config import StageRuntimeConfig
from sglang_omni.config.runtime import (
    resolve_stage_factory_arg_defaults,
    resolve_stage_static_factory_args,
)
from sglang_omni.models.registry import import_pipeline_configs

_KV_BYTES = 2 * 1024**3


def _registered_config_classes():
    configs = import_pipeline_configs("sglang_omni.models", "config")
    # Note (Jiaxin Deng): dedupe aliases so each config class is exercised once.
    seen: dict[type, str] = {}
    for arch, config_cls in sorted(configs.items()):
        seen.setdefault(config_cls, arch)
    return sorted(
        ((arch, config_cls) for config_cls, arch in seen.items()),
        key=lambda item: item[1].__name__,
    )


def _stage_runtime_with_kv_budget(stage) -> StageRuntimeConfig:
    """Rebuild the stage runtime with a byte budget through real validation."""
    data = stage.runtime.model_dump()
    data["memory"] = {"kv_cache_bytes": _KV_BYTES}
    data["resources"] = {"total_gpu_memory_fraction": None}
    return StageRuntimeConfig(**data)


def test_registry_discovery_covers_every_model_package():
    """import_pipeline_configs skips import failures; a skipped package would
    make the walk below silently vacuous for that model, so pin coverage."""
    packages_with_configs = {
        path.parent.name
        for path in Path(sglang_omni.models.__file__).parent.glob("*/config.py")
    }
    discovered_packages = {
        config_cls.__module__.rsplit(".", 2)[-2]
        for _, config_cls in _registered_config_classes()
    }

    missing = packages_with_configs - discovered_packages
    assert not missing, f"registry silently skipped model packages: {sorted(missing)}"


@pytest.mark.parametrize(
    ("arch", "config_cls"),
    _registered_config_classes(),
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_every_registered_model_stage_accepts_a_kv_byte_budget(arch, config_cls):
    del arch
    config = config_cls(model_path="dummy")

    accepted = 0
    for stage in config.stages:
        pinned_typed_fraction = (
            stage.runtime.sglang_server_args.mem_fraction_static is not None
        )
        try:
            stage.runtime = _stage_runtime_with_kv_budget(stage)
        except ValueError as exc:
            assert pinned_typed_fraction, (
                f"stage {stage.name!r} rejected a byte budget without a pinned "
                f"typed mem_fraction_static: {exc}"
            )
            assert "cannot be set together with" in str(exc)
            continue

        args = resolve_stage_static_factory_args(stage, config)
        defaults = resolve_stage_factory_arg_defaults(stage, config, gpu_id=0)

        assert defaults["kv_cache_bytes"] == _KV_BYTES
        assert "kv_cache_bytes" not in args
        accepted += 1

    assert accepted > 0, f"{config_cls.__name__} accepted a byte budget on no stage"
