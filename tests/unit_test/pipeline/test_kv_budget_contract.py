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
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
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


def _stage_with_kv_budget(stage):
    """Rebuild the stage with a byte budget through real validation."""
    data = stage.model_dump()
    engine = dict(data.get("engine") or {})
    engine["kv_cache_bytes"] = _KV_BYTES
    engine["mem_fraction_static"] = None
    data["engine"] = engine
    return type(stage).model_validate(data)


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

    if not any(
        config_cls.stage_config_cls(stage.name).engine_stage for stage in config.stages
    ):
        pytest.skip(
            f"{config_cls.__name__} declares no SGLang engine stage; byte "
            "budgets do not apply"
        )

    accepted = 0
    for index, stage in enumerate(config.stages):
        if not config_cls.stage_config_cls(stage.name).engine_stage:
            continue
        budgeted = _stage_with_kv_budget(stage)
        config.stages[index] = budgeted

        # The budget rides StageLaunchConfig.kv_cache_bytes, which the spec
        # constructors read from stage.engine; it must never surface as a
        # factory kwarg or a ServerArgs override.
        assert budgeted.engine.kv_cache_bytes == _KV_BYTES
        kwargs = resolve_stage_factory_kwargs(budgeted, config)
        typed = resolve_stage_typed_kwargs(budgeted)
        assert "kv_cache_bytes" not in kwargs
        assert "kv_cache_bytes" not in (kwargs.get("server_args_overrides") or {})
        assert "kv_cache_bytes" not in (typed.get("server_args_overrides") or {})
        accepted += 1

    assert accepted > 0, f"{config_cls.__name__} accepted a byte budget on no stage"
