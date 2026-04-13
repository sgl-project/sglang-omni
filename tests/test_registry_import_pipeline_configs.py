# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline config discovery import error handling."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from sglang_omni.models.registry import import_pipeline_configs


def _clear_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def _build_package(
    base_dir: Path,
    package_name: str,
    model_pkg_name: str,
    config_source: str | None,
) -> None:
    package_dir = base_dir / package_name
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    model_dir = package_dir / model_pkg_name
    model_dir.mkdir(parents=True)
    (model_dir / "__init__.py").write_text("", encoding="utf-8")

    if config_source is not None:
        (model_dir / "config.py").write_text(config_source, encoding="utf-8")


def test_missing_config_submodule_respects_strict_flag(monkeypatch, tmp_path: Path):
    package_name = "tmp_models_missing_cfg"
    _build_package(
        tmp_path,
        package_name=package_name,
        model_pkg_name="model_a",
        config_source=None,
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    import_pipeline_configs.cache_clear()
    _clear_modules(package_name)

    discovered = import_pipeline_configs(package_name, "config", strict=False)
    assert discovered == {}

    import_pipeline_configs.cache_clear()
    _clear_modules(package_name)

    with pytest.raises(ModuleNotFoundError):
        import_pipeline_configs(package_name, "config", strict=True)


def test_config_internal_import_error_respects_strict_flag(
    monkeypatch, tmp_path: Path, caplog
):
    package_name = "tmp_models_broken_cfg"
    _build_package(
        tmp_path,
        package_name=package_name,
        model_pkg_name="model_b",
        config_source="import definitely_missing_dependency\n",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    import_pipeline_configs.cache_clear()
    _clear_modules(package_name)

    caplog.set_level(logging.WARNING)
    discovered = import_pipeline_configs(package_name, "config", strict=False)
    assert discovered == {}
    assert any(
        f"Ignore import error when loading {package_name}.model_b.config"
        in record.message
        for record in caplog.records
    )

    import_pipeline_configs.cache_clear()
    _clear_modules(package_name)

    with pytest.raises(ModuleNotFoundError):
        import_pipeline_configs(package_name, "config", strict=True)
