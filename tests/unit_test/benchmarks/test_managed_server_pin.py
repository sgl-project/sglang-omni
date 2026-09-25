# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.benchmarker import utils
from sglang_omni.config.manager import ConfigManager


def config_with(model_path: str | None):
    return SimpleNamespace(config=SimpleNamespace(model_path=model_path))


def test_managed_server_serves_the_config_pin_for_the_same_repo(monkeypatch) -> None:
    monkeypatch.setattr(
        ConfigManager, "from_file", lambda path: config_with("org/model@abc123")
    )
    assert (
        utils._pinned_model_path("org/model", "cfg.yaml") == "org/model@abc123"
    )  # noqa: leading-underscore  # production name
    assert (
        utils._pinned_model_path("org/other", "cfg.yaml") == "org/other"
    )  # noqa: leading-underscore  # production name
    assert (
        utils._pinned_model_path("org/model@def456", "cfg.yaml")
        == "org/model@def456"  # noqa: leading-underscore  # production name
    )


def test_managed_server_keeps_the_flag_without_a_config_or_pin(monkeypatch) -> None:
    assert (
        utils._pinned_model_path("org/model", None) == "org/model"
    )  # noqa: leading-underscore  # production name
    monkeypatch.setattr(
        ConfigManager, "from_file", lambda path: config_with("org/model")
    )
    assert (
        utils._pinned_model_path("org/model", "cfg.yaml") == "org/model"
    )  # noqa: leading-underscore  # production name
