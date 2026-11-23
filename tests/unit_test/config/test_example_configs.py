# SPDX-License-Identifier: Apache-2.0
"""The shipped example configs must load, compile and resolve as documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.cli.serve import apply_default_draft_engine_overrides
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_args,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.arkasr.config import ArkasrPipelineConfig
from sglang_omni.platforms import current_platform
from sglang_omni.speculative.draft_registry import DRAFT_REGISTRY
from tests.unit_test.pipeline.helpers import build_compiled_process_topology

REPO_ROOT = Path(__file__).resolve().parents[3]

EXAMPLE_CONFIG_PATHS = sorted((REPO_ROOT / "examples" / "configs").glob("*.yaml"))


def test_example_configs_are_found() -> None:
    assert len(EXAMPLE_CONFIG_PATHS) > 30


@pytest.mark.parametrize(
    "config_path", EXAMPLE_CONFIG_PATHS, ids=lambda config_path: config_path.name
)
def test_example_config_loads_compiles_and_resolves_every_factory(
    config_path: Path,
) -> None:
    config = ConfigManager.from_file(str(config_path)).config
    build_compiled_process_topology(config)
    for stage in config.stages:
        resolve_stage_factory_args(stage, config)


def test_disable_speculative_decoding_drops_only_the_draft_keys() -> None:
    manager = ConfigManager.from_file(
        str(REPO_ROOT / "examples" / "configs" / "qwen3_asr_1_7b_dflash.yaml")
    )
    drafted = resolve_stage_factory_args(
        manager.config.stage_named("asr"), manager.config
    )["server_args_overrides"]
    config = manager.merge_config(
        manager.parse_extra_args(["--asr.engine.disable_speculative_decoding", "true"])
    )
    overrides = resolve_stage_factory_args(config.stage_named("asr"), config)[
        "server_args_overrides"
    ]
    assert "speculative_algorithm" in drafted
    assert overrides == {
        key: value
        for key, value in drafted.items()
        if not key.startswith("speculative_")
    }


@pytest.mark.parametrize(
    "model_path",
    [
        "AutoArk-AI/ARK-ASR-3B",
        "AutoArk-AI/ARK-ASR-3B@pinned-revision",
        "/cache/models--AutoArk-AI--ARK-ASR-3B/snapshots/pinned-revision",
    ],
)
def test_arkasr_default_draft_reaches_runtime_and_can_be_disabled(
    model_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI and a normal model launch must serve the same default draft."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    manager = ConfigManager(ArkasrPipelineConfig(model_path=model_path))
    launched = apply_default_draft_engine_overrides(manager.config)
    drafted = resolve_stage_typed_kwargs(launched.stage_named("asr"))[
        "server_args_overrides"
    ]
    registered = next(
        draft for draft in DRAFT_REGISTRY if draft.model_path == "AutoArk-AI/ARK-ASR-3B"
    )
    assert drafted.items() >= registered.engine_overrides().items()
    plain = manager.merge_config(
        manager.parse_extra_args(["--asr.engine.disable_speculative_decoding", "true"])
    )
    plain = apply_default_draft_engine_overrides(plain)
    assert resolve_stage_typed_kwargs(plain.stage_named("asr"))[
        "server_args_overrides"
    ] == {
        name: value
        for name, value in drafted.items()
        if not name.startswith("speculative_")
    }


def test_arkasr_explicit_draft_does_not_inherit_default_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the default draft must not retain its checkpoint revision."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    manager = ConfigManager(ArkasrPipelineConfig(model_path="AutoArk-AI/ARK-ASR-3B"))
    config = manager.merge_config(
        manager.parse_extra_args(
            ["--asr.engine.speculative_draft_model_path", "example/custom-ark-draft"]
        )
    )
    config = apply_default_draft_engine_overrides(config)
    overrides = resolve_stage_typed_kwargs(config.stage_named("asr"))[
        "server_args_overrides"
    ]
    assert overrides["speculative_draft_model_path"] == "example/custom-ark-draft"
    assert "speculative_draft_model_revision" not in overrides


@pytest.mark.parametrize(
    "model_path, cuda_available",
    [("example/ark-finetune", True), ("AutoArk-AI/ARK-ASR-3B", False)],
)
def test_arkasr_default_draft_requires_supported_model_and_device(
    model_path: str,
    cuda_available: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default speculation must not change an unqualified model or device."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda_available)
    config = ArkasrPipelineConfig(model_path=model_path)
    config = apply_default_draft_engine_overrides(config)
    overrides = resolve_stage_typed_kwargs(config.stage_named("asr"))[
        "server_args_overrides"
    ]
    assert not any(name.startswith("speculative_") for name in overrides)
