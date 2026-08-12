# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.models.ming_omni.talker.configuration_bailing_talker import (
    MingOmniTalkerConfig,
    normalize_max_conc,
    resolve_talker_max_conc,
)


def test_normalize_max_conc_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="max_conc must be >= 1"):
        normalize_max_conc(0)


def test_from_pretrained_dir_reads_max_conc(tmp_path: Path) -> None:
    talker_dir = tmp_path / "talker"
    talker_dir.mkdir()
    (talker_dir / "config.json").write_text(
        json.dumps({"steps": 10, "max_conc": 3}),
        encoding="utf-8",
    )

    config = MingOmniTalkerConfig.from_pretrained_dir(str(talker_dir))

    assert config.max_conc == 3
    assert config.steps == 10


def test_from_pretrained_dir_defaults_max_conc_to_one(tmp_path: Path) -> None:
    talker_dir = tmp_path / "talker"
    talker_dir.mkdir()
    (talker_dir / "config.json").write_text("{}", encoding="utf-8")

    config = MingOmniTalkerConfig.from_pretrained_dir(str(talker_dir))

    assert config.max_conc == 1


def test_resolve_talker_max_conc_override_wins(tmp_path: Path) -> None:
    talker_dir = tmp_path / "talker"
    talker_dir.mkdir()
    (talker_dir / "config.json").write_text(
        json.dumps({"max_conc": 2}),
        encoding="utf-8",
    )

    assert resolve_talker_max_conc(str(talker_dir), max_conc=4) == 4
    assert resolve_talker_max_conc(str(talker_dir)) == 2


def test_resolve_talker_max_conc_missing_config_defaults_to_one(
    tmp_path: Path,
) -> None:
    assert resolve_talker_max_conc(str(tmp_path / "missing")) == 1


def _install_talker_executor_fake(
    monkeypatch,
    *,
    tmp_path: Path,
    captured: dict[str, object],
) -> None:
    class FakeMingTalkerExecutor:
        def __init__(self, **kwargs):
            captured["executor_kwargs"] = kwargs

        def should_generate_audio(self, payload):
            return True

        def build_empty_audio_result(self, payload):
            return payload

        async def start(self):
            return None

        async def add_request(self, payload):
            return None

        async def get_result(self):
            return SimpleNamespace(request_id="req")

    talker_module = ModuleType(
        "sglang_omni.models.ming_omni.components.talker_executor"
    )
    talker_module.MingTalkerExecutor = FakeMingTalkerExecutor
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.components.talker_executor",
        talker_module,
    )

    weight_loader_module = ModuleType("sglang_omni.models.weight_loader")
    weight_loader_module.resolve_model_path = lambda model_path: str(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.weight_loader",
        weight_loader_module,
    )


def test_create_talker_executor_wires_scheduler_max_concurrency(
    monkeypatch,
    tmp_path: Path,
) -> None:
    talker_dir = tmp_path / "talker"
    talker_dir.mkdir()
    (talker_dir / "config.json").write_text(
        json.dumps({"max_conc": 3}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    _install_talker_executor_fake(monkeypatch, tmp_path=tmp_path, captured=captured)

    from sglang_omni.models.ming_omni.stages import create_talker_executor

    scheduler = create_talker_executor(
        model_path=str(tmp_path),
        talker_model_path=str(talker_dir),
        device="cuda:1",
        voice="DB30",
    )

    assert scheduler._max_concurrency == 3
    assert captured["executor_kwargs"]["max_conc"] == 3


def test_create_talker_executor_cli_override_beats_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    talker_dir = tmp_path / "talker"
    talker_dir.mkdir()
    (talker_dir / "config.json").write_text(
        json.dumps({"max_conc": 2}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    _install_talker_executor_fake(monkeypatch, tmp_path=tmp_path, captured=captured)

    from sglang_omni.models.ming_omni.stages import create_talker_executor

    scheduler = create_talker_executor(
        model_path=str(tmp_path),
        talker_model_path=str(talker_dir),
        max_conc=5,
    )

    assert scheduler._max_concurrency == 5
    assert captured["executor_kwargs"]["max_conc"] == 5
