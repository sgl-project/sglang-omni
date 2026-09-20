# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle

from sglang_omni.cli.sglang_backend import run
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.manager import ConfigManager


class _DummyManager:
    def __init__(self, model_path: str):
        self.config = PipelineConfig(
            model_path=model_path,
            stages=[
                StageConfig(
                    name="stage",
                    process="pipeline",
                    factory_path="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                    terminal=True,
                )
            ],
        )

    def parse_extra_args(self, args):
        assert args == []
        return {}

    def merge_config(self, extra_args, *, extra_patches=None):
        assert extra_args == {}
        return self.config


def _run_and_expect_success(*argv: str) -> None:
    request = SimpleNamespace(argv=argv)
    with pytest.raises(SystemExit) as exc_info:
        run(request)
    assert exc_info.value.code == 0


def test_distribution_registers_backend_without_owning_sglang_executable() -> None:
    pyproject = (Path(__file__).parents[3] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    scripts = pyproject.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    entry_points = pyproject.split('[project.entry-points."sglang.serve_backends"]', 1)[
        1
    ].split("\n[", 1)[0]

    assert 'sgl-omni = "sglang_omni.cli:app"' in scripts
    assert "\nsglang = " not in scripts
    assert 'omni = "sglang_omni.cli.sglang_backend:create_backend"' in entry_points


def test_factory_implements_sglang_backend_contract() -> None:
    serve_backends = pytest.importorskip(
        "sglang.cli.serve_backends",
        reason="requires the SGLang serve-backend API",
    )
    from sglang_omni.cli.sglang_backend import create_backend

    backend = create_backend()

    assert isinstance(backend, serve_backends.ServeBackend)
    assert backend.api_version == serve_backends.SERVE_BACKEND_API_VERSION
    assert backend.run is run
    assert backend.detect is None
    assert backend.requires_model_path is False


def test_adapter_forwards_model_path_and_server_options(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        ConfigManager,
        "from_model_path",
        staticmethod(_DummyManager),
    )
    monkeypatch.setattr(
        "sglang_omni.cli.serve.launch_server",
        lambda config, **kwargs: captured.update(config=config, kwargs=kwargs),
    )

    _run_and_expect_success(
        "--model-path",
        "org/omni-model",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--model-name",
        "omni-alias",
    )

    assert captured["config"].model_path == "org/omni-model"
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8123
    assert captured["kwargs"]["model_name"] == "omni-alias"


def test_adapter_preserves_config_only_launch(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        ConfigManager,
        "from_file",
        staticmethod(lambda path: _DummyManager(f"from:{path}")),
    )
    monkeypatch.setattr(
        "sglang_omni.cli.serve.launch_server",
        lambda config, **kwargs: captured.update(config=config, kwargs=kwargs),
    )

    _run_and_expect_success("--config", "pipeline.yaml", "--port", "8124")

    assert captured["config"].model_path == "from:pipeline.yaml"
    assert captured["kwargs"]["port"] == 8124


def test_adapter_owns_backend_specific_help(capsys) -> None:
    _run_and_expect_success("--help")

    output = unstyle(capsys.readouterr().out)
    assert "Usage: sglang serve" in output
    assert "--model-path" in output
