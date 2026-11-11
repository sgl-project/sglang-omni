# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle

from sglang_omni.cli import app
from sglang_omni.cli.sglang_backend import run
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.manager import ConfigManager


def single_stage_config(model_path: str) -> PipelineConfig:
    return PipelineConfig(
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


class DummyManager:
    def __init__(self, model_path: str):
        self.config = single_stage_config(model_path)

    def parse_extra_args(self, args):
        assert args == []
        return {}

    def merge_config(self, extra_args, *, extra_patches=None):
        assert extra_args == {}
        return self.config


def run_and_expect_success(*argv: str) -> None:
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
        staticmethod(DummyManager),
    )
    monkeypatch.setattr(
        "sglang_omni.cli.serve.launch_server",
        lambda config, **kwargs: captured.update(config=config, kwargs=kwargs),
    )

    run_and_expect_success(
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
        staticmethod(lambda path: DummyManager(f"from:{path}")),
    )
    monkeypatch.setattr(
        "sglang_omni.cli.serve.launch_server",
        lambda config, **kwargs: captured.update(config=config, kwargs=kwargs),
    )

    run_and_expect_success("--config", "pipeline.yaml", "--port", "8124")

    assert captured["config"].model_path == "from:pipeline.yaml"
    assert captured["kwargs"]["port"] == 8124


@dataclass(kw_only=True, frozen=True)
class ServeOutcome:
    exit_code: int | str | None
    launched_config_json: str | None
    launch_options: str | None
    root_log_level: int
    console_log_line: str


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ServeOutcome]]:
    """Record what serve would launch, with the real config resolution."""
    root_logger = logging.getLogger()
    # Swapped out, never closed: pytest keeps using these after the test.
    saved_handlers = root_logger.handlers
    saved_level = root_logger.level
    recorded: list[ServeOutcome] = []

    def record_launch(config: PipelineConfig, **launch_options: object) -> None:
        recorded.append(
            ServeOutcome(
                exit_code=None,
                launched_config_json=config.model_dump_json(),
                launch_options=repr(sorted(launch_options.items())),
                root_log_level=root_logger.level,
                console_log_line=format_console_log_line(),
            )
        )

    monkeypatch.setattr(
        ConfigManager,
        "from_model_path",
        staticmethod(
            lambda model_path, variant=None: ConfigManager(
                single_stage_config(f"{model_path}:{variant}")
            )
        ),
    )
    monkeypatch.setattr(
        ConfigManager,
        "from_file",
        staticmethod(lambda path: ConfigManager(single_stage_config(f"file:{path}"))),
    )
    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", record_launch)
    yield recorded
    root_logger.handlers = saved_handlers
    root_logger.setLevel(saved_level)


def install_warning_root_handler() -> None:
    """Leave root logging the way importing sglang.cli.serve leaves it."""
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.handlers = [console_handler]
    root_logger.setLevel(logging.WARNING)


def format_console_log_line() -> str:
    log_record = logging.LogRecord(
        "parity", logging.INFO, __file__, 0, "message", None, None
    )
    log_record.created = 0.0
    log_record.msecs = 0.0
    return logging.getLogger().handlers[0].format(log_record)


def serve_with_sgl_omni(
    argv: tuple[str, ...], launches: list[ServeOutcome]
) -> ServeOutcome:
    install_warning_root_handler()
    with pytest.raises(SystemExit) as exc_info:
        app(["serve", *argv], prog_name="sgl-omni")
    return outcome_after_exit(exc_info.value.code, launches)


def serve_with_sglang(
    argv: tuple[str, ...],
    launches: list[ServeOutcome],
    monkeypatch: pytest.MonkeyPatch,
) -> ServeOutcome:
    sglang_cli_serve = pytest.importorskip(
        "sglang.cli.serve", reason="requires the SGLang serve-backend API"
    )
    monkeypatch.setattr(
        sglang_cli_serve, "kill_process_tree", lambda pid, include_parent: None
    )
    monkeypatch.setattr("sglang.srt.plugins.load_plugins", lambda: None)
    install_warning_root_handler()
    with pytest.raises(SystemExit) as exc_info:
        sglang_cli_serve.serve(None, [*argv, "--model-type", "omni"])
    return outcome_after_exit(exc_info.value.code, launches)


def outcome_after_exit(
    exit_code: int | str | None, launches: list[ServeOutcome]
) -> ServeOutcome:
    if launches:
        return replace(launches.pop(), exit_code=exit_code)
    else:
        return ServeOutcome(
            exit_code=exit_code,
            launched_config_json=None,
            launch_options=None,
            root_log_level=logging.getLogger().level,
            console_log_line=format_console_log_line(),
        )


class ExpectedResult(Enum):
    LAUNCHED = "launched"
    HELP = "help"
    USAGE_ERROR = "usage_error"


EXIT_CODES: dict[ExpectedResult, int] = {
    ExpectedResult.LAUNCHED: 0,
    ExpectedResult.HELP: 0,
    ExpectedResult.USAGE_ERROR: 2,
}

EQUIVALENT_COMMAND_LINES = [
    (("/models/whisper-tiny",), ExpectedResult.LAUNCHED),
    (
        (
            "/models/whisper-tiny",
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
            "--model-name",
            "whisper-tiny-cli-smoke",
        ),
        ExpectedResult.LAUNCHED,
    ),
    (
        ("--model-path", "/models/whisper-tiny", "--log-level", "warning"),
        ExpectedResult.LAUNCHED,
    ),
    (
        ("--model-path=/models/whisper-tiny", "--text-only", "--stage.gpu", "1"),
        ExpectedResult.LAUNCHED,
    ),
    (
        ("/models/whisper-tiny", "--stage.gpu=1", "--enable-realtime"),
        ExpectedResult.LAUNCHED,
    ),
    (("--config", "pipeline.yaml", "--port", "8124"), ExpectedResult.LAUNCHED),
    (
        ("--config", "pipeline.yaml", "--model-path", "/models/override"),
        ExpectedResult.LAUNCHED,
    ),
    (("-h",), ExpectedResult.HELP),
    (("--port", "8124"), ExpectedResult.USAGE_ERROR),
    (("/models/whisper-tiny", "--stage.gpu"), ExpectedResult.USAGE_ERROR),
    (("/models/whisper-tiny", "--log-level", "verbose"), ExpectedResult.USAGE_ERROR),
]


@pytest.mark.parametrize(("argv", "expected_result"), EQUIVALENT_COMMAND_LINES)
def test_sglang_serve_matches_sgl_omni_serve(
    argv: tuple[str, ...],
    expected_result: ExpectedResult,
    launches: list[ServeOutcome],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sgl_omni_outcome = serve_with_sgl_omni(argv, launches)
    sglang_outcome = serve_with_sglang(argv, launches, monkeypatch)

    assert sgl_omni_outcome.exit_code == EXIT_CODES[expected_result]
    assert (sgl_omni_outcome.launched_config_json is not None) == (
        expected_result is ExpectedResult.LAUNCHED
    )
    assert sglang_outcome == sgl_omni_outcome


def test_positional_model_path_matches_model_path_flag(
    launches: list[ServeOutcome],
) -> None:
    positional = serve_with_sgl_omni(
        ("/models/whisper-tiny", "--port", "18080"), launches
    )
    flagged = serve_with_sgl_omni(
        ("--model-path", "/models/whisper-tiny", "--port", "18080"), launches
    )

    assert positional.exit_code == 0
    assert positional == flagged


def test_short_help_prints_serve_usage(
    launches: list[ServeOutcome], capsys: pytest.CaptureFixture[str]
) -> None:
    outcome = serve_with_sgl_omni(("-h",), launches)

    assert outcome.exit_code == 0
    assert outcome.launched_config_json is None
    assert "sgl-omni serve [OPTIONS]" in unstyle(capsys.readouterr().out)


def test_log_level_and_format_apply_over_an_existing_console_handler(
    launches: list[ServeOutcome],
) -> None:
    outcome = serve_with_sgl_omni(("/models/whisper-tiny",), launches)

    assert outcome.root_log_level == logging.INFO
    assert outcome.console_log_line.endswith(" [INFO] parity: message")


def test_serve_keeps_handlers_it_does_not_own(launches: list[ServeOutcome]) -> None:
    install_warning_root_handler()
    plugin_handler = logging.NullHandler()
    logging.getLogger().addHandler(plugin_handler)

    with pytest.raises(SystemExit):
        app(["serve", "/models/whisper-tiny"], prog_name="sgl-omni")

    assert plugin_handler in logging.getLogger().handlers
