# SPDX-License-Identifier: Apache-2.0
"""Root command options and subcommand dispatch."""

import subprocess
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sglang_omni import __version__
from sglang_omni.cli import app


@pytest.mark.parametrize(
    "arguments",
    [
        ["--version"],
        ["--version", "serve", "--model-path", "unused"],
    ],
)
def test_version_exits_before_model_resolution(arguments: list[str]) -> None:
    """Version queries succeed without resolving or starting a model."""
    with (
        patch(
            "sglang_omni.cli.serve.ConfigManager.from_model_path",
            side_effect=AssertionError("Version queries must not resolve a model"),
        ) as resolve_model,
        patch(
            "sglang_omni.cli.serve.launch_server",
            side_effect=AssertionError("Version queries must not start a server"),
        ) as launch_server,
    ):
        result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert result.stdout == f"sgl-omni {__version__}\n"
    assert result.stderr == ""
    resolve_model.assert_not_called()
    launch_server.assert_not_called()


@pytest.mark.parametrize(
    "command",
    [
        ["sgl-omni"],
        [sys.executable, "-m", "sglang_omni.cli"],
    ],
)
def test_version_from_command_line_entry_points(command: list[str]) -> None:
    """Installed and module entry points expose the same package version."""
    result = subprocess.run(
        [*command, "--version"], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"sgl-omni {__version__}\n"


def test_root_help_lists_version_and_existing_commands() -> None:
    """The new root option remains discoverable alongside existing commands."""
    result = CliRunner().invoke(app, ["--help"], color=False)

    assert result.exit_code == 0, result.output
    assert "--version" in result.stdout
    for command in ("config", "check-gpu", "serve"):
        assert command in result.stdout


@pytest.mark.parametrize(
    ("arguments", "exit_code"),
    [
        ([], 2),
        (["config", "--help"], 0),
        (["check-gpu", "--help"], 0),
        (["serve", "--help"], 0),
    ],
)
def test_root_options_preserve_existing_help(
    arguments: list[str], exit_code: int
) -> None:
    """Normal invocations retain their help and missing-command behavior."""
    result = CliRunner().invoke(app, arguments, color=False)

    assert result.exit_code == exit_code, result.output
    assert "Usage:" in result.output
    assert f"sgl-omni {__version__}\n" not in result.stdout
