# SPDX-License-Identifier: Apache-2.0
"""FDB setup preserves a prepared serving environment."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock

import pytest

from benchmarks.duplex.fdb import setup_assets


@pytest.fixture
def requirements(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(
        setup_assets,
        "requires",
        lambda name: [
            "numpy>=1.24.0",
            'einops>=0.8.1; extra == "minicpm-o"',
            'onnx>=1.18.0; extra == "minicpm-o"',
        ],
    )
    install = Mock()
    monkeypatch.setattr(setup_assets.subprocess, "run", install)
    return install


def test_setup_preserves_satisfied_model_extra(
    monkeypatch: pytest.MonkeyPatch, requirements: Mock
) -> None:
    monkeypatch.setattr(
        setup_assets,
        "version",
        lambda name: {
            "einops": "0.8.2",
            "onnx": "1.22.0",
        }[name],
        raising=False,
    )
    setup_assets.setup_model_extra()
    requirements.assert_not_called()


@pytest.mark.parametrize("installed", [None, "1.17.0"])
def test_setup_installs_a_missing_or_outdated_model_extra(
    monkeypatch: pytest.MonkeyPatch, requirements: Mock, installed: str | None
) -> None:
    def installed_version(name: str) -> str:
        if name == "einops":
            return "0.8.2"
        elif installed is None:
            raise PackageNotFoundError(name)
        else:
            return installed

    monkeypatch.setattr(setup_assets, "version", installed_version, raising=False)
    setup_assets.setup_model_extra()
    requirements.assert_called_once()
    command = requirements.call_args.args[0]
    assert "onnx>=1.18.0" in command
    assert "einops>=0.8.1" not in command
    assert requirements.call_args.kwargs["check"] is True
