# SPDX-License-Identifier: Apache-2.0
"""Managed server configuration, health checks, and proxy isolation."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import getproxies_environment

import pytest
import requests.utils

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


def test_managed_server_health_bypasses_system_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "health").write_text("healthy", encoding="utf-8")
    with socket.socket() as health_socket, socket.socket() as proxy_socket:
        health_socket.bind(("127.0.0.1", 0))
        port = health_socket.getsockname()[1]
        proxy_socket.bind(("127.0.0.1", 0))
        health_socket.close()
        proxy_url = f"http://127.0.0.1:{proxy_socket.getsockname()[1]}"

        def discover_proxies() -> dict[str, str]:
            return getproxies_environment() or {"http": proxy_url}

        monkeypatch.setattr(requests.utils, "getproxies", discover_proxies)
        monkeypatch.setattr(requests.utils, "proxy_bypass", Mock(return_value=False))
        process = utils.start_server_from_cmd(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(tmp_path),
            ],
            tmp_path / "server.log",
            port,
            timeout=10,
        )
        try:
            assert process.poll() is None
        finally:
            utils.stop_server(process)
        assert process.poll() is not None


@pytest.mark.parametrize("configured", [False, True])
def test_disable_proxy_restores_environment_after_nested_error(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    monkeypatch.setattr(requests.utils, "proxy_bypass", Mock(return_value=False))
    proxy_names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )
    for name in proxy_names:
        if configured:
            monkeypatch.setenv(name, f"original-{name}")
        else:
            monkeypatch.delenv(name, raising=False)
    original_environment = dict(os.environ)
    with utils.disable_proxy():
        outer_environment = dict(os.environ)
        with pytest.raises(RuntimeError, match="nested failure"):
            with utils.disable_proxy():
                for host in ("localhost", "127.0.0.1", "[::1]"):
                    assert requests.utils.should_bypass_proxies(
                        f"http://{host}:8000/health", no_proxy=None
                    )
                raise RuntimeError("nested failure")
        assert dict(os.environ) == outer_environment
    assert dict(os.environ) == original_environment
