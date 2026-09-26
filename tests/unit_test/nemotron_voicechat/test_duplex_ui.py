# SPDX-License-Identifier: Apache-2.0
"""The example serves browser assets and releases workers on shutdown."""

import importlib.util
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def example_module() -> ModuleType:
    repository_root = Path(__file__).resolve().parents[3]
    module_spec = importlib.util.spec_from_file_location(
        "voicechat_example",
        repository_root / "examples/run_nemotron_voicechat_duplex.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_browser_assets_are_served_with_javascript_content_type(
    example_module: ModuleType,
) -> None:
    application = FastAPI()
    example_module.mount_example_ui(application)
    with TestClient(application) as client:
        home_response = client.get("/")
        assert home_response.status_code == 200
        for filename in ["app.mjs", "audio.mjs", "capture-worklet.js"]:
            response = client.get(f"/voicechat-assets/{filename}")
            assert response.status_code == 200
            assert "javascript" in response.headers["content-type"]
        assert client.get("/voicechat-assets/missing.js").status_code == 404


def test_workers_stop_after_sessions_close(example_module: ModuleType) -> None:
    lifecycle_events: list[str] = []

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        lifecycle_events.append("start")
        yield
        lifecycle_events.append("close_sessions")

    async def stop_workers() -> None:
        lifecycle_events.append("stop_workers")

    application = FastAPI(lifespan=lifespan)
    example_module.close_workers_on_shutdown(application, stop_workers)
    with TestClient(application):
        assert lifecycle_events == ["start"]
    assert lifecycle_events == ["start", "close_sessions", "stop_workers"]


def test_workers_stop_when_session_shutdown_raises(example_module: ModuleType) -> None:
    lifecycle_events: list[str] = []

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        raise RuntimeError("session shutdown failed")

    async def stop_workers() -> None:
        lifecycle_events.append("stop_workers")

    application = FastAPI(lifespan=lifespan)
    example_module.close_workers_on_shutdown(application, stop_workers)
    with pytest.raises(RuntimeError, match="session shutdown failed"):
        with TestClient(application):
            pass
    assert lifecycle_events == ["stop_workers"]
