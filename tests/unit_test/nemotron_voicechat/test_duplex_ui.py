# SPDX-License-Identifier: Apache-2.0
"""The example serves worklet modules from the same origin as realtime."""

import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def example_module():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "voicechat_example", root / "examples/run_nemotron_voicechat_duplex.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_browser_assets(example_module):
    module = example_module
    app = FastAPI()
    module.mount_example_ui(app)
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert 'id="start"' in home.text
        for filename in ["app.mjs", "audio.mjs", "capture-worklet.js"]:
            response = client.get(f"/voicechat-assets/{filename}")
            assert response.status_code == 200
            assert "javascript" in response.headers["content-type"]
        assert client.get("/voicechat-assets/missing.js").status_code == 404


def test_worker_cleanup_follows_application_shutdown(example_module):
    from contextlib import asynccontextmanager

    module = example_module
    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("start")
        yield
        events.append("close_sessions")

    async def stop():
        events.append("stop_workers")

    app = FastAPI(lifespan=lifespan)
    module.close_workers_on_shutdown(app, stop)
    with TestClient(app):
        assert events == ["start"]
    assert events == ["start", "close_sessions", "stop_workers"]
