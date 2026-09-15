# SPDX-License-Identifier: Apache-2.0
"""The example serves worklet modules from the same origin as realtime."""

import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_browser_assets():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "voicechat_example", root / "examples/run_nemotron_voicechat_duplex.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
