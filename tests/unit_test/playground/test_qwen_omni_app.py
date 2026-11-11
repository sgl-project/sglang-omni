# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

APP_PATH = Path(__file__).resolve().parents[3] / "playground" / "qwen-omni" / "app.py"


def test_file_route_serves_only_media_and_only_to_its_own_origin(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("qwen_omni_playground", APP_PATH)
    playground = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(playground)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    secret = tmp_path / "id_ed25519"
    secret.write_text("private key")
    client = TestClient(playground.app)
    origin = {"Origin": "http://evil.example"}

    media = client.get("/v1/fs/file", params={"path": str(clip)}, headers=origin)
    refused = client.get("/v1/fs/file", params={"path": str(secret)}, headers=origin)

    assert media.status_code == 200
    assert media.content == b"RIFF"
    assert "access-control-allow-origin" not in media.headers
    assert refused.status_code == 403
    assert "private key" not in refused.text
