# SPDX-License-Identifier: Apache-2.0


from fastapi.testclient import TestClient

from playground.realtime.app import create_app
from sglang_omni.serve.webrtc_api import _load_rtc_configuration_from_env


def test_realtime_frontend_injects_ice_config(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_OMNI_ICE_URLS",
        "turn:turn.example.com:443?transport=tcp,stun:stun.example.com:3478",
    )
    monkeypatch.setenv("SGLANG_OMNI_ICE_USERNAME", "demo-user")
    monkeypatch.setenv("SGLANG_OMNI_ICE_CREDENTIAL", "demo-pass")

    client = TestClient(create_app("http://localhost:8000"))
    response = client.get("/")

    assert response.status_code == 200
    assert "window.SGLANG_OMNI_ICE_CONFIG" in response.text
    assert "turn:turn.example.com:443?transport=tcp" in response.text
    assert "stun:stun.example.com:3478" in response.text
    assert "demo-user" in response.text


def test_realtime_backend_loads_ice_config_from_env(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_OMNI_ICE_URLS",
        "turn:turn.example.com:443?transport=tcp,stun:stun.example.com:3478",
    )
    monkeypatch.setenv("SGLANG_OMNI_ICE_USERNAME", "demo-user")
    monkeypatch.setenv("SGLANG_OMNI_ICE_CREDENTIAL", "demo-pass")

    cfg = _load_rtc_configuration_from_env()

    assert cfg is not None
    assert len(cfg.iceServers) == 1
    server = cfg.iceServers[0]
    assert list(server.urls) == [
        "turn:turn.example.com:443?transport=tcp",
        "stun:stun.example.com:3478",
    ]
    assert server.username == "demo-user"
    assert server.credential == "demo-pass"


def test_realtime_frontend_prefers_scoped_ice_config(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_OMNI_ICE_URLS", "turn:generic.example.com:443?transport=tcp"
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_FRONTEND_ICE_URLS",
        "turn:frontend.example.com:443?transport=tcp",
    )
    monkeypatch.setenv("SGLANG_OMNI_ICE_USERNAME", "generic-user")
    monkeypatch.setenv("SGLANG_OMNI_FRONTEND_ICE_USERNAME", "frontend-user")

    client = TestClient(create_app("http://localhost:8000"))
    response = client.get("/")

    assert response.status_code == 200
    assert "turn:frontend.example.com:443?transport=tcp" in response.text
    assert "turn:generic.example.com:443?transport=tcp" not in response.text
    assert "frontend-user" in response.text


def test_realtime_backend_prefers_scoped_ice_config(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_OMNI_ICE_URLS", "turn:generic.example.com:443?transport=tcp"
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_BACKEND_ICE_URLS",
        "turn:backend.example.com:443?transport=tcp",
    )
    monkeypatch.setenv("SGLANG_OMNI_ICE_USERNAME", "generic-user")
    monkeypatch.setenv("SGLANG_OMNI_BACKEND_ICE_USERNAME", "backend-user")
    monkeypatch.setenv("SGLANG_OMNI_ICE_CREDENTIAL", "generic-pass")
    monkeypatch.setenv("SGLANG_OMNI_BACKEND_ICE_CREDENTIAL", "backend-pass")

    cfg = _load_rtc_configuration_from_env()

    assert cfg is not None
    assert len(cfg.iceServers) == 1
    server = cfg.iceServers[0]
    assert list(server.urls) == ["turn:backend.example.com:443?transport=tcp"]
    assert server.username == "backend-user"
    assert server.credential == "backend-pass"
