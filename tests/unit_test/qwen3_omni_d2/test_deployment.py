# SPDX-License-Identifier: Apache-2.0
"""Manifest-selected D2 session contracts at the serving entry point."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni_d2.config import Qwen3OmniD2PipelineConfig
from sglang_omni.serve.openai_api import create_app
from sglang_omni.utils.imports import import_string


@pytest.mark.parametrize("interval_ms", [80, 160, 320, 640, 1040])
def test_manifest_selects_the_session_cadence(tmp_path: Path, interval_ms: int) -> None:
    (tmp_path / "d2.json").write_text(
        json.dumps(
            {"format": "d2.release.v1", "family": "qwen", "latency_ms": interval_ms}
        )
    )
    config = ConfigManager.from_model_path(str(tmp_path)).config
    assert isinstance(config, Qwen3OmniD2PipelineConfig)
    client = Client(Mock())
    deployment = import_string(config.realtime_deployment_factory)(client, config)
    assert deployment.max_connections == 1
    with TestClient(
        create_app(client, model_name="d2", realtime_deployment=deployment)
    ) as http:
        capabilities = http.get("/v1/realtime/capabilities").json()
        assert capabilities["native_full_duplex"]
        assert capabilities["native_unit_ms"] == interval_ms
        assert capabilities["input_audio_format"] == {
            "type": "audio/pcm",
            "rate": 16000,
        }
        assert capabilities["output_audio_format"] == {
            "type": "audio/pcm",
            "rate": 24000,
        }
        assert capabilities["supports_server_interrupt"]
        assert not capabilities["supports_cancel"]
        assert not capabilities["supports_custom_instructions"]
        with http.websocket_connect("/v1/realtime") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            websocket.send_json(
                {
                    "event_id": "custom-prompt",
                    "type": "session.update",
                    "session": {"instructions": "Read aloud."},
                }
            )
            rejection = websocket.receive_json()
            assert rejection["type"] == "error"
            assert rejection["error"]["code"] == "not_applicable"


@pytest.mark.parametrize(("family", "interval_ms"), [("llama", 100), ("qwen", 100)])
def test_other_release_formats_do_not_select_qwen_d2(
    tmp_path: Path, family: str, interval_ms: int
) -> None:
    (tmp_path / "d2.json").write_text(
        json.dumps(
            {"format": "d2.release.v1", "family": family, "latency_ms": interval_ms}
        )
    )
    with pytest.raises(ValueError):
        ConfigManager.from_model_path(str(tmp_path))
