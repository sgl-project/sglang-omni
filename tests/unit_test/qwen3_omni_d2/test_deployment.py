# SPDX-License-Identifier: Apache-2.0
"""Manifest-selected D2 session contracts at the serving entry point."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import constants

from sglang_omni.client.client import Client
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni_d2.config import Qwen3OmniD2PipelineConfig
from sglang_omni.models.qwen3_omni_d2.runtime import D2Runtime
from sglang_omni.models.qwen3_omni_d2.stages import create_scheduler
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


def test_cached_release_serves_without_repository_metadata_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline startup used to fetch .gitattributes despite both weights being cached."""
    revision = "1" * 40
    repository = tmp_path / "models--example--d2"
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "d2.json").write_text(
        json.dumps({"format": "d2.release.v1", "family": "qwen", "latency_ms": 160})
    )
    for weight_filename in ("d2.safetensors", "encoder.safetensors"):
        (snapshot / weight_filename).write_bytes(b"weights")
    tree_files = {
        path.name: {"size": path.stat().st_size, "blob_id": None}
        for path in snapshot.iterdir()
    }
    tree_files[".gitattributes"] = {"size": 100, "blob_id": None}
    (repository / "trees").mkdir()
    (repository / "trees" / f"{revision}.json").write_text(
        json.dumps({"format_version": 1, "files": tree_files})
    )
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    runtime = Mock(spec=D2Runtime)
    load_runtime = Mock(return_value=runtime)
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni_d2.stages.D2Runtime", load_runtime
    )

    scheduler = create_scheduler(
        f"example/d2@{revision}",
        device="cpu",
        runtime_python="python",
        kv_budget=128,
        seed=1,
    )
    assert load_runtime.call_args.kwargs["model_path"] == str(snapshot)
    scheduler.warm_up_serving_thread()
    runtime.step.assert_called_once_with(bytes(5120))
