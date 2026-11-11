# SPDX-License-Identifier: Apache-2.0
"""Speech input media policies and public input-error classification."""

from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    global_http_connection,
    resource_connector_scope,
)
from sglang_omni.serve.openai_api import create_app
from sglang_omni.utils.audio import decode_audio_data_uri
from tests.unit_test.fixtures.moss_speech import (
    SpeechClient,
    audio_reference,
    make_codec,
)


@pytest.mark.parametrize("case", ["allowed", "forbidden", "redirect"])
def test_speech_input_enforces_domain_policy_and_redirects(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ALLOWED_MEDIA_DOMAINS", "allowed.example")
    connector = MultiModalResourceConnector()
    fetched = []
    encoded = decode_audio_data_uri(audio_reference())

    def respond(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        if case == "redirect":
            return httpx.Response(
                302, headers={"location": "https://forbidden.example/voice.wav"}
            )
        return httpx.Response(200, content=encoded)

    codec = make_codec()
    codec.encode_tokens = Mock(wraps=codec.encode_tokens)
    source = (
        f"https://{'forbidden' if case == 'forbidden' else 'allowed'}.example/voice.wav"
    )
    with httpx.Client(
        transport=httpx.MockTransport(respond), trust_env=False
    ) as client:
        monkeypatch.setattr(global_http_connection, "get_sync_client", lambda: client)
        with resource_connector_scope(connector):
            if case == "allowed":
                assert codec.encode_audio(source)
                codec.encode_tokens.assert_called_once()
            else:
                with pytest.raises(
                    ValueError, match="Domain forbidden.example is not allowed"
                ):
                    codec.encode_audio(source)
                codec.encode_tokens.assert_not_called()
    assert fetched == ([] if case == "forbidden" else [source])


@pytest.mark.parametrize("case", ["allowed", "file_uri", "outside", "symlink"])
def test_speech_input_enforces_local_directory_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "voice.wav"
    outside = tmp_path / "voice.wav"
    encoded = decode_audio_data_uri(audio_reference())
    assert encoded is not None
    inside.write_bytes(encoded)
    outside.write_bytes(encoded)
    symlink = allowed / "link.wav"
    symlink.symlink_to(outside)
    monkeypatch.setenv("SGLANG_OMNI_ALLOWED_LOCAL_MEDIA_PATH", str(allowed))
    sources = {
        "allowed": str(inside),
        "file_uri": inside.as_uri(),
        "outside": str(outside),
        "symlink": str(symlink),
    }
    codec = make_codec()
    codec.encode_tokens = Mock(wraps=codec.encode_tokens)
    with resource_connector_scope(MultiModalResourceConnector()):
        if case in ("allowed", "file_uri"):
            assert codec.encode_audio(sources[case])
            codec.encode_tokens.assert_called_once()
        else:
            with pytest.raises(ValueError, match="not within allowed directory"):
                codec.encode_audio(sources[case])
            codec.encode_tokens.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["404", "timeout", "missing_file", "oom", "internal"]
)
def test_chat_classifies_resource_errors_without_hiding_compute_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("Timed out", request=request)
        return httpx.Response(404)

    codec = make_codec()
    if failure in ("oom", "internal"):
        error = (
            torch.OutOfMemoryError("test OOM")
            if failure == "oom"
            else RuntimeError("test encoder failure")
        )
        codec.encode_tokens = Mock(side_effect=error)
        source = audio_reference()
    elif failure == "missing_file":
        source = str(tmp_path / "missing.wav")
    else:
        source = "https://voice.example/input.wav"
    with httpx.Client(
        transport=httpx.MockTransport(respond), trust_env=False
    ) as client:
        monkeypatch.setattr(global_http_connection, "get_sync_client", lambda: client)
        with (
            resource_connector_scope(MultiModalResourceConnector()),
            TestClient(
                create_app(SpeechClient(codec), model_name="moss-speech"),
                raise_server_exceptions=False,
            ) as http,
        ):
            response = http.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio_url", "audio_url": {"url": source}}
                            ],
                        }
                    ],
                    "modalities": ["text"],
                    "stream": False,
                },
            )
    assert response.status_code == (
        500 if failure in ("oom", "internal") else 400
    ), response.text
