# SPDX-License-Identifier: Apache-2.0
"""Exercise the embedded app's media policy before generation."""

import base64
import ipaddress
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from sglang_omni.preprocessing.resource_connector import global_http_connection
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.speech_limits import MAX_REFERENCE_AUDIO_BYTES
from tests.unit_test.fixtures.moss_speech import (
    SpeechClient,
    audio_reference,
    make_codec,
)


def chat_body(source: str, *, reference: bool) -> dict:
    content = (
        "Hello" if reference else [{"type": "audio_url", "audio_url": {"url": source}}]
    )
    body = {"messages": [{"role": "user", "content": content}]}
    if reference:
        body.update(modalities=["audio"], audio={"ref_audio": source})
    return body


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    "case", ["path", "file_uri", "outside", "symlink", "oversize", "disabled"]
)
def test_local_media_policy(tmp_path: Path, case: str, reference: bool) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "voice.wav"
    outside = tmp_path / "other.wav"
    inside.write_bytes(base64.b64decode(audio_reference().split(",")[1]))
    outside.write_bytes(inside.read_bytes())
    link = allowed / "link.wav"
    link.symlink_to(outside)
    if case == "oversize":
        with inside.open("ab") as stream:
            stream.truncate(MAX_REFERENCE_AUDIO_BYTES + 1)
    source = {
        "file_uri": inside.as_uri(),
        "outside": str(outside),
        "symlink": str(link),
    }.get(case, str(inside))
    with TestClient(
        create_app(
            SpeechClient(make_codec()),
            architectures=["MossSpeechForCausalLM"],
            allowed_local_media_path=None if case == "disabled" else str(allowed),
        )
    ) as http:
        response = http.post(
            "/v1/chat/completions", json=chat_body(source, reference=reference)
        )
    assert response.status_code == (
        200 if case in ("path", "file_uri") else 400
    ), response.text


@pytest.mark.parametrize("reference", [True, False])
@pytest.mark.parametrize("case", ["allowed", "redirect", "404", "timeout", "oversize"])
def test_remote_media_policy(
    monkeypatch: pytest.MonkeyPatch, reference: bool, case: str
) -> None:
    fetched = []

    def respond(request: httpx.Request) -> httpx.Response:
        fetched.append(request.url.host)
        if case == "redirect":
            return httpx.Response(
                302, headers={"location": "https://other.example/voice"}
            )
        if case == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if case == "oversize":
            return httpx.Response(
                200, headers={"content-length": str(MAX_REFERENCE_AUDIO_BYTES + 1)}
            )
        return httpx.Response(
            404 if case == "404" else 200,
            content=base64.b64decode(audio_reference().split(",")[1]),
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(global_http_connection, "get_sync_client", lambda: client)
        with (
            patch(
                "sglang_omni.preprocessing.resource_connector.resolve_remote_addresses",
                return_value=(ipaddress.ip_address("93.184.216.34"),),
            ),
            TestClient(
                create_app(
                    SpeechClient(make_codec()),
                    architectures=["MossSpeechForCausalLM"],
                    allowed_media_domains=["voice.example"],
                )
            ) as http,
        ):
            response = http.post(
                "/v1/chat/completions",
                json=chat_body("https://voice.example/voice", reference=reference),
            )
    assert response.status_code == (200 if case == "allowed" else 400), response.text
    assert fetched == ["voice.example"]


@pytest.mark.parametrize(
    "error", [torch.OutOfMemoryError("test OOM"), RuntimeError("test encoder failure")]
)
def test_compute_error_is_server_error(error: Exception) -> None:
    codec = make_codec()
    codec.encode_tokens = Mock(side_effect=error)
    with TestClient(
        create_app(SpeechClient(codec), architectures=["MossSpeechForCausalLM"]),
        raise_server_exceptions=False,
    ) as http:
        response = http.post(
            "/v1/chat/completions", json=chat_body(audio_reference(), reference=False)
        )
    assert response.status_code == 500
