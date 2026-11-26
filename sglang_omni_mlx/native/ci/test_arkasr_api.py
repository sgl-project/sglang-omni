# SPDX-License-Identifier: Apache-2.0
"""API tests for arkasr_server, run against the real model.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_arkasr_api.py
"""

from __future__ import annotations

import http.client
import json
import time
import uuid
from collections.abc import Iterator

import pytest
from test_server_api import (
    DATA_ROOT,
    RUNTIME_BIN,
    Server,
    clip,
    long_wav,
    model_directory,
    pcm16,
    sse_events,
    wav_bytes,
)
from websockets.exceptions import InvalidHandshake
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
ARKASR_REPO = "Edge0/ARK-ASR-3B"
ARKASR_DRAFT_REPO = "wilsonzheng0327/ARK-ASR-3B-dflash"


@pytest.fixture(scope="module", params=[False], ids=["plain"])
def arkasr_server(request: pytest.FixtureRequest) -> Iterator[Server]:
    flags = (
        ("--draft-model-path", str(model_directory(ARKASR_DRAFT_REPO)))
        if request.param
        else ()
    )
    running = Server("arkasr_server", "arkasr", model_directory(ARKASR_REPO), flags)
    yield running
    running.stop()


def test_arkasr_final_request_streams_text_and_generation_metadata(
    arkasr_server: Server,
) -> None:
    assert arkasr_server.ready["model_name"].startswith("voxt-arkasr-")
    status, body = arkasr_server.post_form(
        {"stream": "true", "language": "zh", "include_generation_metadata": "true"},
        clip("0152_zh_short"),
    )
    assert status == 200
    [done, end] = sse_events(body)
    assert end == "[DONE]"
    assert done["type"] == "transcript.text.done"
    assert done["text"]
    metadata = done["generation_metadata"]
    assert metadata["language"] == "zh"
    assert metadata["finish_reason"] == "stop"
    assert metadata["generated_token_count"] > 0


def test_arkasr_plain_request_returns_json_text(arkasr_server: Server) -> None:
    status, body = arkasr_server.post_form({}, clip("0001_en_short"))
    assert status == 200
    assert json.loads(body)["text"]


@pytest.mark.parametrize(
    ("arkasr_server", "drafted", "budget_fields"),
    [(False, False, {}), (False, False, {"max_new_tokens": "0"}), (True, True, {})],
    ids=["plain", "plain-zero-budget", "dflash"],
    indirect=["arkasr_server"],
)
def test_arkasr_decodes_long_audio_one_window_at_a_time(
    arkasr_server: Server, drafted: bool, budget_fields: dict[str, str]
) -> None:
    # Note (AzizbekMuminoff): the checkpoint's processor stops at 30 s, so the
    # text past it comes only from the later windows. The default budget,
    # unset or 0, must not cut a long request short either.
    fields = {"stream": "true", "include_generation_metadata": "true", **budget_fields}
    status, body = arkasr_server.post_form(fields, long_wav(30))
    assert status == 200
    one_window = sse_events(body)[0]["generation_metadata"]
    status, body = arkasr_server.post_form(fields, long_wav(75))
    assert status == 200
    three_windows = sse_events(body)[0]["generation_metadata"]
    assert three_windows["finish_reason"] == "stop"
    assert (
        three_windows["generated_token_count"] > 2 * one_window["generated_token_count"]
    )
    assert len(three_windows["window_texts"]) == 3
    assert ("verify_round_count" in three_windows) == drafted


@pytest.mark.parametrize(
    ("arkasr_server", "drafted"),
    [(False, False), (True, True)],
    ids=["plain", "dflash"],
    indirect=["arkasr_server"],
)
def test_arkasr_token_budget_covers_the_whole_request(
    arkasr_server: Server, drafted: bool
) -> None:
    # Like Cohere, an explicit max_new_tokens is one budget for every window
    # of the request together, not a fresh one per 30 s window.
    status, body = arkasr_server.post_form(
        {
            "stream": "true",
            "include_generation_metadata": "true",
            "max_new_tokens": "2",
        },
        long_wav(75),
    )
    assert status == 200
    metadata = sse_events(body)[0]["generation_metadata"]
    assert metadata["finish_reason"] == "length"
    assert metadata["generated_token_count"] == 2
    assert ("verify_round_count" in metadata) == drafted


def test_arkasr_zero_token_budget_means_the_default(
    arkasr_server: Server,
) -> None:
    status, body = arkasr_server.post_form(
        {"max_new_tokens": "0"}, clip("0001_en_short")
    )
    assert status == 200
    assert json.loads(body)["text"]


def test_arkasr_zero_padded_windows_add_no_text(arkasr_server: Server) -> None:
    # Note (AzizbekMuminoff): zero padding that the 30 s split turns into its
    # own window reads as silence to the checkpoint, which hallucinates text
    # for it ("Okay." on the frozen silence controls). Padding at either end
    # of a request must not change the transcript.
    speech = pcm16("0152_zh_short")
    status, body = arkasr_server.post_form({}, wav_bytes(speech))
    assert status == 200
    reference = json.loads(body)["text"]
    assert reference
    # One silent sample past 30 s: the split's second window is all zeros.
    status, body = arkasr_server.post_form(
        {}, wav_bytes(speech + b"\0" * (2 * 480001 - len(speech)))
    )
    assert status == 200
    assert json.loads(body)["text"] == reference
    # 29.2 s of leading silence: the split's first window is all zeros.
    status, body = arkasr_server.post_form({}, wav_bytes(b"\0" * (2 * 467200) + speech))
    assert status == 200
    assert json.loads(body)["text"] == reference


@pytest.mark.parametrize(
    ("fields", "wav"),
    [
        ({}, None),
        ({}, b"not audio"),
        ({"include_generation_metadata": "true"}, "wav"),
        ({"max_new_tokens": "many"}, "wav"),
        ({"max_new_tokens": "-1"}, "wav"),
    ],
)
def test_arkasr_invalid_requests_are_rejected(
    arkasr_server: Server, fields: dict[str, str], wav: bytes | str | None
) -> None:
    status, body = arkasr_server.post_form(
        fields, clip("0006_en_short") if wav == "wav" else wav
    )
    assert status == 400
    assert "detail" in json.loads(body)


def test_arkasr_disconnected_stream_stops_its_decode(
    arkasr_server: Server,
) -> None:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        ).encode()
        + long_wav(600)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection("127.0.0.1", arkasr_server.port, timeout=30)
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    time.sleep(2.0)
    assert json.loads(arkasr_server.request("GET", "/health")[1])["request_states"] == {
        "running": 1
    }
    closed_at = time.monotonic()
    response.close()
    connection.close()
    while (
        json.loads(arkasr_server.request("GET", "/health")[1])["request_states"] != {}
    ):
        assert (
            time.monotonic() - closed_at < 2.0
        ), "the decode kept running after its client left"
        time.sleep(0.05)


def test_arkasr_serves_no_realtime_api(arkasr_server: Server) -> None:
    with pytest.raises(InvalidHandshake):
        connect(f"ws://127.0.0.1:{arkasr_server.port}/v1/realtime")
