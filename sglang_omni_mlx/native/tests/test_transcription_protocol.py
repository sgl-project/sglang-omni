# SPDX-License-Identifier: Apache-2.0
"""POST /v1/audio/transcriptions, the JSON and event-stream API every native ASR server shares."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from native_server import SupervisedServer, multipart_form, silent_wav, sse_payloads

FRAME_COUNT = 1600
SEGMENT = {"start": 0.5, "end": 1.5, "speaker": "S01", "text": str(FRAME_COUNT)}


def test_a_request_returns_the_transcript_as_json(asr_server: SupervisedServer) -> None:
    response = asr_server.post_form({}, silent_wav(FRAME_COUNT))
    assert response.status == 200
    assert response.headers["Content-Type"] == "application/json"
    assert response.json() == {"text": str(FRAME_COUNT)}


def test_a_streamed_request_ends_with_its_metadata_and_done(
    asr_server: SupervisedServer,
) -> None:
    response = asr_server.post_form(
        {
            "stream": "true",
            "include_generation_metadata": "true",
            "language": "English",
            "max_new_tokens": "7",
        },
        silent_wav(FRAME_COUNT),
    )
    assert response.status == 200
    assert response.headers["Content-Type"] == "text/event-stream"
    payloads = sse_payloads(response.body)
    assert payloads[-1] == "[DONE]"
    assert [json.loads(payload) for payload in payloads[:-1]] == [
        {
            "type": "transcript.text.done",
            "text": str(FRAME_COUNT),
            "generation_metadata": {
                "generated_token_count": 7,
                "language": "English",
                "finish_reason": "stop",
            },
        }
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_speaker_segments_come_with_the_text(
    asr_server: SupervisedServer, stream: bool
) -> None:
    response = asr_server.post_form(
        {"outcome": "segments", "stream": str(stream).lower()}, silent_wav(FRAME_COUNT)
    )
    assert response.status == 200
    if stream:
        done = json.loads(sse_payloads(response.body)[0])
        assert done["segments"] == [SEGMENT]
    else:
        assert response.json() == {"text": str(FRAME_COUNT), "segments": [SEGMENT]}


@pytest.mark.parametrize(
    ("fields", "wav", "detail"),
    [
        ({}, None, "file is required"),
        ({"file": "audio.wav"}, None, "file is required"),
        ({}, b"not a wav file", "audio must be a RIFF/WAVE file"),
        ({}, silent_wav(FRAME_COUNT, sample_rate=8000), "audio must be 16 kHz mono"),
        ({}, silent_wav(FRAME_COUNT, channel_count=2), "audio must be 16 kHz mono"),
        ({"include_generation_metadata": "true"}, silent_wav(FRAME_COUNT), "include_generation_metadata requires stream=true"),
        ({"max_new_tokens": "ten"}, silent_wav(FRAME_COUNT), "max_new_tokens must be an integer"),
    ],
)  # fmt: skip
def test_invalid_requests_get_400(
    asr_server: SupervisedServer,
    fields: dict[str, str],
    wav: bytes | None,
    detail: str,
) -> None:
    response = asr_server.post_form(fields, wav)
    assert response.status == 400
    assert detail in response.json()["detail"]


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_transcription_is_reported(
    asr_server: SupervisedServer, stream: bool
) -> None:
    response = asr_server.post_form(
        {"outcome": "fail", "stream": str(stream).lower()}, silent_wav(FRAME_COUNT)
    )
    if stream:
        assert response.status == 200
        payloads = sse_payloads(response.body)
        assert json.loads(payloads[0])["error"]["code"] == "transcription_failed"
        assert payloads[1:] == ["[DONE]"]
    else:
        assert response.status == 500
        assert response.json() == {"detail": "Transcription failed."}


def test_a_disconnected_stream_cancels_its_transcription(
    asr_server: SupervisedServer,
) -> None:
    """A client that leaves a streamed request must not hold the worker."""
    boundary, body = multipart_form(
        {"outcome": "block", "stream": "true"}, silent_wav(FRAME_COUNT)
    )
    client = asr_server.open_request(
        "/v1/audio/transcriptions", f"multipart/form-data; boundary={boundary}", body
    )
    asr_server.wait_for_request_states({"running": 1})
    queued = asr_server.open_request(
        "/v1/audio/transcriptions", f"multipart/form-data; boundary={boundary}", body
    )
    asr_server.wait_for_request_states({"running": 1, "queued": 1})
    client.close()
    queued.close()
    asr_server.wait_for_request_states({})
    assert asr_server.post_form({}, silent_wav(FRAME_COUNT)).status == 200


def test_shutdown_cancels_a_running_transcription(
    stub_server_binary: Path, model_directory: Path
) -> None:
    server = SupervisedServer(stub_server_binary, "stub_asr", model_directory)
    boundary, body = multipart_form({"outcome": "block"}, silent_wav(FRAME_COUNT))
    client = server.open_request(
        "/v1/audio/transcriptions", f"multipart/form-data; boundary={boundary}", body
    )
    server.wait_for_request_states({"running": 1})
    server.send_line('{"command": "shutdown"}')
    assert server.read_event() == {"event": "stopped"}
    assert server.wait() == 0
    client.close()
