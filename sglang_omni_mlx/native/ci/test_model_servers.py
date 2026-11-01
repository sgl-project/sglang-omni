# SPDX-License-Identifier: Apache-2.0
"""API tests for the native model servers beside qwen3_asr_server, run against
the real models.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_model_servers.py
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import socket
import struct
import subprocess
import time
import uuid
import wave
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_server_api import (
    DATA_ROOT,
    RUNTIME_BIN,
    Server,
    clip,
    long_wav,
    pcm16,
    sse_events,
)
from websockets.exceptions import InvalidHandshake
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
WHISPER_LARGE_V3_REPO = "mlx-community/whisper-large-v3-mlx"
WHISPER_SMALL_REPO = "mlx-community/whisper-small-mlx"
COHERE_REPO = "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
VAD_REPO = "mlx-community/silero-vad-v6"
# Voxt's settings for cutting long audio at speech.
VAD_FIELDS = {
    "vad_threshold": "0.5",
    "vad_min_speech_ms": "220",
    "vad_min_silence_ms": "420",
    "vad_speech_pad_ms": "180",
    "vad_merge_gap_seconds": "1.0",
    "vad_max_chunk_seconds": "24.0",
}
MOSS_REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
NEMOTRON_REPO = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
SENSEVOICE_REPO = "mlx-community/SenseVoiceSmall"
# Voxt's SenseVoice settings for long audio: speech runs cut into overlapping chunks.
SENSEVOICE_VAD_FIELDS = {
    "vad_threshold": "0.5",
    "vad_min_speech_ms": "220",
    "vad_min_silence_ms": "420",
    "vad_speech_pad_ms": "180",
    "vad_max_chunk_seconds": "24.0",
    "vad_chunk_overlap_seconds": "0.35",
}


class ModelServer(Server):
    """Another model's server binary, started and spoken to as qwen3_asr_server is."""

    def __init__(
        self, binary: str, model_kind: str, repo: str, flags: tuple[str, ...] = ()
    ) -> None:
        super().__init__(
            binary,
            model_kind,
            Path(DATA_ROOT) / "models" / repo.replace("/", "_"),
            flags,
        )


@pytest.fixture(scope="module")
def whisper_server() -> Iterator[ModelServer]:
    running = ModelServer("whisper_server", "whisper", WHISPER_REPO)
    yield running
    running.stop()


def test_whisper_ready_event_names_a_loopback_endpoint(
    whisper_server: ModelServer,
) -> None:
    assert whisper_server.ready["event"] == "ready"
    assert whisper_server.ready["host"] == "127.0.0.1"
    assert whisper_server.ready["server_pid"] == whisper_server.process.pid
    assert whisper_server.ready["model_name"].startswith("voxt-whisper-")
    status, body = whisper_server.request("GET", "/health")
    assert (status, json.loads(body)) == (
        200,
        {"status": "healthy", "running": True, "request_states": {}},
    )


def test_whisper_final_request_streams_text_and_generation_metadata(
    whisper_server: ModelServer,
) -> None:
    status, body = whisper_server.post_form(
        {
            "stream": "true",
            "language": "en",
            "max_new_tokens": "1024",
            "temperature": "0.0",
            "include_generation_metadata": "true",
        },
        clip("0006_en_short"),
    )
    assert status == 200
    assert sse_events(body) == [
        {
            "type": "transcript.text.done",
            "text": "Surely you are not thinking of going off there.",
            "generation_metadata": {
                "generated_token_count": 10,
                "language": "en",
                "finish_reason": "stop",
            },
        },
        "[DONE]",
    ]


def test_whisper_plain_request_returns_json_text(whisper_server: ModelServer) -> None:
    status, body = whisper_server.post_form({"language": "zh"}, clip("0152_zh_short"))
    assert (status, json.loads(body)) == (
        200,
        {"text": "互联网结合了大众传播和人际传播的要素"},
    )


@pytest.mark.parametrize("repo", [WHISPER_LARGE_V3_REPO, WHISPER_SMALL_REPO])
def test_whisper_fp16_variant_server_transcribes(repo: str) -> None:
    server = ModelServer("whisper_server", "whisper", repo)
    try:
        status, body = server.post_form(
            {"language": "en", "max_new_tokens": "128"}, clip("0006_en_short")
        )
        assert status == 200
        assert json.loads(body)["text"].strip()
    finally:
        server.stop()


def corrupt_npz(path: Path, field: str) -> None:
    """A stored one-member archive whose directory lies about a size."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("w.npy", b"\0" * 64)
    data = bytearray(path.read_bytes())
    end = len(data) - 22
    directory = struct.unpack_from("<I", data, end + 16)[0]
    if field == "directory_offset":
        struct.pack_into("<I", data, end + 16, len(data))
    else:
        struct.pack_into("<H", data, directory + 28, 0xFFFF)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("field", ["directory_offset", "name_length"])
def test_whisper_corrupt_npz_fails_to_start(tmp_path: Path, field: str) -> None:
    source = Path(DATA_ROOT) / "models" / WHISPER_LARGE_V3_REPO.replace("/", "_")
    for item in source.iterdir():
        if item.name != "weights.npz":
            (tmp_path / item.name).symlink_to(item)
    corrupt_npz(tmp_path / "weights.npz", field)
    server = Server("whisper_server", "whisper", tmp_path)
    assert server.ready["event"] == "failed"
    assert "corrupt" in server.ready["reason"]
    assert server.process.wait(timeout=30) != 0


@pytest.mark.parametrize(
    ("fields", "wav"),
    [
        ({}, None),
        ({}, b"not audio"),
        ({"include_generation_metadata": "true"}, "wav"),
        ({"max_new_tokens": "many"}, "wav"),
        ({"temperature": "warm"}, "wav"),
        ({"temperature": "nan"}, "wav"),
        ({"temperature": "inf"}, "wav"),
        ({"temperature": "-1"}, "wav"),
        ({"max_new_tokens": "-1"}, "wav"),
    ],
)
def test_whisper_invalid_requests_are_rejected(
    whisper_server: ModelServer, fields: dict[str, str], wav: bytes | str | None
) -> None:
    status, body = whisper_server.post_form(
        fields, clip("0006_en_short") if wav == "wav" else wav
    )
    assert status == 400
    assert "detail" in json.loads(body)


def test_whisper_bad_number_error_names_its_field(whisper_server: ModelServer) -> None:
    status, body = whisper_server.post_form(
        {"max_new_tokens": "many"}, clip("0006_en_short")
    )
    assert status == 400
    assert "max_new_tokens" in json.loads(body)["detail"]


def test_whisper_language_with_a_truncated_utf8_sequence_is_unknown(
    whisper_server: ModelServer,
) -> None:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n'.encode()
        + b"en\xf0\r\n"
        + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'.encode()
        + clip("0006_en_short")
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, response = whisper_server.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    # Note (khazic): an unknown language sends no language token, as Swift does; the
    # truncated sequence must not be read past.
    assert status == 200
    assert "text" in json.loads(response)
    assert whisper_server.request("GET", "/health")[0] == 200


def test_whisper_disconnected_stream_stops_its_decode(
    whisper_server: ModelServer,
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
    connection = http.client.HTTPConnection(
        "127.0.0.1", whisper_server.port, timeout=30
    )
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    time.sleep(2.0)
    assert json.loads(whisper_server.request("GET", "/health")[1])[
        "request_states"
    ] == {"running": 1}
    closed_at = time.monotonic()
    # Note (khazic): http.client keeps the socket open while the response object is.
    response.close()
    connection.close()
    while (
        json.loads(whisper_server.request("GET", "/health")[1])["request_states"] != {}
    ):
        assert (
            time.monotonic() - closed_at < 2.0
        ), "the decode kept running after its client left"
        time.sleep(0.05)


def test_whisper_serves_no_realtime_api(whisper_server: ModelServer) -> None:
    with pytest.raises(InvalidHandshake):
        connect(f"ws://127.0.0.1:{whisper_server.port}/v1/realtime")


def test_whisper_shutdown_reports_stopped() -> None:
    running = ModelServer("whisper_server", "whisper", WHISPER_REPO)
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0


@pytest.mark.parametrize(
    ("binary", "other_kind"),
    [
        ("whisper_server", "qwen3_asr"),
        ("cohere_transcribe_server", "whisper"),
        ("nemotron_asr_server", "qwen3_asr"),
        ("parakeet_server", "cohere_transcribe"),
        ("sensevoice_server", "cohere_transcribe"),
        ("music3_server", "qwen3_asr"),
    ],
)
def test_server_serves_only_its_model_kind(binary: str, other_kind: str) -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / binary),
            "--supervised",
            "--model-kind",
            other_kind,
            "--model-directory",
            "x",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.fixture(scope="module")
def cohere_server() -> Iterator[ModelServer]:
    running = ModelServer("cohere_transcribe_server", "cohere_transcribe", COHERE_REPO)
    yield running
    running.stop()


def test_cohere_final_request_streams_text_and_generation_metadata(
    cohere_server: ModelServer,
) -> None:
    assert cohere_server.ready["model_name"].startswith("voxt-cohere_transcribe-")
    status, body = cohere_server.post_form(
        {
            "stream": "true",
            "language": "zh",
            "use_punctuation": "true",
            "max_new_tokens": "1024",
            "temperature": "0.0",
            "chunk_duration": "1200",
            "min_chunk_duration": "1",
            "include_generation_metadata": "true",
        },
        clip("0152_zh_short"),
    )
    assert status == 200
    assert sse_events(body) == [
        {
            "type": "transcript.text.done",
            "text": "互联网结合了大众传播和人际传播的要素。",
            "generation_metadata": {
                "generated_token_count": 20,
                "language": "zh",
                "finish_reason": "stop",
            },
        },
        "[DONE]",
    ]


def test_cohere_cuts_long_audio_at_speech(cohere_server: ModelServer) -> None:
    vad_directory = str(Path(DATA_ROOT) / "models" / VAD_REPO.replace("/", "_"))
    status, body = cohere_server.post_form(
        {"language": "en", "vad_model_directory": vad_directory, **VAD_FIELDS},
        clip("0344_en_long"),
    )
    assert status == 200
    text = json.loads(body)["text"]
    assert text.startswith("In every way they sought to undermine the authority")
    assert len(text.split("\n")) > 1


def test_cohere_voice_activity_needs_every_setting(cohere_server: ModelServer) -> None:
    fields = {"vad_model_directory": "x", **VAD_FIELDS}
    fields.pop("vad_threshold")
    status, body = cohere_server.post_form(fields, clip("0006_en_short"))
    assert status == 400
    assert "vad_threshold" in json.loads(body)["detail"]


def test_cohere_reports_the_language_it_decoded(cohere_server: ModelServer) -> None:
    status, body = cohere_server.post_form(
        {"language": "fr-FR", "stream": "true", "include_generation_metadata": "true"},
        clip("0006_en_short"),
    )
    assert status == 200
    assert sse_events(body)[0]["generation_metadata"]["language"] == "en"


@pytest.mark.parametrize("temperature", ["nan", "-1"])
def test_cohere_rejects_a_bad_temperature(
    cohere_server: ModelServer, temperature: str
) -> None:
    status, body = cohere_server.post_form(
        {"temperature": temperature}, clip("0006_en_short")
    )
    assert status == 400
    assert "temperature" in json.loads(body)["detail"]


@pytest.fixture(scope="module")
def moss_server() -> Iterator[ModelServer]:
    running = ModelServer(
        "moss_transcribe_diarize_server", "moss_transcribe_diarize", MOSS_REPO
    )
    yield running
    running.stop()


def two_speaker_wav() -> bytes:
    """Two LibriSpeech speakers half a second apart."""
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(
            pcm16("0000_en_short") + bytes(16000) + pcm16("0001_en_short")
        )
    return out.getvalue()


def test_moss_request_returns_speaker_segments(moss_server: ModelServer) -> None:
    assert moss_server.ready["model_name"].startswith("voxt-moss_transcribe_diarize-")
    status, body = moss_server.post_form({}, two_speaker_wav())
    assert status == 200
    result = json.loads(body)
    # Greedy decoding follows the chip's arithmetic, so the exact text and
    # timestamps are a golden-file concern: assert the structure here.
    segments = result["segments"]
    assert [segment["speaker"] for segment in segments] == ["S01", "S02"]
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )
    assert all(segment["end"] >= segment["start"] for segment in segments)
    assert "Socrates begins the timaeus" in segments[0]["text"]
    assert "no signs here" in segments[1]["text"]
    assert "[S01]" in result["text"] and "[S02]" in result["text"]


def test_moss_streamed_request_ends_with_segments(moss_server: ModelServer) -> None:
    status, body = moss_server.post_form(
        {"stream": "true", "include_generation_metadata": "true"},
        clip("0006_en_short"),
    )
    assert status == 200
    done, end = sse_events(body)
    assert end == "[DONE]"
    assert done["type"] == "transcript.text.done"
    assert done["segments"] and done["generation_metadata"]["language"] is None


@pytest.mark.parametrize(
    "fields",
    [
        {"prompt": "<|audio_pad|> and <|audio_pad|>"},
        {"max_new_tokens": "-1"},
        {"max_new_tokens": "many"},
    ],
)
def test_moss_invalid_requests_are_rejected(
    moss_server: ModelServer, fields: dict[str, str]
) -> None:
    status, body = moss_server.post_form(fields, clip("0006_en_short"))
    assert status == 400
    assert "detail" in json.loads(body)


def test_moss_realtime_rejects_a_prompt_it_cannot_render(
    moss_server: ModelServer,
) -> None:
    with connect(f"ws://127.0.0.1:{moss_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "prompt": "<|audio_pad|><|audio_pad|>",
                    },
                }
            )
        )
        error = json.loads(socket.recv())
        assert error["type"] == "error"
        assert error["error"]["type"] == "invalid_request_error"
        assert error["error"]["code"] == "invalid_prompt"
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"


def test_moss_realtime_finalizes_windows_with_segments(
    moss_server: ModelServer,
) -> None:
    pcm = pcm16("0344_en_long")[: 10 * 32000]
    with connect(f"ws://127.0.0.1:{moss_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        for start in range(0, len(pcm), 3200):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[start : start + 3200]).decode(),
                    }
                )
            )
            # Twice real time: a window decodes well within its 2 s of sending.
            time.sleep(0.05)
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = []
        while not events or events[-1]["type"] != "transcription.completed":
            events.append(json.loads(socket.recv()))
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    # Windows are finalized as they fill; one that fills during a decode waits
    # for it, and may be joined by the tail at the commit.
    assert len(finals) >= 2
    assert [event["segment_id"] for event in finals] == list(range(len(finals)))
    completed = events[-1]
    assert completed["text"] == "\n".join(event["text"] for event in finals)
    starts = [segment["start"] for segment in completed["segments"]]
    assert starts == sorted(starts)
    assert finals[1]["segments"][0]["start"] >= 4.0


def test_moss_server_serves_only_moss() -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "moss_transcribe_diarize_server"),
            "--supervised",
            "--model-kind",
            "qwen3_asr",
            "--model-directory",
            "x",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.fixture(scope="module")
def nemotron_server() -> Iterator[ModelServer]:
    running = ModelServer("nemotron_asr_server", "nemotron_asr", NEMOTRON_REPO)
    yield running
    running.stop()


@pytest.fixture(scope="module")
def parakeet_server() -> Iterator[ModelServer]:
    running = ModelServer("parakeet_server", "parakeet", PARAKEET_REPO)
    yield running
    running.stop()


def realtime_events(
    server: ModelServer,
    pcm: bytes,
    session: dict[str, object],
    append_bytes: int = 3200,
) -> list[dict[str, object]]:
    """Every event of one realtime session fed pcm in appends of append_bytes."""
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, **session},
                }
            )
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        for start in range(0, len(pcm), append_bytes):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(
                            pcm[start : start + append_bytes]
                        ).decode(),
                    }
                )
            )
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = []
        while not events or events[-1]["type"] != "transcription.completed":
            events.append(json.loads(socket.recv()))
    return events


def test_nemotron_final_request_streams_text_and_segments(
    nemotron_server: ModelServer,
) -> None:
    assert nemotron_server.ready["model_name"].startswith("voxt-nemotron_asr-")
    status, body = nemotron_server.post_form(
        {"stream": "true", "include_generation_metadata": "true", "language": "en-US"},
        clip("0006_en_short"),
    )
    assert status == 200
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    done = events[0]
    assert done["type"] == "transcript.text.done"
    assert done["text"].lower().startswith("surely you are not thinking")
    assert done["generation_metadata"]["language"] == "en-US"
    assert done["generation_metadata"]["finish_reason"] == "stop"
    assert all(
        segment["start"] <= segment["end"] and segment["speaker"] == ""
        for segment in done["segments"]
    )


@pytest.mark.parametrize("append_bytes", [3200, None])
def test_nemotron_realtime_at_the_native_chunk_equals_the_final_pass(
    nemotron_server: ModelServer, append_bytes: int | None
) -> None:
    status, body = nemotron_server.post_form({}, clip("0344_en_long"))
    assert status == 200
    pcm = pcm16("0344_en_long")
    # Note (Jiaxin Deng): None sends the whole clip in one append, which the
    # stream still decodes one chunk of audio at a time.
    events = realtime_events(nemotron_server, pcm, {}, append_bytes or len(pcm))
    # Note (Dayuxiaoshui): the stream only encodes mel frames later audio can no
    # longer change, so however the audio is split, the native chunk decodes
    # exactly what one pass over all of it does.
    assert events[-1]["text"] == json.loads(body)["text"]
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["text"] for event in finals] == [events[-1]["text"]]


def test_nemotron_realtime_previews_at_the_live_latency(
    nemotron_server: ModelServer,
) -> None:
    events = realtime_events(
        nemotron_server, pcm16("0006_en_short"), {"chunk_ms": 560, "language": "en-US"}
    )
    previews = [
        event["text"]
        for event in events
        if event["type"] == "transcription.segment" and not event["is_final"]
    ]
    assert previews
    assert events[-1]["text"].lower().startswith("surely you are not thinking")
    indexes = [event["event_index"] for event in events]
    assert indexes == sorted(indexes)


def test_nemotron_realtime_rejects_settings_after_audio(
    nemotron_server: ModelServer,
) -> None:
    with connect(f"ws://127.0.0.1:{nemotron_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16("0006_en_short")[:3200]).decode(),
                }
            )
        )
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "chunk_ms": 80},
                }
            )
        )
        event = json.loads(socket.recv())
        while event["type"] == "transcription.segment":
            event = json.loads(socket.recv())
        assert event["error"]["code"] == "session_started"


@pytest.mark.parametrize("chunk_ms", [0, -80, "560", True])
def test_nemotron_realtime_rejects_a_bad_latency(
    nemotron_server: ModelServer, chunk_ms: object
) -> None:
    with connect(f"ws://127.0.0.1:{nemotron_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "chunk_ms": chunk_ms},
                }
            )
        )
        assert json.loads(socket.recv())["error"]["code"] == "invalid_chunk"


def append_event(pcm: bytes) -> str:
    return json.dumps(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
    )


def test_nemotron_realtime_rejects_audio_after_commit(
    nemotron_server: ModelServer,
) -> None:
    pcm = pcm16("0006_en_short")
    with connect(f"ws://127.0.0.1:{nemotron_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        socket.send(append_event(pcm))
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(append_event(pcm[:3200]))
        event = json.loads(socket.recv())
        while event["type"] == "transcription.segment" and not event["is_final"]:
            event = json.loads(socket.recv())
        assert event["text"].lower().startswith("surely you are not thinking")
        assert json.loads(socket.recv())["error"]["code"] == "session_finished"


def test_nemotron_realtime_disconnect_stops_its_decode(
    nemotron_server: ModelServer,
) -> None:
    clip_pcm = pcm16("0344_en_long")
    pcm = (clip_pcm * (600 * 32000 // len(clip_pcm) + 1))[: 600 * 32000]
    with connect(f"ws://127.0.0.1:{nemotron_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "chunk_ms": 80},
                }
            )
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        for start in range(0, len(pcm), 320000):
            socket.send(append_event(pcm[start : start + 320000]))
        time.sleep(2.0)
        assert json.loads(nemotron_server.request("GET", "/health")[1])[
            "request_states"
        ] == {"running": 1}
        closed_at = time.monotonic()
    # Note (Dayuxiaoshui): the stream checks the cancel flag before each chunk,
    # so the decode of ten minutes of audio stops within a chunk of the close.
    while (
        json.loads(nemotron_server.request("GET", "/health")[1])["request_states"] != {}
    ):
        assert (
            time.monotonic() - closed_at < 1.5
        ), "the decode kept running after its client left"
        time.sleep(0.05)


def test_parakeet_final_request_returns_text_and_sentences(
    parakeet_server: ModelServer,
) -> None:
    assert parakeet_server.ready["model_name"].startswith("voxt-parakeet-")
    status, body = parakeet_server.post_form(
        {"chunk_duration": "1200"}, clip("0006_en_short")
    )
    assert status == 200
    result = json.loads(body)
    assert result["text"] == "Surely you are not thinking of going off there."
    [segment] = result["segments"]
    assert segment["text"].strip() == result["text"]
    assert 0 <= segment["start"] < segment["end"] <= 3.1


def test_parakeet_merges_overlapping_chunks_into_sentences(
    parakeet_server: ModelServer,
) -> None:
    # Note (Dayuxiaoshui): a whole-recording request decodes 5 s chunks that
    # overlap by 1 s, as Voxt's Swift path does.
    status, body = parakeet_server.post_form(
        {
            "chunk_duration": "1200",
            "stream": "true",
            "include_generation_metadata": "true",
        },
        clip("0344_en_long"),
    )
    assert status == 200
    done = sse_events(body)[0]
    assert done["text"].startswith(
        "In every way they sought to undermine the authority"
    )
    assert done["generation_metadata"]["finish_reason"] == "stop"
    status, body = parakeet_server.post_form({}, clip("0344_en_long"))
    assert status == 200
    segments = json.loads(body)["segments"]
    assert len(segments) > 1
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )


@pytest.mark.parametrize("chunk_duration", ["nan", "1", "-5", "600"])
def test_parakeet_rejects_a_bad_chunk_duration(
    parakeet_server: ModelServer, chunk_duration: str
) -> None:
    status, body = parakeet_server.post_form(
        {"chunk_duration": chunk_duration}, clip("0006_en_short")
    )
    assert status == 400
    assert "chunk_duration" in json.loads(body)["detail"]


@pytest.fixture(scope="module")
def sensevoice_server() -> Iterator[ModelServer]:
    running = ModelServer("sensevoice_server", "sensevoice", SENSEVOICE_REPO)
    yield running
    running.stop()


def test_sensevoice_final_request_returns_text_and_its_metadata(
    sensevoice_server: ModelServer,
) -> None:
    assert sensevoice_server.ready["model_name"].startswith("voxt-sensevoice-")
    status, body = sensevoice_server.post_form(
        {"language": "zh", "use_itn": "false"}, clip("0152_zh_short")
    )
    assert status == 200
    result = json.loads(body)
    assert result["text"] == "互联网结合了大众传播和人际传播的要素"
    [segment] = result["segments"]
    assert segment["start"] == 0
    assert segment["text"] == result["text"]
    assert segment["language"] == "zh"
    assert segment["event"] == "Speech"
    assert segment["emotion"]


def test_sensevoice_itn_writes_punctuation(sensevoice_server: ModelServer) -> None:
    status, body = sensevoice_server.post_form(
        {"language": "en", "use_itn": "true"}, clip("0006_en_short")
    )
    assert status == 200
    assert json.loads(body)["text"] == "Surely you are not thinking of going off there."


def test_sensevoice_detects_the_language_without_a_hint(
    sensevoice_server: ModelServer,
) -> None:
    # Note (Dayuxiaoshui): an unsupported hint detects the language, as Voxt
    # sends a hint only for a single main language SenseVoice supports.
    status, body = sensevoice_server.post_form(
        {"language": "fr", "stream": "true", "include_generation_metadata": "true"},
        clip("0152_zh_short"),
    )
    assert status == 200
    assert sse_events(body)[0]["generation_metadata"]["language"] == "zh"


def test_sensevoice_cuts_long_audio_into_overlapping_chunks(
    sensevoice_server: ModelServer,
) -> None:
    vad_directory = str(Path(DATA_ROOT) / "models" / VAD_REPO.replace("/", "_"))
    status, body = sensevoice_server.post_form(
        {
            "language": "en",
            "vad_model_directory": vad_directory,
            **SENSEVOICE_VAD_FIELDS,
        },
        clip("0344_en_long"),
    )
    assert status == 200
    result = json.loads(body)
    assert result["text"].startswith("in every way they sought to undermine")
    segments = result["segments"]
    assert len(segments) > 1
    assert all(segment["end"] - segment["start"] <= 24 for segment in segments)
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )


def test_sensevoice_chunks_overlap_by_voxts_sample_count(
    sensevoice_server: ModelServer,
) -> None:
    # Note (Jiaxin Deng): 4 s chunks split this clip's speech runs, and each
    # chunk of a run starts 5600 samples (0.35 s, truncated in double as Voxt
    # does) before the previous one ends.
    vad_directory = str(Path(DATA_ROOT) / "models" / VAD_REPO.replace("/", "_"))
    status, body = sensevoice_server.post_form(
        {
            "language": "en",
            "vad_model_directory": vad_directory,
            **SENSEVOICE_VAD_FIELDS,
            "vad_max_chunk_seconds": "4.0",
        },
        clip("0344_en_long"),
    )
    assert status == 200
    bounds = [
        (round(segment["start"] * 16000), round(segment["end"] * 16000))
        for segment in json.loads(body)["segments"]
    ]
    overlaps = [
        previous_end - start
        for (_, previous_end), (start, _) in zip(bounds, bounds[1:])
        if start < previous_end
    ]
    assert overlaps
    assert set(overlaps) == {5600}


# Note (Jiaxin Deng): the last case differs in seconds but not in whole
# samples, where chunking would never advance.
@pytest.mark.parametrize(
    ("max_chunk", "overlap"),
    [("24.0", None), ("24.0", "24.0"), ("0.0001", "0.00009")],
)
def test_sensevoice_voice_activity_needs_an_overlap_shorter_than_a_chunk(
    sensevoice_server: ModelServer, max_chunk: str, overlap: str | None
) -> None:
    fields = {
        "vad_model_directory": "x",
        **SENSEVOICE_VAD_FIELDS,
        "vad_max_chunk_seconds": max_chunk,
    }
    if overlap is None:
        fields.pop("vad_chunk_overlap_seconds")
    else:
        fields["vad_chunk_overlap_seconds"] = overlap
    status, body = sensevoice_server.post_form(fields, clip("0006_en_short"))
    assert status == 400
    assert "overlap" in json.loads(body)["detail"]


# Note (Jiaxin Deng): a cold short request takes over a minute on a Mac that
# pages the weights back in, so only the deadline test uses a short timeout.
MUSIC3_REQUEST_TIMEOUT_SECONDS = 600
MUSIC3_SHORT_REQUEST_TIMEOUT_SECONDS = 5
# A cancelled Music3 generation stops at its next frame, acoustic window or
# Euler step.
MUSIC3_CANCEL_SECONDS = 30.0
MUSIC3_LONG_REQUEST = {
    "input": "[Verse]\nWalking through the morning light",
    "instructions": "Gentle acoustic folk",
    "max_new_tokens": 9000,
    "seed": 42,
}


MUSIC3_SERVER_MODES = [
    pytest.param((), id="resident"),
    pytest.param(("--stage-offload-components", " DIT , ar "), id="offload"),
]


@pytest.fixture(scope="module", params=[()], ids=["resident"])
def music3_server(request: pytest.FixtureRequest) -> Iterator[ModelServer]:
    music3_process = ModelServer(
        "music3_server",
        "minimax_music3",
        "mlx-community/MiniMax-Music3-mxfp8",
        (
            *request.param,
            "--request-timeout-s",
            str(MUSIC3_REQUEST_TIMEOUT_SECONDS),
        ),
    )
    yield music3_process
    music3_process.stop()


@pytest.mark.parametrize("offload_components", ["ar", "dit", "ar,dit,other", ""])
@pytest.mark.parametrize("binary_name", ["music3_server", "music3_synthesize"])
def test_music3_rejects_invalid_offload_components(
    tmp_path: Path, binary_name: str, offload_components: str
) -> None:
    process_result = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / binary_name),
            "--model-path",
            str(tmp_path),
            "--stage-offload-components",
            offload_components,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process_result.returncode != 0
    assert "requires both ar,dit" in process_result.stderr


@pytest.mark.parametrize("server_flags", MUSIC3_SERVER_MODES)
@pytest.mark.parametrize(
    ("tensor_name", "expected_error"),
    [
        ("language_model.model.norm.weight", "missing or incompatible Music3 tensor"),
        ("unassigned.weight", "unassigned or ambiguous Music3 tensor"),
    ],
)
def test_music3_rejects_invalid_checkpoint_before_ready(
    tmp_path: Path,
    server_flags: tuple[str, ...],
    tensor_name: str,
    expected_error: str,
) -> None:
    """Offload must reject incompatible and unassigned weights before serving."""
    checkpoint_directory = Path(DATA_ROOT) / "models/mlx-community_MiniMax-Music3-mxfp8"
    for checkpoint_file in checkpoint_directory.iterdir():
        (tmp_path / checkpoint_file.name).symlink_to(checkpoint_file)
    tensor_header = json.dumps(
        {tensor_name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    tensor_header += b" " * (-len(tensor_header) % 8)
    (tmp_path / "zz-invalid.safetensors").write_bytes(
        struct.pack("<Q", len(tensor_header)) + tensor_header + struct.pack("<f", 0)
    )
    music3_process = Server("music3_server", "minimax_music3", tmp_path, server_flags)
    assert music3_process.ready["event"] == "failed"
    assert expected_error in music3_process.ready["reason"]
    assert music3_process.process.wait(timeout=30) != 0


def test_music3_recovers_after_cancellation_at_ar_handoff(tmp_path: Path) -> None:
    """Cancellation after AR must release stage weights and allow seeded retries."""
    synthesis_request = {
        "input": "[Verse]\nWalking through the morning light",
        "instructions": "Gentle acoustic folk",
        "max_new_tokens": 10,
        "dit_steps": 2,
        "seed": 42,
    }
    request_file = tmp_path / "requests.jsonl"
    request_file.write_text(
        "".join(
            json.dumps({"id": case_id, **synthesis_request}) + "\n"
            for case_id in ("after_cancel", "repeat")
        )
    )
    process_result = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "music3_synthesize"),
            "--model-path",
            str(Path(DATA_ROOT) / "models/mlx-community_MiniMax-Music3-mxfp8"),
            "--stage-offload-components",
            " AR, dit,AR, ",
            "--requests",
            str(request_file),
            "--output-directory",
            str(tmp_path / "audio"),
            "--check-cancellation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process_result.returncode == 0, process_result.stderr
    synthesis_results = [
        json.loads(result_line) for result_line in process_result.stdout.splitlines()
    ]
    assert [result["id"] for result in synthesis_results] == ["after_cancel", "repeat"]
    for synthesis_result in synthesis_results:
        assert synthesis_result["cancelled_after_ar"] is True
        assert (
            synthesis_result["cancelled_active_memory_bytes"]
            == synthesis_result["idle_active_memory_bytes"]
        )
        assert synthesis_result["generated_frames"] == 10
    assert (
        Path(synthesis_results[0]["file"]).read_bytes()
        == Path(synthesis_results[1]["file"]).read_bytes()
    )


def music3_request_states(server: ModelServer) -> dict[str, int]:
    return json.loads(server.request("GET", "/health")[1])["request_states"]


def wait_for_music3_states(
    server: ModelServer, states: dict[str, int], timeout_seconds: float
) -> None:
    started_at = time.monotonic()
    while music3_request_states(server) != states:
        assert (
            time.monotonic() - started_at < timeout_seconds
        ), f"Music3 did not reach request states {states}"
        time.sleep(0.05)


def send_music3_request(server: ModelServer, body: dict) -> socket.socket:
    """Sends a Music3 request on a socket of its own, without reading the answer."""
    encoded_body = json.dumps(body)
    client = socket.create_connection(("127.0.0.1", server.port), timeout=300)
    client.sendall(
        b"POST /v1/audio/speech HTTP/1.1\r\nHost: test\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(encoded_body)}\r\n\r\n{encoded_body}".encode()
    )
    return client


@pytest.mark.parametrize(
    ("section", "replacement", "error"),
    [
        ("pre_tokenizer", {"type": "ByteLevel"}, "requires NFC and byte-level BPE"),
        ("model", {"type": "WordPiece"}, "requires NFC and byte-level BPE"),
        (
            "model",
            {"type": "BPE", "continuing_subword_prefix": "##"},
            "requires NFC and byte-level BPE",
        ),
        (
            "model",
            {"type": "BPE", "end_of_word_suffix": "</w>"},
            "requires NFC and byte-level BPE",
        ),
        (
            "added_tokens",
            [
                {
                    "id": 151643,
                    "content": "<|endoftext|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": True,
                    "special": True,
                }
            ],
            "unsupported added token matching rules",
        ),
    ],
)
def test_music3_rejects_unsupported_tokenizer_rules(
    tmp_path: Path,
    section: str,
    replacement: dict[str, str] | list[dict[str, str | int | bool]],
    error: str,
) -> None:
    source = (
        Path(DATA_ROOT)
        / "models/mlx-community_MiniMax-Music3-mxfp8/tokenizer/tokenizer.json"
    )
    tokenizer = json.loads(source.read_text())
    tokenizer[section] = replacement
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer/tokenizer.json").write_text(json.dumps(tokenizer))
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps({"id": "prompt", "input": "Hello", "instructions": "Folk"}) + "\n"
    )
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "music3_synthesize"),
            "--model-path",
            str(tmp_path),
            "--requests",
            str(requests),
            "--prompt-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert error in completed.stderr


def test_music3_health_and_models(music3_server: ModelServer) -> None:
    assert music3_server.ready["event"] == "ready"
    status, body = music3_server.request("GET", "/health")
    assert status == 200
    assert json.loads(body)["request_states"] == {}
    status, body = music3_server.request("GET", "/v1/models")
    assert status == 200
    assert json.loads(body)["data"][0]["id"] == music3_server.ready["model_name"]


@pytest.mark.parametrize(
    "fields",
    [
        {"input": ""},
        {"instructions": ""},
        {"input": None},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_new_tokens": 9001},
        {"max_new_tokens": 1.5},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
        {"dit_steps": 0},
        {"dit_cfg_scale": -1},
        {"dit_cfg_scale": True},
        {"response_format": "mp3"},
        {"stream": True},
        {"input": "hello " * 5100},
        {"input": "x" * (64 * 1024 + 1)},
        {"instructions": "x" * (64 * 1024 + 1)},
        {"input": "[ab" * 13000},
    ],
)
def test_music3_rejects_invalid_requests_before_generation(
    music3_server: ModelServer, fields: dict[str, str | int | float | bool | None]
) -> None:
    request = {
        "input": "[Verse]\nHello, morning light",
        "instructions": "Acoustic folk",
        **fields,
    }
    status, body = music3_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(request).encode(),
        {"Content-Type": "application/json"},
    )
    assert status == 400
    assert json.loads(body)["detail"]
    status, body = music3_server.request("GET", "/health")
    assert json.loads(body)["request_states"] == {}


@pytest.mark.parametrize("music3_server", MUSIC3_SERVER_MODES, indirect=True)
def test_music3_seeded_wav_matches_pcm_and_repeated_requests(
    music3_server: ModelServer,
) -> None:
    synthesis_request = {
        "input": "[Verse]\nWalking through the morning light",
        "instructions": "Gentle acoustic folk",
        "max_new_tokens": 10,
        "dit_steps": 2,
        "seed": 42,
        # No --model-name was chosen, so an OpenAI client's model is accepted.
        "model": "tts-1",
    }
    audio_responses: list[bytes] = []
    for response_format in ("wav", "pcm"):
        status_code, response_body = music3_server.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(
                {**synthesis_request, "response_format": response_format}
            ).encode(),
            {"Content-Type": "application/json"},
        )
        assert status_code == 200
        audio_responses.append(response_body)
    with wave.open(io.BytesIO(audio_responses[0])) as audio:
        assert (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) == (
            2,
            32000,
            2,
        )
        pcm_bytes = audio.readframes(audio.getnframes())
        assert pcm_bytes == audio_responses[1]
        assert any(pcm_bytes)
    status_code, response_body = music3_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps({**synthesis_request, "response_format": "wav"}).encode(),
        {"Content-Type": "application/json"},
    )
    assert status_code == 200
    assert response_body == audio_responses[0]


@pytest.mark.parametrize(
    ("uppercase_tag", "lowercase_tag"),
    [("CHŒUR", "chœur"), ("ΟΣ", "ος"), ("İNTRO", "i\u0307ntro")],
)
def test_music3_unicode_tag_case_preserves_seeded_audio(
    music3_server: ModelServer, uppercase_tag: str, lowercase_tag: str
) -> None:
    """Non-ASCII uppercase section tags must normalize to the same audio."""
    responses = []
    for tag in [uppercase_tag, lowercase_tag]:
        request = {
            "input": f"[{tag}]\nWalking through the morning light",
            "instructions": "Gentle acoustic folk",
            "max_new_tokens": 10,
            "dit_steps": 2,
            "seed": 42,
        }
        status, body = music3_server.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(request).encode(),
            {"Content-Type": "application/json"},
        )
        assert status == 200
        responses.append(body)
    assert responses[0] == responses[1]


@pytest.mark.parametrize("music3_server", MUSIC3_SERVER_MODES, indirect=True)
def test_music3_busy_request_returns_503(music3_server: ModelServer) -> None:
    with send_music3_request(music3_server, MUSIC3_LONG_REQUEST):
        wait_for_music3_states(music3_server, {"running": 1}, MUSIC3_CANCEL_SECONDS)
        for busy_request in (json.dumps(MUSIC3_LONG_REQUEST).encode(), b"{invalid"):
            status, body = music3_server.request(
                "POST",
                "/v1/audio/speech",
                busy_request,
                {"Content-Type": "application/json"},
            )
            assert (status, json.loads(body)) == (503, {"detail": "Music3 is busy"})
    wait_for_music3_states(music3_server, {}, MUSIC3_CANCEL_SECONDS)


def test_music3_disconnect_cancels_its_generation(music3_server: ModelServer) -> None:
    with send_music3_request(music3_server, MUSIC3_LONG_REQUEST):
        wait_for_music3_states(music3_server, {"running": 1}, MUSIC3_CANCEL_SECONDS)
    wait_for_music3_states(music3_server, {}, MUSIC3_CANCEL_SECONDS)
    # The freed slot takes the next request.
    status, _ = music3_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(
            {**MUSIC3_LONG_REQUEST, "max_new_tokens": 10, "dit_steps": 2}
        ).encode(),
        {"Content-Type": "application/json"},
    )
    assert status == 200


def test_music3_half_closed_client_counts_as_gone(music3_server: ModelServer) -> None:
    with send_music3_request(music3_server, MUSIC3_LONG_REQUEST) as client:
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        assert received == b""
    wait_for_music3_states(music3_server, {}, MUSIC3_CANCEL_SECONDS)


# Note (Jiaxin Deng): last, since it replaces the module's server with one
# whose deadline any long song passes.
def test_music3_times_out_a_request_past_its_deadline(
    music3_server: ModelServer,
) -> None:
    music3_server.stop()
    short_deadline_server = ModelServer(
        "music3_server",
        "minimax_music3",
        "mlx-community/MiniMax-Music3-mxfp8",
        ("--request-timeout-s", str(MUSIC3_SHORT_REQUEST_TIMEOUT_SECONDS)),
    )
    try:
        started_at = time.monotonic()
        status, body = short_deadline_server.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(MUSIC3_LONG_REQUEST).encode(),
            {"Content-Type": "application/json"},
        )
        assert (status, json.loads(body)) == (
            504,
            {"detail": "Music3 generation timed out"},
        )
        assert time.monotonic() - started_at >= MUSIC3_SHORT_REQUEST_TIMEOUT_SECONDS
        wait_for_music3_states(short_deadline_server, {}, MUSIC3_CANCEL_SECONDS)
    finally:
        short_deadline_server.stop()
