"""Real-checkpoint coverage for Ming's native transcription server."""

from __future__ import annotations

import http.client
import io
import json
import struct
import subprocess
import time
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from server_test_utils import DATA_ROOT, RUNTIME_BIN, Server, clip, sse_events
from tokenizers import Tokenizer

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
MING_REPO = "mlx-community/Ming-3.0-asr-tiny-4bit"


@pytest.fixture(scope="module")
def ming_server() -> Iterator[Server]:
    running = Server(
        "ming3_asr_server",
        "ming3_asr",
        Path(DATA_ROOT) / "models" / MING_REPO.replace("/", "_"),
    )
    try:
        yield running
    finally:
        running.stop()


def test_ming_supervisor_health_and_model_identity(ming_server: Server) -> None:
    assert ming_server.ready["event"] == "ready"
    assert ming_server.ready["model_name"].startswith("voxt-ming3_asr-")
    status, body = ming_server.request("GET", "/health")
    assert status == 200
    assert json.loads(body) == {
        "status": "healthy",
        "running": True,
        "request_states": {},
    }


@pytest.mark.parametrize(
    ("language", "clip_id"),
    [("en", "0006_en_short"), ("zh", "0152_zh_short"), ("", "0006_en_short")],
)
def test_ming_json_matches_stream_and_resets_request_state(
    ming_server: Server, language: str, clip_id: str
) -> None:
    fields = {"language": language, "max_new_tokens": "128"}
    status, body = ming_server.post_form(fields, clip(clip_id))
    assert status == 200
    text = json.loads(body)["text"]
    assert isinstance(text, str) and text.strip()
    status, streamed = ming_server.post_form(
        {**fields, "stream": "true", "include_generation_metadata": "true"},
        clip(clip_id),
    )
    assert status == 200
    events = sse_events(streamed)
    assert events[-1] == "[DONE]"
    event = events[0]
    assert event["type"] == "transcript.text.done"
    assert event["text"] == text
    metadata = event["generation_metadata"]
    assert metadata["language"] == (language or "en")
    assert 0 < metadata["generated_token_count"] < 128
    assert metadata["finish_reason"] == "stop"


def test_ming_generation_limit_is_reported(ming_server: Server) -> None:
    status, body = ming_server.post_form(
        {
            "language": "en",
            "max_new_tokens": "2",
            "stream": "true",
            "include_generation_metadata": "true",
        },
        clip("0006_en_short"),
    )
    assert status == 200
    event = sse_events(body)[0]
    assert event["generation_metadata"] == {
        "language": "en",
        "generated_token_count": 2,
        "finish_reason": "length",
    }


def test_ming_no_repeat_unigrams_changes_generation_and_resets(
    ming_server: Server,
) -> None:
    """Unigram suppression must remove repeated tokens without leaking across requests."""
    tokenizer = Tokenizer.from_file(
        str(Path(DATA_ROOT) / "models" / MING_REPO.replace("/", "_") / "tokenizer.json")
    )
    outputs = []
    for ngram_size in (0, 1, 0):
        status, body = ming_server.post_form(
            {
                "language": "en",
                "max_new_tokens": "80",
                "no_repeat_ngram_size": str(ngram_size),
            },
            clip("0344_en_long"),
        )
        assert status == 200
        token_ids = tokenizer.encode(
            json.loads(body)["text"], add_special_tokens=False
        ).ids
        assert token_ids
        outputs.append(token_ids)
    assert len(set(outputs[0])) < len(outputs[0])
    assert len(set(outputs[1])) == len(outputs[1])
    assert outputs[0] == outputs[2]


@pytest.mark.parametrize("language", [None, "English", "Chinese"])
def test_ming_cli_and_server_share_request_preparation(
    ming_server: Server, language: str | None
) -> None:
    command = [
        str(Path(RUNTIME_BIN) / "ming3_asr_transcribe"),
        "--model-path",
        str(Path(DATA_ROOT) / "models" / MING_REPO.replace("/", "_")),
        "--max-new-tokens",
        "2",
    ]
    fields = {
        "max_new_tokens": "2",
        "stream": "true",
        "include_generation_metadata": "true",
    }
    if language is not None:
        command.extend(["--language", language])
        fields["language"] = language
    else:
        pass
    command.append(str(Path(DATA_ROOT) / "corpus/v1/clips/0006_en_short.wav"))
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    )
    result = json.loads(completed.stdout)
    status, body = ming_server.post_form(fields, clip("0006_en_short"))
    assert status == 200
    event = sse_events(body)[0]
    assert result["text"] == event["text"]
    assert event["generation_metadata"] == {
        field: result[field]
        for field in ("language", "generated_token_count", "finish_reason")
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"language": "fr"},
        {"max_new_tokens": "0"},
        {"max_new_tokens": "3x"},
        {"max_new_tokens": "99999999999999999999"},
        {"no_repeat_ngram_size": "-1"},
        {"temperature": "0.5"},
        {"temperature": "nan"},
    ],
)
def test_ming_rejects_invalid_options_before_sse_headers(
    ming_server: Server, fields: dict[str, str]
) -> None:
    status, body = ming_server.post_form(
        {**fields, "stream": "true"}, clip("0006_en_short")
    )
    assert status == 400
    assert json.loads(body)["detail"]


@pytest.mark.parametrize("samples", [200, 16000 * 330])
def test_ming_rejects_audio_outside_context_bounds(
    ming_server: Server, samples: int
) -> None:
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * samples)
    status, body = ming_server.post_form({"stream": "true"}, audio.getvalue())
    assert status == 400
    assert json.loads(body)["detail"]


@pytest.mark.parametrize("sample", [float("nan"), float("inf")])
def test_ming_rejects_nonfinite_float_audio(ming_server: Server, sample: float) -> None:
    samples = struct.pack("<256f", *([sample] * 256))
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(samples),
        b"WAVE",
        b"fmt ",
        16,
        3,
        1,
        16000,
        64000,
        4,
        32,
        b"data",
        len(samples),
    )
    status, body = ming_server.post_form({"stream": "true"}, header + samples)
    assert status == 400
    assert "finite" in json.loads(body)["detail"]


def test_ming_accepts_minimum_reflection_length(ming_server: Server) -> None:
    """A 201-sample waveform must not lose its reflected tail and produce zero frames."""
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 201)
    status, body = ming_server.post_form({"max_new_tokens": "1"}, audio.getvalue())
    assert status == 200
    assert isinstance(json.loads(body)["text"], str)


def test_ming_long_audio_crosses_encoder_and_prefill_chunks(
    ming_server: Server,
) -> None:
    status, body = ming_server.post_form(
        {"language": "en", "max_new_tokens": "512"}, clip("0344_en_long")
    )
    assert status == 200
    assert len(json.loads(body)["text"].split()) > 100


@pytest.mark.parametrize("field", ["prompt", "hotwords"])
def test_ming_prompt_controls_consume_context_budget(
    ming_server: Server, field: str
) -> None:
    """Ignoring a prompt control must not turn an over-budget request into success."""
    prefix = (
        "Please recognize the language of this speech and transcribe it, "
        "placing a `\t` between the identified language and the transcribe."
        if field == "prompt"
        else ""
    )
    status, body = ming_server.post_form(
        {field: prefix + " Ming, ASR", "language": "en"}, clip("0006_en_short")
    )
    assert status == 200
    assert json.loads(body)["text"]
    status, body = ming_server.post_form(
        {field: prefix + " Ming" * 9000, "max_new_tokens": "1"},
        clip("0006_en_short"),
    )
    assert status == 400
    assert "context budget" in json.loads(body)["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", "not the Ming instruction"),
        ("prompt", "Please recognize\nthe language"),
        ("hotwords", "Ming\nASR"),
    ],
)
def test_ming_rejects_invalid_prompt_controls(
    ming_server: Server, field: str, value: str
) -> None:
    status, body = ming_server.post_form(
        {field: value, "language": "en"}, clip("0006_en_short")
    )
    assert status == 400
    assert json.loads(body)["detail"]


def test_ming_rejects_wrong_model_kind() -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_server"),
            "--supervised",
            "--model-kind",
            "qwen3_asr",
            "--model-directory",
            "unused",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


def test_ming_has_no_realtime_endpoint(ming_server: Server) -> None:
    status, _ = ming_server.request("GET", "/v1/realtime")
    assert status == 404


def test_ming_disconnect_cancels_active_request(ming_server: Server) -> None:
    boundary = uuid.uuid4().hex
    with wave.open(io.BytesIO(clip("0344_en_long"))) as source:
        frames = source.readframes(source.getnframes())
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(frames * 4)
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="long.wav"\r\n\r\n'
        ).encode()
        + audio.getvalue()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection("127.0.0.1", ming_server.port, timeout=30)
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    try:
        assert response.status == 200
        deadline = time.monotonic() + 5
        while True:
            states = json.loads(ming_server.request("GET", "/health")[1])[
                "request_states"
            ]
            if states.get("running") == 1:
                break
            else:
                assert time.monotonic() < deadline, "request never entered the worker"
    finally:
        response.close()
        connection.close()
    deadline = time.monotonic() + 5
    while json.loads(ming_server.request("GET", "/health")[1])["request_states"]:
        assert time.monotonic() < deadline, "disconnected request did not cancel"
    status, body = ming_server.post_form({"language": "en"}, clip("0006_en_short"))
    assert status == 200 and json.loads(body)["text"].strip()


@pytest.mark.parametrize(
    ("corruption", "reason"),
    [
        ("config", "n_group"),
        ("weight", "model.norm.weight shape: expected"),
        ("safetensors", "safetensors"),
    ],
)
def test_ming_invalid_checkpoint_fails_before_serving(
    tmp_path: Path, corruption: str, reason: str
) -> None:
    model = Path(DATA_ROOT) / "models" / MING_REPO.replace("/", "_")
    config = json.loads((model / "config.json").read_text())
    if corruption == "config":
        config["text_config"]["n_group"] = 0
    elif corruption == "weight":
        for file in model.glob("*.safetensors"):
            (tmp_path / file.name).symlink_to(file)
        save_file(
            {"model.norm.weight": np.ones((1,), dtype=np.float32)},
            tmp_path / "zz-invalid-shape.safetensors",
        )
    else:
        for file in model.glob("*.safetensors"):
            (tmp_path / file.name).symlink_to(file)
        (tmp_path / "000-corrupt.safetensors").write_bytes(b"not a safetensors file")
    (tmp_path / "config.json").write_text(json.dumps(config))
    running = Server("ming3_asr_server", "ming3_asr", tmp_path)
    try:
        assert running.ready["event"] == "failed"
        assert reason in running.ready["reason"]
        assert running.process.wait(timeout=30) != 0
    finally:
        running.stop()
