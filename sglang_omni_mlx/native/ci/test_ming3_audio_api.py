# SPDX-License-Identifier: Apache-2.0
"""Exercise Ming3-Audio chat completions with the real native model."""

from __future__ import annotations

import base64
import http.client
import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
from check_golden import QUALITY_GROUPS, chip, quality

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
MODEL_REPO = "inclusionAI/Ming-3.0-audio-tiny"
ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it with appropriate "
    "punctuation marks, placing a `\t` between the identified language and the transcribe."
)
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)


@contextmanager
def running_chat(
    log_directory: Path, extra_arguments: tuple[str, ...] = ()
) -> Iterator[tuple[int, subprocess.Popen[str]]]:
    log_path = log_directory / "server.log"
    model_directory = Path(DATA_ROOT) / "models" / MODEL_REPO.replace("/", "_")
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / "ming3_audio_server"),
                "--supervised",
                "--model-directory",
                str(model_directory),
                "--model-name",
                MODEL_REPO,
                *extra_arguments,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            ready = process.stdout.readline()
            if not ready:
                log.seek(0)
                pytest.fail(log.read())
            else:
                yield json.loads(ready)["port"], process
        finally:
            if process.poll() is None:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    pytest.fail("server did not shut down")
            else:
                pass
            assert process.returncode == 0


@pytest.fixture(scope="module")
def chat_port(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    with running_chat(tmp_path_factory.mktemp("ming3-audio")) as (port, _):
        yield port


def pending_request(port: int, body: str) -> http.client.HTTPConnection:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body.encode(),
            headers={"Content-Type": "application/json"},
        )
    except OSError:
        connection.close()
        raise
    return connection


def wait_for_running(port: int, running: bool, timeout_seconds: float = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    expected = {"running": 1} if running else {}
    while time.monotonic() < deadline:
        status, response = request(port, method="GET", path="/health")
        assert status == 200, response
        if json.loads(response)["request_states"] == expected:
            return
        else:
            time.sleep(0.01)
    pytest.fail(f"server did not reach request_states={expected}")


def request(
    port: int, body: str = "", method: str = "POST", path: str = "/v1/chat/completions"
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request(
            method,
            path,
            body=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.fixture(scope="module")
def chat_body() -> str:
    audio = (
        Path(DATA_ROOT) / "corpus" / "v1" / "clips" / "0006_en_short.wav"
    ).read_bytes()
    return json.dumps(
        {
            "model": MODEL_REPO,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "format": "wav",
                                "data": base64.b64encode(audio).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": "What language is spoken? Answer with the language name only.",
                        },
                    ],
                }
            ],
            "max_completion_tokens": 32,
        }
    )


@pytest.fixture
def long_chat_body(chat_body: str) -> str:
    body = json.loads(chat_body)
    body["messages"][0]["content"][1]["text"] = (
        "Identify the spoken language, then count from 1 to 1000, "
        "writing every number without skipping any."
    )
    body["max_completion_tokens"] = 1024
    return json.dumps(body)


def test_busy_request_is_rejected(
    tmp_path: Path, chat_body: str, long_chat_body: str
) -> None:
    with running_chat(tmp_path) as (port, _):
        with closing(pending_request(port, long_chat_body)) as first:
            wait_for_running(port, True)
            invalid = json.loads(chat_body)
            invalid["messages"][0]["content"][1]["text"] = "<|audio_pad|>"
            status, response = request(port, json.dumps(invalid))
            assert status == 400, response
            assert "audio placeholders" in json.loads(response)["error"]["message"]
            status, response = request(port, chat_body)
            assert status == 429, response
            assert json.loads(response)["error"]["type"] == "rate_limit_error"
            completed = first.getresponse()
            reply = json.loads(completed.read())
            assert completed.status == 200, reply
            assert reply["choices"][0]["message"]["content"]
            assert reply["usage"]["completion_tokens"] > 0
        wait_for_running(port, False)


def test_disconnect_releases_generation(
    tmp_path: Path, chat_body: str, long_chat_body: str
) -> None:
    with running_chat(tmp_path) as (port, _):
        with closing(pending_request(port, long_chat_body)):
            wait_for_running(port, True)
        wait_for_running(port, False, timeout_seconds=8)
        status, response = request(port, chat_body)
        assert status == 200, response
        choice = json.loads(response)["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"]


def test_shutdown_during_generation(tmp_path: Path, long_chat_body: str) -> None:
    with running_chat(tmp_path) as (port, process):
        with closing(pending_request(port, long_chat_body)):
            wait_for_running(port, True)
            process.stdin.write('{"command":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=10)
            assert process.returncode == 0
            assert json.loads(process.stdout.readline()) == {"event": "stopped"}


def test_audio_answer_and_request_isolation(chat_port: int, chat_body: str) -> None:
    replies = []
    for _ in range(2):
        status, response = request(chat_port, chat_body)
        assert status == 200, response
        reply = json.loads(response)
        assert reply["object"] == "chat.completion"
        assert reply["model"] == MODEL_REPO
        assert reply["created"] > 0
        choice = reply["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"]
        assert choice["finish_reason"] == "stop"
        usage = reply["usage"]
        assert usage["prompt_tokens"] > 0
        assert 0 < usage["completion_tokens"] <= 32
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        replies.append(reply)
    assert replies[0]["id"] != replies[1]["id"]
    assert replies[0]["choices"] == replies[1]["choices"]
    assert replies[0]["usage"] == replies[1]["usage"]


def test_system_and_text_before_audio(chat_port: int, chat_body: str) -> None:
    status, response = request(chat_port, chat_body)
    assert status == 200, response
    plain = json.loads(response)
    body = json.loads(chat_body)
    body["messages"][0]["content"].reverse()
    body["messages"].insert(
        0, {"role": "system", "content": "Answer briefly in English."}
    )
    status, response = request(chat_port, json.dumps(body))
    assert status == 200, response
    reply = json.loads(response)
    assert reply["choices"][0]["message"]["content"]
    assert reply["usage"]["prompt_tokens"] > plain["usage"]["prompt_tokens"]


def test_generation_budget_above_1024(chat_port: int, chat_body: str) -> None:
    """A valid context budget was rejected solely for exceeding 1024 output tokens."""
    body = json.loads(chat_body)
    body["max_completion_tokens"] = 1025
    status, response = request(chat_port, json.dumps(body))
    assert status == 200, response
    reply = json.loads(response)
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["choices"][0]["message"]["content"]
    assert 0 < reply["usage"]["completion_tokens"] <= 1025


@pytest.mark.parametrize("context_tokens", [128, 16384])
def test_configured_context_budget(
    tmp_path: Path, chat_body: str, context_tokens: int
) -> None:
    """Prompt and output budgets must respect the configured context boundary."""
    with running_chat(tmp_path, ("--context-length", str(context_tokens))) as (port, _):
        status, response = request(port, chat_body)
        assert status == 200, response
        prompt_tokens = json.loads(response)["usage"]["prompt_tokens"]
        body = json.loads(chat_body)
        body["max_completion_tokens"] = context_tokens - prompt_tokens
        status, response = request(port, json.dumps(body))
        assert status == 200, response
        assert json.loads(response)["choices"][0]["finish_reason"] == "stop"
        body["max_completion_tokens"] += 1
        status, response = request(port, json.dumps(body))
        assert status == 400, response
        assert "context limit" in json.loads(response)["error"]["message"]


def test_audio_over_30_seconds(chat_port: int, chat_body: str) -> None:
    """Long recordings were rejected instead of being encoded in multiple chunks."""
    audio = (
        Path(DATA_ROOT) / "corpus" / "v1" / "clips" / "0344_en_long.wav"
    ).read_bytes()
    body = json.loads(chat_body)
    body["messages"][0]["content"][0]["input_audio"]["data"] = base64.b64encode(
        audio
    ).decode()
    status, response = request(chat_port, json.dumps(body))
    assert status == 200, response
    reply = json.loads(response)
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["choices"][0]["message"]["content"]
    assert 0 < reply["usage"]["completion_tokens"] <= 32


@pytest.mark.parametrize("clip_id", ["0006_en_short", "0152_zh_short"])
def test_transcription_names_the_language_before_a_tab(
    chat_port: int, chat_body: str, clip_id: str
) -> None:
    audio = (
        Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
    ).read_bytes()
    body = json.loads(chat_body)
    content = body["messages"][0]["content"]
    content[0]["input_audio"]["data"] = base64.b64encode(audio).decode()
    content[1]["text"] = ASR_PROMPT
    content.reverse()
    body["max_completion_tokens"] = 256
    status, response = request(chat_port, json.dumps(body))
    assert status == 200, response
    choice = json.loads(response)["choices"][0]
    assert choice["finish_reason"] == "stop"
    text = choice["message"]["content"]
    language_label, separator, transcript = text.partition("\t")
    assert language_label and separator and transcript.strip(), text


def test_answers_follow_the_recording(chat_port: int, chat_body: str) -> None:
    status, response = request(chat_port, chat_body)
    assert status == 200, response
    english = json.loads(response)["choices"][0]
    audio = (
        Path(DATA_ROOT) / "corpus" / "v1" / "clips" / "0152_zh_short.wav"
    ).read_bytes()
    body = json.loads(chat_body)
    body["messages"][0]["content"][0]["input_audio"]["data"] = base64.b64encode(
        audio
    ).decode()
    status, response = request(chat_port, json.dumps(body))
    assert status == 200, response
    chinese = json.loads(response)["choices"][0]
    assert english["finish_reason"] == chinese["finish_reason"] == "stop"
    assert english["message"]["content"] != chinese["message"]["content"]


def test_frozen_corpus_accuracy(chat_port: int, chat_body: str, tmp_path: Path) -> None:
    manifest = {
        clip["id"]: clip
        for clip in map(
            json.loads,
            (Path(__file__).parent / "corpus" / "manifest.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    body = json.loads(chat_body)
    content = body["messages"][0]["content"]
    content[1]["text"] = ASR_PROMPT
    content.reverse()
    body["max_completion_tokens"] = 1024
    transcripts = {}
    results = {}
    failures = []
    truncated = []
    for clip_id, clip in manifest.items():
        audio = (
            Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
        ).read_bytes()
        content[1]["input_audio"]["data"] = base64.b64encode(audio).decode()
        status, response = request(chat_port, json.dumps(body))
        reply = json.loads(response)
        results[clip_id] = {"status": status, "response": reply}
        if status != 200:
            failures.append(f"{clip_id}: HTTP {status}: {reply}")
        else:
            choice = reply["choices"][0]
            transcripts[clip_id] = choice["message"]["content"].split("\t", 1)[-1]
            if choice["finish_reason"] == "length":
                truncated.append(clip_id)
            else:
                pass
        print(f"{clip_id}: HTTP {status}", flush=True)
    metrics = quality(manifest, transcripts)
    failures.extend(
        f"{clip_id}: speech transcription reached the token limit"
        for clip_id in truncated
        if manifest[clip_id]["stratum"] not in ("silence", "noise")
    )
    pins = json.loads((Path(__file__).parent / "model_pins.json").read_text())
    report = {
        "model": MODEL_REPO,
        "revision": pins[MODEL_REPO]["revision"],
        "device": chip(),
        "max_completion_tokens": body["max_completion_tokens"],
        "metrics": metrics,
        "scored_clips": {
            metric: sum(select(manifest[clip_id]) for clip_id in transcripts)
            for metric, select in QUALITY_GROUPS.items()
        },
        "transcribed": len(transcripts),
        "over_30_seconds": sum(clip["duration"] > 30 for clip in manifest.values()),
        "truncated": truncated,
        "failures": failures,
        "clips": results,
    }
    results_path = tmp_path / "corpus-results.json"
    results_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{json.dumps(metrics)}\nresults: {results_path}", flush=True)
    assert not failures, "\n".join(failures)


def test_completion_limit_overrides_max_tokens(chat_port: int, chat_body: str) -> None:
    body = json.loads(chat_body)
    body["max_completion_tokens"] = 1
    body["max_tokens"] = 32
    status, response = request(chat_port, json.dumps(body))
    assert status == 200, response
    reply = json.loads(response)
    assert reply["usage"]["completion_tokens"] == 1
    assert reply["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream", True),
        ("temperature", 0.7),
        ("modalities", ["text", "audio"]),
        ("tools", []),
        ("model", "another-model"),
        ("max_completion_tokens", 0),
        ("max_completion_tokens", 8192),
        ("max_completion_tokens", 2**32 + 1),
        ("max_completion_tokens", 1.5),
        ("messages", []),
        ("messages", [{"role": "assistant", "content": "hello"}]),
    ],
)
def test_unsupported_requests(
    chat_port: int, chat_body: str, field: str, value: object
) -> None:
    body = json.loads(chat_body)
    body[field] = value
    status, response = request(chat_port, json.dumps(body))
    assert status == 400
    assert json.loads(response)["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("damage", ["base64", "wav", "format", "missing", "multiple"])
def test_invalid_audio(chat_port: int, chat_body: str, damage: str) -> None:
    body = json.loads(chat_body)
    content = body["messages"][0]["content"]
    if damage == "base64":
        content[0]["input_audio"]["data"] = "%%%"
    elif damage == "wav":
        content[0]["input_audio"]["data"] = base64.b64encode(b"not a wave").decode()
    elif damage == "format":
        content[0]["input_audio"]["format"] = "mp3"
    elif damage == "missing":
        content.pop(0)
    else:
        content.append(content[0])
    status, response = request(chat_port, json.dumps(body))
    assert status == 400
    assert json.loads(response)["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("body", ["{", "[]", "null"])
def test_invalid_json(chat_port: int, body: str) -> None:
    status, response = request(chat_port, body)
    assert status == 400
    assert json.loads(response)["error"]["message"]


def test_chat_needs_post_and_serves_no_transcriptions(chat_port: int) -> None:
    status, response = request(chat_port, method="GET")
    assert status == 405
    assert json.loads(response)["error"]["type"] == "invalid_request_error"
    assert request(chat_port, path="/v1/audio/transcriptions")[0] == 404
