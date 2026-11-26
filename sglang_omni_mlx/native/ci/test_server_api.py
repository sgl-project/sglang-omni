# SPDX-License-Identifier: Apache-2.0
"""API tests for the native qwen3_asr_server, run against the real model.

Needs NATIVE_RUNTIME_BIN (the server's directory) and CI_DATA_ROOT (a provisioned root).
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import select
import shutil
import subprocess
import time
import uuid
import wave
from array import array
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from websockets.sync.client import connect

from sglang_omni_mlx.native.ci.long_audio_client import score, stream_recording
from sglang_omni_mlx.native.ci.long_audio_corpus import build_recording, load_clips

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
AUTO_LANGUAGE_TEST_BIN = os.environ.get("NATIVE_QWEN3_ASR_AUTO_LANGUAGE_TEST")
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
MODEL_REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"
DRAFT_REPO = "wilsonzheng0327/Qwen3-ASR-0.6B-dflash"
LARGE_DRAFT_REPO = "wilsonzheng0327/Qwen3-ASR-1.7B-dflash"
# Bounds on a server that hangs, so a test fails instead of waiting forever.
STARTUP_TIMEOUT_SECONDS = 600
SHUTDOWN_TIMEOUT_SECONDS = 10
MODEL_TEST_TIMEOUT_SECONDS = 1800
REALTIME_EVENT_TIMEOUT_SECONDS = 60


def model_directory(repo: str = MODEL_REPO) -> Path:
    return Path(DATA_ROOT) / "models" / repo.replace("/", "_")


def clip_path(clip_id: str) -> Path:
    return Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav"


def clip(clip_id: str) -> bytes:
    return clip_path(clip_id).read_bytes()


def pcm16(clip_id: str) -> bytes:
    with wave.open(io.BytesIO(clip(clip_id))) as reader:
        return reader.readframes(reader.getnframes())


class Server:
    def __init__(
        self,
        binary: str = "qwen3_asr_server",
        model_kind: str = "qwen3_asr",
        directory: Path | None = None,
        flags: tuple[str, ...] = (),
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / binary),
                "--supervised",
                "--model-kind",
                model_kind,
                "--model-directory",
                str(directory or model_directory()),
                *flags,
            ],  # fmt: skip
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        readable, _, _ = select.select(
            [self.process.stdout], [], [], STARTUP_TIMEOUT_SECONDS
        )
        if not readable:
            self.process.kill()
            self.process.wait()
            raise TimeoutError(
                f"{binary} sent no ready event within {STARTUP_TIMEOUT_SECONDS} s"
            )
        else:
            self.ready = json.loads(self.process.stdout.readline())
        self.port = self.ready.get("port")

    def request(
        self, method: str, path: str, body: bytes = b"", headers: dict | None = None
    ):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()

    def post_form(self, fields: dict[str, str], wav: bytes | None):
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        if wav is not None:
            body += (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                    "Content-Type: audio/wav\r\n\r\n"
                ).encode()
                + wav
                + b"\r\n"
            )
        else:
            pass
        body += f"--{boundary}--\r\n".encode()
        return self.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.write('{"command": "shutdown"}\n')
            self.process.stdin.flush()
            try:
                self.process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                # A server left running would keep its model in memory.
                self.process.kill()
                self.process.wait()
                raise
        else:
            pass


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    running = Server()
    yield running
    running.stop()


def sse_events(body: bytes) -> list:
    return [
        (
            line[len("data: ") :]
            if line == "data: [DONE]"
            else json.loads(line[len("data: ") :])
        )
        for line in body.decode().splitlines()
        if line.startswith("data: ")
    ]


def test_ready_event_names_a_loopback_endpoint(server: Server) -> None:
    assert server.ready["event"] == "ready"
    assert server.ready["host"] == "127.0.0.1"
    assert server.ready["server_pid"] == server.process.pid
    assert server.ready["model_name"].startswith("voxt-qwen3_asr-")
    status, body = server.request("GET", "/health")
    assert (status, json.loads(body)) == (
        200,
        {"status": "healthy", "running": True, "request_states": {}},
    )
    status, body = server.request("GET", "/v1/models")
    assert json.loads(body)["data"] == [
        {"id": server.ready["model_name"], "object": "model"}
    ]


def test_voxt_final_request_streams_text_and_generation_metadata(
    server: Server,
) -> None:
    status, body = server.post_form(
        {
            "model": server.ready["model_name"],
            "stream": "true",
            "language": "English",
            "max_new_tokens": "1024",
            "stop_at_end_of_text": "true",
            "stop_on_token_loop": "true",
            "include_generation_metadata": "true",
            "audio_layout": "voxt_swift",
        },
        clip("0006_en_short"),
    )
    assert status == 200
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    assert events[0] == {
        "type": "transcript.text.done",
        "text": "Surely you are not thinking of going off there.",
        "generation_metadata": {
            "generated_token_count": 11,
            "language": "English",
            "finish_reason": "stop",
        },
    }


def test_a_draft_model_serves_the_voxt_final_request() -> None:
    """The draft's proposals are verified in place: Voxt's request is answered as without one,
    a token budget below the block size is kept exactly, and a guess of the text saves the
    draft's rounds when it is right and changes nothing when it is wrong."""
    with_draft = Server(flags=("--draft-model-path", str(model_directory(DRAFT_REPO))))
    final_fields = {
        "model": with_draft.ready["model_name"],
        "stream": "true",
        "language": "English",
        "max_new_tokens": "1024",
        "stop_at_end_of_text": "true",
        "stop_on_token_loop": "true",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }
    try:
        status, body = with_draft.post_form(final_fields, clip("0006_en_short"))
        text = sse_events(body)[0]["text"]
        guessed_events = {}
        for guessed_text in (text, "An unrelated sentence.", text + " And more."):
            guess_status, guess_body = with_draft.post_form(
                final_fields | {"guessed_text": guessed_text},
                clip("0006_en_short"),
            )
            assert guess_status == 200
            guessed_events[guessed_text] = sse_events(guess_body)[0]
        budget_status, budget_body = with_draft.post_form(
            {
                "model": with_draft.ready["model_name"],
                "stream": "true",
                "language": "English",
                "max_new_tokens": "2",
                "include_generation_metadata": "true",
                "audio_layout": "voxt_swift",
            },
            clip("0006_en_short"),
        )
    finally:
        with_draft.stop()
    assert status == 200
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    assert events[0]["type"] == "transcript.text.done"
    assert events[0]["text"]
    assert events[0]["generation_metadata"]["finish_reason"] == "stop"
    assert events[0]["generation_metadata"]["language"] == "English"
    metadata = events[0]["generation_metadata"]
    assert metadata["verify_round_count"] > 0
    assert metadata["proposed_draft_token_count"] > 0
    assert (
        0
        <= metadata["accepted_draft_token_count"]
        <= metadata["proposed_draft_token_count"]
    )
    assert all(event["text"] == text for event in guessed_events.values())
    assert "verify_round_count" not in guessed_events[text]["generation_metadata"]
    assert budget_status == 200
    budget_done = sse_events(budget_body)[0]
    assert budget_done["text"]
    assert budget_done["generation_metadata"]["finish_reason"] == "length"
    assert budget_done["generation_metadata"]["generated_token_count"] == 2
    assert budget_done["generation_metadata"]["verify_round_count"] == 1
    assert budget_done["generation_metadata"]["proposed_draft_token_count"] == 0
    assert budget_done["generation_metadata"]["accepted_draft_token_count"] == 0


def test_a_draft_for_another_target_fails_at_load() -> None:
    """A draft of the wrong hidden size is refused before ready, not at the first request."""
    failed = Server(
        flags=("--draft-model-path", str(model_directory(LARGE_DRAFT_REPO)))
    )
    assert failed.ready["event"] == "failed"
    assert "hidden size 2048" in failed.ready["reason"]
    assert failed.process.wait(timeout=10) == 1


def test_backoff_preserves_recording_quality(tmp_path: Path) -> None:
    """Adaptive decoding hands over to plain decoding when the draft loses, and the transcript stays as accurate."""
    # note (XXXXRT666): the last clip ends before 120 s, so the recording is one chunk.
    recording_path = tmp_path / "recording.wav"
    recording = build_recording(
        load_clips(Path(DATA_ROOT) / "corpus" / "v1", "zh"),
        100,
        0,
        recording_path,
    )
    # note (XXXXRT666): a draft with zero weights loses on any machine, as its every proposal is rejected.
    draft_directory = tmp_path / "draft"
    shutil.copytree(
        model_directory(DRAFT_REPO),
        draft_directory,
        ignore=shutil.ignore_patterns("*.safetensors"),
    )
    for weights in model_directory(DRAFT_REPO).glob("*.safetensors"):
        with weights.open("rb") as reader:
            header_size = reader.read(8)
            header = reader.read(int.from_bytes(header_size, "little"))
        with (draft_directory / weights.name).open("wb") as writer:
            writer.write(header_size + header)
            writer.truncate(weights.stat().st_size)
    outputs = {}
    for policy in ("fixed", "adaptive"):
        result = subprocess.run(
            [
                str(Path(RUNTIME_BIN) / "qwen3_asr_transcribe"),
                "--model-path",
                str(model_directory()),
                "--draft-model-path",
                str(draft_directory),
                "--draft-policy",
                policy,
                "--layout",
                "voxt_swift",
                "--language",
                "Chinese",
                "--max-new-tokens",
                "4096",
                "--cache-limit-bytes",
                str(2 << 30),
                "--stop-at-end-of-text",
                "--stop-on-token-loop",
                str(recording_path),
            ],  # fmt: skip
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        outputs[policy] = json.loads(result.stdout)
        assert outputs[policy]["finish_reason"] == "stop"
        assert ("DFlash backoff after " in result.stderr) == (
            policy == "adaptive"
        ), result.stderr
    fixed = outputs["fixed"]
    adaptive = outputs["adaptive"]
    assert 0 < adaptive["verify_rounds"] < fixed["verify_rounds"]
    fixed_error = score(recording.reference, fixed["text"], "zh").error_rate
    adaptive_error = score(recording.reference, adaptive["text"], "zh").error_rate
    golden = json.loads(
        (
            Path(__file__).parent / "golden" / "qwen3-asr-0.6b-4bit-dflash.json"
        ).read_text()
    )
    # note (ZaneZhang-67): a faster token-loop stop is not a quality-preserving backoff.
    assert (adaptive_error - fixed_error) * 100 <= golden["tolerance"]["metric_pp"], (
        fixed_error,
        adaptive_error,
        fixed["generated_token_count"],
        adaptive["generated_token_count"],
    )


def test_plain_request_returns_json_text(server: Server) -> None:
    status, body = server.post_form({"language": "en"}, clip("0006_en_short"))
    assert (status, json.loads(body)) == (
        200,
        {"text": "Surely you are not thinking of going off there."},
    )


@pytest.mark.parametrize(
    ("fields", "wav"),
    [
        ({}, None),
        ({}, b"not audio"),
        ({"include_generation_metadata": "true"}, "wav"),
        ({"audio_layout": "sideways"}, "wav"),
        ({"max_new_tokens": "many"}, "wav"),
        ({"max_new_tokens": "-1"}, "wav"),
        ({"repetition_penalty": "0"}, "wav"),
        ({"repetition_penalty": "-1"}, "wav"),
        ({"repetition_penalty": "nan"}, "wav"),
    ],
)
def test_invalid_requests_are_rejected(server: Server, fields: dict, wav) -> None:
    status, body = server.post_form(
        fields, clip("0006_en_short") if wav == "wav" else wav
    )
    assert status == 400
    detail = json.loads(body)["detail"]
    assert all(name in detail for name in fields)


def wav_bytes(pcm: bytes) -> bytes:
    """16 kHz mono 16-bit PCM wrapped in a WAV header."""
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm)
    return out.getvalue()


def long_wav(seconds: int) -> bytes:
    """A clip repeated to about seconds long, so a full decode takes several seconds."""
    pcm = pcm16("0344_en_long")
    repeated = pcm * (seconds * 32000 // len(pcm) + 1)
    return wav_bytes(repeated[: seconds * 32000])


def test_repetition_penalty_ends_a_token_loop(server: Server) -> None:
    """Greedy decoding repeats one phrase on this audio until the token budget runs out, and repetition_penalty was ignored."""
    # Note (Minyi Tang): 30.0 s to 31.1 s of this clip is the public loop trigger from issue 3044.
    looping_wav = wav_bytes(
        pcm16("0366_zh_long")[int(30.0 * 32000) : int(31.1 * 32000)]
    )
    finish_reasons = {}
    for repetition_penalty in ("1", "1.2"):
        status, body = server.post_form(
            {
                "stream": "true",
                "language": "zh",
                "include_generation_metadata": "true",
                "stop_on_token_loop": "false",
                "repetition_penalty": repetition_penalty,
            },
            looping_wav,
        )
        assert status == 200
        finish_reasons[repetition_penalty] = sse_events(body)[0]["generation_metadata"][
            "finish_reason"
        ]
    assert finish_reasons == {"1": "length", "1.2": "stop"}


def test_a_disconnected_stream_stops_its_decode(server: Server) -> None:
    transcript_before = server.post_form({}, clip("0006_en_short"))
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="max_new_tokens"\r\n\r\n4096\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        ).encode()
        + long_wav(300)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    # Note (Jiaxin Deng): cancellation is checked between generated tokens, so
    # give the request time to get past encoding and prefill.
    time.sleep(4.0)
    assert json.loads(server.request("GET", "/health")[1])["request_states"] == {
        "running": 1
    }
    closed_at = time.monotonic()
    # Note (Jiaxin Deng): http.client keeps the socket open while the response
    # object lives.
    response.close()
    connection.close()
    while json.loads(server.request("GET", "/health")[1])["request_states"] != {}:
        assert (
            time.monotonic() - closed_at < 1.5
        ), "the decode kept running after its client left"
        time.sleep(0.05)
    assert server.post_form({}, clip("0006_en_short")) == transcript_before


def open_stream(server: Server, wav: bytes) -> http.client.HTTPResponse:
    """Starts a streamed request with a large token budget; closing the response cancels it."""
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="max_new_tokens"\r\n\r\n4096\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        ).encode()
        + wav
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=60)
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    return response


def request_states(server: Server) -> dict:
    return json.loads(server.request("GET", "/health")[1])["request_states"]


def wait_for(condition, timeout_seconds: float, message: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(0.05)


def test_concurrent_requests_decode_together_and_match_serial_output() -> None:
    """Queued requests batch under --max-batch; each finishes on its own and a cancelled one leaves the rest intact."""
    clip_ids = ["0006_en_short", "0007_en_short", "0008_en_short"]
    server = Server(flags=("--max-batch", "4"))
    try:
        # Note: a request alone in the queue runs serially.
        serial = {clip_id: server.post_form({}, clip(clip_id)) for clip_id in clip_ids}
        assert all(status == 200 for status, _ in serial.values())
        # The first request occupies the worker so the next four queue behind
        # it and are taken as one batch once it is cancelled.
        blocker = open_stream(server, long_wav(300))
        wait_for(
            lambda: request_states(server) == {"running": 1},
            10,
            "the blocking request never started",
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            short = {
                clip_id: pool.submit(server.post_form, {}, clip(clip_id))
                for clip_id in clip_ids
            }
            # note (XXXXRT666): longer audio is decoded in chunks, outside the batch.
            canceled = pool.submit(open_stream, server, long_wav(120))
            wait_for(
                lambda: request_states(server).get("queued") == 4,
                10,
                "the requests did not queue behind the blocker",
            )
            blocker.close()
            # The short requests answer while the long one is still decoding.
            results = {clip_id: future.result(60) for clip_id, future in short.items()}
            assert request_states(server) == {"running": 1}
            assert results == serial
            # Cancelling the long row ends only that row.
            canceled.result(10).close()
            wait_for(
                lambda: request_states(server) == {},
                3,
                "the cancelled row kept decoding",
            )
        assert server.post_form({}, clip(clip_ids[0])) == serial[clip_ids[0]]
    finally:
        server.stop()


def test_realtime_accepts_silent_audio_without_a_preview(
    server: Server, tmp_path: Path
) -> None:
    """A session with no preview events must stay open while audio is sent."""
    recording = tmp_path / "silence.wav"
    recording.write_bytes(wav_bytes(b"\x00\x00" * 64000))
    result = stream_recording(
        server.port, recording, "en", 1, 1, tmp_path / "events.jsonl"
    )
    assert result.text == ""
    assert result.final_segment_count == 1


def test_realtime_manual_session(server: Server) -> None:
    pcm = pcm16("0006_en_short")
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": {"type": "server_vad"}},
                }
            )
        )
        assert json.loads(socket.recv())["error"]["code"] == "unsupported_session"
        socket.send("{not json")
        assert json.loads(socket.recv())["error"]["code"] == "invalid_json"
        socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": "***"}))
        assert json.loads(socket.recv())["error"]["code"] == "invalid_audio"
        socket.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": "AAAAAA"})
        )
        assert json.loads(socket.recv())["error"]["code"] == "invalid_audio"
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": "en"},
                }
            )
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
    assert [event["text"] for event in finals] == [events[-1]["text"]]
    assert events[-1]["text"].startswith("Surely you are not thinking")
    indexes = [event["event_index"] for event in events]
    assert indexes == sorted(indexes)


def test_a_segment_final_reuses_its_live_session_cache(server: Server) -> None:
    """A segment Final reuses cache, matches reference decoding, and preserves HTTP reuse."""
    clip_id = "0098_en_mid"
    pcm = pcm16(clip_id)
    segment_pcm = pcm[: 17 * 32000]
    fields = {
        "stream": "true",
        "language": "en",
        "stop_on_token_loop": "true",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }

    def final(extra: dict[str, str], wav: bytes = clip(clip_id)) -> dict:
        status, body = server.post_form(fields | extra, wav)
        assert status == 200
        return sse_events(body)[0]

    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": "en"},
                }
            )
        )
        session_id = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))[
            "session_id"
        ]
        # note (XXXXRT666): 17 s of audio in one append make one refresh, over two windows.
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(segment_pcm).decode(),
                }
            )
        )
        preview = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
        assert preview["type"] == "transcription.segment"
        assert preview["segment_id"] == 0
        assert preview["is_final"] is False
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        finalized = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
        assert finalized["type"] == "transcription.segment"
        assert finalized["segment_id"] == 0
        assert finalized["is_final"] is True
        assert finalized["generation_metadata"]["reused_audio_token_count"] == 208
        assert finalized["generation_metadata"]["reused_prompt_token_count"] > 208
        reference = final({"audio_layout": "reference"}, wav_bytes(segment_pcm))
        plain = final({})
        reused = final({"realtime_session": session_id})
        with_context = final({"realtime_session": session_id, "prompt": "GLOSSARY"})
        other_audio = final(
            {"realtime_session": session_id},
            wav_bytes(pcm16("0100_en_mid")[: len(pcm)]),
        )
        unknown = final({"realtime_session": "sess_0"})
    assert finalized["text"] == reference["text"]
    assert reused["text"] == plain["text"] == unknown["text"]
    # note (XXXXRT666): two complete 8 s windows of 104 audio tokens.
    assert reused["generation_metadata"]["reused_audio_token_count"] == 208
    assert (
        with_context["generation_metadata"]["reused_audio_token_count"] == 208
        and "reused_audio_token_count" not in other_audio["generation_metadata"]
        and "reused_audio_token_count" not in unknown["generation_metadata"]
    )


def test_a_short_segment_final_encodes_audio_without_a_complete_cache(
    server: Server,
) -> None:
    """A Final with no completed encoder window computes its audio instead of claiming reuse."""
    pcm = pcm16("0006_en_short")[: 5 * 32000]
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": "en"},
                }
            )
        )
        json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                }
            )
        )
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        events: list[dict[str, object]] = []
        deadline = time.monotonic() + REALTIME_EVENT_TIMEOUT_SECONDS
        while not events or not events[-1].get("is_final", False):
            remaining = deadline - time.monotonic()
            assert remaining > 0, "timed out waiting for the segment Final"
            events.append(json.loads(socket.recv(timeout=remaining)))
    finalized = events[-1]
    assert finalized["type"] == "transcription.segment"
    assert finalized["generation_metadata"]["reused_audio_token_count"] == 0


def test_a_segment_final_verifies_eligible_rollback_guesses(
    server: Server,
) -> None:
    """Two completed previews make rollback guesses eligible for Segment Final."""
    quiet = array(
        "h",
        (sample // 10 for sample in array("h", pcm16("0098_en_mid")[: 9 * 32000])),
    ).tobytes()
    loud = pcm16("0108_en_mid")[: 2 * 32000]
    chunks = (quiet, loud)
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": "en"},
                }
            )
        )
        updated = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
        assert updated["type"] == "transcription_session.updated"
        preview_indexes = []
        for chunk in chunks:
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )
            preview = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
            assert preview["type"] == "transcription.segment"
            assert preview["segment_id"] == 0
            assert preview["is_final"] is False
            preview_indexes.append(preview["event_index"])
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        finalized = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
    assert preview_indexes == sorted(set(preview_indexes))
    assert finalized["type"] == "transcription.segment"
    assert finalized["segment_id"] == 0
    assert finalized["is_final"] is True
    assert preview_indexes[-1] < finalized["event_index"]
    metadata = finalized["generation_metadata"]
    assert metadata["verified_guess_token_count"] > 0
    assert (
        0
        <= metadata["accepted_guess_token_count"]
        <= metadata["verified_guess_token_count"]
    )


def test_a_final_reuses_the_keys_and_values_of_its_prompt(server: Server) -> None:
    """The Final's dictionary was computed again after Stop, with all the audio after it: the session that gets the same dictionary to prefill has its keys and values already, and the text is the same."""
    clip_id = "0098_en_mid"
    fields = {
        "stream": "true",
        "language": "en",
        "prompt": "GLOSSARY",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "language": "en",
                        "final_prompt": "GLOSSARY",
                    },
                }
            )
        )
        session_id = json.loads(socket.recv())["session_id"]
        # note (XXXXRT666): the prefill starts after the first preview and the windows' part is queued before the second one's event is sent, so the Final after that event finds both.
        for samples in (
            pcm16(clip_id)[: 9 * 32000],
            pcm16(clip_id)[9 * 32000 : 17 * 32000],
        ):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(samples).decode(),
                    }
                )
            )
            assert json.loads(socket.recv())["type"] == "transcription.segment"
        status, body = server.post_form(
            fields | {"realtime_session": session_id}, clip(clip_id)
        )
        assert status == 200
        reused = sse_events(body)[0]
    status, body = server.post_form(fields, clip(clip_id))
    assert status == 200
    assert reused["text"] == sse_events(body)[0]["text"]
    # note (XXXXRT666): the head of the prompt and two 8 s windows of 104 audio tokens.
    assert reused["generation_metadata"]["reused_prompt_token_count"] > 208


@pytest.mark.skipif(
    not AUTO_LANGUAGE_TEST_BIN,
    reason="set NATIVE_QWEN3_ASR_AUTO_LANGUAGE_TEST",
)
@pytest.mark.parametrize("draft_repo", [None, DRAFT_REPO], ids=["greedy", "dflash"])
def test_prefixed_segment_final_preserves_output_with_rollback_guesses(
    draft_repo: str | None,
) -> None:
    assert AUTO_LANGUAGE_TEST_BIN is not None
    draft_directory = (
        model_directory(draft_repo) if draft_repo is not None else Path("-")
    )
    subprocess.run(
        [
            AUTO_LANGUAGE_TEST_BIN,
            str(model_directory()),
            str(draft_directory),
            str(clip_path("0098_en_mid")),
            "rollback_guess_parity",
            str(clip_path("0108_en_mid")),
        ],
        check=True,
        timeout=MODEL_TEST_TIMEOUT_SECONDS,
    )


@pytest.mark.skipif(
    not AUTO_LANGUAGE_TEST_BIN,
    reason="set NATIVE_QWEN3_ASR_AUTO_LANGUAGE_TEST",
)
@pytest.mark.parametrize("draft_repo", [None, DRAFT_REPO], ids=["greedy", "dflash"])
def test_an_incorrect_auto_language_guess_does_not_replace_final_output(
    draft_repo: str | None,
) -> None:
    assert AUTO_LANGUAGE_TEST_BIN is not None
    draft_directory = (
        model_directory(draft_repo) if draft_repo is not None else Path("-")
    )
    subprocess.run(
        [
            AUTO_LANGUAGE_TEST_BIN,
            str(model_directory()),
            str(draft_directory),
            str(clip_path("0098_en_mid")),
            "wrong_guess",
        ],
        check=True,
        timeout=MODEL_TEST_TIMEOUT_SECONDS,
    )


@pytest.mark.skipif(
    not AUTO_LANGUAGE_TEST_BIN,
    reason="set NATIVE_QWEN3_ASR_AUTO_LANGUAGE_TEST",
)
def test_auto_language_final_guess_requires_an_eligible_live_preview() -> None:
    assert AUTO_LANGUAGE_TEST_BIN is not None
    subprocess.run(
        [
            AUTO_LANGUAGE_TEST_BIN,
            str(model_directory()),
            "-",
            str(clip_path("0098_en_mid")),
            "eligibility",
        ],
        check=True,
        timeout=MODEL_TEST_TIMEOUT_SECONDS,
    )


def test_a_louder_frame_after_the_last_window_rebuilds_the_prefill(
    server: Server,
) -> None:
    """Speech that got louder after the last complete window raised the log-mel floor, so the windows encoded before it did not fit the Final, which encoded the recording again; the session compares them with the audio at every preview and computes them again, so the Final has them, with the text of a Final without a session."""
    quiet = array(
        "h",
        (sample // 10 for sample in array("h", pcm16("0098_en_mid")[: 9 * 32000])),
    ).tobytes()
    loud = pcm16("0108_en_mid")[: 2 * 32000]
    recording = quiet + loud
    fields = {
        "stream": "true",
        "language": "en",
        "prompt": "GLOSSARY",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "language": "en",
                        "final_prompt": "GLOSSARY",
                    },
                }
            )
        )
        session_id = json.loads(socket.recv())["session_id"]
        for samples in (quiet, loud):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(samples).decode(),
                    }
                )
            )
            assert json.loads(socket.recv())["type"] == "transcription.segment"
            wait_for(
                lambda: request_states(server) == {}, 30, "the prefill did not finish"
            )
        status, body = server.post_form(
            fields | {"realtime_session": session_id}, wav_bytes(recording)
        )
        assert status == 200
        reused = sse_events(body)[0]
    status, body = server.post_form(fields, wav_bytes(recording))
    assert status == 200
    assert reused["text"] == sse_events(body)[0]["text"]
    # note (XXXXRT666): one complete 8 s window of 104 audio tokens.
    assert reused["generation_metadata"]["reused_audio_token_count"] == 104


def test_a_final_of_the_recordings_rest_reuses_its_windows(server: Server) -> None:
    """A recording over 60 s is recognised in pieces, and its last Final encoded its audio again: the client sends where the Final starts, the session encodes the windows from there, and the Final of that audio has them, with the text of a Final without a session. A start the session does not have is an error."""
    recording = (pcm16("0098_en_mid") + pcm16("0117_en_mid"))[: 37 * 32000]
    start_sample = int(9.5 * 16000)
    rest = recording[start_sample * 2 :]
    fields = {
        "stream": "true",
        "language": "en",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }

    def final(extra: dict[str, str]) -> dict:
        status, body = server.post_form(fields | extra, wav_bytes(rest))
        assert status == 200
        return sse_events(body)[0]

    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "language": "en",
                        "final_prompt": "",
                    },
                }
            )
        )
        session_id = json.loads(socket.recv())["session_id"]
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(recording).decode(),
                }
            )
        )
        # note (XXXXRT666): the 30 s cut's final and the second segment's preview.
        for _ in range(2):
            assert json.loads(socket.recv())["type"] == "transcription.segment"
        for invalid_start in (-1, len(recording), "9"):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.final_start",
                        "start_sample": invalid_start,
                    }
                )
            )
            assert json.loads(socket.recv())["type"] == "error"
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.final_start",
                    "start_sample": start_sample,
                }
            )
        )
        # note (XXXXRT666): the windows from the start on, one background job each.
        wait_for(lambda: request_states(server) == {}, 30, "the prefill did not finish")
        reused = final({"realtime_session": session_id})
    # note (XXXXRT666): 27.5 s of audio has three complete 8 s windows of 104 audio tokens.
    assert reused["generation_metadata"]["reused_audio_token_count"] == 312
    assert reused["generation_metadata"]["reused_prompt_token_count"] > 312
    assert reused["text"] == final({})["text"]


def test_a_final_of_other_audio_reuses_the_dictionary_alone(server: Server) -> None:
    """A recording over 60 s is recognised in pieces, and its last Final computed the dictionary again: the audio is not the session's recording, so only the prompt the session prefilled is reused, and the text is the one without a session."""
    clip_id = "0108_en_mid"
    dictionary = "\n".join(f"term{i}" for i in range(300))
    fields = {
        "stream": "true",
        "language": "en",
        "prompt": dictionary,
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "language": "en",
                        "final_prompt": dictionary,
                    },
                }
            )
        )
        session_id = json.loads(socket.recv())["session_id"]
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm16("0098_en_mid")[: 9 * 32000]
                    ).decode(),
                }
            )
        )
        assert json.loads(socket.recv())["type"] == "transcription.segment"
        # note (XXXXRT666): the prefill's chunks of 256 tokens, then the window.
        wait_for(lambda: request_states(server) == {}, 30, "the prefill did not finish")
        status, body = server.post_form(
            fields | {"realtime_session": session_id}, clip(clip_id)
        )
        assert status == 200
        reused = sse_events(body)[0]
    status, body = server.post_form(fields, clip(clip_id))
    assert status == 200
    assert reused["text"] == sse_events(body)[0]["text"]
    metadata = reused["generation_metadata"]
    assert "reused_audio_token_count" not in metadata
    # note (XXXXRT666): the dictionary is over 600 tokens, and no token of the audio.
    assert metadata["reused_prompt_token_count"] > 600


def test_a_long_final_reuses_the_windows_after_the_first_segment(
    server: Server,
) -> None:
    """The Final of a recording past the previews' 30 s cut reused the first segment's windows alone; the session encodes the others in the background from the whole recording, so the Final has them, and its text is the one without a session."""
    pcm = (pcm16("0098_en_mid") + pcm16("0117_en_mid"))[: 33 * 32000]
    fields = {
        "stream": "true",
        "language": "en",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }

    def final(extra: dict[str, str]) -> dict:
        status, body = server.post_form(fields | extra, wav_bytes(pcm))
        assert status == 200
        return sse_events(body)[0]

    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        # note (XXXXRT666): an empty final_prompt is that of a Final without a dictionary.
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "language": "en",
                        "final_prompt": "",
                    },
                }
            )
        )
        session_id = json.loads(socket.recv())["session_id"]
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                }
            )
        )
        # note (XXXXRT666): the first segment's final and the second's preview.
        initial_segments = [
            json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
            for _ in range(2)
        ]
        assert [event["type"] for event in initial_segments] == [
            "transcription.segment",
            "transcription.segment",
        ]
        assert [event["segment_id"] for event in initial_segments] == [0, 1]
        assert [event["is_final"] for event in initial_segments] == [True, False]
        # note (XXXXRT666): the second preview queued the background jobs, one window each, and the server is idle again once they are done.
        wait_for(lambda: request_states(server) == {}, 30, "the prefill did not finish")
        reused = final({"realtime_session": session_id})
        socket.send(json.dumps({"type": "transcription.done"}))
        second_final = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
        completed = json.loads(socket.recv(timeout=REALTIME_EVENT_TIMEOUT_SECONDS))
        assert second_final["type"] == "transcription.segment"
        assert second_final["segment_id"] == 1
        assert second_final["is_final"] is True
        assert completed["type"] == "transcription.completed"
        assert completed["text"]
        event_indices = [
            initial_segments[0]["event_index"],
            initial_segments[1]["event_index"],
            second_final["event_index"],
            completed["event_index"],
        ]
        assert event_indices == sorted(event_indices)
    # note (XXXXRT666): four complete 8 s windows of 104 audio tokens.
    assert reused["generation_metadata"]["reused_audio_token_count"] == 416
    assert reused["generation_metadata"]["reused_prompt_token_count"] > 416
    assert reused["text"] == final({})["text"]


def test_a_final_reads_a_tail_under_one_second_like_a_whole_recording(
    server: Server,
) -> None:
    """Audio less than 1 s past an 8 s window was encoded alone, with less padding than in a whole recording, and the Final repeated a sentence the recording has once."""
    pcm = pcm16("0228_zh_mid")[: int(8.6 * 32000)]
    fields = {
        "stream": "true",
        "language": "zh",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }

    def final(extra: dict[str, str]) -> dict:
        status, body = server.post_form(fields | extra, wav_bytes(pcm))
        assert status == 200
        return sse_events(body)[0]

    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": "zh"},
                }
            )
        )
        session_id = json.loads(socket.recv())["session_id"]
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                }
            )
        )
        assert json.loads(socket.recv())["type"] == "transcription.segment"
        reused = final({"realtime_session": session_id})
    # note (XXXXRT666): the 8 s window's 104 audio tokens.
    assert reused["generation_metadata"]["reused_audio_token_count"] == 104
    assert reused["text"] == final({})["text"]


def test_a_silent_chunk_of_a_long_request_adds_no_text(server: Server) -> None:
    """A request over 120 s is cut into chunks, and the chunk of its closing silence was decoded like the others, which added words nobody said (The.)."""
    speech = (pcm16("0344_en_long") * 2)[: 114 * 32000]
    recording = speech + bytes(16 * 32000)
    # note (XXXXRT666): the cut is in the middle of the first quiet 100 ms of the 5 s before 120 s.
    first_chunk = recording[: 115 * 32000 + 1600]

    def transcribe(pcm: bytes) -> dict:
        status, body = server.post_form(
            {
                "stream": "true",
                "language": "en",
                "include_generation_metadata": "true",
            },
            wav_bytes(pcm),
        )
        assert status == 200
        return sse_events(body)[0]

    assert transcribe(recording) == transcribe(first_chunk)


def test_a_long_request_is_cut_whatever_a_live_session_has_of_it(
    server: Server,
) -> None:
    """A request over 120 s with a preview's text or a live session was decoded as one prompt, on which the model can repeat a sentence until its tokens run out; both only save work, so the answer is the one without them. The chunks share max_new_tokens."""
    pcm = (pcm16("0344_en_long") * 3)[: 130 * 32000]

    def final(extra: dict[str, str]) -> dict:
        status, body = server.post_form(
            {"stream": "true", "include_generation_metadata": "true", **extra},
            wav_bytes(pcm),
        )
        assert status == 200
        return sse_events(body)[0]

    plain = final({})
    assert final({"guessed_text": plain["text"]}) == plain
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        session_id = json.loads(socket.recv())["session_id"]
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[: 9 * 32000]).decode(),
                }
            )
        )
        assert json.loads(socket.recv())["type"] == "transcription.segment"
        assert final({"realtime_session": session_id}) == plain
    truncated = final({"max_new_tokens": "2"})["generation_metadata"]
    assert truncated["finish_reason"] == "length"
    assert truncated["generated_token_count"] == 2


@pytest.mark.parametrize(
    "fields, finish_reason, is_collapsed",
    [
        ({}, "stop", True),
        ({"stop_on_token_loop": "true"}, "stop", True),
        ({"stop_on_token_loop": "false"}, "length", False),
    ],
)
def test_a_token_loop_stops_with_one_copy_of_its_unit(
    server: Server, fields: dict[str, str], finish_reason: str, is_collapsed: bool
) -> None:
    """A request without stop_on_token_loop repeated 我覺得 to the 128-token budget, and one that sent it still kept about 12 copies."""
    # Note (Minyi Tang): 30.0 s to 31.1 s of this clip is the public loop trigger from issue 2150.
    status, body = server.post_form(
        {
            "stream": "true",
            "language": "zh",
            "include_generation_metadata": "true",
            **fields,
        },
        wav_bytes(pcm16("0366_zh_long")[int(30.0 * 32000) : int(31.1 * 32000)]),
    )
    assert status == 200
    done = sse_events(body)[0]
    metadata = done["generation_metadata"]
    assert (
        metadata["finish_reason"],
        metadata["generated_token_count"] < 128,
        done["text"].count("我覺得") == 1,
    ) == (finish_reason, is_collapsed, is_collapsed)


def test_end_of_stdin_stops_the_server() -> None:
    running = Server()
    running.process.stdin.close()
    assert running.process.wait(timeout=10) == 0


def test_shutdown_reports_stopped() -> None:
    running = Server()
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0
