# SPDX-License-Identifier: Apache-2.0
"""Check long-recording references, scoring and transport error handling."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import types
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from sglang_omni_mlx.native.ci import benchmark_long_audio, long_audio_client
from sglang_omni_mlx.native.ci.long_audio_client import (
    MemorySampler,
    score,
    stream_recording,
    transcribe_http,
)
from sglang_omni_mlx.native.ci.long_audio_corpus import (
    Clip,
    build_recording,
    load_clips,
)


def test_recording_keeps_whole_clips_and_matching_references(tmp_path: Path) -> None:
    """Truncating the last clip would score words that were never sent."""
    (tmp_path / "clips").mkdir()
    records = []
    for index in range(3):
        clip_id = f"{index:04d}_en_mid"
        with wave.open(str(tmp_path / "clips" / f"{clip_id}.wav"), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(bytes([index, 0]) * 16000)
        records.append(
            {
                "id": clip_id,
                "stratum": "en_mid",
                "lang": "en",
                "reference": f"word{index}",
            }
        )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records)
    )
    clips = load_clips(tmp_path, "en")
    result = build_recording(clips, 7.5, 2888, tmp_path / "long.wav")
    reference_by_id = {clip.clip_id: clip.reference for clip in clips}
    assert result["audio_seconds"] == 8
    assert result["reference"] == " ".join(
        reference_by_id[clip_id] for clip_id in result["clip_ids"]
    )
    assert len(set(result["clip_ids"][:3])) == 3
    assert all(
        left != right for left, right in zip(result["clip_ids"], result["clip_ids"][1:])
    )
    with wave.open(str(tmp_path / "long.wav"), "rb") as reader:
        assert reader.getnframes() == 8 * 16000
        expected = b"".join(
            bytes([int(clip_id[:4]), 0]) * 16000 for clip_id in result["clip_ids"]
        )
        assert reader.readframes(reader.getnframes()) == expected
    repeated = build_recording(clips, 7.5, 2888, tmp_path / "again.wav")
    assert repeated == result
    assert (
        result["wav_sha256"]
        == hashlib.sha256((tmp_path / "long.wav").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("language", "reference", "hypothesis", "errors", "reference_tokens"),
    [
        ("en", "It's FIVE words right here.", "it's five words", 2, 5),
        ("zh", "今天天气好。", "今天天气好好好", 2, 5),
    ],
)
def test_score_counts_dropped_tails_and_repeated_text(
    language: str, reference: str, hypothesis: str, errors: int, reference_tokens: int
) -> None:
    result = score(reference, hypothesis, language)
    assert result["errors"] == errors
    assert result["reference_tokens"] == reference_tokens
    assert result["error_rate"] == errors / reference_tokens


def test_empty_reference_is_not_a_perfect_score() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        score("", "hallucinated text", "en")


@pytest.mark.parametrize("send_final", [True, False])
def test_realtime_silent_audio_does_not_time_out_before_input_ends(
    tmp_path: Path, send_final: bool
) -> None:
    """A session with no preview events must remain open while audio is sent."""
    recording = tmp_path / "silence.wav"
    with wave.open(str(recording), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 48000)
    received_frames = []

    def handle(socket: ServerConnection) -> None:
        socket.recv()
        socket.send(json.dumps({"type": "transcription_session.updated"}))
        try:
            for raw in socket:
                event = json.loads(raw)
                if event["type"] == "input_audio_buffer.append":
                    received_frames.append(event["audio"])
                elif event["type"] == "transcription.done" and send_final:
                    socket.send(
                        json.dumps({"type": "transcription.completed", "text": ""})
                    )
                    return
                else:
                    pass
        except ConnectionClosed:
            pass

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            if send_final:
                result = stream_recording(
                    server.socket.getsockname()[1],
                    recording,
                    "en",
                    1,
                    1,
                    tmp_path / "events.jsonl",
                )
                assert result["text"] == ""
            else:
                with pytest.raises(TimeoutError, match="No completed transcript"):
                    stream_recording(
                        server.socket.getsockname()[1],
                        recording,
                        "en",
                        1,
                        1,
                        tmp_path / "events.jsonl",
                    )
            assert len(received_frames) == 30
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_realtime_pacing_starts_after_session_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow handshake must not cause the audio to be sent in a burst."""
    recording = tmp_path / "silence.wav"
    with wave.open(str(recording), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000)

    clock = [0.0]
    sent_at: list[float] = []

    class Socket:
        def __enter__(self) -> Socket:
            return self

        def __exit__(self, *arguments: object) -> None:
            pass

        def send(self, payload: str) -> None:
            if json.loads(payload)["type"] == "input_audio_buffer.append":
                sent_at.append(clock[0])
            else:
                pass

        def recv(self, timeout: float | None = None) -> str:
            if timeout is not None:
                clock[0] += 5
                return json.dumps({"type": "transcription_session.updated"})
            else:
                return json.dumps({"type": "transcription.completed", "text": ""})

        def close(self) -> None:
            pass

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(
        long_audio_client,
        "time",
        types.SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep),
    )
    monkeypatch.setattr(
        long_audio_client, "connect", lambda *arguments, **options: Socket()
    )
    stream_recording(1234, recording, "en", 1, 1, tmp_path / "events.jsonl")
    assert len(sent_at) == 10
    assert sent_at[1] - sent_at[0] == pytest.approx(0.1)


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS process accounting")
@pytest.mark.parametrize("limit_bytes,limit_seconds", [(1, 60), (2**40, 0)])
def test_resource_guard_stops_its_child(
    tmp_path: Path, limit_bytes: int, limit_seconds: float
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "input()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sampler = MemorySampler(
        process, tmp_path / "memory.jsonl", limit_bytes, limit_seconds
    )
    try:
        sampler.thread.start()
        assert process.wait(timeout=5) != 0
        sampler.thread.join(timeout=5)
        assert not sampler.thread.is_alive()
        if limit_seconds == 0:
            assert sampler.error == "Case wall-clock deadline exceeded"
        else:
            assert sampler.exceeded
        assert sampler.samples[0]["footprint_bytes"] > 0
    finally:
        sampler.done.set()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        else:
            pass


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS process accounting")
def test_resource_guard_stops_child_when_trace_cannot_be_written(
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "input()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sampler = MemorySampler(process, tmp_path / "missing" / "memory.jsonl", 2**40, 60)
    try:
        sampler.thread.start()
        assert process.wait(timeout=5) != 0
        sampler.thread.join(timeout=5)
        assert not sampler.thread.is_alive()
        assert "Memory trace failed" in sampler.error
    finally:
        sampler.done.set()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        else:
            pass


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_cannot_create_an_unbounded_recording(
    tmp_path: Path, seconds: float
) -> None:
    clips = [
        Clip(
            clip_id=str(index),
            path=tmp_path / f"{index}.wav",
            reference="hello",
            frames=16000,
            sha256="unused",
        )
        for index in range(2)
    ]
    with pytest.raises(ValueError):
        build_recording(clips, seconds, 2888, tmp_path / "invalid.wav")
    assert not (tmp_path / "invalid.wav").exists()


@pytest.mark.parametrize(
    "reply",
    [
        b'data: {"type":"transcript.text.done","text":"actual transcript"}\n\n',
        b':\n\ndata: {"type":"transcript.text.done","text":"actual transcript"}\n\n',
        b'data: {"type":"error","error":"failed"}\n\n',
        b":\n\n",
    ],
)
def test_http_requires_a_successful_final_event(reply: bytes) -> None:
    submitted = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            submitted.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(reply)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            if b"transcript.text.done" in reply:
                result = transcribe_http(server.server_port, b"audio", "en", 2)
                assert result["text"] == "actual transcript"
            else:
                with pytest.raises(RuntimeError):
                    transcribe_http(server.server_port, b"audio", "en", 2)
            assert b'name="audio_layout"\r\n\r\nvoxt_swift' in submitted[0]
            assert b'name="stop_on_token_loop"\r\n\r\ntrue' in submitted[0]
        finally:
            thread.join(timeout=5)


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS process accounting")
@pytest.mark.parametrize(
    "exit_policy", ["normal", "early_zero", "early_23", "shutdown_23"]
)
def test_unexpected_server_exit_fails_the_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_policy: str
) -> None:
    """An exit after recovery must not be reported as a successful measurement."""
    corpus = tmp_path / "corpus"
    (corpus / "clips").mkdir(parents=True)
    records = []
    for index in range(2):
        clip_id = f"{index:04d}_en_short"
        with wave.open(str(corpus / "clips" / f"{clip_id}.wav"), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * 16000)
        records.append(
            {"id": clip_id, "stratum": "en_short", "lang": "en", "reference": "hello"}
        )
    (corpus / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records)
    )
    model = tmp_path / "model"
    model.mkdir()
    server = tmp_path / "fake_server.py"
    server.write_text(
        f"#!{sys.executable}\n"
        + r"""import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

policy = EXIT_POLICY
post_count = 0
lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_POST(self):
        global post_count
        self.rfile.read(int(self.headers["Content-Length"]))
        with lock:
            post_count += 1
            count = post_count
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b":\n\ndata: {\"type\":\"transcript.text.done\",\"text\":\"hello\"}\n\n")
        self.wfile.flush()
        if count == 3 and policy.startswith("early_"):
            os._exit(0 if policy == "early_zero" else 23)

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(json.dumps({"event": "ready", "port": server.server_port}), flush=True)
for line in sys.stdin:
    if json.loads(line).get("command") == "shutdown":
        server.shutdown()
        sys.exit(23 if policy == "shutdown_23" else 0)
""".replace(
            "EXIT_POLICY", repr(exit_policy)
        )
    )
    server.chmod(0o755)
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_long_audio",
            "--server",
            str(server),
            "--model-directory",
            str(model),
            "--corpus-directory",
            str(corpus),
            "--output",
            str(output),
            "--mode",
            "http",
            "--minutes",
            "0.001",
            "--languages",
            "en",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        benchmark_long_audio.main()
    result = json.loads((output / "en-0.001m.result.json").read_text())
    assert result["recovery_matches_warmup"] is True
    if exit_policy == "normal":
        assert stopped.value.code == 0
        assert result["status"] == "measured"
        assert result["server_exit_code"] == 0
    else:
        assert stopped.value.code != 0
        assert result["status"] == "failed"
        assert result["server_exit_code"] == (0 if exit_policy == "early_zero" else 23)
        assert (
            "during shutdown" if exit_policy == "shutdown_23" else "before shutdown"
        ) in result["error"]
