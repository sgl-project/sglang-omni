# SPDX-License-Identifier: Apache-2.0
"""Check long-recording references, scoring, transports and whole benchmark cases."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import threading
import types
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from native_server import REQUEST_TIMEOUT_SECONDS, SupervisedServer, silent_wav
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from sglang_omni_mlx.native.ci import benchmark_long_audio, long_audio_client
from sglang_omni_mlx.native.ci.benchmark_long_audio import stop_server
from sglang_omni_mlx.native.ci.long_audio_client import (
    MemorySampler,
    score,
    stream_recording,
    transcribe_chunks,
    transcribe_http,
    voxt_token_budget,
)
from sglang_omni_mlx.native.ci.long_audio_corpus import (
    build_recording,
    chunk_frame_counts,
    load_clips,
)

DONE_EVENT = (
    b'data: {"type":"transcript.text.done","text":"actual transcript",'
    b'"generation_metadata":{"generated_token_count":2,"language":"en","finish_reason":"stop"}}\n\n'
)
STUB_KIND_FLAGS = ("--", "--model-kind", "stub_asr")


def write_corpus(corpus: Path) -> None:
    """Three one-second English clips whose samples name their index."""
    (corpus / "clips").mkdir(parents=True)
    records = []
    for index in range(3):
        clip_id = f"{index:04d}_en_mid"
        with wave.open(str(corpus / "clips" / f"{clip_id}.wav"), "wb") as writer:
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
    (corpus / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records)
    )


def run_benchmark(
    monkeypatch: pytest.MonkeyPatch, data_root: Path, runtime_bin: Path, *flags: str
) -> tuple[int, Path, dict[str, object]]:
    """One 3-second case: its exit code, output directory and result fields."""
    write_corpus(data_root / "corpus" / "v1")
    (data_root / "models" / "owner_model").mkdir(parents=True)
    output = data_root / "results"
    monkeypatch.setattr(benchmark_long_audio, "MEMORY_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_long_audio",
            "--runtime-bin",
            str(runtime_bin),
            "--data-root",
            str(data_root),
            "--model",
            "owner/model",
            "--output",
            str(output),
            "--languages",
            "en",
            "--minutes",
            "0.04",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        benchmark_long_audio.main()
    result = json.loads((output / "en-0.04m.result.json").read_text())
    return stopped.value.code, output, result


@pytest.fixture
def stub_runtime_bin(stub_server_binary: Path, tmp_path: Path) -> Path:
    """A runtime directory whose qwen3_asr_server runs the stub server."""
    directory = tmp_path / "bin"
    directory.mkdir()
    server = directory / "qwen3_asr_server"
    server.write_text(f'#!/bin/sh\nexec {shlex.quote(str(stub_server_binary))} "$@"\n')
    server.chmod(0o755)
    return directory


def test_recording_keeps_whole_clips_and_matching_references(tmp_path: Path) -> None:
    """Truncating the last clip would score words that were never sent."""
    write_corpus(tmp_path)
    clips = load_clips(tmp_path, "en")
    result = build_recording(clips, 7.5, 2888, tmp_path / "long.wav")
    reference_by_id = {clip.clip_id: clip.reference for clip in clips}
    assert result.audio_seconds == 8
    assert result.reference == " ".join(
        reference_by_id[clip_id] for clip_id in result.clip_ids
    )
    assert len(set(result.clip_ids[:3])) == 3
    assert all(
        left != right for left, right in zip(result.clip_ids, result.clip_ids[1:])
    )
    with wave.open(str(tmp_path / "long.wav"), "rb") as reader:
        assert reader.getnframes() == 8 * 16000
        expected = b"".join(
            bytes([int(clip_id[:4]), 0]) * 16000 for clip_id in result.clip_ids
        )
        assert reader.readframes(reader.getnframes()) == expected
    repeated = build_recording(clips, 7.5, 2888, tmp_path / "again.wav")
    assert repeated == result
    assert (
        result.wav_sha256
        == hashlib.sha256((tmp_path / "long.wav").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("clip_frames", "chunk_seconds", "expected"),
    [
        ([16000, 16000], 2, [32000]),
        ([16000, 16000, 16000], 2, [32000, 16000]),
        ([48000, 16000], 2, [48000, 16000]),
    ],
)
def test_chunks_keep_whole_clips_within_the_limit(
    clip_frames: list[int], chunk_seconds: float, expected: list[int]
) -> None:
    """A clip cut in two would score words on audio split across requests."""
    assert chunk_frame_counts(clip_frames, chunk_seconds) == expected


@pytest.mark.parametrize(
    ("audio_seconds", "budget"), [(1, 256), (60, 1744), (1800, 8192)]
)
def test_token_budget_matches_voxt(audio_seconds: float, budget: int) -> None:
    """The values come from Voxt's ASRTuning.tokenBudget."""
    assert voxt_token_budget(audio_seconds) == budget


def test_chunked_http_sends_each_chunk_with_its_own_budget(
    asr_server: SupervisedServer, tmp_path: Path
) -> None:
    """The stub answers with each request's sample count and token budget."""
    recording = tmp_path / "long.wav"
    recording.write_bytes(silent_wav(176000))
    result = transcribe_chunks(
        asr_server.port, recording, [16000, 160000], "en", REQUEST_TIMEOUT_SECONDS
    )
    assert result.text == "16000 160000"
    assert [
        metadata.generated_token_count for metadata in result.generation_metadata
    ] == [256, 344]


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
    assert result.errors == errors
    assert result.reference_tokens == reference_tokens
    assert result.error_rate == errors / reference_tokens


def test_realtime_silent_audio_does_not_time_out_before_input_ends(
    asr_server: SupervisedServer, tmp_path: Path
) -> None:
    """A session with no preview events must remain open while audio is sent."""
    recording = tmp_path / "silence.wav"
    recording.write_bytes(silent_wav(32000))
    result = stream_recording(
        asr_server.port, recording, "en", 1, 1, tmp_path / "events.jsonl"
    )
    assert result.text == "32000"
    assert result.preview_event_count == 0


def test_realtime_times_out_without_a_completed_transcript(tmp_path: Path) -> None:
    recording = tmp_path / "silence.wav"
    recording.write_bytes(silent_wav(1600))

    def handle(socket: ServerConnection) -> None:
        socket.recv()
        socket.send(json.dumps({"type": "transcription_session.updated"}))
        try:
            for _ in socket:
                pass
        except ConnectionClosed:
            pass

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with pytest.raises(TimeoutError, match="No completed transcript"):
                stream_recording(
                    server.socket.getsockname()[1],
                    recording,
                    "en",
                    1,
                    0.1,
                    tmp_path / "events.jsonl",
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_realtime_pacing_starts_after_session_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow handshake must not cause the audio to be sent in a burst."""
    recording = tmp_path / "silence.wav"
    recording.write_bytes(silent_wav(16000))

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


def test_resource_guard_stops_its_child(tmp_path: Path) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "input()"], stdin=subprocess.PIPE
    ) as process:
        with MemorySampler(process, tmp_path / "memory.jsonl", 1) as sampler:
            assert process.wait(timeout=5) != 0
    assert sampler.exceeded
    assert sampler.samples[0].footprint_bytes > 0


def test_peak_footprint_includes_a_spike_freed_before_the_first_sample(
    tmp_path: Path,
) -> None:
    """A short allocation between samples must still reach the peak and the guard."""
    spike_bytes = 256 * 1024**2
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"spike = bytearray(b'x') * {spike_bytes}; del spike; print('freed', flush=True); input()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        process.stdout.readline()
        sampler = MemorySampler(process, tmp_path / "memory.jsonl", spike_bytes // 2)
        sampler.sample()
    assert sampler.exceeded
    assert sampler.samples[0].peak_footprint_bytes >= spike_bytes
    assert sampler.samples[0].footprint_bytes < spike_bytes // 2


@pytest.mark.parametrize(
    "reply",
    [
        b":\n\n" + DONE_EVENT,
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
                assert result.text == "actual transcript"
            else:
                with pytest.raises(RuntimeError):
                    transcribe_http(server.server_port, b"audio", "en", 2)
            assert b'name="audio_layout"\r\n\r\nvoxt_swift' in submitted[0]
            assert b'name="stop_on_token_loop"\r\n\r\ntrue' in submitted[0]
        finally:
            thread.join(timeout=5)


@pytest.mark.parametrize(
    ("script", "timeout_seconds", "exit_code", "error"),
    [
        ("input()", 10, 0, None),
        ("input(); raise SystemExit(23)", 10, 23, "during shutdown"),
        ("import time; time.sleep(60)", 0.2, -9, "did not shut down"),
    ],
)
def test_stop_server_reports_how_the_server_stopped(
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    timeout_seconds: float,
    exit_code: int,
    error: str | None,
) -> None:
    monkeypatch.setattr(
        benchmark_long_audio, "SHUTDOWN_TIMEOUT_SECONDS", timeout_seconds
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    stop_error = stop_server(process)
    assert process.returncode == exit_code
    if error is None:
        assert stop_error is None
    else:
        assert error in stop_error


@pytest.mark.parametrize("exit_code", [0, 23])
def test_stop_server_fails_a_server_that_already_exited(exit_code: int) -> None:
    """An exit before shutdown is a failure even when its code is zero."""
    process = subprocess.Popen(
        [sys.executable, "-c", f"raise SystemExit({exit_code})"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    process.wait(timeout=5)
    assert "before shutdown" in stop_server(process)
    assert process.returncode == exit_code


@pytest.mark.parametrize(
    ("flags", "text"),
    [
        (("--mode", "http"), "48000"),
        (("--mode", "http", "--chunk-seconds", "1"), "16000 16000 16000"),
        (("--mode", "realtime", "--speed", "20"), "48000"),
    ],
)
def test_benchmark_measures_a_case_on_the_stub_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_runtime_bin: Path,
    flags: tuple[str, ...],
    text: str,
) -> None:
    exit_code, _, result = run_benchmark(
        monkeypatch, tmp_path, stub_runtime_bin, *flags, *STUB_KIND_FLAGS
    )
    assert exit_code == 0
    assert result["status"] == "measured"
    assert result["error"] == ""
    assert result["transcription"]["text"] == text
    assert result["score"]["reference_tokens"] == 3
    assert result["recovery_matches_warmup"] is True
    assert result["server_exit_code"] == 0
    assert result["peak_footprint_bytes"] >= result["warmup_peak_footprint_bytes"] > 0


def test_a_failed_shutdown_fails_a_measured_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_runtime_bin: Path
) -> None:
    """An exit after recovery must not be reported as a successful measurement."""

    def failing_stop_server(process: subprocess.Popen[bytes]) -> str:
        stop_server(process)
        return "Server exited before shutdown with code 23"

    monkeypatch.setattr(benchmark_long_audio, "stop_server", failing_stop_server)
    exit_code, _, result = run_benchmark(
        monkeypatch, tmp_path, stub_runtime_bin, "--mode", "http", *STUB_KIND_FLAGS
    )
    assert exit_code == 1
    assert result["status"] == "failed"
    assert result["recovery_matches_warmup"] is True
    assert "before shutdown" in result["error"]


def test_a_failed_model_load_reports_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime_bin: Path
) -> None:
    """The reason arrives on stdout, so only the case's error can carry it."""
    exit_code, _, result = run_benchmark(
        monkeypatch, tmp_path, runtime_bin, "--mode", "http"
    )
    assert exit_code == 1
    assert result["status"] == "failed"
    assert "model load failed: " in result["error"]


def test_flags_after_the_separator_reach_the_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_runtime_bin: Path
) -> None:
    exit_code, output, _ = run_benchmark(
        monkeypatch,
        tmp_path,
        stub_runtime_bin,
        "--mode",
        "http",
        *STUB_KIND_FLAGS,
        "--no-such-flag",
        "value",
    )
    assert exit_code == 1
    assert (
        "unknown argument --no-such-flag"
        in (output / "en-0.04m.server.log").read_text()
    )
