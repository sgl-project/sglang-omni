# SPDX-License-Identifier: Apache-2.0
"""Native-model regression checks for the long-audio benchmark."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from sglang_omni_mlx.native.ci import benchmark_long_audio
from sglang_omni_mlx.native.ci.long_audio_client import stream_recording


@pytest.mark.skipif(
    not os.environ.get("NATIVE_RUNTIME_BIN") or not os.environ.get("CI_DATA_ROOT"),
    reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT for the native model",
)
def test_native_realtime_accepts_silent_audio_without_preview(tmp_path: Path) -> None:
    """The native server need not emit previews while silent audio arrives."""
    recording = tmp_path / "silence.wav"
    with wave.open(str(recording), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 64000)
    server = subprocess.Popen(
        [
            str(Path(os.environ["NATIVE_RUNTIME_BIN"]) / "qwen3_asr_server"),
            "--supervised",
            "--model-directory",
            str(
                Path(os.environ["CI_DATA_ROOT"])
                / "models"
                / "mlx-community_Qwen3-ASR-0.6B-4bit"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(server.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=60):
                raise TimeoutError("Native server startup deadline exceeded")
            else:
                ready = json.loads(server.stdout.readline())
        assert ready["event"] == "ready"
        result = stream_recording(
            ready["port"], recording, "en", 1, 1, tmp_path / "events.jsonl"
        )
        assert result["text"] == ""
        assert result["final_segments"] == 1
    finally:
        if server.poll() is None:
            try:
                server.stdin.write('{"command":"shutdown"}\n')
                server.stdin.flush()
                server.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                server.kill()
                server.wait(timeout=10)
        else:
            pass
        server.stdin.close()
        server.stdout.close()


@pytest.mark.skipif(
    not os.environ.get("NATIVE_RUNTIME_BIN") or not os.environ.get("CI_DATA_ROOT"),
    reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT for the native model",
)
def test_incomparable_cancellation_is_reported_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results"
    data_root = Path(os.environ["CI_DATA_ROOT"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_long_audio",
            "--server",
            str(Path(os.environ["NATIVE_RUNTIME_BIN"]) / "qwen3_asr_server"),
            "--model-directory",
            str(data_root / "models" / "mlx-community_Qwen3-ASR-0.6B-4bit"),
            "--corpus-directory",
            str(data_root / "corpus" / "v1"),
            "--output",
            str(output),
            "--mode",
            "realtime",
            "--speed",
            "5",
            "--minutes",
            "0.2",
            "--languages",
            "en",
            "--max-footprint-gib",
            "16",
        ],
    )
    monkeypatch.setattr(
        benchmark_long_audio,
        "compare_cancellation_timing",
        lambda heartbeat_to_completion_seconds, heartbeat_to_disconnect_seconds, disconnect_to_idle_seconds: (
            None,
            None,
        ),
    )
    with pytest.raises(SystemExit) as stopped:
        benchmark_long_audio.main()
    assert stopped.value.code == 1
    result = json.loads((output / "en-0.2m.result.json").read_text())
    assert result["status"] == "inconclusive"
    assert result["cancellation"]["idle_before_paired_baseline"] is None
    assert result["recovery_matches_warmup"] is True
