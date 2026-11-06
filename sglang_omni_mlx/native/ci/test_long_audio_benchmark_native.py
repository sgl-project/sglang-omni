# SPDX-License-Identifier: Apache-2.0
"""Native-model regression checks for the long-audio benchmark."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import wave
from pathlib import Path

import pytest

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
