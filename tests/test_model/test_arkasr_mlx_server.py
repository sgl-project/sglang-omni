# SPDX-License-Identifier: Apache-2.0
"""Opt-in ARK-ASR MLX server integration test."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.utils import server_log_file, start_server_from_cmd, stop_server

CHECKPOINT_ENV = "ARKASR_MLX_SERVER_CHECKPOINT"
AUDIO_ENV = "ARKASR_MLX_SERVER_AUDIO"
QUANTIZATION_ENV = "ARKASR_MLX_SERVER_QUANTIZATION"
EXPECTED_TEXT_ENV = "ARKASR_MLX_SERVER_EXPECTED_TEXT"
DEFAULT_AUDIO = Path(__file__).parent.parent / "data" / "query_to_cars.wav"
DEFAULT_EXPECTED_TEXT = "how many cars are there in the picture"
STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 180


def _normalize_transcript(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


@pytest.mark.accelerator
def test_arkasr_mlx_server_transcribes_audio(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    if not checkpoint:
        pytest.skip(f"Set {CHECKPOINT_ENV} to run the MLX server integration test")

    audio_override = os.environ.get(AUDIO_ENV)
    audio_path = Path(audio_override or DEFAULT_AUDIO)
    if not audio_path.is_file():
        pytest.fail(f"ARK-ASR integration audio does not exist: {audio_path}")
    expected_text = os.environ.get(EXPECTED_TEXT_ENV)
    if audio_override and expected_text is None:
        pytest.fail(
            f"Set {EXPECTED_TEXT_ENV} when overriding the integration audio"
        )
    expected_text = expected_text or DEFAULT_EXPECTED_TEXT

    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "arkasr_mlx_server_logs")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        checkpoint,
        "--model-name",
        "ark-asr-3b",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    quantization = os.environ.get(QUANTIZATION_ENV)
    if quantization:
        cmd.extend(["--asr.engine.quantization", quantization])
    proc = start_server_from_cmd(
        cmd,
        log_file,
        port,
        timeout=STARTUP_TIMEOUT,
        env={"SGLANG_USE_MLX": "1"},
    )
    try:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                files={
                    "file": (
                        audio_path.name,
                        audio_file,
                        "audio/wav",
                    )
                },
                data={
                    "model": "ark-asr-3b",
                    "response_format": "json",
                    "temperature": "0",
                },
                timeout=REQUEST_TIMEOUT,
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert isinstance(payload.get("text"), str)
        assert _normalize_transcript(payload["text"]) == _normalize_transcript(
            expected_text
        )
    finally:
        stop_server(proc)
