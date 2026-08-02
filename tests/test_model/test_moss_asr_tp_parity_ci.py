# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP parity test for MOSS-TD.

This test launches real GPU servers, so it is skipped by default for local machines
without the MOSS-TD checkpoint and CUDA resources.
"""

from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

from tests.utils import (
    TPParityServerProcess,
    extract_text,
    get_available_port,
    post_json,
    stop_server_process,
    wait_health,
)

MODEL_NAME = os.environ.get("MOSS_TD_MODEL_NAME", "moss-transcribe-diarize")
MODEL_PATH = os.environ.get("MOSS_TD_MODEL_PATH", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
RUN_FLAG = os.environ.get("RUN_MOSS_TD_TP_PARITY", "1") == "1"


def _post_audio(port: int, audio_path: Path, timeout: float = 180.0) -> Any:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"file://{audio_path}"},
                    }
                ],
            }
        ],
    }
    return post_json(port, payload, timeout=timeout)


def start_server(
    port: int, tp_size: int, cuda_visible_devices: str, tmp_path: Path
) -> TPParityServerProcess:
    cwd = Path.cwd()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    pythonpath = str(cwd)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join((pythonpath, env["PYTHONPATH"]))
    env["PYTHONPATH"] = pythonpath

    log_path = tmp_path / f"moss_td_tp{tp_size}.log"
    log_handle = log_path.open("wb")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-name",
        MODEL_NAME,
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--stages.0.tp_size",
        str(tp_size),
    ]

    print("starting local server with:", " ".join(cmd))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise

    return TPParityServerProcess(process=process, log_handle=log_handle)


def _create_dummy_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00" * 16000)


@pytest.mark.benchmark
@pytest.mark.skipif(not RUN_FLAG, reason="Skipping MOSS-TD TP parity test")
def test_moss_asr_tp1_and_tp2_deterministic_text_match(tmp_path: Path) -> None:
    cuda_devices = os.environ.get("MOSS_TP_PARITY_CUDA_VISIBLE_DEVICES", "0,1")
    tp_size = int(os.environ.get("MOSS_TP_PARITY_TP_SIZE", "2"))
    visible_device_count = len(
        [device for device in cuda_devices.split(",") if device.strip()]
    )
    assert tp_size >= 2
    assert (
        visible_device_count >= tp_size
    ), f"Not enough visible devices for TP size {tp_size}"

    audio_file = tmp_path / "dummy.wav"
    _create_dummy_wav(audio_file)

    port1 = get_available_port()
    tp1_process = start_server(
        port=port1,
        tp_size=1,
        cuda_visible_devices=cuda_devices,
        tmp_path=tmp_path,
    )
    try:
        wait_health(port1, tp1_process)
        tp1_text = extract_text(_post_audio(port1, audio_file))
    finally:
        stop_server_process(tp1_process)

    port2 = get_available_port()
    tpx_process = start_server(
        port=port2,
        tp_size=tp_size,
        cuda_visible_devices=cuda_devices,
        tmp_path=tmp_path,
    )
    try:
        wait_health(port2, tpx_process)
        tpx_text = extract_text(_post_audio(port2, audio_file))
    finally:
        stop_server_process(tpx_process)

    assert (
        tp1_text == tpx_text
    ), f"TP=1 text '{tp1_text}' does not match TP={tp_size} text '{tpx_text}'"
