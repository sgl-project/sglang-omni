# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ming-Omni TP parity test.

This launches real GPU servers, so it is skipped by default for local machines
without the Ming checkpoint and CUDA resources.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from tests.utils import (
    TPParityServerProcess,
    extract_text,
    post_json,
    stop_server_process,
    wait_health,
)

MODEL_NAME = os.environ.get("MING_OMNI_MODEL_NAME", "ming-omni")
MODEL_PATH = os.environ.get("MING_OMNI_MODEL_PATH", "inclusionAI/Ming-flash-omni-2.0")
RUN_FLAG = os.environ.get("RUN_MING_TP_PARITY", "0") == "1"


@dataclass(frozen=True)
class MingServerProcess:
    process: subprocess.Popen[bytes]
    log_handle: BinaryIO


def _start_server(
    port: int,
    tp_size: int,
    cuda_visible_devices: str,
    gpu_talker: int,
    tmp_path: Path,
) -> TPParityServerProcess:
    cwd = Path.cwd()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    pythonpath = str(cwd)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join((pythonpath, env["PYTHONPATH"]))
    env["PYTHONPATH"] = pythonpath

    log_path = tmp_path / f"ming_tp{tp_size}.log"
    log_handle = log_path.open("wb")
    cmd = [
        sys.executable,
        "-u",
        str(cwd / "examples" / "run_ming_omni_speech_server.py"),
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--tp-size",
        str(tp_size),
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        str(gpu_talker),
        "--mem-fraction-static",
        "0.80",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    return TPParityServerProcess(process=process, log_handle=log_handle)


def _chat_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text"],
        "max_tokens": 16,
        "temperature": 0,
        "top_p": 1,
    }


def _collect_outputs(port: int, prompts: list[str]) -> list[str]:
    return [extract_text(post_json(port, _chat_payload(prompt))) for prompt in prompts]


@pytest.mark.benchmark
@pytest.mark.skipif(not RUN_FLAG, reason="set RUN_MING_TP_PARITY=1 to run")
def test_ming_tp1_and_tp4_deterministic_text_match(tmp_path: Path) -> None:
    cuda_devices = os.environ.get("MING_TP_PARITY_CUDA_VISIBLE_DEVICES", "0,1,2,3,4")
    tp_size = int(os.environ.get("MING_TP_PARITY_TP_SIZE", "4"))
    visible_device_count = len(
        [device for device in cuda_devices.split(",") if device.strip()]
    )
    assert tp_size >= 2
    assert visible_device_count >= tp_size + 1

    prompts = [
        "What is the capital of Japan? Answer with exactly one word.",
        "What is 17+25? Answer with exactly one number.",
    ]

    tp1_process = _start_server(
        port=18101,
        tp_size=1,
        cuda_visible_devices=cuda_devices,
        gpu_talker=tp_size,
        tmp_path=tmp_path,
    )
    try:
        wait_health(18101, tp1_process)
        tp1_outputs = _collect_outputs(18101, prompts)
    finally:
        stop_server_process(tp1_process)

    tpn_process = _start_server(
        port=18104,
        tp_size=tp_size,
        cuda_visible_devices=cuda_devices,
        gpu_talker=tp_size,
        tmp_path=tmp_path,
    )
    try:
        wait_health(18104, tpn_process)
        tpn_outputs = _collect_outputs(18104, prompts)
    finally:
        stop_server_process(tpn_process)

    assert tpn_outputs == tp1_outputs
