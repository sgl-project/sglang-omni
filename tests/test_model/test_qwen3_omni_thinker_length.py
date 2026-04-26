# SPDX-License-Identifier: Apache-2.0
"""Integration tests for thinker length validation and finish_reason propagation.

Starts a real text-only Qwen3-Omni server and verifies that:
1. overlong prompts return HTTP 400 with SGLang-aligned wording;
2. prompt + max_tokens overflow returns HTTP 400 with SGLang-aligned wording;
3. decode hitting max_tokens returns HTTP 200 with finish_reason="length".
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.utils import disable_proxy, start_server_from_cmd, stop_server

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"
THINKER_MAX_SEQ_LEN = 128
STARTUP_TIMEOUT = 300
REQUEST_TIMEOUT = 120


def _post_chat(
    port: int, payload: dict, timeout: int = REQUEST_TIMEOUT
) -> requests.Response:
    with disable_proxy():
        return requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = tmp_path_factory.mktemp("thinker_length_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--thinker-max-seq-len",
        str(THINKER_MAX_SEQ_LEN),
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
    yield proc
    stop_server(proc)


def test_overlong_prompt_returns_400(server_process: subprocess.Popen) -> None:
    resp = _post_chat(
        server_process.port,
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "a " * 10000,
                }
            ],
            "max_tokens": 16,
            "stream": False,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "The input (" in body["detail"]
    assert "is longer than the model's context length" in body["detail"]


def test_total_token_overflow_returns_400(server_process: subprocess.Popen) -> None:
    resp = _post_chat(
        server_process.port,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 200,
            "stream": False,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert (
        "Requested token count exceeds the model's maximum context length"
        in body["detail"]
    )


def test_length_finish_reason_is_preserved(server_process: subprocess.Popen) -> None:
    resp = _post_chat(
        server_process.port,
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 20, separated by commas.",
                }
            ],
            "max_tokens": 2,
            "stream": False,
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "length"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
