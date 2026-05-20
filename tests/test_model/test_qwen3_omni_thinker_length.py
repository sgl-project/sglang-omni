# SPDX-License-Identifier: Apache-2.0
"""Integration tests for thinker length validation and finish_reason propagation.

Starts router-backed colocated Qwen3-Omni workers and verifies that:
1. overlong prompts return HTTP 400 with SGLang-aligned wording;
2. prompt + max_tokens overflow returns HTTP 400 with SGLang-aligned wording;
3. decode hitting max_tokens returns HTTP 200 with finish_reason="length".
"""

from __future__ import annotations

import sys

import pytest
import requests

from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.utils import disable_proxy

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"
THINKER_MAX_SEQ_LEN = 128
ROUTER_WAIT_TIMEOUT = 180
REQUEST_TIMEOUT = 120

pytestmark = pytest.mark.benchmark


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
def router_server(tmp_path_factory: pytest.TempPathFactory):
    worker_extra_args = (
        "--config examples/configs/qwen3_omni_colocated.yaml "
        "--colocate "
        f"--stages.0.factory-args.thinker-max-seq-len {THINKER_MAX_SEQ_LEN} "
        f"--stages.4.factory-args.thinker-max-seq-len {THINKER_MAX_SEQ_LEN}"
    )
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        worker_extra_args=worker_extra_args,
        wait_timeout=ROUTER_WAIT_TIMEOUT,
        log_prefix="thinker_length_router_logs",
    ) as router:
        yield router


def test_overlong_prompt_returns_400(router_server: ManagedRouterHandle) -> None:
    resp = _post_chat(
        router_server.port,
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


def test_total_token_overflow_returns_400(router_server: ManagedRouterHandle) -> None:
    resp = _post_chat(
        router_server.port,
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


def test_length_finish_reason_is_preserved(router_server: ManagedRouterHandle) -> None:
    resp = _post_chat(
        router_server.port,
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 20, separated by commas.",
                }
            ],
            "max_tokens": 1,
            "stream": False,
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "length"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
