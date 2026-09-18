# SPDX-License-Identifier: Apache-2.0
"""Integration tests for thinker length validation and finish_reason propagation.

Starts the selected Omni CI server with a short context and verifies that:
1. overlong prompts return HTTP 400 with the model's length-limit error;
2. prompt + max_tokens overflow returns HTTP 400 with that length-limit error;
3. decode hitting max_tokens returns HTTP 200 with finish_reason="length".
"""

from __future__ import annotations

import re
import sys

import pytest
import requests

from tests.test_model.omni_ci_config import OmniCiModelPreset
from tests.test_model.omni_router_utils import ManagedRouterHandle
from tests.utils import disable_proxy

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


def _assert_minicpmo_length_error(
    detail: str, *, max_new_tokens: int, overlong_prompt: bool
) -> None:
    match = re.search(
        r"Request requires more tokens than the thinker KV cache can hold "
        r"\(input_tokens=(\d+), max_new_tokens=(\d+), "
        r"required_tokens=(\d+), kv_capacity=(\d+)\)\.",
        detail,
    )
    assert match, detail
    input_tokens, requested_tokens, required_tokens, capacity = map(int, match.groups())
    assert requested_tokens == max_new_tokens, detail
    # The short-context fixture caps requests at context_length - 1.
    assert capacity == 127, detail
    assert required_tokens == input_tokens + requested_tokens, detail
    assert required_tokens > capacity, detail
    if overlong_prompt:
        assert input_tokens > capacity, detail
    else:
        assert 0 < input_tokens < capacity, detail


def test_overlong_prompt_returns_400(
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ManagedRouterHandle,
) -> None:
    resp = _post_chat(
        omni_ci_server.port,
        {
            "model": omni_ci_model.name,
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
    if omni_ci_model.name == "minicpmo":
        _assert_minicpmo_length_error(
            body["detail"], max_new_tokens=16, overlong_prompt=True
        )
    else:
        assert "The input (" in body["detail"]
        assert "is longer than the model's context length" in body["detail"]


def test_total_token_overflow_returns_400(
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ManagedRouterHandle,
) -> None:
    resp = _post_chat(
        omni_ci_server.port,
        {
            "model": omni_ci_model.name,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 200,
            "stream": False,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    if omni_ci_model.name == "minicpmo":
        _assert_minicpmo_length_error(
            body["detail"], max_new_tokens=200, overlong_prompt=False
        )
    else:
        assert (
            "Requested token count exceeds the model's maximum context length"
            in body["detail"]
        )


def test_length_finish_reason_is_preserved(
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ManagedRouterHandle,
) -> None:
    resp = _post_chat(
        omni_ci_server.port,
        {
            "model": omni_ci_model.name,
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
