# SPDX-License-Identifier: Apache-2.0
"""Request-seeded sampling through the MiniCPM-o Thinker DSpark HTTP path."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.test_model.omni_ci_config import OMNI_CI_PRESETS
from tests.utils import server_log_file, start_server_from_cmd, stop_server

MODEL = OMNI_CI_PRESETS["minicpmo"]
DRAFT_MODEL = "Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm"


@pytest.fixture(scope="module")
def dspark_server_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = find_available_port()
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        os.environ.get("SGLANG_OMNI_TEST_MINICPMO_MODEL", MODEL.model_path),
        "--model-name",
        MODEL.name,
        "--text-only",
        "--thinker.engine.speculative_algorithm",
        "DSPARK",
        "--thinker.engine.speculative_draft_model_path",
        os.environ.get("SGLANG_OMNI_TEST_MINICPMO_DRAFT_MODEL", DRAFT_MODEL),
        "--thinker.factory.max_seq_len",
        "2048",
        "--thinker.engine.max_running_requests",
        "4",
        "--thinker.engine.max_total_tokens",
        "4096",
        # Keep prefill identical on the first request and its repeat.
        "--thinker.engine.disable_radix_cache",
        "true",
    ]
    proc = start_server_from_cmd(
        cmd, server_log_file(tmp_path_factory, "minicpmo_dspark_seed"), port
    )
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        stop_server(proc)


@pytest.mark.benchmark
def test_seeded_repeat_after_unrelated_request(dspark_server_url: str) -> None:
    prompt = "Write a two-sentence story about a robot discovering a garden."
    with requests.Session() as client:
        client.trust_env = False

        def generate(prompt: str, seed: int | None = None) -> str:
            response = client.post(
                dspark_server_url,
                json={
                    "model": MODEL.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 64,
                    "return_spec_tokens_details": True,
                    **({"seed": seed} if seed is not None else {}),
                },
                timeout=120,
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["sglext"]["spec_tokens_details"]["spec_verify_ct"] > 0, body
            text = body["choices"][0]["message"]["content"]
            assert isinstance(text, str) and text.strip(), body
            return text

        first = generate(prompt, seed=42)
        generate("Name three planets in our solar system.")
        assert generate(prompt, seed=42) == first
