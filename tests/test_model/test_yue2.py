# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint acoustic and HTTP serving checks for one CUDA device."""

import concurrent.futures
import io
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.models.yue2.acoustic import synthesize_latents
from sglang_omni.models.yue2.request import CODEC_OFFSET
from sglang_omni.models.yue2.runtime import Yue2Runtime

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not os.environ.get("YUE2_MODEL_PATH"),
        reason="Set YUE2_MODEL_PATH to run real-checkpoint tests",
    ),
]

VAE_PATH = os.environ.get("YUE2_VAE_PATH", "m-a-p/YuE2-Vae")
LYRICS = "[Verse]\nMorning light is shining through"
STYLE = "Gentle acoustic pop, warm female vocals"
TOKEN_BUDGET = 16


@pytest.fixture(scope="module", params=[False, True, "fused"])
def runtime(request: pytest.FixtureRequest) -> Yue2Runtime:
    return Yue2Runtime.from_pretrained(
        os.environ["YUE2_MODEL_PATH"],
        VAE_PATH,
        torch.device("cuda:0"),
        vae_core_frames=1024,
        vae_halo_frames=16,
        nar_cuda_graph=bool(request.param),
        fuse_projections=bool(request.param),
        fused_kernels=request.param == "fused",
    )


def test_acoustic_graph_preserves_solver_and_abort_recovery(
    runtime: Yue2Runtime,
) -> None:
    prefix, codecs = [1, 2, 3], list(range(64))
    expected = synthesize_latents(runtime.model, prefix, codecs, 42, 7, lambda: False)
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 5

    with pytest.raises(InterruptedError):
        synthesize_latents(
            runtime.model, prefix, codecs, 42, 7, cancelled, cuda_graph=True
        )
    actual = synthesize_latents(
        runtime.model, prefix, codecs, 42, 7, lambda: False, cuda_graph=True
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_acoustic_render_returns_stereo_audio(runtime: Yue2Runtime) -> None:
    tokens = [CODEC_OFFSET + token for token in range(TOKEN_BUDGET)]
    audio = runtime.render([1, 2, 3], tokens, 42, 2, lambda: False)
    assert audio.shape == (2, TOKEN_BUDGET * 1920 - 64)
    assert audio.dtype == torch.float32 and torch.isfinite(audio).all()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            os.environ["YUE2_MODEL_PATH"],
            "--yue2_acoustic.factory.vae_path",
            VAE_PATH,
            "--port",
            str(port),
        ],
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            assert process.poll() is None, "YuE2 server exited during startup"
            try:
                if httpx.get(f"{base}/health", timeout=3).status_code == 200:
                    break
                else:
                    pass
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("YuE2 server did not become ready")
        yield base
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=120)


def speech_request(
    seed: int, *, token_budget: int = TOKEN_BUDGET, **options: object
) -> dict[str, object]:
    return {
        "model": os.environ["YUE2_MODEL_PATH"],
        "input": LYRICS,
        "instructions": STYLE,
        "seed": seed,
        "response_format": "wav",
        "max_new_tokens": token_budget,
        "stage_params": {"synthesis": {"abc_max_tokens": 8, "ode_steps": 2, **options}},
    }


def synthesize(base: str, body: dict[str, object]) -> httpx.Response:
    return httpx.post(f"{base}/v1/audio/speech", json=body, timeout=600)


def test_concurrent_modes_return_truncated_stereo_songs(server: str) -> None:
    bodies = [
        speech_request(42, cot="full"),
        speech_request(43, cot="melody", cfg_scale=1.5),
        speech_request(44, cot="off"),
        speech_request(
            45, cot="melody", abc="X:1\nT:Morning\nM:4/4\nL:1/4\nK:C\nC D E F |"
        ),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        responses = list(pool.map(lambda body: synthesize(server, body), bodies))
    for response in responses:
        assert response.status_code == 200, response.text
        audio, rate = sf.read(io.BytesIO(response.content), always_2d=True)
        assert rate == 48000 and audio.shape == (TOKEN_BUDGET * 1920 - 64, 2)
        assert np.isfinite(audio).all()
        assert response.headers["X-Finish-Reason"] == "length"


def test_seed_reproduces_and_invalid_requests_fail_cleanly(server: str) -> None:
    body = speech_request(42, cot="melody", cfg_scale=1.5)
    first = synthesize(server, body)
    second = synthesize(server, body)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    missing_style = dict(body)
    missing_style.pop("instructions")
    assert synthesize(server, missing_style).status_code == 400
    assert synthesize(server, speech_request(42, token_budget=25000)).status_code == 400
    with pytest.raises(httpx.TimeoutException):
        httpx.post(
            f"{server}/v1/audio/speech",
            json=speech_request(42, token_budget=2000),
            timeout=0.2,
        )
    assert synthesize(server, body).content == first.content
