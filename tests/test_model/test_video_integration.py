# SPDX-License-Identifier: Apache-2.0
"""Backend integration test: video + text -> thinker -> correct text output.

Starts the sglang-omni server, sends a video with a text prompt, and checks
that the response is semantically correct and the server remains stable.

Usage:
    pytest tests/test_video_integration.py -s -x
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.test_model.helpers import disable_proxy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SERVER_PORT = 18899
API_BASE = f"http://localhost:{SERVER_PORT}"
VIDEO_PATH = Path(__file__).parent.parent.joinpath("data/draw.mp4")

# Keywords that indicate the model understood the video content.
# The video shows a transit hub (airport/train station) with a gate number "12",
# a "UCI Health" advertisement, seating areas, and large arched architecture.
EXPECTED_KEYWORDS = [
    "stylus",
    "guitar",
]

STARTUP_TIMEOUT = 600  # seconds
REQUEST_TIMEOUT = 300  # seconds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def server_process():
    """Start the sglang-omni backend server and wait until healthy."""
    assert os.path.isfile(VIDEO_PATH), f"Test video not found: {VIDEO_PATH}"

    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--relay-backend",
        "nixl",
        "--port",
        str(SERVER_PORT),
    ]

    t_start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    # Wait for health endpoint
    healthy = False
    for _ in range(STARTUP_TIMEOUT):
        if proc.poll() is not None:
            # Server exited early — dump output for debugging
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"Server exited with code {proc.returncode}.\n{out}")
        try:
            with disable_proxy():
                resp = requests.get(f"{API_BASE}/health", timeout=2)
            if resp.status_code == 200 and "healthy" in resp.text:
                healthy = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)

    startup_time = time.monotonic() - t_start

    if not healthy:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
        out = proc.stdout.read() if proc.stdout else ""
        pytest.fail(f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n{out}")

    print(f"\n[PERF] Server startup time: {startup_time:.1f}s")

    yield proc, startup_time

    # Teardown
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_two_round_conversation(server_process):
    """Two-round video conversation: location question then school question.

    Round 1: video + "Where am I right now?" -> expects transit hub keywords
    Round 2: follow-up "Is there a specific school shown in the video?" -> expects "UCI"

    Also measures per-round latency to check whether the second round benefits
    from KV cache reuse (it should be significantly faster if prefix caching works).
    """
    proc, startup_time = server_process
    video_abs = os.path.abspath(VIDEO_PATH)

    # ------------------------------------------------------------------
    # Round 1: location question
    # ------------------------------------------------------------------
    round1_messages = [
        {"role": "user", "content": "What is the girl holding in her hand?"},
    ]
    payload_r1 = {
        "model": "qwen3-omni",
        "messages": round1_messages,
        "videos": [video_abs],
        "modalities": ["text"],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }

    t1_start = time.monotonic()
    with disable_proxy():
        resp_r1 = requests.post(
            f"{API_BASE}/v1/chat/completions",
            json=payload_r1,
            timeout=REQUEST_TIMEOUT,
        )
    t1_elapsed = time.monotonic() - t1_start

    assert (
        resp_r1.status_code == 200
    ), f"Round 1 failed with status {resp_r1.status_code}: {resp_r1.text}"

    body_r1 = resp_r1.json()
    content_r1 = body_r1["choices"][0]["message"].get("content", "")

    print(f"\n{'='*60}")
    print(f"[PERF] Server startup : {startup_time:.1f}s")
    print(f"[PERF] Round 1 latency: {t1_elapsed:.1f}s")
    print(f"[R1 RESPONSE] {content_r1}")

    content_r1_lower = content_r1.lower()
    assert "stylus" in content_r1_lower, (
        f"Round 1: response does not mention expected keyword 'stylus'.\n"
        f"Response: {content_r1}"
    )

    # ------------------------------------------------------------------
    # Round 2: follow-up about the school (multi-turn with context)
    # ------------------------------------------------------------------
    round2_messages = [
        {"role": "user", "content": "What is the girl holding in her hand?"},
        {"role": "assistant", "content": content_r1},
        {"role": "user", "content": "What is the girl drawing?"},
    ]
    payload_r2 = {
        "model": "qwen3-omni",
        "messages": round2_messages,
        "videos": [video_abs],
        "modalities": ["text"],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }

    t2_start = time.monotonic()
    with disable_proxy():
        resp_r2 = requests.post(
            f"{API_BASE}/v1/chat/completions",
            json=payload_r2,
            timeout=REQUEST_TIMEOUT,
        )
    t2_elapsed = time.monotonic() - t2_start

    assert (
        resp_r2.status_code == 200
    ), f"Round 2 failed with status {resp_r2.status_code}: {resp_r2.text}"

    body_r2 = resp_r2.json()
    content_r2 = body_r2["choices"][0]["message"].get("content", "")

    print(f"[PERF] Round 2 latency: {t2_elapsed:.1f}s")
    print(f"[R2 RESPONSE] {content_r2}")
    print(f"{'='*60}")

    # Round 2 should mention UCI
    content_r2_lower = content_r2.lower()
    assert "guitar" in content_r2_lower, (
        f"Round 2: response does not mention 'guitar'.\n" f"Response: {content_r2}"
    )

    # ------------------------------------------------------------------
    # Performance comparison
    # ------------------------------------------------------------------
    speedup = t1_elapsed / t2_elapsed if t2_elapsed > 0 else float("inf")
    print(f"\n[PERF] Round 1 / Round 2 speedup: {speedup:.2f}x")
    if t2_elapsed >= t1_elapsed * 0.8:
        print(
            "[WARN] Round 2 is NOT significantly faster than Round 1. "
            "KV cache / prefix caching may not be effective."
        )

    # Server should still be healthy
    with disable_proxy():
        health = requests.get(f"{API_BASE}/health", timeout=5)
    assert health.status_code == 200, "Server unhealthy after round 2"


def test_server_stability_after_request(server_process):
    """Verify the server process is still alive after the conversation."""
    proc, _ = server_process
    assert (
        proc.poll() is None
    ), f"Server process died with return code {proc.returncode}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
