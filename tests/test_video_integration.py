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

import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "QWEN3_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
)
SERVER_PORT = int(os.environ.get("TEST_SERVER_PORT", "18899"))
API_BASE = f"http://localhost:{SERVER_PORT}"
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "test_file.webm")
# Allow both the project root and sglang_omni/ locations
if not os.path.isfile(VIDEO_PATH):
    VIDEO_PATH = os.path.join(
        os.path.dirname(__file__), "..", "sglang_omni", "test_file.webm"
    )

# Keywords that indicate the model understood the airport video
AIRPORT_KEYWORDS = [
    "airport",
    "terminal",
    "flight",
    "gate",
    "departure",
    "arrival",
    "boarding",
    "airplane",
    "plane",
    "aviation",
    "runway",
    "luggage",
    "baggage",
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
        "shm",
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
        pytest.fail(
            f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n{out}"
        )

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


def test_video_text_airport(server_process):
    """Send video + 'Where am I right now?' and expect airport-related answer."""
    proc, startup_time = server_process

    video_abs = os.path.abspath(VIDEO_PATH)
    payload = {
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "Where am I right now?"}],
        "videos": [video_abs],
        "modalities": ["text"],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }

    t_req_start = time.monotonic()
    resp = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    e2e_latency = time.monotonic() - t_req_start

    assert resp.status_code == 200, (
        f"Request failed with status {resp.status_code}: {resp.text}"
    )

    body = resp.json()
    content = body["choices"][0]["message"].get("content", "")
    print(f"\n[PERF] E2E latency: {e2e_latency:.1f}s")
    print(f"[PERF] Server startup: {startup_time:.1f}s")
    print(f"[RESPONSE] {content}")

    # Check the model recognized the airport
    content_lower = content.lower()
    matched = [kw for kw in AIRPORT_KEYWORDS if kw in content_lower]
    assert matched, (
        f"Response does not mention airport-related keywords.\n"
        f"Keywords checked: {AIRPORT_KEYWORDS}\n"
        f"Response: {content}"
    )

    # Verify server is still healthy after the request
    health = requests.get(f"{API_BASE}/health", timeout=5)
    assert health.status_code == 200, "Server unhealthy after request"


def test_server_stability_after_request(server_process):
    """Verify the server process is still alive after the video request."""
    proc, _ = server_process
    assert proc.poll() is None, (
        f"Server process died with return code {proc.returncode}"
    )
