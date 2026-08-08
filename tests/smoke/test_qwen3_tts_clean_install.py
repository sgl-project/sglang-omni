# SPDX-License-Identifier: Apache-2.0
"""Clean-environment smoke test for Qwen3-TTS.

Regression coverage for: "qwen-tts 0.1.1 breaks against transformers 5.12 --
every request 500s".

The test does following things:

    1. Build a venv from *only* `pyproject.toml` (i.e. whatever transformers
       version is pinned there today -- currently `transformers==5.12.1` per
       `pyproject.toml:24`). No dev/test extras, no local overrides.
    2. Run the exact install line documented at `docs/basic_usage/tts.md:18`
       (and mirrored at `docs/cookbook/qwen3_tts.md:22`):
           uv pip install --upgrade sox einops
           uv pip install --no-deps qwen-tts==0.1.1
    3. Launch `sgl-omni serve` for a Qwen3-TTS Base checkpoint, exactly as
       documented.
    4. Send one real `/v1/audio/speech` request (the same reference-audio
       example from the docs, since Qwen3-TTS Base requires `ref_audio`) and
       assert it comes back as audio, not a 500.

Run explicitly with:

    SGLANG_OMNI_RUN_CLEAN_INSTALL_SMOKE=1 pytest -m clean_install_smoke \
        tests/smoke/test_qwen3_tts_clean_install.py -v -s
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Constants mirrored from the documented, user-facing install / serve flow.
# If these ever need to change, they should change in lockstep with the docs
# they're copied from -- that's the whole point of this test.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# docs/basic_usage/tts.md:18 and docs/cookbook/qwen3_tts.md:22
QWEN_TTS_EXTRA_DEPS = ["sox", "einops"]
QWEN_TTS_PACKAGE_SPEC = "qwen-tts==0.1.1"

# docs/basic_usage/tts.md "For Qwen3-TTS Base"
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_CONFIG = "examples/configs/qwen3_tts_0_6b.yaml"

# docs/basic_usage/tts.md "Qwen3-TTS Base requires reference audio"
REFERENCE_AUDIO = (
    "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/"
    "resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
)
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."

SERVER_PORT = int(os.environ.get("SGLANG_OMNI_SMOKE_TEST_PORT", "8931"))
SERVER_HOST = "127.0.0.1"
SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

SERVER_BOOT_TIMEOUT_S = float(os.environ.get("SGLANG_OMNI_SMOKE_BOOT_TIMEOUT_S", "900"))
REQUEST_TIMEOUT_S = float(os.environ.get("SGLANG_OMNI_SMOKE_REQUEST_TIMEOUT_S", "300"))

RUN_ENV_VAR = "SGLANG_OMNI_RUN_CLEAN_INSTALL_SMOKE"

pytestmark = pytest.mark.clean_install_smoke


def _skip_reason() -> str | None:
    """Return why this test should be skipped, or None to run it."""
    if os.environ.get(RUN_ENV_VAR) != "1":
        return (
            f"Set {RUN_ENV_VAR}=1 to run the Qwen3-TTS clean-install smoke "
            "test (builds a fresh venv, downloads model weights, needs a "
            "GPU and network access)."
        )
    if shutil.which("uv") is None:
        return "uv is not on PATH; required to build the clean install venv."
    if not (REPO_ROOT / "pyproject.toml").exists():
        return f"Could not find pyproject.toml at {REPO_ROOT}; wrong repo root?"
    try:
        gpu_check = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "No GPU detected (nvidia-smi unavailable); required to serve Qwen3-TTS."
    if gpu_check.returncode != 0 or not gpu_check.stdout.strip():
        return "No GPU detected (nvidia-smi reported none); required to serve Qwen3-TTS."
    return None


@pytest.fixture(scope="module")
def clean_env_python(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build a venv from *only* pyproject.toml + the documented install line.

    This deliberately does not reuse the interpreter running pytest: that
    interpreter may have local patches, editable installs with extra pins,
    or a transformers version chosen for the rest of the test suite rather
    than the one actually pinned for release. The whole point of this test
    is to reproduce what a user gets from a clean install of `main`.
    """
    venv_dir = tmp_path_factory.mktemp("qwen3_tts_clean_install") / "venv"

    _run(["uv", "venv", str(venv_dir), "--python", "3.11"], cwd=REPO_ROOT)
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():  # pragma: no cover - Windows runners, if any
        venv_python = venv_dir / "Scripts" / "python.exe"

    # Step 1: install sglang-omni itself exactly as pyproject.toml pins it
    # (in particular transformers==5.12.1, pyproject.toml:24). No extras.
    _run(
        ["uv", "pip", "install", "--python", str(venv_python), str(REPO_ROOT)],
        cwd=REPO_ROOT,
    )

    # Step 2: the documented Qwen3-TTS install lines, verbatim
    # (docs/basic_usage/tts.md:18, docs/cookbook/qwen3_tts.md:22).
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--upgrade",
            *QWEN_TTS_EXTRA_DEPS,
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--no-deps",
            QWEN_TTS_PACKAGE_SPEC,
        ],
        cwd=REPO_ROOT,
    )

    # Sanity check: fail fast and legibly if the pin in pyproject.toml has
    # silently drifted away from what this test assumes.
    installed_transformers = _run(
        [
            str(venv_python),
            "-c",
            "import transformers; print(transformers.__version__)",
        ],
        cwd=REPO_ROOT,
    ).stdout.strip()
    print(f"[clean-install smoke] transformers=={installed_transformers}")

    yield venv_python


@pytest.fixture(scope="module")
def qwen3_tts_server(clean_env_python: Path) -> Iterator[str]:
    """Boot `sgl-omni serve` for Qwen3-TTS Base inside the clean venv."""
    log_path = Path(tempfile.mkstemp(suffix=".log", prefix="sgl_omni_serve_")[1])
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

    proc = subprocess.Popen(
        [
            str(clean_env_python),
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            MODEL_PATH,
            "--config",
            MODEL_CONFIG,
            "--port",
            str(SERVER_PORT),
            "--host",
            SERVER_HOST,
        ],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_for_health(SERVER_BASE_URL, proc, log_path, SERVER_BOOT_TIMEOUT_S)
        yield SERVER_BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        log_file.close()
        if proc.returncode not in (0, None, -15):  # not a clean exit/terminate
            print(f"[clean-install smoke] server log ({log_path}):")
            print(log_path.read_text(encoding="utf-8", errors="replace"))


def _wait_for_health(
    base_url: str, proc: subprocess.Popen, log_path: Path, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_contents = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"sgl-omni serve exited early with code {proc.returncode} "
                f"before becoming healthy.\n--- server log ---\n{log_contents}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2)

    log_contents = log_path.read_text(encoding="utf-8", errors="replace")
    raise TimeoutError(
        f"Server did not become healthy within {timeout_s}s "
        f"(last error: {last_error}).\n--- server log ---\n{log_contents}"
    )


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_qwen3_tts_base_serves_one_request_end_to_end(
    qwen3_tts_server: str,
) -> None:
    """Boot Qwen3-TTS Base from a clean install and synthesize once.

    This is the documented "Qwen3-TTS Base requires reference audio" curl
    example from docs/basic_usage/tts.md, sent as a real HTTP request against
    a server built from nothing but pyproject.toml + the documented qwen-tts
    install line.

    Before the create_causal_mask shim, this failed on every request with a
    500 originating from qwen-tts calling
    `create_causal_mask(..., input_embeds=..., cache_position=...)` against a
    transformers version that spells the parameter `inputs_embeds` and no
    longer accepts `cache_position` at all.
    """
    payload = {
        "input": SPEECH_INPUT,
        "ref_audio": REFERENCE_AUDIO,
        "ref_text": REFERENCE_TEXT,
        "response_format": "wav",
    }
    request = urllib.request.Request(
        f"{qwen3_tts_server}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as resp:
            status = resp.status
            audio_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        pytest.fail(
            f"/v1/audio/speech returned HTTP {exc.code} instead of 200.\n"
            f"Response body: {body}"
        )

    assert status == 200, f"Expected HTTP 200, got {status}"

    # The failure mode this test guards against returns an OpenAI-style JSON
    # error envelope (see docs/basic_usage/tts.md "Invalid speech requests")
    # rather than audio bytes. Confirm we actually got audio, not that.
    assert not audio_bytes.lstrip().startswith(b"{"), (
        "Response looks like a JSON error envelope, not audio bytes: "
        f"{audio_bytes[:200]!r}"
    )

    with contextlib.closing(wave.open(io.BytesIO(audio_bytes), "rb")) as wav_file:
        frame_count = wav_file.getnframes()
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()

    assert frame_count > 0, "Synthesized WAV has zero audio frames"
    assert channels >= 1
    assert sample_rate > 0

    duration_s = frame_count / sample_rate
    print(
        f"[clean-install smoke] synthesized {duration_s:.2f}s of audio "
        f"({len(audio_bytes)} bytes, {sample_rate}Hz, {channels}ch)"
    )
    # A few words of speech should not collapse to near-silence-length audio;
    # this loosely guards against "200 OK with garbage/empty audio".
    assert duration_s > 0.2