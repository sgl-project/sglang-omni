# SPDX-License-Identifier: Apache-2.0
"""Exercise the persistent worker and PCM transport across two interpreter environments."""

import os
import select
import signal
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.models.qwen3_omni_d2.runtime import D2Runtime


@pytest.fixture
def runtime_python(tmp_path: Path) -> str:
    venv.EnvBuilder(with_pip=False).create(tmp_path)
    interpreter = tmp_path / "bin" / "python"
    site_packages = subprocess.check_output(
        [interpreter, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    inference = Path(site_packages) / "d2_qwen" / "inference"
    inference.mkdir(parents=True)
    (inference / "session.py").write_text(
        "import os, signal\n"
        "from types import SimpleNamespace\n"
        "INPUT_FRAME_SAMPLES = 1280\n"
        "class D2LiveSession:\n"
        "    async def push_pcm16(self, pcm16: bytes) -> SimpleNamespace:\n"
        "        if pcm16 == b'wait':\n"
        "            print(os.getpid(), flush=True)\n"
        "            signal.pause()\n"
        "        event = {'sequence': 0, 'token_id': 42, 'text_delta': 'heard'}\n"
        "        return SimpleNamespace(pcm_frames=(pcm16,), frame_events=(event,))\n"
        "    async def close(self) -> None: pass\n"
    )
    (inference / "runtime.py").write_text(
        "from types import SimpleNamespace\n"
        "from .session import D2LiveSession\n"
        "class Runtime:\n"
        "    def __init__(self, model_path: str, device: str, kv_budget: int, seed: int) -> None:\n"
        "        self.model = SimpleNamespace(interrupt_id=42)\n"
        "        self.max_context_frames = 1\n"
        "    async def load(self) -> None: pass\n"
        "    async def create_session(self) -> D2LiveSession: return D2LiveSession()\n"
        "    async def close(self) -> None: pass\n"
    )
    return str(interpreter)


def test_worker_reopens_with_a_full_budget_and_lossless_pcm(
    runtime_python: str,
) -> None:
    runtime = D2Runtime("fixture", "cpu", runtime_python, 128, 1)
    pcm16 = bytes(range(256)) * 10
    try:
        for _ in range(2):
            runtime.open_session()
            with pytest.raises(ContextExhaustedError):
                runtime.step(pcm16 + b"\0\0")
            (frame,) = runtime.step(pcm16)
            assert frame.pcm16 == pcm16
            assert frame.is_interruption and frame.text_delta == "heard"
            with pytest.raises(ContextExhaustedError):
                runtime.step(b"\0\0")
            runtime.close_session()
    finally:
        runtime.shutdown()
    assert runtime.process.returncode == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux GPU stage watchdog")
def test_worker_dies_with_its_stage_during_a_blocking_forward(
    runtime_python: str,
) -> None:
    stage_program = (
        "import sys\n"
        "from sglang_omni.models.qwen3_omni_d2.runtime import D2Runtime\n"
        "runtime = D2Runtime('fixture', 'cpu', sys.argv[1], 128, 1)\n"
        "runtime.open_session()\n"
        "runtime.step(b'wait')\n"
    )
    stage = subprocess.Popen(
        [sys.executable, "-c", stage_program, runtime_python], stdout=subprocess.PIPE
    )
    try:
        assert stage.stdout is not None and select.select([stage.stdout], [], [], 30)[0]
        worker_pid = int(stage.stdout.readline())
        worker_descriptor = os.pidfd_open(worker_pid)
        try:
            stage.kill()
            stage.wait(timeout=5)
            if not select.select([worker_descriptor], [], [], 5)[0]:
                signal.pidfd_send_signal(worker_descriptor, signal.SIGKILL)
                pytest.fail("D2 worker still owns the device after the stage died")
        finally:
            os.close(worker_descriptor)
    finally:
        stage.kill()
        stage.wait(timeout=5)
