# SPDX-License-Identifier: Apache-2.0
"""Real-weight stream schedule and pinned-device codec golden checks."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

MODEL_REPOSITORY = "mlx-community_Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def stream_schedule_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    executable = os.environ.get("NATIVE_CODEC_STREAM_TEST")
    data_root = os.environ.get("CI_DATA_ROOT")
    if not executable or not data_root:
        pytest.skip("set NATIVE_CODEC_STREAM_TEST and CI_DATA_ROOT")
    else:
        pass
    output = tmp_path_factory.mktemp("qwen3-stream-schedule")
    command = [
        executable,
        str(Path(data_root) / "models" / MODEL_REPOSITORY),
        str(HERE / "model_golden/qwen3-tts-12hz-0.6b-customvoice.json"),
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    return output


def test_stream_schedule_and_tail(stream_schedule_output: Path) -> None:
    assert (stream_schedule_output / "long-default.f32").is_file()
    assert (stream_schedule_output / "long-bridge.f32").is_file()


def test_stream_codec_golden(stream_schedule_output: Path) -> None:
    golden_path = os.environ.get("NATIVE_CODEC_GOLDEN")
    if not golden_path:
        pytest.skip("set NATIVE_CODEC_GOLDEN for the matching device and MLX build")
    else:
        pass
    golden = json.loads(Path(golden_path).read_text())
    assert set(golden["sha256_f32"]) == {"long-default", "long-bridge"}
    for name, expected in golden["sha256_f32"].items():
        audio = (stream_schedule_output / f"{name}.f32").read_bytes()
        assert len(audio) == golden["samples"] * 4
        assert hashlib.sha256(audio).hexdigest() == expected
