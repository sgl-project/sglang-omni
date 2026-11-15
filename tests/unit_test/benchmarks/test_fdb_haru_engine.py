# SPDX-License-Identifier: Apache-2.0
"""Haru profile and engine of the Full-Duplex-Bench runbook; no weights are loaded."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.duplex.fdb.common import load_settings
from benchmarks.duplex.fdb.engines import ENGINES
from benchmarks.duplex.fdb.servers import model_server_command
from benchmarks.duplex.profiles import PROFILES

HARU_PROFILE = "haru-native-mlx"


@pytest.fixture
def haru_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "GPU",
        "SERVER_PORT",
        "JUDGE_PORT",
        "HARU_SERVER_BIN",
        "HARU_SERVER_SEED",
        "HARU_REFERENCE_AUDIO",
        "HARU_INSTRUCTIONS_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENGINE", "haru")
    monkeypatch.setenv("FDB_WORK", str(tmp_path / "fdb"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "Haru-4B"))
    return tmp_path


def test_haru_profile_declares_the_wire_contract() -> None:
    profile = PROFILES[HARU_PROFILE]
    assert profile.native_unit_ms == 400
    assert profile.output_sample_rate == 24000
    assert profile.protocol == "native"
    assert profile.stop_requires_eos is True
    assert profile.continuous_output is False


def test_haru_engine_records_the_native_profile() -> None:
    engine = ENGINES["haru"]
    assert engine.record_profile == HARU_PROFILE
    assert engine.turn_detection is None
    assert engine.ready_text == '"native_full_duplex":true'
    assert engine.server_gpu_count == 1
    assert engine.max_sessions == 1
    assert engine.needs_aligner is False


def test_haru_server_command_takes_voice_and_card_from_configuration(
    haru_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARU_SERVER_BIN", str(haru_env / "haru_server"))
    monkeypatch.setenv("HARU_SERVER_SEED", "7")
    monkeypatch.setenv("HARU_REFERENCE_AUDIO", str(haru_env / "voice.wav"))
    monkeypatch.setenv("HARU_INSTRUCTIONS_FILE", str(haru_env / "card.txt"))
    command = model_server_command(load_settings("haru-run"))
    arguments = dict(zip(command[1::2], command[2::2]))
    assert command[0] == str(haru_env / "haru_server")
    assert arguments["--model-path"] == str(haru_env / "Haru-4B")
    assert arguments["--quantize"] == "8"
    assert arguments["--seed"] == "7"
    assert arguments["--input-sample-rate"] == "16000"
    assert arguments["--reference-audio"] == str(haru_env / "voice.wav")
    assert arguments["--instructions-file"] == str(haru_env / "card.txt")
    assert arguments["--port"] == "8097"
    assert "sglang_omni.cli" not in command


def test_haru_server_command_requires_the_binary(
    haru_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARU_REFERENCE_AUDIO", str(haru_env / "voice.wav"))
    monkeypatch.setenv("HARU_INSTRUCTIONS_FILE", str(haru_env / "card.txt"))
    with pytest.raises(SystemExit, match="HARU_SERVER_BIN"):
        model_server_command(load_settings("haru-run"))


@pytest.mark.parametrize("missing", ["HARU_REFERENCE_AUDIO", "HARU_INSTRUCTIONS_FILE"])
def test_haru_server_command_refuses_to_guess_the_voice_or_card(
    haru_env: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    present = {
        "HARU_REFERENCE_AUDIO": str(haru_env / "voice.wav"),
        "HARU_INSTRUCTIONS_FILE": str(haru_env / "card.txt"),
    }
    present.pop(missing)
    for name, value in present.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match="HARU_REFERENCE_AUDIO"):
        model_server_command(load_settings("haru-run"))


def test_other_engines_still_launch_sglang_omni(
    haru_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENGINE", "minicpmo")
    command = model_server_command(load_settings("minicpmo-run"))
    assert command[1:4] == ["-m", "sglang_omni.cli", "serve"]
