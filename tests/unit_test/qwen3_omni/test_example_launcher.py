# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from examples.launchers.qwen3_omni import _parse_thinker_tp_gpu_list
from examples.launchers.qwen3_omni import (
    launch_qwen_speech_server as _launch_speech_server,
)
from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples"


def test_unified_launcher_preset_help():
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_DIR / "run_omni.py"),
            "qwen3-text-server",
            "--help",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def fresh_process_log_level(preset: str, loglevel: str | None) -> str:
    code = (
        "import logging\n"
        "from examples import _omni_launcher\n"
        "try:\n"
        f"    _omni_launcher.run_preset({preset!r}, ['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(logging.getLevelName(logging.getLogger().level))\n"
    )
    process_env = os.environ.copy()
    if loglevel is None:
        process_env.pop("LOGLEVEL", None)
    else:
        process_env["LOGLEVEL"] = loglevel
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXAMPLES_DIR.parent,
        env=process_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_unified_launcher_honors_loglevel_override():
    assert fresh_process_log_level("qwen3-text-server", "DEBUG") == "DEBUG"


@pytest.mark.parametrize(
    ("preset", "expected"),
    [("qwen3-text-server", "INFO"), ("ming-text", "DEBUG")],
)
def test_unified_launcher_preserves_default_log_levels(preset, expected):
    assert fresh_process_log_level(preset, None) == expected


def test_unified_ming_offline_launcher_applies_tp_and_overrides(monkeypatch):
    from examples import _omni_launcher as registry
    from examples.launchers import ming_omni as launcher

    captured = {}

    async def fake_run(config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(launcher, "run_speech_request", fake_run)
    args = registry.parse_preset_args(
        "ming-speech",
        [
            "--model-path",
            "dummy",
            "--tp-size",
            "2",
            "--gpu-talker",
            "2",
            "--voice",
            "CUSTOM_VOICE",
            "--cpu-offload-gb",
            "4",
            "--mem-fraction-static",
            "0.8",
        ],
    )

    asyncio.run(launcher.run_ming_speech(args))

    stages = {stage.name: stage for stage in captured["config"].stages}
    thinker = stages["thinker"]
    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]
    assert stages["talker"].gpu == 2
    assert stages["talker"].factory.voice == "CUSTOM_VOICE"
    assert thinker.engine.overrides() == {
        "disable_custom_all_reduce": True,
        "cpu_offload_gb": 4.0,
        "mem_fraction_static": 0.8,
    }


def test_unified_ming_text_applies_thinker_max_seq_len():
    from examples import _omni_launcher as registry
    from examples.launchers import ming_omni as launcher

    args = registry.parse_preset_args(
        "ming-text",
        [
            "--model-path",
            "dummy",
            "--thinker-max-seq-len",
            "1234",
            "--cpu-offload-gb",
            "0",
        ],
    )

    config = launcher.build_ming_text_config(args)

    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    assert thinker.factory.thinker_max_seq_len == 1234


@pytest.mark.parametrize(
    "script",
    [
        "run_qwen3_omni_server.py",
        "run_qwen3_omni_speech.py",
        "run_qwen3_omni_speech_server.py",
        "run_ming_omni_server.py",
        "run_ming_omni_speech.py",
        "run_ming_omni_speech_server.py",
        "run_ming_omni_text_first.py",
    ],
)
def test_example_script_help(script):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script), "--help"],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        gpu_thinker=0,
        gpu_talker=None,
        gpu_code_predictor=None,
        gpu_code2wav=None,
        gpu_image_encoder=None,
        gpu_audio_encoder=None,
        thinker_tp_size=1,
        gpu_thinker_tp=None,
        thinker_max_seq_len=None,
        talker_max_seq_len=None,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
        enable_partial_start=None,
        partial_start_min_chunks=5,
        colocated=False,
        host="0.0.0.0",
        port=8000,
        model_name="qwen3-omni",
        enable_realtime=False,
        skip_server_warmup=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_stage(config, name: str):
    return next(s for s in config.stages if s.name == name)


@pytest.fixture()
def mock_launch_server():
    mock_fn = MagicMock()
    fake_serve = ModuleType("sglang_omni.serve")
    fake_serve.launch_server = mock_fn
    with patch.dict(sys.modules, {"sglang_omni.serve": fake_serve}):
        yield mock_fn


def test_tp2_config_contract(mock_launch_server):
    """The thinker TP flags land in tp_size and the GPU list for TP=2."""
    args = make_args(thinker_tp_size=2, gpu_thinker_tp="0,1")
    with patch(
        "sglang_omni.utils.gpu_compat.should_disable_custom_all_reduce_for_gpus",
        return_value=True,
    ):
        _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = make_stage(config, "thinker")

    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]
    assert thinker.engine.overrides()["disable_custom_all_reduce"] is True


def test_tp2_enables_custom_all_reduce_on_p2p_mesh(mock_launch_server):
    """A P2P-capable (e.g. NVLink) TP thinker keeps custom all-reduce enabled."""
    args = make_args(thinker_tp_size=2, gpu_thinker_tp="0,1")
    with patch(
        "sglang_omni.utils.gpu_compat.should_disable_custom_all_reduce_for_gpus",
        return_value=False,
    ):
        _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = make_stage(config, "thinker")
    assert thinker.engine.overrides()["disable_custom_all_reduce"] is False


def test_tp1_default_config_contract(mock_launch_server):
    args = make_args()
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = make_stage(config, "thinker")
    talker = make_stage(config, "talker_ar")
    code2wav = make_stage(config, "code2wav")

    assert thinker.tp_size == 1
    assert thinker.gpu == 0
    assert talker.gpu == 1
    assert code2wav.gpu == 0


def test_speech_server_code2wav_follows_relocated_thinker(mock_launch_server):
    args = make_args(gpu_thinker=2, gpu_talker=3)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    assert make_stage(config, "thinker").gpu == 2
    assert make_stage(config, "talker_ar").gpu == 3
    assert make_stage(config, "code2wav").gpu == 2


def test_speech_server_forwards_realtime_flag(mock_launch_server):
    args = make_args(enable_realtime=True)
    _launch_speech_server(args)

    assert mock_launch_server.call_args.kwargs["enable_realtime"] is True


def test_mem_fractions_applied(mock_launch_server):
    args = make_args(
        thinker_mem_fraction_static=0.55,
        talker_mem_fraction_static=0.20,
    )
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = make_stage(config, "thinker")
    talker = make_stage(config, "talker_ar")

    assert thinker.engine.overrides()["mem_fraction_static"] == 0.55
    assert talker.engine.overrides()["mem_fraction_static"] == 0.20


def test_talker_max_seq_len_applied(mock_launch_server):
    args = make_args(talker_max_seq_len=128)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.max_seq_len == 128


def test_partial_start_updates_talker_factory_args(mock_launch_server):
    args = make_args(enable_partial_start=True, partial_start_min_chunks=7)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.enable_partial_start is True
    assert talker.factory.partial_start_min_chunks == 7


def test_partial_start_defaults_on(mock_launch_server):
    args = make_args()
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.enable_partial_start is True


def test_partial_start_colocated_defaults_off(mock_launch_server):
    args = make_args(colocated=True)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.enable_partial_start is False


def test_partial_start_colocated_can_be_enabled(mock_launch_server):
    args = make_args(colocated=True, enable_partial_start=True)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.enable_partial_start is True


def test_partial_start_can_be_disabled(mock_launch_server):
    args = make_args(enable_partial_start=False)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.enable_partial_start is False


def test_partial_start_disabled_does_not_propagate_subfloor_min_chunks(
    mock_launch_server,
):
    args = make_args(enable_partial_start=False, partial_start_min_chunks=2)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = make_stage(config, "talker_ar")

    assert talker.factory.enable_partial_start is False
    assert talker.factory.partial_start_min_chunks >= MIN_PARTIAL_START_CHUNKS


def test_partial_start_min_chunks_rejects_below_floor(mock_launch_server):
    args = make_args(enable_partial_start=True, partial_start_min_chunks=2)
    with pytest.raises(ValueError, match="partial-start-min-chunks must be >= 3"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()


def test_colocated_defaults_use_thinker_gpu_for_gpu_stages(mock_launch_server):
    args = make_args(colocated=True, gpu_thinker=3)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    assert make_stage(config, "image_encoder").gpu == 3
    assert make_stage(config, "audio_encoder").gpu == 3
    assert make_stage(config, "thinker").gpu == 3
    assert make_stage(config, "talker_ar").gpu == 3
    assert make_stage(config, "code2wav").gpu == 3


def test_colocated_rejects_conflicting_stage_gpu(mock_launch_server):
    args = make_args(colocated=True, gpu_talker=1)
    with pytest.raises(ValueError, match="--colocated requires all GPU stage flags"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()


def test_parse_thinker_tp_rejects_length_mismatch():
    with pytest.raises(ValueError, match="1 entries.*thinker-tp-size=2"):
        _parse_thinker_tp_gpu_list("0", tp_size=2)


def test_parse_thinker_tp_rejects_duplicates():
    with pytest.raises(ValueError, match="distinct"):
        _parse_thinker_tp_gpu_list("0,0", tp_size=2)


def test_parse_thinker_tp_rejects_negative_ids():
    with pytest.raises(ValueError, match="must be >= 0"):
        _parse_thinker_tp_gpu_list("-1,0", tp_size=2)


def test_parse_thinker_tp_rejects_non_integers():
    with pytest.raises(ValueError, match="comma-separated list of integers"):
        _parse_thinker_tp_gpu_list("x,1", tp_size=2)


def test_tp_greater_than_1_requires_gpu_thinker_tp(mock_launch_server):
    args = make_args(thinker_tp_size=2, gpu_thinker_tp=None)
    with pytest.raises(ValueError, match="requires --gpu-thinker-tp"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()


def test_gpu_thinker_tp_rejected_when_tp1(mock_launch_server):
    args = make_args(thinker_tp_size=1, gpu_thinker_tp="0,1")
    with pytest.raises(ValueError, match="only applies when.*thinker-tp-size > 1"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()
