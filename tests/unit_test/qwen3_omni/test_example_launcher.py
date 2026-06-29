# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples"


@pytest.mark.parametrize(
    "script",
    [
        "run_qwen3_omni_server.py",
        "run_qwen3_omni_speech.py",
    ],
)
def test_example_script_help(script):
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES_DIR / script), "--help"],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


_EXAMPLE_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "examples"
    / "run_qwen3_omni_speech_server.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location(
        "run_qwen3_omni_speech_server", _EXAMPLE_MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_example_module()
_launch_speech_server = _mod._launch_speech_server
_parse_thinker_tp_gpu_list = _mod._parse_thinker_tp_gpu_list


def _make_args(**overrides) -> argparse.Namespace:
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
        relay_backend="shm",
        thinker_max_seq_len=8192,
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
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _stage(config, name: str):
    return next(s for s in config.stages if s.name == name)


@pytest.fixture()
def mock_launch_server():
    mock_fn = MagicMock()
    fake_serve = ModuleType("sglang_omni.serve")
    fake_serve.launch_server = mock_fn
    with patch.dict(sys.modules, {"sglang_omni.serve": fake_serve}):
        yield mock_fn


def test_tp2_config_contract(mock_launch_server):
    """tp_size and parallelism.tp must stay in sync for TP=2."""
    args = _make_args(thinker_tp_size=2, gpu_thinker_tp="0,1")
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = _stage(config, "thinker")

    assert thinker.tp_size == 2
    assert thinker.parallelism.tp == 2
    assert thinker.gpu == [0, 1]
    assert (
        thinker.factory_args["server_args_overrides"]["disable_custom_all_reduce"]
        is True
    )


def test_tp1_default_config_contract(mock_launch_server):
    args = _make_args()
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = _stage(config, "thinker")
    talker = _stage(config, "talker_ar")
    code2wav = _stage(config, "code2wav")

    assert thinker.tp_size == 1
    assert thinker.parallelism.tp == 1
    assert thinker.gpu == 0
    assert talker.gpu == 1
    assert code2wav.gpu == 0


def test_mem_fractions_applied(mock_launch_server):
    args = _make_args(
        thinker_mem_fraction_static=0.55,
        talker_mem_fraction_static=0.20,
    )
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    thinker = _stage(config, "thinker")
    talker = _stage(config, "talker_ar")

    assert thinker.factory_args["server_args_overrides"]["mem_fraction_static"] == 0.55
    assert talker.factory_args["server_args_overrides"]["mem_fraction_static"] == 0.20


def test_talker_max_seq_len_applied(mock_launch_server):
    args = _make_args(talker_max_seq_len=128)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["talker_max_seq_len"] == 128


def test_partial_start_updates_talker_factory_args(mock_launch_server):
    args = _make_args(enable_partial_start=True, partial_start_min_chunks=7)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["enable_partial_start"] is True
    assert talker.factory_args["partial_start_min_chunks"] == 7


def test_partial_start_defaults_on(mock_launch_server):
    args = _make_args()
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["enable_partial_start"] is True


def test_partial_start_colocated_defaults_off(mock_launch_server):
    args = _make_args(colocated=True)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["enable_partial_start"] is False


def test_partial_start_colocated_can_be_enabled(mock_launch_server):
    args = _make_args(colocated=True, enable_partial_start=True)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["enable_partial_start"] is True


def test_partial_start_can_be_disabled(mock_launch_server):
    args = _make_args(enable_partial_start=False)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["enable_partial_start"] is False


def test_partial_start_disabled_does_not_propagate_subfloor_min_chunks(
    mock_launch_server,
):
    args = _make_args(enable_partial_start=False, partial_start_min_chunks=2)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    talker = _stage(config, "talker_ar")

    assert talker.factory_args["enable_partial_start"] is False
    assert talker.factory_args["partial_start_min_chunks"] >= MIN_PARTIAL_START_CHUNKS


def test_partial_start_min_chunks_rejects_below_floor(mock_launch_server):
    args = _make_args(enable_partial_start=True, partial_start_min_chunks=2)
    with pytest.raises(ValueError, match="partial-start-min-chunks must be >= 3"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()


def test_colocated_defaults_use_thinker_gpu_for_gpu_stages(mock_launch_server):
    args = _make_args(colocated=True)
    _launch_speech_server(args)

    config = mock_launch_server.call_args[0][0]
    assert _stage(config, "image_encoder").gpu == 0
    assert _stage(config, "audio_encoder").gpu == 0
    assert _stage(config, "thinker").gpu == 0
    assert _stage(config, "talker_ar").gpu == 0
    assert _stage(config, "code2wav").gpu == 0


def test_colocated_rejects_conflicting_stage_gpu(mock_launch_server):
    args = _make_args(colocated=True, gpu_talker=1)
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
    args = _make_args(thinker_tp_size=2, gpu_thinker_tp=None)
    with pytest.raises(ValueError, match="requires --gpu-thinker-tp"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()


def test_gpu_thinker_tp_rejected_when_tp1(mock_launch_server):
    args = _make_args(thinker_tp_size=1, gpu_thinker_tp="0,1")
    with pytest.raises(ValueError, match="only applies when.*thinker-tp-size > 1"):
        _launch_speech_server(args)

    mock_launch_server.assert_not_called()
