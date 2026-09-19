# SPDX-License-Identifier: Apache-2.0
"""Model selection and serving contracts for shared Omni CI stages."""

from __future__ import annotations

import shlex
import tomllib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.minicpm_o.config import (
    MiniCPMOPipelineConfig,
    MiniCPMOSpeechPipelineConfig,
)
from tests.test_model import conftest, omni_router_utils
from tests.test_model.omni_ci_config import OMNI_CI_PRESETS
from tests.test_model.rust_router_config import CiRouterTopology


@pytest.mark.parametrize(
    "module, fixture",
    [
        ("thinker_length", "qwen3_omni_bf16_tp2_server"),
        ("tts_ci", "qwen3_omni_bf16_colocated_server"),
        ("mmmu_ci", "qwen3_omni_fp8_colocated_server"),
        ("mmmu_talker_ci", "qwen3_omni_bf16_disagg_server"),
        ("mmsu_ci", "qwen3_omni_bf16_colocated_thinker_server"),
        ("mmsu_talker_ci", "qwen3_omni_fp8_tp2_server"),
        ("videomme_ci", "qwen3_omni_bf16_disagg_server"),
        ("videomme_talker_ci", "qwen3_omni_bf16_disagg_server"),
        ("videoamme_ci", "qwen3_omni_fp8_colocated_server"),
        ("videoamme_talker_tp2_ci", "qwen3_omni_fp8_tp2_server"),
    ],
)
def test_qwen_stages_reuse_existing_server_fixtures(module: str, fixture: str) -> None:
    selected = []
    server = object()
    request = SimpleNamespace(
        module=SimpleNamespace(__name__=f"tests.test_model.test_qwen3_omni_{module}"),
        getfixturevalue=lambda name: selected.append(name) or server,
    )

    lifecycle = conftest.omni_ci_server.__wrapped__(
        request, None, OMNI_CI_PRESETS["qwen3-omni"]
    )
    assert next(lifecycle) is server
    with pytest.raises(StopIteration):
        next(lifecycle)
    assert selected == [fixture]


@pytest.mark.parametrize(
    "module, audio_output, context_length",
    [
        ("thinker_length", False, 128),
        ("mmmu_ci", False, 8192),
        ("tts_ci", True, 8192),
        ("mmsu_talker_ci", True, 8192),
        ("videomme_ci", False, 32768),
        ("videoamme_talker_tp2_ci", True, 32768),
    ],
)
def test_minicpm_stages_launch_two_single_gpu_workers(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    audio_output: bool,
    context_length: int,
) -> None:
    events = []
    launch_args = {}
    server = object()

    @contextmanager
    def launch(**kwargs):
        launch_args.update(kwargs)
        events.append("start")
        try:
            yield server
        finally:
            events.append("stop")

    monkeypatch.setattr(omni_router_utils, "launch_managed_router", launch)
    monkeypatch.setenv("SGLANG_OMNI_TEST_MINICPMO_MODEL", "/models/minicpmo")
    request = SimpleNamespace(
        module=SimpleNamespace(__name__=f"tests.test_model.test_qwen3_omni_{module}")
    )
    lifecycle = conftest.omni_ci_server.__wrapped__(
        request, None, OMNI_CI_PRESETS["minicpmo"]
    )
    assert next(lifecycle) is server
    lifecycle.close()

    assert events == ["start", "stop"]
    assert launch_args["model_path"] == "/models/minicpmo"
    assert launch_args["model_name"] == "minicpmo"
    assert launch_args["num_workers"] == 2
    assert launch_args["num_gpus_per_worker"] == 1
    assert launch_args["router_topology"] == (
        CiRouterTopology.OMNI_AUDIO if audio_output else CiRouterTopology.OMNI_TEXT
    )
    assert launch_args["generation_streaming"] is not audio_output
    argv = shlex.split(launch_args["worker_extra_args"])
    assert "--colocate" not in argv
    assert "--variant" not in argv
    if audio_output:
        config = MiniCPMOSpeechPipelineConfig(model_path="/models/minicpmo")
        assert "--text-only" not in argv
    else:
        config = MiniCPMOPipelineConfig(model_path="/models/minicpmo")
        argv.remove("--text-only")
    manager = ConfigManager(config)
    resolved = manager.merge_config(manager.parse_extra_args(argv))
    thinker = next(stage for stage in resolved.stages if stage.name == "thinker")
    assert thinker.factory.max_seq_len == context_length
    assert thinker.engine.mem_fraction_static == (0.55 if audio_output else 0.80)
    if audio_output:
        talker = next(stage for stage in resolved.stages if stage.name == "talker")
        assert talker.engine.mem_fraction_static == 0.15


@pytest.mark.parametrize("streaming", [True, False])
def test_router_config_declares_actual_generation_stream_modes(
    tmp_path_factory: pytest.TempPathFactory, streaming: bool
) -> None:
    config_path = omni_router_utils._write_router_config(
        tmp_path_factory,
        topology=CiRouterTopology.OMNI_AUDIO,
        router_port=8000,
        worker_urls=["http://127.0.0.1:8001", "http://127.0.0.1:8002"],
        model_name="minicpmo",
        generation_streaming=streaming,
    )
    config = tomllib.loads(config_path.read_text())
    for worker in config["workers"]:
        profile = worker["service_profiles"][0]
        assert profile["output_modalities"] == ["text", "audio"]
        assert profile["stream_modes"] == (
            ["non_streaming", "streaming"] if streaming else ["non_streaming"]
        )


def test_omni_ci_model_option_rejects_unknown_models() -> None:
    assert conftest._parse_omni_ci_model(" MiniCPMO ") == "minicpmo"
    with pytest.raises(pytest.UsageError, match="Unsupported OMNI_CI_MODEL"):
        conftest._parse_omni_ci_model("unknown")


def test_omni_ci_model_defaults_to_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_CI_MODEL", raising=False)
    assert conftest.omni_ci_model.__wrapped__().name == "qwen3-omni"
    monkeypatch.setenv("OMNI_CI_MODEL", "minicpmo")
    assert conftest.omni_ci_model.__wrapped__().name == "minicpmo"
