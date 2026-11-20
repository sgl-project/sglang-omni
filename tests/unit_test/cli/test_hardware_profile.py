# SPDX-License-Identifier: Apache-2.0
"""sgl-omni serve applies a measured replica profile only on its GPU and only with MPS."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from sglang_omni import hardware_profiles
from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager
from sglang_omni.hardware_profiles import (
    PROFILE_DIRS,
    PROFILE_ROOT,
    ProfileGpu,
    select_hardware_profile,
)
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.platforms import current_platform
from tests.unit_test.pipeline.helpers import build_compiled_process_topology

RTX_4090_D = "NVIDIA GeForce RTX 4090 D"
PROFILE_PATHS = sorted(PROFILE_ROOT.glob("*/*.yaml"))


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)

    def set_gpu(name: str | None, mps_unsupported_reason: str | None = None) -> None:
        monkeypatch.setattr(
            hardware_profiles,
            "inspect_profile_gpu",
            lambda: ProfileGpu(name, mps_unsupported_reason),
        )

    set_gpu(RTX_4090_D)
    return set_gpu


def serve(model_path: str, *args: str, config_file: Path | None = None):
    """Run sgl-omni serve up to launch and return the config it would launch."""
    from_model_path = lambda path, variant=None: ConfigManager(  # noqa: E731
        Qwen3ASRPipelineConfig(model_path=path)
    )
    source = ["--config", str(config_file)] if config_file else []
    with (
        mock.patch("sglang_omni.cli.serve.launch_server") as launch_server,
        mock.patch.object(ConfigManager, "from_model_path", from_model_path),
    ):
        result = CliRunner().invoke(
            app, ["serve", "--model-path", model_path, *source, *args]
        )
    assert result.exit_code == 0, result.output
    (launched,) = launch_server.call_args.args
    return launched


def replicas(config) -> int:
    return config.processes["asr"].num_replicas if "asr" in config.processes else 1


def test_every_profile_names_a_known_gpu_and_resolves() -> None:
    assert {path.parent.name for path in PROFILE_PATHS} <= set(PROFILE_DIRS.values())
    for path in PROFILE_PATHS:
        config = ConfigManager.from_file(str(path)).config
        build_compiled_process_topology(config)


@pytest.mark.parametrize(
    "model_path",
    [
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5",
        "/hf/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5",
    ],
)
def test_serve_applies_the_profile_of_the_detected_gpu(gpu, model_path: str) -> None:
    launched = serve(model_path)
    asr = launched.stage_named("asr")
    assert launched.mps == "on"
    assert replicas(launched) == 2
    assert launched.processes["asr"].replica_devices == [0, 0]
    assert asr.gpu_memory_fraction == pytest.approx(0.2799)
    assert asr.engine.mem_fraction_static == pytest.approx(0.8008)


@pytest.mark.parametrize(
    ("gpu_name", "model_path"),
    [
        ("NVIDIA H100 80GB HBM3", "Qwen/Qwen3-ASR-1.7B"),
        (None, "Qwen/Qwen3-ASR-1.7B"),
        (RTX_4090_D, "Qwen/Qwen3-ASR-0.6B"),
    ],
)
def test_serve_keeps_the_default_without_a_matching_profile(
    gpu, gpu_name: str | None, model_path: str
) -> None:
    gpu(gpu_name)
    launched = serve(model_path)
    assert launched.mps == "off"
    assert replicas(launched) == 1


def test_serve_keeps_the_default_when_the_gpu_cannot_run_mps(gpu) -> None:
    gpu(RTX_4090_D, "MIG mode is enabled")
    launched = serve("Qwen/Qwen3-ASR-1.7B")
    assert launched.mps == "off"
    assert replicas(launched) == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--mps", "off"],
        ["--asr.engine.max_running_requests", "8"],
        ["--mem-fraction-static", "0.5"],
    ],
)
def test_serve_keeps_the_default_when_the_user_sets_a_profile_setting(
    gpu, args: list[str]
) -> None:
    launched = serve("Qwen/Qwen3-ASR-1.7B", *args)
    assert replicas(launched) == 1


@pytest.mark.parametrize("mode", ["on", "auto"])
def test_serve_keeps_the_users_mps_mode_with_the_profile(gpu, mode: str) -> None:
    launched = serve("Qwen/Qwen3-ASR-1.7B", "--mps", mode)
    assert launched.mps == mode
    assert replicas(launched) == 2


def test_serve_with_a_config_file_ignores_profiles(gpu, tmp_path: Path) -> None:
    config_file = tmp_path / "qwen3_asr.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Qwen3ASRPipelineConfig",
                "model_path": "Qwen/Qwen3-ASR-1.7B",
            }
        )
    )
    launched = serve("Qwen/Qwen3-ASR-1.7B", config_file=config_file)
    assert replicas(launched) == 1


def test_no_profile_off_cuda(gpu, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")
    assert select_hardware_profile(config, set()) is None
