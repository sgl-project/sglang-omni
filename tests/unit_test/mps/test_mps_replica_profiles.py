# SPDX-License-Identifier: Apache-2.0
"""Measured replica profiles: what a launch on a detected GPU gets, and when it keeps its own config."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.cli.config import resolve_sources
from sglang_omni.cli.serve import apply_engine_writes, patches_from_broadcast_flags
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import ConfigPatch
from sglang_omni.config.resolver import diff_configs
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import patches_from_dotted_cli
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.ming3_tts import config as ming3_tts_config
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.mps import replica_profiles
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.replica_profiles import (
    B200,
    REPLICA_PROFILES,
    ReplicaProfile,
    decide_replica_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
B200_EXAMPLES = sorted((REPO_ROOT / "examples" / "mps_dp" / "b200").glob("*.yaml"))
MIB = 1 << 20
WHISPER = "openai/whisper-large-v3"


class FakeDeviceInfo:
    def __init__(self, device: MpsPhysicalDevice):
        self.device = device
        self.inspected: list[int] = []

    def inspect(self, gpu_ids: Iterable[int]) -> dict[int, MpsPhysicalDevice]:
        gpu_ids = list(gpu_ids)
        self.inspected.extend(gpu_ids)
        return {gpu_id: self.device for gpu_id in gpu_ids}


def idle_b200() -> MpsPhysicalDevice:
    return MpsPhysicalDevice(
        "GPU-b200",
        name=B200.name,
        total_memory_bytes=B200.total_memory_mib * MIB,
        free_memory_bytes=(B200.total_memory_mib - 600) * MIB,
    )


def model_default(
    profile: ReplicaProfile, model_path: str | None = None
) -> PipelineConfig:
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(profile.config_cls)
    return config_cls(model_path=model_path or profile.model_path)


def profile_for(model_path: str) -> ReplicaProfile:
    (profile,) = [p for p in REPLICA_PROFILES if p.model_path == model_path]
    return profile


def cli_patches(config: PipelineConfig, argv: list[str]) -> list[ConfigPatch]:
    return list(
        patches_from_dotted_cli(ConfigManager(config).parse_extra_args(argv), config)
    )


@pytest.fixture
def mps_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA host with the MPS control binary and no inherited MPS settings."""
    monkeypatch.setattr(
        replica_profiles,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )
    monkeypatch.setattr(
        replica_profiles.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("SGLANG_OMNI_WEIGHT_SHARE", raising=False)


def test_every_b200_example_is_a_profile_at_its_replica_count(monkeypatch) -> None:
    """The example files and the defaults stay one measurement: each file is its model's profile at the file's count."""
    # The 0.5B checkpoint's config.json selects Ming3's dense defaults; these tests read no checkpoint.
    monkeypatch.setattr(
        ming3_tts_config,
        "try_resolve_ming3_dense_tts",
        lambda repo_id, revision=None: (
            ming3_tts_config.DENSE_ARCHITECTURE
            if repo_id == "inclusionAI/Ming-3.0-tts-0.5B"
            else None
        ),
    )
    counts_by_model: dict[str, set[int]] = {}
    for example in B200_EXAMPLES:
        measured = ConfigManager.from_file(str(example)).config
        profile = profile_for(measured.model_path)
        num_replicas = int(re.search(r"_dp(\d+)\.yaml$", example.name).group(1))
        writes = {
            **profile.settings,
            "mps": "on",
            **{
                f"processes.{process_name}.num_replicas": num_replicas
                for process_name in profile.replicated_processes
            },
            **{
                f"processes.{process_name}.replica_devices": [0] * num_replicas
                for process_name in profile.replicated_processes
            },
        }
        expected = apply_engine_writes(model_default(profile), writes)
        assert diff_configs(expected, measured) == [], example.name
        counts_by_model.setdefault(profile.model_path, set()).add(num_replicas)
    defaults_without_example = [
        profile.model_path
        for profile in REPLICA_PROFILES
        if profile.num_replicas not in counts_by_model.get(profile.model_path, set())
    ]
    assert defaults_without_example == []


@pytest.mark.parametrize("gpu_argv", [[], ["--asr.gpu", "2"]])
@pytest.mark.parametrize(
    "model_path",
    [
        WHISPER,
        f"{WHISPER}@06f233fe06e710322aca913c1bc4249a0d71fce1",
        "/hf/hub/models--openai--whisper-large-v3/snapshots/06f233fe",
    ],
)
def test_detected_b200_serves_the_measured_replicas_on_the_stage_gpu(
    mps_host, gpu_argv, model_path
) -> None:
    profile = profile_for(WHISPER)
    manager = ConfigManager(model_default(profile, model_path))
    config = manager.merge_config(manager.parse_extra_args(gpu_argv))
    user_patches = cli_patches(manager.config, gpu_argv)
    device_info = FakeDeviceInfo(idle_b200())

    decision = decide_replica_profile(
        config, user_patches=user_patches, device_info=device_info
    )

    gpu_id = 2 if gpu_argv else 0
    assert device_info.inspected == [gpu_id]
    assert decision.profile is profile
    assert decision.writes == {
        **profile.settings,
        "processes.asr.num_replicas": 6,
        "processes.asr.replica_devices": [gpu_id] * 6,
        "mps": "on",
    }
    plan, _ = compile_logical_processes(apply_engine_writes(config, decision.writes))
    assert [(process.name, process.num_replicas) for process in plan.processes] == [
        ("asr", 6)
    ]


@pytest.mark.parametrize(
    "argv, device, environment, reason",
    [
        (
            [],
            MpsPhysicalDevice(
                "GPU-h100",
                name="NVIDIA H100 80GB HBM3",
                total_memory_bytes=81559 * MIB,
                free_memory_bytes=81000 * MIB,
            ),
            {},
            "GPU 0 is NVIDIA H100 80GB HBM3 (81559 MiB); replica counts were measured on NVIDIA B200 (183359 MiB)",
        ),
        (
            [],
            MpsPhysicalDevice("GPU-mig", "MIG mode is enabled"),
            {},
            "GPU 0 cannot run MPS: MIG mode is enabled",
        ),
        (
            [],
            MpsPhysicalDevice(
                "GPU-b200",
                name=B200.name,
                total_memory_bytes=B200.total_memory_mib * MIB,
                free_memory_bytes=40 << 30,
            ),
            {},
            "GPU 0 has 40.0 GiB free and 6 replicas reserve 86.1 GiB",
        ),
        (
            [],
            idle_b200(),
            {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps"},
            "the environment sets CUDA_MPS_PIPE_DIRECTORY",
        ),
        (
            ["--mps", "off"],
            idle_b200(),
            {},
            "mps is set by cli dotted",
        ),
        (
            ["--asr.engine.max_running_requests", "32"],
            idle_b200(),
            {},
            "stages.asr.engine.max_running_requests is set by cli dotted",
        ),
        (
            ["--processes.asr.num_replicas", "1"],
            idle_b200(),
            {},
            "processes.asr.num_replicas is set by cli dotted",
        ),
        (
            ["--asr.engine.kv_cache_bytes", "8GiB"],
            idle_b200(),
            {},
            "stages.asr.engine.kv_cache_bytes is set by cli dotted",
        ),
    ],
)
def test_launch_keeps_its_own_config_when_the_profile_cannot_hold(
    mps_host, monkeypatch, argv, device, environment, reason
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    config = model_default(profile_for(WHISPER))
    user_patches = cli_patches(config, argv)

    decision = decide_replica_profile(
        config, user_patches=user_patches, device_info=FakeDeviceInfo(device)
    )

    assert decision.writes == {}
    assert decision.message.startswith(
        f"Measured replica profile for {WHISPER} not applied: {reason}"
    )


def test_broadcast_memory_flag_keeps_the_launch_config(mps_host) -> None:
    config = model_default(profile_for(WHISPER))
    user_patches = list(patches_from_broadcast_flags(config, mem_fraction_static=0.5))

    decision = decide_replica_profile(
        config, user_patches=user_patches, device_info=FakeDeviceInfo(idle_b200())
    )

    assert decision.writes == {}
    assert "--mem-fraction-static" in decision.message


def test_platform_without_mps_never_inspects_the_gpu(mps_host, monkeypatch) -> None:
    monkeypatch.setattr(replica_profiles.shutil, "which", lambda name: None)
    device_info = FakeDeviceInfo(idle_b200())

    decision = decide_replica_profile(
        model_default(profile_for(WHISPER)), user_patches=[], device_info=device_info
    )

    assert decision.writes == {}
    assert decision.message.endswith("nvidia-cuda-mps-control is not on PATH")
    assert device_info.inspected == []


def test_config_resolve_previews_the_profile_and_names_it(
    mps_host, monkeypatch, tmp_path
) -> None:
    """config resolve and explain show what serve launches on this host, with the profile as the source."""
    monkeypatch.setattr(
        "sglang_omni.cli.config.NvmlDeviceInfo", lambda: FakeDeviceInfo(idle_b200())
    )
    config_file = tmp_path / "whisper.yaml"
    config_file.write_text(
        f"config_cls: WhisperASRPipelineConfig\nmodel_path: {WHISPER}\n"
    )

    resolution = resolve_sources(
        model_path=None,
        config_file=str(config_file),
        text_only=False,
        variant=None,
        mem_fraction_static=None,
        argv=[],
    )

    assert resolution.resolved.config.processes["asr"].num_replicas == 6
    assert resolution.resolved.config.mps == "on"
    assert "replica profile measured on this GPU" in (
        resolution.resolved.provenance.explain("processes.asr.num_replicas")
    )
