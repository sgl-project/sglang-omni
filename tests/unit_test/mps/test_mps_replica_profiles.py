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
from sglang_omni.mps.manager import MpsClientRef
from sglang_omni.mps.replica_profiles import (
    B200,
    REPLICA_PROFILES,
    RTX_4090_D,
    MeasuredGpu,
    ReplicaProfile,
    decide_replica_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_GPUS = {"b200": B200, "rtx4090d": RTX_4090_D}
EXAMPLES = sorted(
    (example, gpu)
    for directory, gpu in EXAMPLE_GPUS.items()
    for example in (REPO_ROOT / "examples" / "mps_dp" / directory).glob("*.yaml")
)
MIB = 1 << 20
WHISPER = "openai/whisper-large-v3"
MOSS_TD = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


class FakeMpsClient:
    """Control I/O of a live daemon: its owner leases are still held, or were left by a serve killed with SIGKILL."""

    def __init__(self, *, owner_leases_held: bool = True):
        self.owner_leases_held = owner_leases_held
        self.queried: list[Path] = []

    def read_daemon_identity(self, pipe_dir: Path) -> int:
        self.queried.append(pipe_dir)
        return 4242

    def snapshot(self, pipe_dir: Path) -> set[MpsClientRef]:
        return set()

    def owner_lease_held(self, lease_file: Path) -> bool:
        return self.owner_leases_held


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


def profile_for(model_path: str, gpu: MeasuredGpu = B200) -> ReplicaProfile:
    (profile,) = [
        p for p in REPLICA_PROFILES if p.model_path == model_path and p.gpu == gpu
    ]
    return profile


def cli_patches(config: PipelineConfig, argv: list[str]) -> list[ConfigPatch]:
    return list(
        patches_from_dotted_cli(ConfigManager(config).parse_extra_args(argv), config)
    )


@pytest.fixture
def mps_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A CUDA host with the MPS control binary, no inherited MPS settings and an empty MPS state root, which it returns."""
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
    state_root = tmp_path / "mps-state"
    monkeypatch.setenv("SGLANG_OMNI_MPS_STATE_ROOT", str(state_root))
    return state_root


def test_every_example_is_a_profile_at_its_replica_count(monkeypatch) -> None:
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
    counts_by_profile: dict[tuple[str, str], set[int]] = {}
    for example, gpu in EXAMPLES:
        measured = ConfigManager.from_file(str(example)).config
        profile = profile_for(measured.model_path, gpu)
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
        counts_by_profile.setdefault((profile.model_path, gpu.name), set()).add(
            num_replicas
        )
    defaults_without_example = [
        (profile.model_path, profile.gpu.name)
        for profile in REPLICA_PROFILES
        if profile.num_replicas
        not in counts_by_profile.get((profile.model_path, profile.gpu.name), set())
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
    mps_client = FakeMpsClient()

    decision = decide_replica_profile(
        config,
        user_patches=user_patches,
        device_info=device_info,
        mps_client=mps_client,
    )

    gpu_id = 2 if gpu_argv else 0
    assert device_info.inspected == [gpu_id]
    assert mps_client.queried == []
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


def idle_4090_d() -> MpsPhysicalDevice:
    return MpsPhysicalDevice(
        "GPU-4090d",
        name=RTX_4090_D.name,
        total_memory_bytes=RTX_4090_D.total_memory_mib * MIB,
        free_memory_bytes=(RTX_4090_D.total_memory_mib - 600) * MIB,
    )


@pytest.mark.parametrize(
    "device, gpu", [(idle_b200(), B200), (idle_4090_d(), RTX_4090_D)]
)
def test_each_gpu_serves_its_own_measured_count(mps_host, device, gpu) -> None:
    profile = profile_for(MOSS_TD, gpu)

    decision = decide_replica_profile(
        model_default(profile),
        user_patches=[],
        device_info=FakeDeviceInfo(device),
        mps_client=FakeMpsClient(),
    )

    assert decision.profile is profile
    assert decision.writes["processes.asr.num_replicas"] == profile.num_replicas


def test_model_measured_on_another_gpu_keeps_its_config_on_a_4090_d(mps_host) -> None:
    config = model_default(profile_for("Qwen/Qwen3-ASR-1.7B"))

    decision = decide_replica_profile(
        config,
        user_patches=[],
        device_info=FakeDeviceInfo(idle_4090_d()),
        mps_client=FakeMpsClient(),
    )

    assert decision.writes == {}
    assert "replica counts were measured on NVIDIA B200" in decision.message


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
        config,
        user_patches=user_patches,
        device_info=FakeDeviceInfo(device),
        mps_client=FakeMpsClient(),
    )

    assert decision.writes == {}
    assert decision.message.startswith(
        f"Measured replica profile for {WHISPER} not applied: {reason}"
    )


def test_broadcast_memory_flag_keeps_the_launch_config(mps_host) -> None:
    config = model_default(profile_for(WHISPER))
    user_patches = list(patches_from_broadcast_flags(config, mem_fraction_static=0.5))

    decision = decide_replica_profile(
        config,
        user_patches=user_patches,
        device_info=FakeDeviceInfo(idle_b200()),
        mps_client=FakeMpsClient(),
    )

    assert decision.writes == {}
    assert "--mem-fraction-static" in decision.message


def test_platform_without_mps_never_inspects_the_gpu(mps_host, monkeypatch) -> None:
    monkeypatch.setattr(replica_profiles.shutil, "which", lambda name: None)
    device_info = FakeDeviceInfo(idle_b200())

    decision = decide_replica_profile(
        model_default(profile_for(WHISPER)),
        user_patches=[],
        device_info=device_info,
        mps_client=FakeMpsClient(),
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


def leave_owner_lease(state_root: Path, gpu_uuid: str) -> Path:
    """The state a serve publishes for one GPU: a pipe directory and an active owner lease."""
    state_dir = state_root / gpu_uuid
    (state_dir / "pipe").mkdir(parents=True)
    (state_dir / "owners").mkdir()
    (state_dir / "owners" / "4242").write_text("active\n")
    return state_dir


def test_state_left_by_a_killed_serve_keeps_the_launch_config(mps_host) -> None:
    """A serve killed with SIGKILL leaves an owner lease no join accepts; the launch serves its own config instead of failing in MPS setup."""
    state_dir = leave_owner_lease(mps_host, "GPU-b200")
    mps_client = FakeMpsClient(owner_leases_held=False)

    decision = decide_replica_profile(
        model_default(profile_for(WHISPER)),
        user_patches=[],
        device_info=FakeDeviceInfo(idle_b200()),
        mps_client=mps_client,
    )

    assert mps_client.queried == [state_dir / "pipe"]
    assert decision.writes == {}
    assert decision.message.endswith(
        "GPU 0 holds MPS state from an earlier run that a new launch cannot "
        f"join, in {state_dir}"
    )


def test_live_daemon_of_another_serve_is_joined_with_the_profile(mps_host) -> None:
    leave_owner_lease(mps_host, "GPU-b200")

    decision = decide_replica_profile(
        model_default(profile_for(WHISPER)),
        user_patches=[],
        device_info=FakeDeviceInfo(idle_b200()),
        mps_client=FakeMpsClient(owner_leases_held=True),
    )

    assert decision.profile is profile_for(WHISPER)


@pytest.mark.parametrize(
    "argv, reason",
    [
        (["--vocoder.gpu", "[1]"], "its GPU stages run on GPUs [0, 1]"),
        (
            ["--vocoder.gpu", "null"],
            "the profile replicates processes without a GPU stage: ['vocoder']",
        ),
    ],
)
def test_stage_placement_the_profile_cannot_replicate_keeps_the_launch_config(
    mps_host, argv, reason
) -> None:
    """A list placement counts as a GPU, and a replicated process left without a GPU stage would fail topology validation."""
    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    manager = ConfigManager(model_default(profile_for(model_path)))
    config = manager.merge_config(manager.parse_extra_args(argv))

    decision = decide_replica_profile(
        config,
        user_patches=cli_patches(manager.config, argv),
        device_info=FakeDeviceInfo(idle_b200()),
        mps_client=FakeMpsClient(),
    )

    assert decision.writes == {}
    assert decision.message.endswith(f"{model_path} not applied: {reason}")


def test_byte_reserves_count_toward_the_memory_the_replicas_need(mps_host) -> None:
    """Drama 3's preprocessing reserves 6 GiB in bytes, once per replica, on top of the fractions."""
    device = MpsPhysicalDevice(
        "GPU-b200",
        name=B200.name,
        total_memory_bytes=B200.total_memory_mib * MIB,
        free_memory_bytes=130 << 30,
    )

    decision = decide_replica_profile(
        model_default(profile_for("fishaudio/drama-3")),
        user_patches=[],
        device_info=FakeDeviceInfo(device),
        mps_client=FakeMpsClient(),
    )

    assert decision.writes == {}
    assert decision.message.endswith(
        "GPU 0 has 130.0 GiB free and 2 replicas reserve 139.4 GiB"
    )
