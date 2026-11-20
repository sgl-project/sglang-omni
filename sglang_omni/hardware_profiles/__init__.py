# SPDX-License-Identifier: Apache-2.0
"""Measured MPS replica profiles that ``sgl-omni serve`` applies by default.

Each directory holds config files measured on one GPU model, as described in
docs/basic_usage/mps_dp.md. A file applies when its pipeline class and
checkpoint match the launched model and logical GPU 0 can run native MPS.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import yaml

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import flatten
from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)

PROFILE_ROOT = Path(__file__).parent

# NVML device name -> directory of profiles measured on that GPU.
PROFILE_DIRS: dict[str, str] = {
    "NVIDIA GeForce RTX 4090 D": "rtx4090d",
}

_MPS_ENV_CONFLICTS = ("CUDA_MPS_PIPE_DIRECTORY", "SGLANG_OMNI_WEIGHT_SHARE")


@dataclass(frozen=True)
class ProfileGpu:
    name: str | None
    mps_unsupported_reason: str | None = None


@dataclass(frozen=True)
class HardwareProfile:
    path: Path
    writes: dict[str, object]


def launches_checkpoint(model_path: str, repo_id: str) -> bool:
    """Whether model_path is repo_id, pinned to a revision or as a hub cache snapshot."""
    return (
        model_path.partition("@")[0] == repo_id
        or "models--" + repo_id.replace("/", "--") in Path(model_path).parts
    )


def inspect_profile_gpu() -> ProfileGpu:
    """Name logical GPU 0 and say why it cannot run native MPS, without a CUDA context."""
    from sglang_omni.mps.devices import NvmlDeviceInfo
    from sglang_omni.utils.gpu_memory import get_gpu_device_info

    name = get_gpu_device_info(0).name
    device = NvmlDeviceInfo().inspect([0])[0]
    if device.unsupported_reason is not None:
        return ProfileGpu(name, device.unsupported_reason)
    else:
        pass
    if shutil.which("nvidia-cuda-mps-control") is None:
        return ProfileGpu(name, "nvidia-cuda-mps-control is not on PATH")
    else:
        pass
    for variable in _MPS_ENV_CONFLICTS:
        if os.environ.get(variable):
            return ProfileGpu(name, f"{variable} is set")
        else:
            pass
    return ProfileGpu(name)


def find_profile(
    gpu_name: str | None, pipeline_config: PipelineConfig
) -> HardwareProfile | None:
    directory = PROFILE_DIRS.get(gpu_name or "")
    if directory is None:
        return None
    else:
        pass
    config_cls = type(pipeline_config)
    for path in sorted((PROFILE_ROOT / directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data["config_cls"] == config_cls.__name__ and launches_checkpoint(
            pipeline_config.model_path, data["model_path"]
        ):
            writes: dict[str, object] = {"mps": "on"}
            for block in ("stages", "processes"):
                writes.update(flatten(block, data.get(block, {}), config_cls))
            return HardwareProfile(path, writes)
        else:
            pass
    return None


def select_hardware_profile(
    pipeline_config: PipelineConfig,
    user_paths: Collection[str],
) -> HardwareProfile | None:
    """The profile to apply, or None to keep the model's default config.

    A profile replaces settings as a set, so it is skipped when the user wrote
    any of them; a user's ``mps`` of on or auto is kept, and off skips it.
    """
    if not current_platform.is_cuda():
        return None
    else:
        pass
    if "mps" in user_paths and pipeline_config.mps == "off":
        return None
    else:
        pass
    gpu = inspect_profile_gpu()
    profile = find_profile(gpu.name, pipeline_config)
    if profile is None:
        return None
    else:
        pass
    overlap = sorted(set(user_paths) & (profile.writes.keys() - {"mps"}))
    if overlap:
        logger.info(
            "Not applying hardware profile %s: the command line sets %s",
            profile.path.name,
            ", ".join(overlap),
        )
        return None
    else:
        pass
    if gpu.mps_unsupported_reason is not None:
        logger.warning(
            "Not applying hardware profile %s, which needs native MPS: %s",
            profile.path.name,
            gpu.mps_unsupported_reason,
        )
        return None
    else:
        pass
    if "mps" in user_paths:
        writes = {key: value for key, value in profile.writes.items() if key != "mps"}
        return HardwareProfile(profile.path, writes)
    else:
        return profile
