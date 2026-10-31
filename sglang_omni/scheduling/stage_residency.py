# SPDX-License-Identifier: Apache-2.0
"""Stable-address CUDA weight residency shared by serial pipelines."""

from __future__ import annotations

import logging
import os
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import torch

try:
    from torch_memory_saver import torch_memory_saver
except ModuleNotFoundError as failure:
    if failure.name != "torch_memory_saver":
        raise
    else:
        torch_memory_saver = None

from sglang_omni.scheduling.weight_cache import RuntimeWeights, StagingRing

logger = logging.getLogger(__name__)
BYTES_PER_GIB = 1024.0**3


def offload_environment(environment: dict[str, str]) -> dict[str, str]:
    """Install the allocator hook before a CUDA worker imports torch."""
    package = find_spec("torch_memory_saver")
    if package is None or package.origin is None or torch.version.cuda is None:
        raise RuntimeError("CUDA offload requires torch_memory_saver==0.0.10")
    else:
        pass
    package_directory = Path(package.origin).parent
    cuda_major = torch.version.cuda.split(".")[0]
    pattern = f"torch_memory_saver_hook_mode_preload_cu{cuda_major}.*.so"
    libraries = [
        library
        for directory in (package_directory, package_directory.parent)
        for library in directory.glob(pattern)
    ]
    if len(libraries) != 1:
        raise RuntimeError(f"Offload requires one allocator library matching {pattern}")
    else:
        pass
    preload = environment.get("LD_PRELOAD", os.environ.get("LD_PRELOAD", ""))
    library_path = str(libraries[0])
    if "torch_memory_saver" not in preload:
        environment = dict(environment)
        environment["LD_PRELOAD"] = (
            f"{library_path}:{preload}" if preload else library_path
        )
    else:
        pass
    return environment


def gpu_memory_note(device: torch.device) -> str:
    """Report allocator and device occupancy, including memory outside torch."""
    if device.type != "cuda":
        return "cpu"
    else:
        pass
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (
        f"free={free_bytes / BYTES_PER_GIB:.2f}/{total_bytes / BYTES_PER_GIB:.2f}GiB "
        f"torch_allocated={torch.cuda.memory_allocated(device) / BYTES_PER_GIB:.2f}GiB "
        f"torch_reserved={torch.cuda.memory_reserved(device) / BYTES_PER_GIB:.2f}GiB"
    )


class StageResidency:
    """Release physical pages and restore values into unchanged virtual storage."""

    def __init__(
        self,
        modules: dict[str, torch.nn.Module],
        device: torch.device,
        *,
        label: str = "stage",
        tags: tuple[str, ...] = (),
        source: Literal["mmap", "ram"] = "ram",
        cache_dir: str | None = None,
        checkpoint_contents: dict[str, str] | None = None,
        folding_backend: str = "none",
        staging_ring: StagingRing | None = None,
    ) -> None:
        if not modules:
            raise ValueError("StageResidency requires at least one module")
        else:
            pass
        self.modules: dict[str, torch.nn.Module] = dict(modules)
        self.device: torch.device = torch.device(device)
        self.is_resident: bool = True
        self.label: str = label
        self.tags: tuple[str, ...] = tags
        self.staging_ring: StagingRing | None = staging_ring
        if self.device.type == "cuda" and (not tags or staging_ring is None):
            raise ValueError(
                "CUDA residency requires managed allocation tags and a staging ring"
            )
        else:
            pass
        if self.device.type == "cuda" and torch_memory_saver is None:
            raise RuntimeError("CUDA stage offload requires torch_memory_saver==0.0.10")
        else:
            pass
        self.weights: RuntimeWeights = RuntimeWeights(
            self.modules,
            source=source,
            cache_dir=cache_dir,
            checkpoint_contents=checkpoint_contents or {},
            folding_backend=folding_backend,
        )
        weight_bytes = sum(storage.numel() for storage in self.weights.storages)
        logger.info(
            f"Stage residency {label}: {weight_bytes / BYTES_PER_GIB:.2f}GiB of "
            f"weights, starts {'resident' if self.is_resident else 'offloaded'} "
            f"({gpu_memory_note(self.device)})"
        )

    @property
    def resident(self) -> bool:
        return self.is_resident

    def sleep(self) -> None:
        """Drop the GPU replica; a cheap no-op once already asleep."""
        if not self.is_resident:
            return
        else:
            pass
        started_at_seconds = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        for tag in self.tags:
            assert torch_memory_saver is not None
            torch_memory_saver.pause(tag=tag)
        self.is_resident = False
        logger.info(
            f"Stage residency {self.label} -> host "
            f"elapsed_seconds={time.perf_counter() - started_at_seconds:.4f} "
            f"({gpu_memory_note(self.device)})"
        )

    def wake(self) -> None:
        """Refill the GPU replica from the host copy; no-op once resident."""
        if self.is_resident:
            return
        else:
            pass
        started_at_seconds = time.perf_counter()
        for tag in self.tags:
            assert torch_memory_saver is not None
            torch_memory_saver.resume(tag=tag)
        self.weights.restore(self.staging_ring)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        self.is_resident = True
        logger.info(
            f"Stage residency {self.label} -> gpu "
            f"elapsed_seconds={time.perf_counter() - started_at_seconds:.4f} "
            f"({gpu_memory_note(self.device)})"
        )
