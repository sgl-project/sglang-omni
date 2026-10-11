# SPDX-License-Identifier: Apache-2.0
"""Serial GPU residency for MiniMax Music 3 AR and DIT/DAV."""

from __future__ import annotations

import logging
import threading
import time
from importlib.metadata import version
from typing import Literal, Protocol

import torch
from torch_memory_saver import torch_memory_saver

from sglang_omni.models.minimax_music3.weight_cache import RuntimeWeights, StagingRing

logger = logging.getLogger(__name__)

STALL_REPORT_SECONDS = 900.0
BYTES_PER_GIB = 1024.0**3


class AcousticRelease(Protocol):
    def __call__(self) -> None: ...


def gpu_memory_note(device: torch.device) -> str:
    """Report allocator and device occupancy; offload leaves the KV cache resident."""
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
        self.tags = tags
        self.staging_ring = staging_ring
        if self.device.type == "cuda" and (not tags or staging_ring is None):
            raise ValueError(
                "CUDA residency requires managed allocation tags and a staging ring"
            )
        else:
            pass
        self.weights = RuntimeWeights(
            self.modules,
            source=source,
            cache_dir=cache_dir,
            checkpoint_contents=checkpoint_contents or {},
            folding_backend=folding_backend,
        )
        weight_bytes = sum(storage.numel() for storage in self.weights.storages)
        logger.info(
            f"MiniMax Music 3 residency {label}: {weight_bytes / BYTES_PER_GIB:.2f}GiB of "
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
            torch_memory_saver.pause(tag=tag)
        self.is_resident = False
        logger.info(
            f"MiniMax Music 3 residency {self.label} -> host "
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
            torch_memory_saver.resume(tag=tag)
        self.weights.restore(self.staging_ring)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        self.is_resident = True
        logger.info(
            f"MiniMax Music 3 residency {self.label} -> gpu "
            f"elapsed_seconds={time.perf_counter() - started_at_seconds:.4f} "
            f"({gpu_memory_note(self.device)})"
        )


class SerialOffloadCoordinator:
    """Coordinate request-scoped AR and DIT/DAV GPU handoffs."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.is_enabled: bool = False
        self.ar_residency: StageResidency | None = None
        self.owner_request_id: str | None = None
        self.phase: Literal["idle", "ar", "acoustic", "failed"] = "idle"
        self.failure: Exception | None = None
        self.paused_at_seconds: float | None = None
        self.has_reported_stall: bool = False
        self.staging_ring: StagingRing | None = None

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    def register_ar(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        source: Literal["mmap", "ram"] = "ram",
        cache_dir: str | None = None,
        checkpoint_contents: dict[str, str] | None = None,
    ) -> None:
        with self.lock:
            if self.is_enabled:
                raise RuntimeError("MiniMax Music 3 serial offload is already enabled")
            else:
                pass
            if device.type == "cuda":
                if version("torch_memory_saver") != "0.0.10":
                    raise RuntimeError(
                        "Music3 CUDA offload requires torch_memory_saver==0.0.10"
                    )
                else:
                    pass
                self.staging_ring = StagingRing(device)
                tags = ("weights", "music3_audio")
            else:
                tags = ()
            self.ar_residency = StageResidency(
                {"ar": model},
                device,
                label="ar",
                tags=tags,
                source=source,
                cache_dir=cache_dir,
                checkpoint_contents=checkpoint_contents,
                staging_ring=self.staging_ring,
            )
            self.is_enabled = True
        logger.info(
            f"MiniMax Music 3 serial offload enabled device={device}; AR "
            "starts GPU-resident, DIT/DAV starts offloaded"
        )

    def pause_ar_for_startup(self) -> None:
        with self.lock:
            self.require_ar_locked()
            try:
                self.ar_residency.sleep()
            except Exception as failure:
                self.failure = failure
                self.phase = "failed"
                raise
            self.phase = "acoustic"

    def restore_ar_after_startup(self) -> None:
        with self.lock:
            self.require_ar_locked()
            try:
                self.ar_residency.wake()
            except Exception as failure:
                self.failure = failure
                self.phase = "failed"
                raise
            self.phase = "idle"

    def ar_can_admit(self) -> bool:
        """Whether AR may admit a new request onto the GPU right now."""
        if not self.is_enabled:
            return True
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            if self.phase == "idle":
                return True
            else:
                pass
            self.report_stall_locked()
            return False

    def try_acquire_ar(self, request_id: str) -> bool:
        """Claim one request until its AR and acoustic work have retired."""
        if not self.is_enabled:
            return True
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            if self.owner_request_id == request_id and self.phase == "ar":
                return True
            elif self.phase == "idle":
                self.owner_request_id = request_id
                self.phase = "ar"
                return True
            else:
                self.report_stall_locked()
                return False

    def begin_dit_handoff(self, request_id: str) -> None:
        """Hand the GPU to DIT/DAV for *request_id* and take AR off it."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            if self.owner_request_id != request_id:
                raise RuntimeError(
                    f"MiniMax Music 3 handoff request={request_id!r} does not own "
                    f"residency (owner={self.owner_request_id!r})"
                )
            elif self.phase == "acoustic":
                return
            else:
                pass
            try:
                self.ar_residency.sleep()
            except Exception as failure:
                self.failure = failure
                self.phase = "failed"
                raise
            self.phase = "acoustic"
            self.paused_at_seconds = time.monotonic()
            self.has_reported_stall = False
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> CPU (DIT/DAV's turn, "
            f"request={request_id})"
        )

    def require_acoustic(self, request_id: str) -> None:
        """Reject acoustic compute outside the owning request's handoff."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            if self.owner_request_id != request_id or self.phase != "acoustic":
                raise RuntimeError(
                    f"MiniMax Music 3 acoustic request={request_id!r} does not own "
                    f"residency (owner={self.owner_request_id!r}, phase={self.phase})"
                )
            else:
                pass

    def fail_transition(self, failure: Exception) -> None:
        with self.lock:
            self.failure = failure
            self.phase = "failed"

    def end_dit_handoff(
        self,
        request_id: str,
        *,
        release_acoustic: AcousticRelease | None = None,
    ) -> None:
        """Release acoustic weights and restore AR only for the current owner."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            if self.owner_request_id != request_id or self.phase != "acoustic":
                return
            else:
                pass
            try:
                if release_acoustic is not None:
                    release_acoustic()
                else:
                    pass
                self.ar_residency.wake()
            except Exception as failure:
                self.failure = failure
                self.phase = "failed"
                raise
            self.owner_request_id = None
            self.phase = "idle"
            self.paused_at_seconds = None
            self.has_reported_stall = False
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> GPU (AR's turn, "
            f"request={request_id})"
        )

    def cancel_ar(self, request_id: str) -> None:
        """Release an AR owner after scheduler compute stops, before handoff."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.lock:
            if self.owner_request_id == request_id and self.phase == "ar":
                self.owner_request_id = None
                self.phase = "idle"
            else:
                pass

    def require_ar_locked(self) -> None:
        if self.failure is not None:
            raise RuntimeError(
                "MiniMax Music 3 serial offload is unavailable after a residency "
                "transition failed"
            ) from self.failure
        elif self.ar_residency is None:
            raise RuntimeError(
                "MiniMax Music 3 serial offload is enabled but the AR "
                "backbone was never registered"
            )
        else:
            pass

    def report_stall_locked(self) -> None:
        """Name the outstanding requests once AR has been parked too long."""
        if self.paused_at_seconds is None or self.has_reported_stall:
            return
        else:
            pass
        elapsed_seconds = time.monotonic() - self.paused_at_seconds
        if elapsed_seconds < STALL_REPORT_SECONDS:
            return
        else:
            pass
        self.has_reported_stall = True
        logger.error(
            f"MiniMax Music 3 serial offload: AR has been off the GPU for "
            f"{elapsed_seconds:.0f}s and is still waiting on "
            f"{self.owner_request_id!r}; AR admits nothing until DIT/DAV "
            "retires them, so this server needs a restart if the requests "
            "are gone"
        )


COORDINATOR = SerialOffloadCoordinator()


def get_coordinator() -> SerialOffloadCoordinator:
    return COORDINATOR


__all__ = [
    "STALL_REPORT_SECONDS",
    "StageResidency",
    "SerialOffloadCoordinator",
    "get_coordinator",
]
