# SPDX-License-Identifier: Apache-2.0
"""Serial GPU residency for the MiniMax Music 3 AR and DIT/DAV stages.

``--layerwise-offload-components ar,dit`` colocates both stages in one process
and lets only one of them hold GPU weights at a time: AR generates a whole
request, hands off to DIT/DAV, and stays off the GPU until DIT/DAV is finished
with that request.

The handoff is keyed by request id rather than by a single boolean. AR wakes
when nothing is outstanding, so a duplicated, out-of-order, or late terminal
event cannot wake AR while DIT/DAV is still decoding, and an event that names
a request nobody handed off cannot wake AR early.

A handoff that never ends stops admission for good, because ``ar_can_admit``
gates the AR scheduler's queue. That is reported rather than force-corrected:
waking AR while DIT/DAV still holds the GPU is how this configuration runs out
of memory, so a stuck server is preferable to a crashed one, and the log names
the requests still outstanding.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

STALL_REPORT_SECONDS = 900.0


def move_module_to_device(module: Any, device: torch.device) -> None:
    source_device = None
    try:
        source_device = next(module.parameters()).device
    except StopIteration:
        pass
    module.to(device)
    if source_device is not None and source_device.type == "cuda":
        torch.cuda.synchronize(source_device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.cuda.empty_cache()


class SerialOffloadCoordinator:
    """Process-wide GPU residency coordinator (see module docstring)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._ar_active = True
        self._ar_model: Any | None = None
        self._ar_device: torch.device | None = None
        self._outstanding: set[str] = set()
        self._paused_at: float | None = None
        self._stall_reported = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def register_ar(self, model: Any, device: torch.device) -> None:
        with self._lock:
            self._ar_model = model
            self._ar_device = device
            self._enabled = True
            self._ar_active = True
        logger.info(
            f"MiniMax Music 3 serial offload enabled device={device}; AR "
            "starts GPU-resident, DIT/DAV starts offloaded"
        )

    def ar_can_admit(self) -> bool:
        """Whether AR may admit a new request onto the GPU right now."""
        if not self._enabled:
            return True
        with self._lock:
            if self._ar_active:
                return True
            self._report_stall_locked()
            return False

    def begin_dit_handoff(self, request_id: str) -> None:
        """Hand the GPU to DIT/DAV for *request_id* and take AR off it."""
        if not self._enabled:
            return
        with self._lock:
            self._require_ar_locked()
            self._outstanding.add(request_id)
            if not self._ar_active:
                return
            move_module_to_device(self._ar_model, torch.device("cpu"))
            self._ar_active = False
            self._paused_at = time.monotonic()
            self._stall_reported = False
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> CPU (DIT/DAV's turn, "
            f"request={request_id})"
        )

    def end_dit_handoff(self, request_id: str) -> None:
        """Retire *request_id*; give the GPU back once nothing is outstanding.

        Safe to call for a request that never handed off, and safe to call
        more than once: both leave the outstanding set unchanged.
        """
        if not self._enabled:
            return
        with self._lock:
            self._require_ar_locked()
            self._outstanding.discard(request_id)
            if self._ar_active or self._outstanding:
                return
            move_module_to_device(self._ar_model, self._ar_device)
            self._ar_active = True
            self._paused_at = None
            self._stall_reported = False
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> GPU (AR's turn, "
            f"request={request_id})"
        )

    def _require_ar_locked(self) -> None:
        if self._ar_model is None:
            raise RuntimeError(
                "MiniMax Music 3 serial offload is enabled but the AR "
                "backbone was never registered"
            )

    def _report_stall_locked(self) -> None:
        """Name the outstanding requests once AR has been parked too long."""
        if self._paused_at is None or self._stall_reported:
            return
        elapsed = time.monotonic() - self._paused_at
        if elapsed < STALL_REPORT_SECONDS:
            return
        self._stall_reported = True
        logger.error(
            f"MiniMax Music 3 serial offload: AR has been off the GPU for "
            f"{elapsed:.0f}s and is still waiting on "
            f"{sorted(self._outstanding)}; AR admits nothing until DIT/DAV "
            "retires them, so this server needs a restart if the requests "
            "are gone"
        )


_COORDINATOR = SerialOffloadCoordinator()


def get_coordinator() -> SerialOffloadCoordinator:
    return _COORDINATOR


__all__ = [
    "STALL_REPORT_SECONDS",
    "SerialOffloadCoordinator",
    "get_coordinator",
    "move_module_to_device",
]
