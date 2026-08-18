# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

logger = logging.getLogger(__name__)


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
            return self._ar_active

    def begin_dit_handoff(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if self._ar_model is None:
                raise RuntimeError(
                    "MiniMax Music 3 serial offload is enabled but the AR "
                    "backbone was never registered"
                )
            if not self._ar_active:
                return
            move_module_to_device(self._ar_model, torch.device("cpu"))
            self._ar_active = False
        logger.info("MiniMax Music 3 serial offload: AR -> CPU (DIT/DAV's turn)")

    def end_dit_handoff(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if self._ar_model is None:
                raise RuntimeError(
                    "MiniMax Music 3 serial offload is enabled but the AR "
                    "backbone was never registered"
                )
            if self._ar_active:
                return
            move_module_to_device(self._ar_model, self._ar_device)
            self._ar_active = True
        logger.info("MiniMax Music 3 serial offload: AR -> GPU (AR's turn)")


_COORDINATOR = SerialOffloadCoordinator()


def get_coordinator() -> SerialOffloadCoordinator:
    return _COORDINATOR


__all__ = ["SerialOffloadCoordinator", "get_coordinator", "move_module_to_device"]
