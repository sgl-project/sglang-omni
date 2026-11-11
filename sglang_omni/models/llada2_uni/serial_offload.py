# SPDX-License-Identifier: Apache-2.0
"""Exclusive CUDA residency for the single-image LLaDA2-Uni pipeline."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


class SerialOffloadCoordinator:
    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device
        self.lock: threading.Lock = threading.Lock()
        self.failure: BaseException | None = None

    def acquire(self) -> None:
        self.lock.acquire()
        if self.failure is not None:
            self.lock.release()
            raise RuntimeError(
                "LLaDA2-Uni offload is unavailable after a residency transition failed"
            ) from self.failure
        else:
            pass

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass

    def empty_cache(self) -> None:
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        else:
            pass

    @contextmanager
    def use_module(self, module: torch.nn.Module, label: str) -> Iterator[None]:
        self.acquire()
        try:
            started_seconds = time.perf_counter()
            try:
                module.to(self.device)
                self.synchronize()
            except BaseException as failure:
                self.failure = failure
                raise
            logger.info(
                f"LLaDA2-Uni {label} restore_seconds={time.perf_counter() - started_seconds:.4f}"
            )
            try:
                yield
            finally:
                started_seconds = time.perf_counter()
                try:
                    self.synchronize()
                    module.to("cpu")
                    self.empty_cache()
                except BaseException as failure:
                    self.failure = failure
                    raise
                logger.info(
                    f"LLaDA2-Uni {label} release_seconds={time.perf_counter() - started_seconds:.4f}"
                )
        finally:
            self.lock.release()


@lru_cache(maxsize=None)
def get_offload_coordinator(device: torch.device) -> SerialOffloadCoordinator:
    """Share a residency lock between stages colocated in the same process."""
    return SerialOffloadCoordinator(device)
