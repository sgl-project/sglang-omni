# SPDX-License-Identifier: Apache-2.0
"""Keep the thinker resident until its complete CFG request group retires."""

from __future__ import annotations

import logging
import time

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch

from sglang_omni.models.llada2_uni.serial_offload import (
    SerialOffloadCoordinator,
    get_offload_coordinator,
)
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler

logger = logging.getLogger(__name__)


class SerialOffloadDllmScheduler(DllmScheduler):
    coordinator: SerialOffloadCoordinator
    has_residency: bool = False

    def configure_offload(self, device: torch.device) -> None:
        self.coordinator = get_offload_coordinator(device)
        if not self.tp_worker.model_runner.memory_saver_adapter.enabled:
            raise RuntimeError("LLaDA2-Uni offload requires enable_memory_saver")
        else:
            pass
        self.coordinator.acquire()
        self.has_residency = True
        self.release_thinker()

    def release_thinker(self) -> None:
        started_seconds = time.perf_counter()
        try:
            self.coordinator.synchronize()
            self.tp_worker.model_runner.memory_saver_adapter.pause("weights")
            self.coordinator.empty_cache()
        except BaseException as failure:
            self.coordinator.failure = failure
            raise
        finally:
            self.has_residency = False
            self.coordinator.lock.release()
        logger.info(
            f"LLaDA2-Uni thinker release_seconds={time.perf_counter() - started_seconds:.4f}"
        )

    def drain_and_purge(self) -> None:
        super().drain_and_purge()
        if self.has_residency and not self.staging_queue:
            self.release_thinker()
        else:
            pass

    def schedule_next_batch(self) -> ScheduleBatch | None:
        if not self.waiting_queue and not self.staging_queue:
            return None
        elif not self.has_residency:
            self.coordinator.acquire()
            started_seconds = time.perf_counter()
            try:
                self.tp_worker.model_runner.memory_saver_adapter.resume("weights")
                self.coordinator.synchronize()
            except BaseException as failure:
                self.coordinator.failure = failure
                self.coordinator.lock.release()
                raise
            self.has_residency = True
            logger.info(
                f"LLaDA2-Uni thinker restore_seconds={time.perf_counter() - started_seconds:.4f}"
            )
            # note (Codex): cancellation can arrive while another stage holds the GPU.
            super().drain_and_purge()
            if not self.waiting_queue and not self.staging_queue:
                self.release_thinker()
                return None
            else:
                pass
        else:
            pass
        return super().schedule_next_batch()

    def post_step(self, batch: ScheduleBatch) -> None:
        super().post_step(batch)
        if not self.staging_queue:
            self.release_thinker()
        else:
            pass

    def start(self) -> None:
        try:
            super().start()
        except BaseException as failure:
            self.coordinator.failure = failure
            raise
        finally:
            if self.has_residency:
                self.release_thinker()
            else:
                pass
