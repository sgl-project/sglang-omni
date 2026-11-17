# SPDX-License-Identifier: Apache-2.0
"""Serial stage execution with cancellation-safe GPU ownership."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from sglang_omni.models.auk.batch_compute import BatchCompute
from sglang_omni.models.auk.serial_offload import AuKOffloadCoordinator, StageName
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class AuKOffloadScheduler(SimpleScheduler[StagePayload, StagePayload]):
    """Hold request ownership until the terminal result has been routed."""

    def __init__(
        self,
        compute_batch: BatchCompute,
        coordinator: AuKOffloadCoordinator,
        stage: StageName,
    ) -> None:
        self.compute_batch: BatchCompute = compute_batch
        self.coordinator: AuKOffloadCoordinator = coordinator
        self.stage: StageName = stage
        self.stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=coordinator.device)
            if coordinator.device.type == "cuda"
            else None
        )
        super().__init__(self.execute, abort_callback=coordinator.cancel)
        coordinator.inboxes.append(self.inbox)

    @torch.inference_mode()
    def execute(self, payload: StagePayload) -> StagePayload:
        self.coordinator.begin(payload.request_id, self.stage, self.is_aborted)
        succeeded = False
        try:
            with (
                torch.cuda.stream(self.stream)
                if self.stream is not None
                else nullcontext()
            ):
                result = self.compute_batch([payload])[0]
            succeeded = True
            return result
        finally:
            try:
                if self.stream is not None:
                    torch.cuda.synchronize(self.coordinator.device)
                else:
                    pass
            except Exception as failure:
                self.coordinator.fail(failure)
                raise
            finally:
                self.coordinator.finish(
                    payload.request_id, self.stage, succeeded=succeeded
                )

    def next_message(self) -> IncomingMessage:
        # note (guozhihao-224): Transition failures must escape SimpleScheduler's per-request catch.
        self.coordinator.require_healthy()
        message = super().next_message()
        self.coordinator.require_healthy()
        return message

    def stop(self) -> None:
        self.coordinator.stop()
        super().stop()

    def claim_result(self, result: object, *, terminal: bool) -> bool:
        if not isinstance(result, StagePayload):
            raise TypeError("AuK scheduler results must be StagePayload")
        elif terminal != (self.stage == "decode"):
            raise RuntimeError(
                "AuK offload requires conditioning -> auk_engine -> decode"
            )
        else:
            return self.coordinator.claim_result(result.request_id)

    def release_result(self, result: object, *, delivered: bool) -> None:
        if not isinstance(result, StagePayload):
            raise TypeError("AuK scheduler results must be StagePayload")
        else:
            self.coordinator.release_result(
                result.request_id, terminal=self.stage == "decode", delivered=delivered
            )
