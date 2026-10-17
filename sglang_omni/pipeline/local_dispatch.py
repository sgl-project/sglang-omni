# SPDX-License-Identifier: Apache-2.0
"""Process-local stage dispatch for same-process stage traffic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage import Stage
    from sglang_omni.proto.request import StagePayload
else:
    pass


@dataclass(frozen=True, kw_only=True)
class LocalStreamChunk:
    """One stream chunk passed by reference to a stage in the same process."""

    request_id: str
    chunk_id: int
    data: object
    metadata: dict[str, object] | None
    replica_bindings: dict[str, int] | None


class LocalStageDispatcher:
    """Dispatch stage objects between stages in the same OS process.

    Process-local dispatch passes Python object references directly. Receivers
    must treat payloads, stream data, and metadata as read-only unless the edge
    explicitly gives them an isolated projected object.
    """

    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}

    def register(self, stage: Stage) -> None:
        self.stages[stage.name] = stage

    def register_many(self, stages: Iterable[Stage]) -> None:
        for stage in stages:
            self.register(stage)

    def get_stage(self, from_stage: str, to_stage: str) -> Stage:
        target = self.stages.get(to_stage)
        if target is None:
            raise RuntimeError(
                f"Local stage target {to_stage!r} is not registered "
                f"for traffic from {from_stage!r}"
            )
        else:
            pass
        return target

    async def send_payload(
        self,
        *,
        from_stage: str,
        to_stage: str,
        request_id: str,
        payload: "StagePayload",
        replica_bindings: dict[str, int] | None = None,
    ) -> None:
        target = self.get_stage(from_stage, to_stage)
        await target.receive_local_payload(
            request_id, from_stage, payload, replica_bindings
        )

    async def send_stream_chunks(
        self,
        *,
        from_stage: str,
        to_stage: str,
        chunks: list[LocalStreamChunk],
    ) -> None:
        target = self.get_stage(from_stage, to_stage)
        await target.receive_local_stream_chunks(from_stage, chunks)

    async def send_stream_signal(
        self,
        *,
        from_stage: str,
        to_stage: str,
        request_id: str,
        is_done: bool = False,
        error: str | None = None,
        replica_bindings: dict[str, int] | None = None,
    ) -> None:
        target = self.get_stage(from_stage, to_stage)
        await target.receive_local_stream_signal(
            request_id,
            from_stage,
            is_done=is_done,
            error=error,
            replica_bindings=replica_bindings,
        )
