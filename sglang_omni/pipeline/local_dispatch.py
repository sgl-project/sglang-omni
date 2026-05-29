# SPDX-License-Identifier: Apache-2.0
"""Process-local stage dispatch for same-process stage traffic."""

from __future__ import annotations

from typing import Any, Iterable


class LocalStageDispatcher:
    """Dispatch stage objects between stages in the same OS process.

    Process-local dispatch passes Python object references directly. Receivers
    must treat payloads, stream data, and metadata as read-only unless the edge
    explicitly gives them an isolated projected object.
    """

    def __init__(self) -> None:
        self._stages: dict[str, Any] = {}

    def register(self, stage: Any) -> None:
        self._stages[stage.name] = stage

    def register_many(self, stages: Iterable[Any]) -> None:
        for stage in stages:
            self.register(stage)

    def _get_stage(self, from_stage: str, to_stage: str) -> Any:
        target = self._stages.get(to_stage)
        if target is None:
            raise RuntimeError(
                f"Local stage target {to_stage!r} is not registered "
                f"for traffic from {from_stage!r}"
            )
        return target

    async def send_payload(
        self,
        *,
        from_stage: str,
        to_stage: str,
        request_id: str,
        payload: Any,
    ) -> None:
        target = self._get_stage(from_stage, to_stage)
        await target.receive_local_payload(request_id, from_stage, payload)

    async def send_stream_chunk(
        self,
        *,
        from_stage: str,
        to_stage: str,
        request_id: str,
        chunk_id: int,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        target = self._get_stage(from_stage, to_stage)
        await target.receive_local_stream_chunk(
            request_id,
            from_stage,
            chunk_id,
            data,
            metadata,
        )

    async def send_stream_signal(
        self,
        *,
        from_stage: str,
        to_stage: str,
        request_id: str,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        target = self._get_stage(from_stage, to_stage)
        await target.receive_local_stream_signal(
            request_id,
            from_stage,
            is_done=is_done,
            error=error,
        )
