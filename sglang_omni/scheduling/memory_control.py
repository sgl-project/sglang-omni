# SPDX-License-Identifier: Apache-2.0
"""Coordinate memory transitions across schedulers sharing an allocator."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from sglang_omni.proto.admin import (
    ADMIN_RELEASE_MEMORY_OCCUPATION,
    ADMIN_RESUME_MEMORY_OCCUPATION,
)

logger = logging.getLogger(__name__)


class WorkerMemoryControl:
    """Freeze every scheduler before touching process-wide allocation tags."""

    def __init__(
        self,
        handlers: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
        *,
        worker: str,
    ) -> None:
        if not handlers:
            raise ValueError("Memory control requires at least one scheduler")
        self.handlers = dict(handlers)
        self.worker = worker
        self.lock = threading.Lock()

    def dispatch(
        self, stage: str, phase: str, action: str, tags: list[str]
    ) -> dict[str, Any]:
        result = self.handlers[stage]({"phase": phase, "action": action, "tags": tags})
        if not result["success"]:
            raise RuntimeError(
                f"Stage {stage}: {result.get('error') or result['message']}"
            )
        return result["data"]

    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action not in {
            ADMIN_RELEASE_MEMORY_OCCUPATION,
            ADMIN_RESUME_MEMORY_OCCUPATION,
        }:
            raise ValueError(f"Unsupported memory action: {action}")
        all_tags = {"weights", "kv_cache", "cuda_graph"}
        raw_tags = payload.get("tags")
        if raw_tags is not None and (
            not isinstance(raw_tags, list)
            or any(not isinstance(tag, str) or tag not in all_tags for tag in raw_tags)
        ):
            raise ValueError("tags must contain weights, kv_cache, or cuda_graph")
        requested = sorted(set(raw_tags or all_tags))
        releasing = action == ADMIN_RELEASE_MEMORY_OCCUPATION
        with self.lock:
            prepared: dict[str, dict[str, Any]] = {}
            attempted: list[str] = []
            mutated = False
            try:
                for stage in self.handlers:
                    attempted.append(stage)
                    prepared[stage] = self.dispatch(stage, "prepare", action, requested)
                first = next(iter(prepared))
                tags = prepared[first]["tags"]
                if any(state != prepared[first] for state in prepared.values()):
                    raise RuntimeError("Worker stages have inconsistent memory state")
                if tags:
                    mutated = True
                    if not releasing:
                        self.dispatch(first, "apply", action, tags)
                    for stage in prepared:
                        self.dispatch(stage, "buffers", action, tags)
                    if releasing:
                        self.dispatch(first, "apply", action, tags)
                results = {
                    stage: self.dispatch(stage, "commit", action, tags)
                    for stage in prepared
                }
            except Exception:
                for stage in attempted:
                    try:
                        self.dispatch(
                            stage, "fail" if mutated else "cancel", action, []
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to suspend memory control for {stage}"
                        )
                raise
            return {
                "success": True,
                "message": "memory released" if releasing else "memory resumed",
                "data": {
                    "released_memory_tags": results[first]["released_memory_tags"],
                    "engine_paused": any(
                        state["engine_paused"] for state in results.values()
                    ),
                    "worker": self.worker,
                    "affected_stages": list(self.handlers),
                    "stage_states": results,
                },
            }
