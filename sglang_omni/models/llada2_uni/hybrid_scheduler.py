# SPDX-License-Identifier: Apache-2.0
"""Hybrid thinker scheduler for LLaDA2-Uni text and image requests."""

from __future__ import annotations

import logging
import queue as _queue_mod
import threading
from collections.abc import Callable
from typing import Any

from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


def is_llada2_image_generation_payload(payload: Any) -> bool:
    state = LLaDA2UniPipelineState.from_dict(getattr(payload, "data", None))
    return state.image_generation.get("type") == "image"


class LLaDA2HybridThinkerScheduler:
    """Route image requests to HF ``generate_image`` and text to SGLang dLLM."""

    def __init__(
        self,
        *,
        text_scheduler_factory: Callable[[], Any],
        image_compute_fn: Callable[[Any], Any],
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout = True

        self._text_scheduler_factory = text_scheduler_factory
        self._image_compute_fn = image_compute_fn
        self._text_scheduler: Any | None = None
        self._text_thread: threading.Thread | None = None
        self._running = False
        self._aborted: set[str] = set()
        self._abort_lock = threading.Lock()

    def _ensure_text_scheduler(self) -> Any:
        if self._text_scheduler is not None:
            return self._text_scheduler

        scheduler = self._text_scheduler_factory()
        self._text_scheduler = scheduler
        self._text_thread = threading.Thread(
            target=scheduler.start,
            name="llada2-text-dllm-scheduler",
            daemon=True,
        )
        self._text_thread.start()
        return scheduler

    def _drain_text_outbox(self) -> None:
        scheduler = self._text_scheduler
        if scheduler is None:
            return
        while True:
            try:
                msg = scheduler.outbox.get_nowait()
            except _queue_mod.Empty:
                break
            self.outbox.put(msg)

    def _consume_aborted(self, request_id: str) -> bool:
        with self._abort_lock:
            if request_id not in self._aborted:
                return False
            self._aborted.discard(request_id)
            return True

    def start(self) -> None:
        self._running = True
        while self._running:
            self._drain_text_outbox()
            try:
                msg = self.inbox.get(timeout=0.01)
            except _queue_mod.Empty:
                continue

            if msg.type != "new_request":
                self._ensure_text_scheduler().inbox.put(msg)
                continue

            if self._consume_aborted(msg.request_id):
                continue

            if not is_llada2_image_generation_payload(msg.data):
                self._ensure_text_scheduler().inbox.put(msg)
                continue

            try:
                result = self._image_compute_fn(msg.data)
            except Exception as exc:
                if self._consume_aborted(msg.request_id):
                    continue
                logger.exception(
                    "LLaDA2 image token generation failed for %s", msg.request_id
                )
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )
                continue

            if self._consume_aborted(msg.request_id):
                continue
            self.outbox.put(
                OutgoingMessage(
                    request_id=msg.request_id,
                    type="result",
                    data=result,
                )
            )

    def stop(self) -> None:
        self._running = False
        if self._text_scheduler is not None:
            self._text_scheduler.stop()

    def abort(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted.add(request_id)
        if self._text_scheduler is not None:
            self._text_scheduler.abort(request_id)
