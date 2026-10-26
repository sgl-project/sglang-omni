# SPDX-License-Identifier: Apache-2.0
"""Serial audio generation with frame-boundary cancellation and backpressure."""

from __future__ import annotations

import logging
import queue
import threading
from collections import OrderedDict
from typing import Generator, Protocol

import numpy as np
from numpy.typing import NDArray

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


class AudioOutbox(queue.Queue[OutgoingMessage]):
    """Bound audio/result buffering while allowing small admission errors.

    Stage calls enqueue on its asyncio thread. Rejection must neither block
    that thread nor raise out of Stage's receive loop when audio is backed up.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.buffered_chunks = 0

    def _put(
        self, message: OutgoingMessage
    ) -> None:  # noqa: leading-underscore - queue.Queue hook
        if message.type not in {"error", "wakeup"}:
            if self.buffered_chunks >= self.capacity:
                raise queue.Full
            else:
                pass
            self.buffered_chunks += 1
        else:
            pass
        super()._put(message)  # noqa: leading-underscore - queue.Queue hook

    def _get(self) -> OutgoingMessage:  # noqa: leading-underscore - queue.Queue hook
        message = super()._get()  # noqa: leading-underscore - queue.Queue hook
        if message.type not in {"error", "wakeup"}:
            self.buffered_chunks -= 1
        else:
            pass
        return message


class AudioBackend(Protocol):
    sample_rate: int
    max_samples: int

    def iter_audio(
        self, payload: StagePayload
    ) -> Generator[NDArray[np.float32], None, None]: ...


class VoXtreamScheduler:
    """One active generator, with no interleaving of model or codec state.

    Use enqueue() for admission, as Stage does. stop() and abort() only signal
    cancellation; generator cleanup belongs to the scheduler thread. They
    cannot interrupt an already executing model frame or prompt preparation.
    This first integration accepts complete text, not incremental input.
    """

    requires_tp_work_fanout = True

    def warm_up_serving_thread(self) -> None:
        """Model initialization belongs to the isolated worker process."""

    def __init__(
        self,
        backend: AudioBackend,
        *,
        max_pending: int,
        output_buffer_chunks: int,
    ) -> None:
        if max_pending < 1 or output_buffer_chunks < 1:
            raise ValueError("Queue limits must be positive")
        else:
            pass
        if backend.sample_rate <= 0 or backend.max_samples <= 0:
            raise ValueError("Backend audio limits must be positive")
        else:
            pass
        self.backend = backend
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue(max_pending)
        self.outbox: queue.Queue[OutgoingMessage] = AudioOutbox(output_buffer_chunks)
        self.max_pending_requests = max_pending
        self.lock = threading.Lock()
        self.request_cancellations: dict[str, threading.Event] = {}
        self.early_aborts: OrderedDict[str, None] = OrderedDict()
        self.stopped = threading.Event()

    def enqueue(self, msg: IncomingMessage) -> None:
        if msg.type != "new_request":
            raise ValueError("VoXtream2 currently accepts complete text requests only")
        else:
            pass
        if (
            not isinstance(msg.data, StagePayload)
            or msg.data.request_id != msg.request_id
        ):
            raise ValueError("Expected a StagePayload with a matching request id")
        else:
            pass
        with self.lock:
            if self.stopped.is_set():
                raise RuntimeError("Scheduler has stopped")
            else:
                pass
            if msg.request_id in self.request_cancellations:
                raise ValueError("Request id is already active or queued")
            else:
                pass
            if msg.request_id in self.early_aborts:
                self.early_aborts.pop(msg.request_id)
                return
            else:
                pass
            if len(self.request_cancellations) >= self.max_pending_requests:
                self.outbox.put_nowait(
                    OutgoingMessage(
                        msg.request_id,
                        "error",
                        QueueFullError(),
                    )
                )
                return
            else:
                pass
            self.inbox.put_nowait(msg)
            self.request_cancellations[msg.request_id] = threading.Event()

    def abort(self, request_id: str) -> None:
        with self.lock:
            cancelled = self.request_cancellations.get(request_id)
            if cancelled is not None:
                cancelled.set()
            else:
                self.early_aborts[request_id] = None
                # Bound unknown/late aborts; admitted requests have their own
                # events and are never evicted from cancellation tracking.
                if len(self.early_aborts) > 1024:
                    self.early_aborts.popitem(last=False)
                else:
                    pass

        cancel = getattr(self.backend, "cancel", None)
        if cancel is not None:
            cancel(request_id)
        else:
            pass

    def stop(self) -> None:
        with self.lock:
            self.stopped.set()
            for cancelled in self.request_cancellations.values():
                cancelled.set()

        cancel_active = getattr(self.backend, "cancel_active", None)
        if cancel_active is not None:
            cancel_active()
        else:
            pass

    def emit(self, message: OutgoingMessage, cancelled: threading.Event) -> bool:
        while not self.stopped.is_set():
            # No output is enqueued after abort() returns. Already queued
            # messages are still filtered by the Stage request lifecycle.
            with self.lock:
                if cancelled.is_set() or self.stopped.is_set():
                    return False
                else:
                    pass
                try:
                    self.outbox.put_nowait(message)
                    return True
                except queue.Full:
                    pass
            self.stopped.wait(0.01)
        return False

    def start(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    msg = self.inbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                with self.lock:
                    cancelled = self.request_cancellations[msg.request_id]
                try:
                    if not cancelled.is_set():
                        self.run_request(msg, cancelled)
                    else:
                        pass
                except Exception as exc:
                    logger.exception(f"VoXtream2 request {msg.request_id} failed")
                    self.emit(OutgoingMessage(msg.request_id, "error", exc), cancelled)
                finally:
                    with self.lock:
                        self.request_cancellations.pop(msg.request_id, None)
        finally:
            with self.lock:
                self.request_cancellations.clear()
                self.early_aborts.clear()
                while not self.inbox.empty():
                    self.inbox.get_nowait()

            close = getattr(self.backend, "close", None)
            if close is not None:
                close()
            else:
                pass

    def run_request(self, msg: IncomingMessage, cancelled: threading.Event) -> None:
        payload = msg.data
        streaming = bool(payload.request.params.get("stream", False))
        chunks = []
        samples = 0
        iterator = self.backend.iter_audio(payload)
        try:
            while not cancelled.is_set() and not self.stopped.is_set():
                try:
                    audio = next(iterator)
                except StopIteration:
                    break
                if cancelled.is_set() or self.stopped.is_set():
                    return
                else:
                    pass
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                    raise ValueError("Backend must produce finite non-empty mono audio")
                else:
                    pass
                samples += audio.size
                if samples > self.backend.max_samples:
                    raise ValueError("Backend exceeded its configured audio limit")
                else:
                    pass
                if streaming:
                    data = audio_waveform_payload(
                        audio, sample_rate=self.backend.sample_rate, modality="audio"
                    )
                    if not self.emit(
                        OutgoingMessage(
                            msg.request_id,
                            "stream",
                            data,
                            metadata={"modality": "audio"},
                        ),
                        cancelled,
                    ):
                        return
                    else:
                        pass
                else:
                    # A backend may reuse its numpy buffer on the next frame.
                    chunks.append(audio.copy())
        finally:
            iterator.close()
        if not samples and not cancelled.is_set() and not self.stopped.is_set():
            raise ValueError("VoXtream2 produced no audio")
        else:
            pass
        if cancelled.is_set() or self.stopped.is_set():
            return
        else:
            pass
        data = {"sample_rate": self.backend.sample_rate, "modality": "audio"}
        if not streaming:
            data.update(audio_waveform_payload(np.concatenate(chunks)))
        else:
            pass
        result = StagePayload(msg.request_id, payload.request, data)
        self.emit(OutgoingMessage(msg.request_id, "result", result), cancelled)
