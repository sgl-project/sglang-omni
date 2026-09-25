# SPDX-License-Identifier: Apache-2.0
"""Audio output streaming after complete AuK latent generation."""

from __future__ import annotations

import asyncio
import time
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable

import torch

from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


class AuKDecodeScheduler(SimpleScheduler):
    """Keep ordinary batching; stream requests decode serially in this worker.

    Cancellation is cooperative between VAE windows, not kernel preemption.
    The stage runtime drains stream messages while this worker decodes.
    """

    def __init__(
        self,
        compute_batch: Callable[[list[Any]], list[Any]],
        *,
        vae: BigVGANFlowVAE,
        device: torch.device,
        chunk_frames: int,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 10,
    ) -> None:
        self.vae = vae
        self.device = device
        self.chunk_frames = chunk_frames
        self.decode_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.closing = False

        @torch.inference_mode()
        def run(payloads: list[Any]) -> list[Any]:
            with self.device_context():
                return compute_batch(payloads)

        super().__init__(
            lambda payload: run([payload])[0],
            batch_compute_fn=run,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_wait_when_idle=False,
        )

    def device_context(self) -> AbstractContextManager[Any]:
        return (
            torch.cuda.stream(self.decode_stream)
            if self.decode_stream is not None
            else nullcontext()
        )

    @staticmethod
    def wants_stream(msg: IncomingMessage) -> bool:
        return bool((msg.data.request.params or {}).get("stream", False))

    def run_batch(
        self, batch: list[IncomingMessage], loop: asyncio.AbstractEventLoop
    ) -> None:
        if not any(self.wants_stream(msg) for msg in batch):
            return super().run_batch(batch, loop)
        # A failure in one streamed request must not error earlier completed
        # requests in this mixed batch. Non-stream requests remain non-streaming.
        for msg in batch:
            try:
                self.run_single(msg, loop)
            except Exception as exc:
                if not self.consume_if_aborted(msg.request_id):
                    self.emit_error(msg.request_id, exc, self.outbox)

    def run_single(self, msg: IncomingMessage, loop: asyncio.AbstractEventLoop) -> None:
        if not self.wants_stream(msg):
            return super().run_single(msg, loop)
        if self.consume_if_aborted(msg.request_id) or self.closing:
            return
        payload = msg.data
        state = load_state(payload, AuKState)
        chunks = None
        started = time.perf_counter()
        try:
            if self.chunk_frames <= 0:
                raise ValueError(
                    "AuK stream=true requires decode.factory.chunk_frames > 0"
                )
            with torch.inference_mode(), self.device_context():
                latents = state.latent.unsqueeze(0).to(self.device)
                chunks = self.vae.iter_decode_chunks(latents, self.chunk_frames)
                count = 0
                while not self.closing:
                    if self.consume_if_aborted(msg.request_id):
                        return
                    try:
                        waveform = next(chunks)
                    except StopIteration:
                        break
                    if self.consume_if_aborted(msg.request_id) or self.closing:
                        return
                    if not torch.isfinite(waveform).all():
                        raise RuntimeError("AuK generated audio contains NaN/Inf")
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=msg.request_id,
                            type="stream",
                            data=audio_waveform_payload(
                                waveform,
                                sample_rate=state.sample_rate,
                                modality="audio",
                                source_hint="AuK",
                            ),
                            metadata={"modality": "audio"},
                        )
                    )
                    count += 1
                if self.closing or self.consume_if_aborted(msg.request_id):
                    return
                if count == 0:
                    raise RuntimeError("AuK decoder produced no audio chunks")
            state.latent = None
            state.engine_time_s += time.perf_counter() - started
            result = store_state(payload, state)
            result.data.update(
                sample_rate=state.sample_rate,
                modality="audio",
                usage=build_usage(state),
            )
            # Metadata only: emitted audio must not be sent again at completion.
            self.emit_result(msg.request_id, result, self.outbox)
        finally:
            if chunks is not None:
                chunks.close()
            state.latent = None
            payload.data.pop("latent", None)

    def stop(self) -> None:
        self.closing = True
        super().stop()
