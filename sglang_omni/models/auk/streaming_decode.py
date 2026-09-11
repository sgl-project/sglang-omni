# SPDX-License-Identifier: Apache-2.0
"""Audio output streaming after complete AuK latent generation."""

import time
from contextlib import nullcontext

import torch

from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.scheduling.messages import OutgoingMessage
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
        compute_batch,
        *,
        vae,
        device,
        chunk_frames,
        max_batch_size=4,
        max_batch_wait_ms=10,
    ):
        self._vae = vae
        self._device = device
        self._chunk_frames = chunk_frames
        self._decode_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._closing = False

        @torch.inference_mode()
        def run(payloads):
            with self._device_context():
                return compute_batch(payloads)

        super().__init__(
            lambda payload: run([payload])[0],
            batch_compute_fn=run,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_wait_when_idle=False,
        )

    def _device_context(self):
        return (
            torch.cuda.stream(self._decode_stream)
            if self._decode_stream is not None
            else nullcontext()
        )

    @staticmethod
    def _wants_stream(msg):
        return bool((msg.data.request.params or {}).get("stream", False))

    def _run_batch(self, batch, loop):
        if not any(self._wants_stream(msg) for msg in batch):
            return super()._run_batch(batch, loop)
        # A failure in one streamed request must not error earlier completed
        # requests in this mixed batch. Non-stream requests remain non-streaming.
        for msg in batch:
            try:
                self._run_single(msg, loop)
            except Exception as exc:
                if not self._consume_if_aborted(msg.request_id):
                    self._emit_error(msg.request_id, exc, self.outbox)

    def _run_single(self, msg, loop):
        if not self._wants_stream(msg):
            return super()._run_single(msg, loop)
        if self._consume_if_aborted(msg.request_id) or self._closing:
            return
        payload = msg.data
        state = load_state(payload, AuKState)
        chunks = None
        started = time.perf_counter()
        try:
            if self._chunk_frames <= 0:
                raise ValueError(
                    "AuK stream=true requires decode.factory.chunk_frames > 0"
                )
            with torch.inference_mode(), self._device_context():
                latents = state.latent.unsqueeze(0).to(self._device)
                chunks = self._vae.iter_decode_chunks(latents, self._chunk_frames)
                count = 0
                while not self._closing:
                    if self._consume_if_aborted(msg.request_id):
                        return
                    try:
                        waveform = next(chunks)
                    except StopIteration:
                        break
                    if self._consume_if_aborted(msg.request_id) or self._closing:
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
                if self._closing or self._consume_if_aborted(msg.request_id):
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
            self._emit_result(msg.request_id, result, self.outbox)
        finally:
            if chunks is not None:
                chunks.close()
            state.latent = None
            payload.data.pop("latent", None)

    def stop(self):
        self._closing = True
        super().stop()
