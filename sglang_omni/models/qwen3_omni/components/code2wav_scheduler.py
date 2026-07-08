# SPDX-License-Identifier: Apache-2.0
"""Code2Wav scheduler — streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), accumulates them,
runs vocoder incrementally, outputs final audio via outbox.
"""

from __future__ import annotations

import logging
import os
import queue as _queue_mod
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


@dataclass
class _DecodePlan:
    request_id: str
    start: int
    end: int
    trim: int
    codes: torch.Tensor


def load_code2wav_model(
    model_path: str, *, device: str = "cuda", dtype: str | None = None
):
    """Load Code2Wav model from HF checkpoint."""
    from transformers import AutoConfig

    from sglang_omni.models.weight_loader import load_module, resolve_dtype

    torch_dtype = resolve_dtype(dtype)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    code2wav_config = config.code2wav_config

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model = Qwen3OmniMoeCode2Wav._from_config(code2wav_config)
    model = load_module(
        model,
        model_path,
        prefix="code2wav.",
        dtype=torch_dtype,
        device=device,
        strict=False,
    )
    return model


class Code2WavScheduler(StreamingSimpleScheduler):
    """Streaming vocoder scheduler. Same inbox/outbox interface as OmniScheduler."""

    def __init__(
        self,
        model: Any,
        device: str,
        stream_chunk_size: int = 10,
        non_stream_chunk_size: int | None = None,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
        max_batch_size: int = 16,
        max_batch_wait_ms: int = 1,
        enable_batched_decode: bool = True,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        env_non_stream_chunk_size = os.getenv(
            "SGLANG_OMNI_QWEN3_CODE2WAV_NON_STREAM_CHUNK_SIZE"
        )
        if env_non_stream_chunk_size is not None:
            non_stream_chunk_size = int(env_non_stream_chunk_size)
        self._non_stream_chunk_size = (
            self._stream_chunk_size
            if non_stream_chunk_size is None
            else max(int(non_stream_chunk_size), 1)
        )
        env_left_context_size = os.getenv(
            "SGLANG_OMNI_QWEN3_CODE2WAV_LEFT_CONTEXT_SIZE"
        )
        if env_left_context_size is not None:
            left_context_size = int(env_left_context_size)
        self._left_context_size = max(int(left_context_size), 0)
        self._sample_rate = sample_rate
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(model.total_upsample)
        self._decode_max_batch_size = max(int(max_batch_size), 1)
        env_batch_wait_ms = os.getenv(
            "SGLANG_OMNI_QWEN3_CODE2WAV_MAX_BATCH_WAIT_MS"
        )
        if env_batch_wait_ms is not None:
            max_batch_wait_ms = float(env_batch_wait_ms)
        self._decode_max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._enable_batched_decode = bool(enable_batched_decode)

        # Per-request state
        self._code_chunks: dict[str, list[torch.Tensor]] = {}
        self._emitted: dict[str, int] = {}
        self._audio_chunks: dict[str, list[np.ndarray]] = {}
        self._stream_enabled: dict[str, bool] = {}
        super().__init__(compute_fn=None)
        self._payloads = self._stream_payloads

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        del payload
        self._ensure_request_state(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        self._code_chunks.pop(request_id, None)
        self._emitted.pop(request_id, None)
        self._audio_chunks.pop(request_id, None)
        self._stream_enabled.pop(request_id, None)

    def _fail_request(self, request_id: str, error: Exception) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )
        self.abort(request_id)

    def _ensure_request_state(self, request_id: str) -> None:
        if request_id in self._code_chunks:
            return
        self._code_chunks[request_id] = []
        self._emitted[request_id] = 0
        self._audio_chunks[request_id] = []

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        self._append_stream_chunk(request_id, chunk)
        if self._is_aborted(request_id):
            return []
        return self._decode_ready_requests(request_ids=[request_id])

    def _handle_stream_chunk(self, request_id: str, item: Any) -> None:
        if not isinstance(item, StreamItem):
            raise TypeError(
                f"{self.__class__.__name__} expected StreamItem for "
                f"{request_id!r}, got {type(item).__name__}"
            )
        with self._state_lock:
            self._append_stream_chunk(request_id, item)
            if self._is_aborted(request_id):
                return
            if self._enable_batched_decode:
                self._collect_more_stream_chunks()
                messages = self._decode_ready_requests()
            else:
                messages = self._decode_ready_requests(request_ids=[request_id])
            for out in messages:
                if not self._is_aborted(out.request_id):
                    self.outbox.put(out)

    def _handle_stream_done(self, request_id: str) -> None:
        with self._state_lock:
            done_request_ids = [request_id]
            if self._enable_batched_decode:
                done_request_ids.extend(self._collect_more_stream_done())

            ready_to_finalize: list[str] = []
            for done_request_id in done_request_ids:
                if done_request_id not in self._stream_payloads:
                    if done_request_id in self._completed_non_streaming_request_ids:
                        continue
                    self._pending_done.add(done_request_id)
                    continue
                ready_to_finalize.append(done_request_id)

            if not ready_to_finalize:
                return

            force_ids = set(ready_to_finalize)
            request_ids = None if self._enable_batched_decode else ready_to_finalize
            for out in self._decode_ready_requests(
                force_request_ids=force_ids,
                request_ids=request_ids,
            ):
                if not self._is_aborted(out.request_id):
                    self.outbox.put(out)

            for done_request_id in ready_to_finalize:
                if self._is_aborted(done_request_id):
                    continue
                for out in self._finalize_request(done_request_id):
                    if not self._is_aborted(out.request_id):
                        self.outbox.put(out)
                if not self._is_aborted(done_request_id):
                    self._clear_request_state(done_request_id)

    def _append_stream_chunk(self, request_id: str, chunk: StreamItem) -> None:
        self._ensure_request_state(request_id)

        # Latch the stream flag from talker's metadata once per request.
        # Talker contract: always populate metadata['stream']; a missing
        # field means the upstream changed shape.
        if request_id not in self._stream_enabled:
            meta = chunk.metadata if isinstance(chunk.metadata, dict) else None
            if meta is None or "stream" not in meta:
                self._fail_request(
                    request_id,
                    RuntimeError(
                        f"code2wav got a chunk for {request_id!r} without "
                        "metadata['stream']; talker_model_runner must "
                        "populate it."
                    ),
                )
                return
            self._stream_enabled[request_id] = bool(meta["stream"])

        codes = chunk.data.to(device=self._device, dtype=torch.long)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Code2Wav chunk req=%s shape=%s first_codes=%s",
                request_id,
                tuple(codes.shape),
                codes.reshape(-1)[:8].tolist(),
            )

        # Skip EOS
        if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Code2Wav skip EOS req=%s codes=%s", request_id, codes.tolist()
                )
            return
        self._code_chunks[request_id].append(codes)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        messages: list[OutgoingMessage] = []
        messages.extend(self._decode_ready_requests(force_request_ids={request_id}))
        messages.extend(self._finalize_request(request_id))
        return messages

    def _finalize_request(self, request_id: str) -> list[OutgoingMessage]:
        messages: list[OutgoingMessage] = []

        # Build final output
        audio_parts = self._audio_chunks.get(request_id, [])
        if not audio_parts:
            self._fail_request(
                request_id,
                RuntimeError(f"code2wav produced no audio for {request_id!r}"),
            )
            return []
        full_audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        payload = self._payloads[request_id]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Code2Wav finalize req=%s code_chunks=%s audio_parts=%s final_samples=%s",
                request_id,
                len(self._code_chunks[request_id]),
                len(audio_parts),
                int(full_audio.shape[0]),
            )
        # Streaming clients already received per-chunk audio; final result is
        # metadata-only to avoid IPC-ing full audio that the HTTP layer drops.
        # Default False so missing latch falls back to non-streaming (safe:
        # may waste bandwidth, never starves a non-streaming client).
        if self._stream_enabled.get(request_id, False):
            final_data: dict[str, Any] = {
                "modality": "audio",
                "sample_rate": self._sample_rate,
            }
        else:
            final_data = self._build_audio_payload(full_audio)
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def _decode_ready_requests(
        self,
        *,
        force_request_ids: set[str] | None = None,
        request_ids: list[str] | None = None,
    ) -> list[OutgoingMessage]:
        plans = self._build_decode_plans(
            force_request_ids=force_request_ids or set(),
            request_ids=request_ids,
        )
        if not plans:
            return []

        messages: list[OutgoingMessage] = []
        for group in self._group_decode_plans(plans):
            for plan, audio in zip(group, self._decode_plan_batch(group)):
                self._emitted[plan.request_id] = plan.end
                if audio.size == 0:
                    continue
                is_first = not self._audio_chunks[plan.request_id]
                self._audio_chunks[plan.request_id].append(audio)
                if is_first:
                    _emit_event(
                        request_id=plan.request_id,
                        stage=None,
                        event_name="code2wav_first_audio",
                        metadata={"samples": int(audio.shape[0])},
                    )
                if self._stream_enabled.get(plan.request_id, True):
                    messages.append(
                        OutgoingMessage(
                            request_id=plan.request_id,
                            type="stream",
                            target=None,
                            data=self._build_audio_payload(audio),
                            metadata={"modality": "audio"},
                        )
                    )
        return messages

    def _build_decode_plans(
        self,
        *,
        force_request_ids: set[str],
        request_ids: list[str] | None,
    ) -> list[_DecodePlan]:
        plans: list[_DecodePlan] = []
        candidate_ids = (
            request_ids if request_ids is not None else list(self._code_chunks)
        )
        for request_id in candidate_ids:
            if self._is_aborted(request_id):
                continue
            code_chunks = self._code_chunks.get(request_id)
            if not code_chunks:
                continue
            start = self._emitted.get(request_id, 0)
            end = len(code_chunks)
            ready = end - start
            if ready <= 0:
                continue
            if (
                ready < self._decode_ready_threshold(request_id)
                and request_id not in force_request_ids
            ):
                continue
            context = min(self._left_context_size, start)
            window = torch.stack(code_chunks[start - context : end], dim=0)
            codes = window.transpose(0, 1)
            plans.append(
                _DecodePlan(
                    request_id=request_id,
                    start=start,
                    end=end,
                    trim=context * self._total_upsample,
                    codes=codes,
                )
            )
        return plans

    def _group_decode_plans(self, plans: list[_DecodePlan]) -> list[list[_DecodePlan]]:
        if not self._enable_batched_decode:
            return [[plan] for plan in plans]

        grouped: dict[tuple[tuple[int, ...], int], list[_DecodePlan]] = {}
        ordered_keys: list[tuple[tuple[int, ...], int]] = []
        for plan in plans:
            key = (tuple(plan.codes.shape), plan.trim)
            if key not in grouped:
                grouped[key] = []
                ordered_keys.append(key)
            grouped[key].append(plan)

        batches: list[list[_DecodePlan]] = []
        for key in ordered_keys:
            group = grouped[key]
            for start in range(0, len(group), self._decode_max_batch_size):
                batches.append(group[start : start + self._decode_max_batch_size])
        return batches

    def _decode_plan_batch(self, plans: list[_DecodePlan]) -> list[np.ndarray]:
        if not plans:
            return []

        codes = torch.stack([plan.codes for plan in plans], dim=0)
        with torch.no_grad():
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            wav = self._model(codes)
        batch_size = len(plans)
        if batch_size == 1:
            wav = wav.reshape(1, -1)
        elif int(wav.shape[0]) != batch_size:
            raise RuntimeError(
                "code2wav batched decode returned incompatible batch dimension: "
                f"expected {batch_size}, got shape {tuple(wav.shape)}"
            )
        else:
            wav = wav.reshape(batch_size, -1)

        trim = plans[0].trim
        if trim:
            wav = wav[:, trim:]
        audio_batch = wav.detach().to(device="cpu", dtype=torch.float32).numpy()
        audio_parts = [audio_batch[i].copy() for i in range(batch_size)]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Code2Wav decode batch window=%s batch=%s start_end=%s trim=%s samples=%s",
                tuple(codes.shape),
                batch_size,
                [(plan.start, plan.end) for plan in plans],
                trim,
                [int(audio.shape[0]) for audio in audio_parts],
            )
        return audio_parts

    def _ready_request_count(self) -> int:
        count = 0
        for request_id, chunks in self._code_chunks.items():
            if self._is_aborted(request_id):
                continue
            ready = len(chunks) - self._emitted.get(request_id, 0)
            if ready >= self._decode_ready_threshold(request_id):
                count += 1
        return count

    def _decode_ready_threshold(self, request_id: str) -> int:
        if not self._stream_enabled.get(request_id, True):
            return self._non_stream_chunk_size
        return self._stream_chunk_size

    def _collect_more_stream_chunks(self) -> None:
        ready_count = self._ready_request_count()
        if self._decode_max_batch_size <= 1 or ready_count == 0:
            return

        deadline = time.monotonic() + self._decode_max_batch_wait_s
        while ready_count < self._decode_max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if self._is_aborted(msg.request_id):
                continue
            if msg.type != "stream_chunk" or not isinstance(msg.data, StreamItem):
                self._pending_messages.append(msg)
                break
            self._append_stream_chunk(msg.request_id, msg.data)
            ready_count = self._ready_request_count()

    def _collect_more_stream_done(self) -> list[str]:
        if self._decode_max_batch_size <= 1:
            return []

        request_ids: list[str] = []
        deadline = time.monotonic() + self._decode_max_batch_wait_s
        while len(request_ids) + 1 < self._decode_max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if self._is_aborted(msg.request_id):
                continue
            if msg.type != "stream_done":
                self._pending_messages.append(msg)
                break
            request_ids.append(msg.request_id)
        return request_ids

    def _build_audio_payload(self, audio: np.ndarray) -> dict[str, Any]:
        return audio_waveform_payload(
            audio.astype(np.float32, copy=False),
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Qwen3-Omni code2wav",
        )


def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    non_stream_chunk_size: int | None = None,
    left_context_size: int = 25,
    max_batch_size: int = 16,
    max_batch_wait_ms: int = 1,
    enable_batched_decode: bool = True,
):
    """Factory: returns Code2WavScheduler."""
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        non_stream_chunk_size=non_stream_chunk_size,
        left_context_size=left_context_size,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        enable_batched_decode=enable_batched_decode,
    )
