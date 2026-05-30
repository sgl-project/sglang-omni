# SPDX-License-Identifier: Apache-2.0
"""Code2Wav scheduler — streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), accumulates them,
runs vocoder incrementally, outputs final audio via outbox.
"""
from __future__ import annotations

import logging
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
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._left_context_size = max(int(left_context_size), 0)
        self._sample_rate = sample_rate
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(model.total_upsample)

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
                return []
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
            return []
        self._code_chunks[request_id].append(codes)
        ready = len(self._code_chunks[request_id]) - self._emitted[request_id]
        if ready >= self._stream_chunk_size:
            return self._decode_and_emit(request_id)
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        # Decode remaining
        chunks = self._code_chunks[request_id]
        emitted = self._emitted[request_id]
        messages: list[OutgoingMessage] = []
        if chunks and emitted < len(chunks):
            messages.extend(self._decode_and_emit(request_id))

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

    def _decode_and_emit(self, request_id: str) -> list[OutgoingMessage]:
        chunks = self._code_chunks[request_id]
        start = self._emitted[request_id]
        end = len(chunks)
        audio = self._decode_incremental(request_id, chunks, start, end)
        self._emitted[request_id] = end
        messages: list[OutgoingMessage] = []
        if audio.size > 0:
            is_first = not self._audio_chunks[request_id]
            self._audio_chunks[request_id].append(audio)
            if is_first:
                _emit_event(
                    request_id=request_id,
                    stage=None,
                    event_name="code2wav_first_audio",
                    metadata={"samples": int(audio.shape[0])},
                )
            if self._stream_enabled.get(request_id, True):
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        target=None,
                        data=self._build_audio_payload(audio),
                        metadata={"modality": "audio"},
                    )
                )
        return messages

    def _decode_incremental(
        self, request_id: str, code_chunks, start, end
    ) -> np.ndarray:
        if start >= end:
            return np.zeros((0,), dtype=np.float32)
        context = min(self._left_context_size, start)
        window = torch.stack(code_chunks[start - context : end], dim=0)
        codes = window.transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            wav = self._model(codes)
        trim = context * self._total_upsample
        if trim:
            wav = wav[..., trim:]
        audio = wav.reshape(-1).detach().cpu().float().numpy().copy()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Code2Wav decode window=%s start=%s end=%s trim=%s samples=%s",
                tuple(codes.shape),
                start,
                end,
                trim,
                int(audio.shape[0]),
            )
        return audio

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
    left_context_size: int = 25,
):
    """Factory: returns Code2WavScheduler."""
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
    )
