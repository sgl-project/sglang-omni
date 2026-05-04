# SPDX-License-Identifier: Apache-2.0
"""Code2Wav scheduler — streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), accumulates them,
runs vocoder incrementally, outputs final audio via outbox.
"""
from __future__ import annotations

import logging
import queue as _queue_mod
from typing import Any

import numpy as np
import torch

from sglang_omni_v1.proto import StagePayload
from sglang_omni_v1.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


def load_code2wav_model(
    model_path: str, *, device: str = "cuda", dtype: str | None = None
):
    """Load Code2Wav model from HF checkpoint."""
    from transformers import AutoConfig

    from sglang_omni_v1.models.weight_loader import load_module, resolve_dtype

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


class Code2WavScheduler:
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
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._left_context_size = max(int(left_context_size), 0)
        self._sample_rate = sample_rate
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(model.total_upsample)
        self._running = False

        # Per-request state
        self._code_chunks: dict[str, list[torch.Tensor]] = {}
        self._emitted: dict[str, int] = {}
        self._audio_chunks: dict[str, list[np.ndarray]] = {}
        self._payloads: dict[str, StagePayload] = {}
        self._pending_done: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue

            if msg.type == "new_request":
                self._ensure_request_state(msg.request_id)
                self._payloads[msg.request_id] = msg.data
                if msg.request_id in self._pending_done:
                    self._pending_done.discard(msg.request_id)
                    self._on_done(msg.request_id)

            elif msg.type == "stream_chunk":
                self._on_chunk(msg.request_id, msg.data)

            elif msg.type == "stream_done":
                self._on_done(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._code_chunks.pop(request_id, None)
        self._emitted.pop(request_id, None)
        self._audio_chunks.pop(request_id, None)
        self._payloads.pop(request_id, None)
        self._pending_done.discard(request_id)

    def _ensure_request_state(self, request_id: str) -> None:
        if request_id in self._code_chunks:
            return
        self._code_chunks[request_id] = []
        self._emitted[request_id] = 0
        self._audio_chunks[request_id] = []

    def _on_chunk(self, request_id: str, chunk: Any) -> None:
        self._ensure_request_state(request_id)

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
        ready = len(self._code_chunks[request_id]) - self._emitted[request_id]
        if ready >= self._stream_chunk_size:
            self._decode_and_emit(request_id)

    def _on_done(self, request_id: str) -> None:
        if request_id not in self._code_chunks:
            self._pending_done.add(request_id)
            return
        if request_id not in self._payloads:
            self._pending_done.add(request_id)
            return

        # Decode remaining
        chunks = self._code_chunks[request_id]
        emitted = self._emitted[request_id]
        if chunks and emitted < len(chunks):
            self._decode_and_emit(request_id)

        # Build final output
        audio_parts = self._audio_chunks.get(request_id, [])
        if audio_parts:
            full_audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        else:
            full_audio = np.zeros((0,), dtype=np.float32)
        payload = self._payloads[request_id]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Code2Wav finalize req=%s code_chunks=%s audio_parts=%s final_samples=%s",
                request_id,
                len(self._code_chunks[request_id]),
                len(audio_parts),
                int(full_audio.shape[0]),
            )
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=self._build_audio_payload(full_audio),
                ),
            )
        )

        # Cleanup
        self._code_chunks.pop(request_id, None)
        self._emitted.pop(request_id, None)
        self._audio_chunks.pop(request_id, None)
        self._payloads.pop(request_id, None)

    def _decode_and_emit(self, request_id: str) -> None:
        chunks = self._code_chunks[request_id]
        start = self._emitted[request_id]
        end = len(chunks)
        audio = self._decode_incremental(request_id, chunks, start, end)
        self._emitted[request_id] = end
        if audio.size > 0:
            self._audio_chunks[request_id].append(audio)

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
        audio = audio.astype(np.float32, copy=False)
        return {
            "audio_waveform": audio.tobytes(),
            "audio_waveform_shape": list(audio.shape),
            "audio_waveform_dtype": "float32",
            "sample_rate": self._sample_rate,
            "modality": "audio",
        }


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
