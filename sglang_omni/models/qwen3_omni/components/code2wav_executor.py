# SPDX-License-Identifier: Apache-2.0
"""Code2Wav executor with incremental waveform streaming."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from sglang_omni.executors import Executor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def load_code2wav_model(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    weight_prefix: str = "code2wav.",
):
    """Load Code2Wav model from HF checkpoint with the given weight prefix."""
    from transformers import AutoConfig

    from sglang_omni.models.weight_loader import load_module, resolve_dtype

    torch_dtype = resolve_dtype(dtype)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    code2wav_config = getattr(config, "code2wav_config", None)
    if code2wav_config is None:
        raise ValueError(f"No code2wav_config found in {model_path}")

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model = Qwen3OmniMoeCode2Wav._from_config(code2wav_config)

    model = load_module(
        model,
        model_path,
        prefix=weight_prefix,
        dtype=torch_dtype,
        device=device,
        strict=False,
    )
    return model


class _Code2WavStreamingExecutor(Executor):
    """Decode codec chunks incrementally and emit audio stream chunks."""

    def __init__(
        self,
        model,
        *,
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
        self._total_upsample = int(getattr(model, "total_upsample", 1))
        self._stream_queue: Any | None = None
        self._done: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[StagePayload]] = {}
        self._stream_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._aborted: set[str] = set()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        self._stream_queues[request_id] = asyncio.Queue()
        task = asyncio.create_task(self._run_request(payload))
        self._tasks[request_id] = task
        task.add_done_callback(lambda _task: self._done.put_nowait(request_id))

    async def get_result(self) -> StagePayload:
        while True:
            request_id = await self._done.get()
            task = self._tasks.pop(request_id, None)
            if task is None:
                continue
            if request_id in self._aborted:
                self._stream_queues.pop(request_id, None)
                continue
            try:
                return await task
            except Exception as exc:
                exc.request_id = request_id
                raise

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        task = self._tasks.pop(request_id, None)
        if task is not None:
            task.cancel()
        queue = self._stream_queues.pop(request_id, None)
        if queue is not None:
            queue.put_nowait(None)

    async def stream(self, request_id: str):
        queue = self._stream_queues.get(request_id)
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    async def _run_request(self, payload: StagePayload) -> StagePayload:
        request_id = payload.request_id
        if self._stream_queue is None:
            raise RuntimeError("Code2Wav executor requires a stream queue")

        queue = self._stream_queues[request_id]
        loop = asyncio.get_running_loop()
        code_chunks: list[torch.Tensor] = []
        audio_chunks: list[np.ndarray] = []
        emitted_positions = 0

        try:
            while True:
                if request_id in self._aborted:
                    raise asyncio.CancelledError()

                item = await self._stream_queue.get(request_id)
                if item is None:
                    break

                codes = item.data.to(device=self._device, dtype=torch.long)
                # Skip EOS position — codec_eos_token_id (2150) should not be
                # decoded into audio. HF's generate() stops before emitting EOS
                # into hidden_states, but our streaming pipeline includes it.
                if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
                    continue
                code_chunks.append(codes)
                ready_positions = len(code_chunks) - emitted_positions
                if ready_positions < self._stream_chunk_size:
                    continue

                audio = await self._decode_async(
                    loop,
                    code_chunks,
                    emitted_positions,
                    len(code_chunks),
                )
                emitted_positions = len(code_chunks)
                if audio.size == 0:
                    continue
                audio_chunks.append(audio)
                await queue.put(self._build_audio_payload(audio))

            if code_chunks and emitted_positions < len(code_chunks):
                audio = await self._decode_async(
                    loop,
                    code_chunks,
                    emitted_positions,
                    len(code_chunks),
                )
                if audio.size > 0:
                    audio_chunks.append(audio)
                    await queue.put(self._build_audio_payload(audio))

            await queue.put(None)

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
            else:
                full_audio = np.zeros((0,), dtype=np.float32)

            payload.data = self._build_audio_payload(full_audio)
            return payload
        finally:
            self._stream_queues.pop(request_id, None)

    async def _decode_async(
        self,
        loop: asyncio.AbstractEventLoop,
        code_chunks: list[torch.Tensor],
        start_index: int,
        end_index: int,
    ) -> np.ndarray:
        if self._device.type == "cpu":
            return self._decode_incremental(code_chunks, start_index, end_index)
        return await loop.run_in_executor(
            None,
            self._decode_incremental,
            code_chunks,
            start_index,
            end_index,
        )

    def _decode_incremental(
        self,
        code_chunks: list[torch.Tensor],
        start_index: int,
        end_index: int,
    ) -> np.ndarray:
        if start_index >= end_index:
            return np.zeros((0,), dtype=np.float32)

        context_size = min(self._left_context_size, start_index)
        window = torch.stack(code_chunks[start_index - context_size : end_index], dim=0)
        codes = window.transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            wav = self._model(codes)

        trim = context_size * self._total_upsample
        if trim:
            wav = wav[..., trim:]
        return wav.reshape(-1).detach().cpu().float().numpy().copy()

    def _build_audio_payload(self, audio: np.ndarray) -> dict[str, Any]:
        audio = audio.astype(np.float32, copy=False)
        return {
            "audio_waveform": audio.tobytes(),
            "audio_waveform_shape": list(audio.shape),
            "audio_waveform_dtype": "float32",
            "sample_rate": self._sample_rate,
            "modality": "audio",
        }


def create_code2wav_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_batch_size: int = 32,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
) -> Executor:
    """Create Code2Wav executor that streams waveform chunks."""
    del max_batch_size
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return _Code2WavStreamingExecutor(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
    )
