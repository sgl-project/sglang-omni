# SPDX-License-Identifier: Apache-2.0
"""Code2Wav executor — streaming vocoder with opportunistic micro-batching."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sglang_omni.executors import Executor
from sglang_omni.executors._batching import run_batch_loop
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_DEFAULT_NUM_QUANTIZERS: int = 16
_DEFAULT_PAD_TOKEN_ID: int = 0


@dataclass
class _DecodeRequest:
    request_id: str
    codes_window: torch.Tensor
    trim_samples: int
    result_future: asyncio.Future[np.ndarray] | None = None


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
        max_batch_size: int = 16,
        num_quantizers: int = _DEFAULT_NUM_QUANTIZERS,
        pad_token_id: int = _DEFAULT_PAD_TOKEN_ID,
        warmup: bool = True,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._left_context_size = max(int(left_context_size), 0)
        self._sample_rate = sample_rate
        self._codec_eos_token_id = codec_eos_token_id
        self._pad_token_id = int(pad_token_id)
        self._total_upsample = int(getattr(model, "total_upsample", 1))
        self._max_batch_size = max(int(max_batch_size), 1)
        self._num_quantizers = int(num_quantizers)
        self._stream_queue: Any | None = None
        self._done: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[StagePayload]] = {}
        self._stream_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._aborted: set[str] = set()

        self._pending: asyncio.Queue[_DecodeRequest] = asyncio.Queue()
        self._batch_loop_task: asyncio.Task[None] | None = None
        self._forward_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="code2wav-fwd",
        )
        self._output_len_by_seq: dict[int, int] = {}
        self._output_len_offset: int | None = None

        if warmup:
            self.warmup()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        self._ensure_batch_loop()
        self._stream_queues[request_id] = asyncio.Queue()
        task = asyncio.create_task(self._run_request(payload))
        self._tasks[request_id] = task

        def _cleanup(_task: asyncio.Task, rid: str = request_id) -> None:
            self._aborted.discard(rid)
            self._done.put_nowait(rid)

        task.add_done_callback(_cleanup)

    async def get_result(self) -> StagePayload:
        while True:
            request_id = await self._done.get()
            task = self._tasks.pop(request_id, None)
            if task is None:
                continue
            try:
                return await task
            except asyncio.CancelledError:
                continue
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

    async def stop(self) -> None:
        if self._batch_loop_task is not None and not self._batch_loop_task.done():
            self._batch_loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._batch_loop_task
        self._batch_loop_task = None

        while not self._pending.empty():
            try:
                req = self._pending.get_nowait()
            except asyncio.QueueEmpty:
                break
            fut = req.result_future
            if fut is not None and not fut.done():
                fut.cancel()

        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        self._tasks.clear()

        self._forward_pool.shutdown(wait=False)

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

        queue = self._stream_queues.get(request_id)
        if queue is None:
            raise asyncio.CancelledError()
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
                if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
                    continue
                code_chunks.append(codes)
                ready_positions = len(code_chunks) - emitted_positions
                if ready_positions < self._stream_chunk_size:
                    continue

                audio = await self._submit_decode(
                    request_id,
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
                audio = await self._submit_decode(
                    request_id,
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

    def _prepare_window(
        self,
        code_chunks: list[torch.Tensor],
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, int]:
        context_size = min(self._left_context_size, start_index)
        window = torch.stack(
            code_chunks[start_index - context_size : end_index],
            dim=0,
        )
        codes_window = window.transpose(0, 1)
        trim_samples = context_size * self._total_upsample
        return codes_window, trim_samples

    async def _submit_decode(
        self,
        request_id: str,
        code_chunks: list[torch.Tensor],
        start_index: int,
        end_index: int,
    ) -> np.ndarray:
        if start_index >= end_index:
            return np.zeros((0,), dtype=np.float32)

        codes_window, trim_samples = self._prepare_window(
            code_chunks,
            start_index,
            end_index,
        )

        if self._device.type == "cpu":
            return self._vocoder_forward(codes_window, trim_samples)

        loop = asyncio.get_running_loop()
        req = _DecodeRequest(
            request_id=request_id,
            codes_window=codes_window,
            trim_samples=trim_samples,
            result_future=loop.create_future(),
        )
        self._pending.put_nowait(req)
        return await req.result_future

    def _ensure_batch_loop(self) -> None:
        if self._batch_loop_task is None or self._batch_loop_task.done():
            self._batch_loop_task = asyncio.create_task(
                run_batch_loop(self._batch_decode_step, "Code2Wav")
            )

    async def _batch_decode_step(self) -> None:
        loop = asyncio.get_running_loop()

        first = await self._pending.get()
        batch: list[_DecodeRequest] = [first]

        while not self._pending.empty() and len(batch) < self._max_batch_size:
            try:
                batch.append(self._pending.get_nowait())
            except asyncio.QueueEmpty:
                break

        live: list[_DecodeRequest] = []
        for r in batch:
            if r.request_id in self._aborted:
                fut = r.result_future
                if fut is not None and not fut.done():
                    fut.cancel()
            else:
                live.append(r)
        if not live:
            return

        def _finalize(req: _DecodeRequest, audio: np.ndarray) -> None:
            fut = req.result_future
            if req.request_id in self._aborted:
                if fut is not None and not fut.done():
                    fut.cancel()
                return
            if fut is not None and not fut.done():
                fut.set_result(audio)

        try:
            audios = await loop.run_in_executor(
                self._forward_pool,
                self._forward_batch,
                live,
            )
            for req, audio in zip(live, audios):
                _finalize(req, audio)
        except Exception as exc:
            for req in live:
                fut = req.result_future
                if fut is not None and not fut.done():
                    fut.set_exception(exc)
            raise

    @torch.no_grad()
    def warmup(self) -> None:
        """Prime conv kernel caches + record actual output length per seq_len."""
        if self._device.type != "cuda":
            return

        warmup_lens: set[int] = set()
        warmup_lens.add(self._stream_chunk_size)
        warmup_lens.add(self._stream_chunk_size + self._left_context_size)
        if self._left_context_size > 0:
            step = max(1, self._left_context_size // 4)
            for ctx in range(step, self._left_context_size, step):
                warmup_lens.add(self._stream_chunk_size + ctx)
        for flush_len in (1, self._stream_chunk_size // 2):
            if flush_len >= 1:
                warmup_lens.add(flush_len)

        sizes = sorted(
            {
                b
                for b in {1, 2, 4, 8, self._max_batch_size}
                if 1 <= b <= self._max_batch_size
            }
        )
        ordered_lens = sorted(warmup_lens)
        logger.info(
            f"Code2Wav warmup: priming {len(sizes) * len(ordered_lens)} "
            f"(batch_size, seq_len) pairs on {self._device} "
            f"(sizes={sizes}, seq_lens={ordered_lens})"
        )
        torch.cuda.set_device(self._device)
        start = time.perf_counter()
        for seq_len in ordered_lens:
            for b in sizes:
                dummy = torch.full(
                    (b, self._num_quantizers, seq_len),
                    fill_value=self._pad_token_id,
                    dtype=torch.long,
                    device=self._device,
                )
                out = self._model(dummy)
                actual = int(out.shape[-1])
                prev = self._output_len_by_seq.get(seq_len)
                if prev is not None and prev != actual:
                    logger.warning(
                        f"Code2Wav output length non-deterministic for "
                        f"seq_len={seq_len}: prev={prev} vs now={actual}"
                    )
                self._output_len_by_seq[seq_len] = actual
                if self._output_len_offset is None:
                    self._output_len_offset = seq_len * self._total_upsample - actual
        torch.cuda.synchronize(self._device)
        logger.info(
            f"Code2Wav warmup complete in {time.perf_counter() - start:.2f}s; "
            f"output_len_offset={self._output_len_offset}"
        )

    def _compute_valid_samples(self, seq_len: int) -> int:
        # Architecture-constant offset invariant: output_len == seq_len * total_upsample - offset
        # holds for all seq_lens on this vocoder. Guarded by
        # tests/test_real_code2wav_parity.py::test_output_length_offset_is_architecture_constant.
        # If a new vocoder breaks this, re-enable probing by dropping the offset branch.
        cached = self._output_len_by_seq.get(seq_len)
        if cached is not None:
            return cached
        if self._output_len_offset is not None:
            return seq_len * self._total_upsample - self._output_len_offset
        actual = self._probe_output_len(seq_len)
        self._output_len_by_seq[seq_len] = actual
        self._output_len_offset = seq_len * self._total_upsample - actual
        return actual

    @torch.no_grad()
    def _probe_output_len(self, seq_len: int) -> int:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        probe = torch.full(
            (1, self._num_quantizers, seq_len),
            fill_value=self._pad_token_id,
            dtype=torch.long,
            device=self._device,
        )
        out = self._model(probe)
        return int(out.shape[-1])

    @torch.no_grad()
    def _vocoder_forward(
        self,
        codes_window: torch.Tensor,
        trim_samples: int,
    ) -> np.ndarray:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        codes = codes_window.unsqueeze(0)
        wav = self._model(codes)
        if trim_samples:
            wav = wav[..., trim_samples:]
        return wav.reshape(-1).detach().cpu().float().numpy().copy()

    @torch.no_grad()
    def _forward_batch(
        self,
        batch: list[_DecodeRequest],
    ) -> list[np.ndarray]:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        max_len = max(r.codes_window.shape[1] for r in batch)
        num_q = batch[0].codes_window.shape[0]

        padded = torch.full(
            (len(batch), num_q, max_len),
            fill_value=self._pad_token_id,
            dtype=batch[0].codes_window.dtype,
            device=self._device,
        )
        for i, req in enumerate(batch):
            seq_len = req.codes_window.shape[1]
            padded[i, :, :seq_len] = req.codes_window

        wav_batch = self._model(padded)

        results: list[np.ndarray] = []
        for i, req in enumerate(batch):
            wav = wav_batch[i]
            seq_len = req.codes_window.shape[1]
            valid_samples = self._compute_valid_samples(seq_len)
            if req.trim_samples:
                wav = wav[..., req.trim_samples : valid_samples]
            else:
                wav = wav[..., :valid_samples]
            results.append(wav.reshape(-1).detach().cpu().float().numpy().copy())

        return results

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
    max_batch_size: int = 16,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
    num_quantizers: int = _DEFAULT_NUM_QUANTIZERS,
    pad_token_id: int = _DEFAULT_PAD_TOKEN_ID,
    warmup: bool = True,
) -> Executor:
    """Create Code2Wav executor that streams waveform chunks."""
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return _Code2WavStreamingExecutor(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        max_batch_size=max_batch_size,
        num_quantizers=num_quantizers,
        pad_token_id=pad_token_id,
        warmup=warmup,
    )


def create_code2wav_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    server_args_overrides: dict[str, Any] | None = None,
    dtype: str | None = None,
) -> Executor:
    """Factory mirroring create_code_predictor_executor_from_config."""
    overrides = dict(server_args_overrides or {})
    max_batch_size = int(overrides.pop("code2wav_max_batch_size", 16))
    warmup = bool(overrides.pop("code2wav_warmup", True))
    stream_chunk_size = int(overrides.pop("code2wav_stream_chunk_size", 10))
    left_context_size = int(overrides.pop("code2wav_left_context_size", 25))
    pad_token_id = int(overrides.pop("code2wav_pad_token_id", _DEFAULT_PAD_TOKEN_ID))
    num_quantizers = int(
        overrides.pop("code2wav_num_quantizers", _DEFAULT_NUM_QUANTIZERS)
    )
    return create_code2wav_executor(
        model_path=model_path,
        gpu_id=gpu_id,
        dtype=dtype,
        max_batch_size=max_batch_size,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        num_quantizers=num_quantizers,
        pad_token_id=pad_token_id,
        warmup=warmup,
    )
