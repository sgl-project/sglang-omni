# SPDX-License-Identifier: Apache-2.0
"""Precompute Ming ASR audio embeddings before LM admission."""

from __future__ import annotations

import concurrent.futures
import contextlib
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from sglang_omni.modules.ming_audio.audio_encoder import Ming3AudioEncoder
from sglang_omni.scheduling.pre_lm_encoder import (
    PreLMEncoderService,
    QueueEntry,
    QueueSignal,
)

MelBatch = list[torch.Tensor]


@dataclass(frozen=True, kw_only=True, slots=True)
class EncodedMelBatch:
    embeddings: list[torch.Tensor]
    chunk_counts: list[int]


class Ming3ASRPreLMEncoderService(
    PreLMEncoderService[MelBatch, EncodedMelBatch, torch.Tensor]
):
    ENCODE_TIMEOUT_S = 300.0

    def __init__(
        self,
        encoder: Ming3AudioEncoder,
        *,
        max_batch_size: int,
        max_batch_wait_ms: int,
    ) -> None:
        self.encoder = encoder
        reference = next(encoder.parameters())
        self.device = reference.device
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.max_batch_size = max_batch_size
        self.max_batch_wait_s = max_batch_wait_ms / 1000.0
        self.lifecycle_lock = threading.Lock()
        self.closed = False
        super().__init__(worker_name="ming3-asr-audio-encode")

    def close(self) -> None:
        with self.lifecycle_lock:
            if self.closed:
                return
            else:
                pass
            self.closed = True
            self.queue.put(QueueSignal.SHUTDOWN)
        self.thread.join(timeout=5)

    def enqueue(
        self,
        mels: MelBatch,
        future: concurrent.futures.Future[torch.Tensor],
    ) -> None:
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError("Ming ASR pre-LM encoder service is closed")
            else:
                pass
            self.queue.put(QueueEntry(item=mels, future=future))

    def encode(self, mels: MelBatch) -> torch.Tensor:
        return self.submit(mels).result(timeout=self.ENCODE_TIMEOUT_S)

    def drain_batch(
        self,
    ) -> tuple[list[QueueEntry[MelBatch, torch.Tensor]], bool]:
        first = self.queue.get()
        if first is QueueSignal.SHUTDOWN:
            return [], True
        else:
            pass
        batch = [first]
        deadline = time.monotonic() + self.max_batch_wait_s
        shutdown = False
        while len(batch) < self.max_batch_size:
            try:
                remaining_s = deadline - time.monotonic()
                queued = (
                    self.queue.get(timeout=remaining_s)
                    if remaining_s > 0
                    else self.queue.get_nowait()
                )
            except queue.Empty:
                break
            if queued is QueueSignal.SHUTDOWN:
                shutdown = True
                break
            else:
                pass
            batch.append(queued)
        return batch, shutdown

    def next_batch(
        self,
    ) -> tuple[list[QueueEntry[MelBatch, torch.Tensor]], bool]:
        return self.drain_batch()

    @contextlib.contextmanager
    def batch_context(self) -> Iterator[None]:
        with torch.inference_mode():
            if self.stream is None:
                yield
            else:
                with torch.cuda.stream(self.stream):
                    yield

    def encode_batch(self, requests: list[MelBatch]) -> EncodedMelBatch:
        chunk_size = self.encoder.config.whisper_chunk_size
        mel_chunks: list[torch.Tensor] = []
        chunk_counts: list[int] = []
        for mels in requests:
            count = 0
            for mel in mels:
                parts = mel.split(chunk_size)
                count += len(parts)
                mel_chunks.extend(parts)
            chunk_counts.append(count)
        return EncodedMelBatch(
            embeddings=self.encoder.embed_mels(mel_chunks),
            chunk_counts=chunk_counts,
        )

    def split_embeddings(
        self,
        requests: list[MelBatch],
        encoded: EncodedMelBatch,
    ) -> list[torch.Tensor]:
        offset = 0
        embeddings: list[torch.Tensor] = []
        for count in encoded.chunk_counts:
            embeddings.append(torch.cat(encoded.embeddings[offset : offset + count]))
            offset += count
        return embeddings

    def attach_embedding(self, mels: MelBatch, embedding: torch.Tensor) -> None:
        if self.stream is not None and embedding.is_cuda:
            embedding.record_stream(torch.cuda.default_stream(self.device))
        else:
            pass

    def synchronize_batch(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        else:
            pass
