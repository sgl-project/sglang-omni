# SPDX-License-Identifier: Apache-2.0
"""Serialized Voxtral MLX loading and speech generation."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from sglang_omni_mlx.voxtral_tts.audio import encode_wav
from sglang_omni_mlx.voxtral_tts.config import VoxtralConfig, read_voxtral_config
from sglang_omni_mlx.voxtral_tts.model import VoxtralMLXModel, load_voxtral_model
from sglang_omni_mlx.voxtral_tts.synthesis import AudioFinishReason, VoxtralSynthesizer
from sglang_omni_mlx.voxtral_tts.tokenizer import (
    VoxtralSpeechTokenizer,
    load_voice_embeddings,
)

DEFAULT_VOICE = "cheerful_female"
DEFAULT_MAX_NEW_TOKENS = 4096
MAX_NEW_TOKENS_LIMIT = 4096
MAX_QUEUED_REQUESTS = 2
REQUEST_QUEUE_TIMEOUT_SECONDS = 30
REQUEST_GENERATION_TIMEOUT_SECONDS = 600


class SpeechQueueFullError(RuntimeError):
    """The bounded Voxtral request queue did not admit a request in time."""


class SpeechGenerationTimeoutError(RuntimeError):
    """Voxtral speech generation exceeded its request deadline."""


@dataclass(frozen=True, kw_only=True)
class SpeechResult:
    audio: bytes
    finish_reason: AudioFinishReason = "stop"


class VoxtralTTSWorker:
    def __init__(self, checkpoint_directory: Path) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="voxtral-mlx"
        )
        self.config, self.model, self.tokenizer, self.voice_embeddings = (
            self.executor.submit(self.load, checkpoint_directory).result()
        )
        self.state_lock = threading.Lock()
        self.queued_requests = 0
        self.running_requests = 0
        self.request_slots = asyncio.Semaphore(MAX_QUEUED_REQUESTS + 1)

    @staticmethod
    def load(
        checkpoint_directory: Path,
    ) -> tuple[
        VoxtralConfig,
        VoxtralMLXModel,
        VoxtralSpeechTokenizer,
        dict[str, mx.array],
    ]:
        config = read_voxtral_config(checkpoint_directory)
        model = load_voxtral_model(config)
        tokenizer = VoxtralSpeechTokenizer(checkpoint_directory)
        voice_embeddings = load_voice_embeddings(config)
        return config, model, tokenizer, voice_embeddings

    def request_states(self) -> dict[str, int]:
        with self.state_lock:
            return {
                "queued": self.queued_requests,
                "running": self.running_requests,
            }

    def run(
        self,
        text: str,
        voice: str,
        max_new_tokens: int,
        cancellation_event: threading.Event,
        seed: int,
    ) -> SpeechResult:
        with self.state_lock:
            self.queued_requests -= 1
            self.running_requests += 1
        try:
            prompt_token_ids = self.tokenizer.encode(text, voice)
            synthesizer = VoxtralSynthesizer(
                self.model,
                self.config.audio.audio_token_id,
                max_new_tokens,
            )
            synthesis_result = synthesizer.synthesize(
                prompt_token_ids,
                self.voice_embeddings[voice],
                cancellation_event,
                seed,
            )
            audio_tokenizer = self.model.audio_tokenizer
            assert audio_tokenizer is not None
            mx.eval(synthesis_result.waveform)
            audio_bytes = encode_wav(
                np.asarray(synthesis_result.waveform.astype(mx.float32)),
                audio_tokenizer.sampling_rate,
            )
            return SpeechResult(
                audio=audio_bytes,
                finish_reason=synthesis_result.finish_reason,
            )
        finally:
            with self.state_lock:
                self.running_requests -= 1

    def complete_request(
        self,
        future: Future[SpeechResult],
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        if future.cancelled():
            with self.state_lock:
                self.queued_requests -= 1
        else:
            pass
        event_loop.call_soon_threadsafe(self.request_slots.release)

    async def synthesize(
        self, text: str, voice: str, max_new_tokens: int, seed: int
    ) -> SpeechResult:
        try:
            await asyncio.wait_for(
                self.request_slots.acquire(),
                timeout=REQUEST_QUEUE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise SpeechQueueFullError(
                "Voxtral speech request queue is full"
            ) from error
        cancellation_event = threading.Event()
        with self.state_lock:
            self.queued_requests += 1
        event_loop = asyncio.get_running_loop()
        try:
            future = self.executor.submit(
                self.run,
                text,
                voice,
                max_new_tokens,
                cancellation_event,
                seed,
            )
        except Exception:
            with self.state_lock:
                self.queued_requests -= 1
            self.request_slots.release()
            raise
        future.add_done_callback(
            lambda completed_future: self.complete_request(completed_future, event_loop)
        )
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=REQUEST_GENERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            cancellation_event.set()
            future.cancel()
            raise SpeechGenerationTimeoutError(
                "Voxtral speech generation timed out"
            ) from error
        except asyncio.CancelledError:
            cancellation_event.set()
            future.cancel()
            raise
