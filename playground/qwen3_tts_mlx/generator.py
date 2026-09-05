# SPDX-License-Identifier: Apache-2.0
"""Lazily loaded MLX Qwen3-TTS generator shared by the playground UI.

Unlike the other playgrounds, this one has no HTTP backend: MLX serving is not
wired into Omni's scheduler yet, so the UI drives the model in process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

FRAME_RATE_HZ = 12.5
SAMPLE_RATE = 24000


class ModelUnsupportedError(RuntimeError):
    """Raised when the checkpoint cannot clone a voice."""


@dataclass
class CloneResult:
    """One completed synthesis, with the numbers the UI reports."""

    audio: np.ndarray
    frames: int
    prompt_tokens: int
    reference_frames: int
    reference_cached: bool
    prefill_seconds: float
    decode_seconds: float
    vocoder_seconds: float

    @property
    def audio_seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds + self.vocoder_seconds

    @property
    def realtime_factor(self) -> float:
        return self.total_seconds / max(self.audio_seconds, 1e-6)


class LazyGenerator:
    """Loads the checkpoint on first use and keeps it warm.

    Generation is serialised: MLX arrays and the reference cache are shared
    mutable state, and the playground is single-user anyway.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._generator = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._generator is not None

    def load(self):
        """Return the generator, loading it the first time. Raises on non-Base."""
        with self._lock:
            if self._generator is not None:
                return self._generator

            from sglang_omni.models.qwen3_tts.mlx.generate import Qwen3TTSMlxGenerator

            generator = Qwen3TTSMlxGenerator.from_pretrained(self.model_path)
            if not generator.speech_tokenizer.has_encoder:
                raise ModelUnsupportedError(
                    f"{self.model_path} has no speech-tokenizer encoder, so it "
                    "cannot encode reference audio. Voice cloning needs a "
                    "*-Base checkpoint, for example "
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base."
                )
            self._generator = generator
            return generator

    def clone(
        self,
        *,
        text: str,
        ref_audio: str,
        ref_text: str,
        language: str,
        max_frames: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int | None,
        on_frame: Callable[[int], None] | None = None,
    ) -> CloneResult:
        """Synthesise one request, reporting progress per generated frame."""
        import mlx.core as mx

        from sglang_omni.models.qwen3_tts.mlx.generate import (
            CloneRequest,
            _load_audio_24k,
        )
        from sglang_omni.models.qwen3_tts.mlx.sampling import SamplingParams

        generator = self.load()
        params = SamplingParams(
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        request = CloneRequest(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            max_frames=int(max_frames),
            semantic=params,
            subtalker=SamplingParams(
                temperature=temperature, top_k=int(top_k), top_p=top_p
            ),
            seed=seed,
        )

        with self._lock:
            if seed is not None:
                mx.random.seed(seed)

            audio = _load_audio_24k(ref_audio)
            references_before = generator.cached_reference_count

            started = time.perf_counter()
            prompt = generator.build_icl_prompt(text, audio, ref_text, language)
            mx.eval(prompt.input_embeds, prompt.ref_codes)
            prefill_seconds = time.perf_counter() - started
            reference_cached = (
                generator.cached_reference_count == references_before
            )

            started = time.perf_counter()
            frames = []
            for frame in generator.generate_frames(prompt, request):
                frames.append(frame)
                if on_frame is not None:
                    on_frame(len(frames))
            decode_seconds = time.perf_counter() - started

            started = time.perf_counter()
            waveform = generator.decode_frames(frames, prompt.ref_codes)
            vocoder_seconds = time.perf_counter() - started

        return CloneResult(
            audio=waveform,
            frames=len(frames),
            prompt_tokens=int(prompt.input_embeds.shape[1]),
            reference_frames=int(prompt.ref_codes.shape[-1]),
            reference_cached=reference_cached,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            vocoder_seconds=vocoder_seconds,
        )
