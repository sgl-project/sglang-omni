# SPDX-License-Identifier: Apache-2.0
"""Adapter for a loaded author SpeechGenerator; model loading is external."""

from __future__ import annotations

import itertools
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Generator, Iterator, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from sglang_omni.proto import StagePayload


class GeneratorConfiguration(Protocol):
    mimi_sr: int
    max_audio_length_ms: int


class GeneratorContext(Protocol):
    device: str | torch.device


class ModelCache(Protocol):
    def reset_caches(self) -> None: ...


class StreamingCodec(Protocol):
    def reset_streaming(self) -> None: ...


class SpeechGenerator(Protocol):
    config: GeneratorConfiguration
    ctx: GeneratorContext
    model: ModelCache
    mimi: StreamingCodec
    _mimi_streaming_started: (
        bool  # noqa: leading-underscore - author generator contract
    )

    def generate_stream(
        self,
        reference_audio: Path,
        text: str,
        *,
        speaking_rate: Iterator[float],
        enhance_prompt: bool,
        apply_vad: bool,
    ) -> Generator[tuple[NDArray[np.float32], float], None, None]: ...


class VoXtreamBackend:
    """Own one generator, used exclusively by one serial scheduler thread.

    The reference path is configured by the operator. This adapter does not
    resolve arbitrary client paths or change the author's module globals.
    """

    def __init__(self, generator: SpeechGenerator, reference_audio: str | Path) -> None:
        self.generator = generator
        self.cleanup_error: Exception | None = None
        self.reference_audio = Path(reference_audio)
        if not self.reference_audio.is_file():
            raise FileNotFoundError(self.reference_audio)
        else:
            pass
        self.sample_rate = int(generator.config.mimi_sr)
        self.max_samples = int(
            generator.config.max_audio_length_ms * self.sample_rate / 1000
        )

    def iter_audio(
        self, payload: StagePayload
    ) -> Generator[NDArray[np.float32], None, None]:
        if self.cleanup_error is not None:
            raise RuntimeError(
                "VoXtream2 state reset failed; recreate the backend"
            ) from self.cleanup_error
        else:
            pass
        inputs = payload.request.inputs
        text = (
            inputs
            if isinstance(inputs, str)
            else inputs.get("text") if isinstance(inputs, dict) else None
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "VoXtream2 invalid request: VoXtream2 requires non-empty complete text"
            )
        else:
            pass
        params = payload.request.params
        stage_params = params.get("stage_params", {}).get("tts_engine", {})
        rate = stage_params.get(
            "speaking_rate_sps", params.get("speaking_rate_sps", 3.0)
        )
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(
                "VoXtream2 invalid request: speaking_rate_sps must be a finite positive number"
            )
        else:
            pass
        tts_params = payload.request.metadata.get("tts_params", {})
        unsupported = set(tts_params.get("explicit_generation_params", ())) - {"seed"}
        if unsupported:
            raise ValueError(
                "VoXtream2 invalid request: VoXtream2 does not support per-request "
                + ", ".join(sorted(unsupported))
            )
        else:
            pass
        if tts_params.get("instructions"):
            raise ValueError(
                "VoXtream2 invalid request: VoXtream2 does not support style instructions"
            )
        else:
            pass
        if tts_params.get("language") not in (None, "English", "Auto"):
            raise ValueError(
                "VoXtream2 invalid request: This VoXtream2 integration has only been validated for English"
            )
        else:
            pass
        seed = params.get("seed", tts_params.get("seed"))
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
        ):
            raise ValueError(
                "VoXtream2 invalid request: seed must be an integer in [0, 2**63)"
            )
        else:
            pass
        devices: list[int] = []
        if seed is not None and str(self.generator.ctx.device).startswith("cuda"):
            device_index = torch.device(self.generator.ctx.device).index
            devices = [
                torch.cuda.current_device() if device_index is None else device_index
            ]
        else:
            pass
        rng = (
            torch.random.fork_rng(devices=devices)
            if seed is not None
            else nullcontext()
        )
        with rng:
            if seed is not None:
                torch.default_generator.manual_seed(seed)
                for device_index in devices:
                    torch.cuda.default_generators[device_index].manual_seed(seed)
            else:
                pass
            yield from self.generate_audio(text, rate)

    def generate_audio(
        self, text: str, rate: float
    ) -> Generator[NDArray[np.float32], None, None]:
        stream = self.generator.generate_stream(
            self.reference_audio,
            text,
            speaking_rate=itertools.repeat(float(rate)),
            enhance_prompt=False,
            apply_vad=False,
        )
        try:
            for audio, _generation_seconds in stream:
                yield audio
        finally:
            # Generator.close() skips the author's code after the yield loop.
            # Reset on this same worker thread before another request starts.
            try:
                stream.close()
            finally:
                try:
                    # Mimi streaming tensors are created in inference mode.
                    with torch.inference_mode():
                        self.generator.model.reset_caches()
                        if (
                            self.generator._mimi_streaming_started
                        ):  # noqa: leading-underscore - author generator contract
                            self.generator.mimi.reset_streaming()
                        else:
                            pass
                except Exception as exc:
                    self.cleanup_error = exc
                    raise
