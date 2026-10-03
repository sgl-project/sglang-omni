# SPDX-License-Identifier: Apache-2.0
"""Preprocessing and cooperative codec generation for native MLX serving."""

from __future__ import annotations

import logging
import queue
import secrets
import time
from collections import deque
from collections.abc import Generator
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import snapshot_download

from sglang_omni.models.qwen3_tts.config import (
    load_qwen3_tts_checkpoint_config,
    normalize_qwen3_tts_model_type,
)
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    QWEN3_TTS_TASK_CUSTOM_VOICE,
    build_qwen3_tts_state,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)
SUPPORTED_MLX_GENERATION_FIELDS = frozenset(
    {"max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"}
)


class MlxCodecGeneration(Protocol):
    def __call__(
        self, payload: StagePayload
    ) -> Generator[OutgoingMessage, None, None]: ...


class Qwen3TTSMlxScheduler(SimpleScheduler):
    """Advance active requests round-robin on one MLX thread."""

    def __init__(self, generate: MlxCodecGeneration, *, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("Qwen3-TTS MLX max_concurrency must be positive")
        else:
            pass
        super().__init__(generate, max_concurrency=max_concurrency)
        self.fn: MlxCodecGeneration = generate

    def start(self) -> None:
        self.running = True
        active: deque[tuple[str, Generator[OutgoingMessage, None, None]]] = deque()
        try:
            while self.running:
                while len(active) < self.max_concurrency and self.running:
                    try:
                        message = self.inbox.get(timeout=0.0 if active else 0.1)
                    except queue.Empty:
                        break
                    if message.type != "new_request" or self.consume_if_aborted(
                        message.request_id
                    ):
                        continue
                    else:
                        active.append((message.request_id, self.fn(message.data)))
                if not active:
                    continue
                else:
                    request_id, messages = active.popleft()
                if self.consume_if_aborted(request_id):
                    messages.close()
                    continue
                else:
                    pass
                try:
                    outgoing = next(messages)
                except StopIteration:
                    messages.close()
                except Exception as error:
                    logger.exception(
                        f"Qwen3-TTS MLX generation failed for {request_id}"
                    )
                    if not self.consume_if_aborted(request_id):
                        self.emit_error(request_id, error, self.outbox)
                    else:
                        pass
                    messages.close()
                else:
                    if self.consume_if_aborted(request_id):
                        messages.close()
                    else:
                        self.outbox.put(outgoing)
                        if outgoing.type == "result":
                            messages.close()
                        else:
                            active.append((request_id, messages))
        finally:
            for _, messages in active:
                messages.close()


def resolve_mlx_model_dir(
    model_path: str,
    *,
    gpu_id: int | None,
    mlx_model_path: str | None,
    mlx_model_revision: str | None,
) -> Path:
    """Validate the optional backend and resolve its converted checkpoint."""
    from sglang.srt.hardware_backend.mlx.runtime import use_mlx

    if not current_platform.is_mps() or not use_mlx():
        raise RuntimeError("Qwen3-TTS MLX requires Apple Metal and SGLANG_USE_MLX=1")
    else:
        pass
    if gpu_id not in (None, 0):
        raise ValueError("Qwen3-TTS MLX supports only Metal device 0")
    else:
        pass
    if not mlx_model_path:
        raise ValueError("Qwen3-TTS MLX requires factory.mlx_model_path")
    else:
        pass
    checkpoint_config = load_qwen3_tts_checkpoint_config(model_path)
    model_type = normalize_qwen3_tts_model_type(checkpoint_config.get("tts_model_type"))
    if model_type != "custom_voice":
        raise ValueError(
            "Qwen3-TTS MLX currently supports CustomVoice checkpoints only"
        )
    else:
        pass
    converted_dir = Path(mlx_model_path).expanduser()
    if converted_dir.is_dir():
        return converted_dir
    else:
        return Path(
            snapshot_download(repo_id=mlx_model_path, revision=mlx_model_revision)
        )


def create_mlx_preprocessing_executor() -> SimpleScheduler:
    """Normalize and validate requests without loading model weights."""

    def preprocess(payload: StagePayload) -> StagePayload:
        tts_params = (payload.request.metadata or {}).get("tts_params")
        if not isinstance(tts_params, dict):
            tts_params = {}
        else:
            pass
        if float(tts_params.get("speed", 1.0)) != 1.0:
            raise ValueError("Qwen3-TTS MLX currently supports speed=1.0 only")
        else:
            pass
        state = build_qwen3_tts_state(payload, default_stream_codec_output=True)
        if state.task_type != QWEN3_TTS_TASK_CUSTOM_VOICE:
            raise ValueError("Qwen3-TTS MLX currently supports CustomVoice only")
        else:
            pass
        if state.instructions is not None:
            raise ValueError("Qwen3-TTS MLX currently does not support instructions")
        else:
            pass
        unsupported = state.generation_kwargs.keys() - SUPPORTED_MLX_GENERATION_FIELDS
        if unsupported:
            raise ValueError(
                f"Qwen3-TTS MLX does not support generation parameters: {sorted(unsupported)}"
            )
        else:
            pass
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(preprocess)


def create_mlx_tts_executor(
    model_path: str,
    *,
    stream_chunk_frames: int,
    max_concurrency: int,
    gpu_id: int | None = None,
    mlx_model_path: str | None = None,
    mlx_model_revision: str | None = None,
) -> Qwen3TTSMlxScheduler:
    """Load shared talker weights and publish request-local codec chunks."""
    import mlx.core as mx

    from sglang_omni.models.qwen3_tts.mlx.generate import Qwen3TTSMlxCodeGenerator

    if stream_chunk_frames <= 0:
        raise ValueError("Qwen3-TTS MLX stream_chunk_frames must be positive")
    else:
        pass
    converted_dir = resolve_mlx_model_dir(
        model_path,
        gpu_id=gpu_id,
        mlx_model_path=mlx_model_path,
        mlx_model_revision=mlx_model_revision,
    )
    generator = Qwen3TTSMlxCodeGenerator(converted_dir)
    speakers = {
        name.casefold(): name for name in generator.talker.artifact.talker_config.spk_id
    }

    def generate(payload: StagePayload) -> Generator[OutgoingMessage, None, None]:
        state = Qwen3TTSState.from_dict(payload.data)
        voice = speakers.get((state.voice or "").casefold())
        if voice is None:
            raise ValueError(f"Qwen3-TTS MLX does not support voice {state.voice!r}")
        else:
            pass
        generation = state.generation_kwargs
        started = time.perf_counter()
        token_count = 0
        with closing(
            generator.generate_codes(
                text=state.text,
                voice=voice,
                language=state.language,
                max_new_tokens=int(generation["max_new_tokens"]),
                temperature=float(generation.get("temperature", 0.9)),
                top_k=int(generation.get("top_k", 50)),
                top_p=float(generation.get("top_p", 1.0)),
                repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
                seed=state.seed if state.seed is not None else secrets.randbits(32),
            )
        ) as frames:
            while chunk := list(islice(frames, stream_chunk_frames)):
                token_count += len(chunk)
                yield OutgoingMessage(
                    request_id=payload.request_id,
                    type="stream",
                    data=np.asarray(mx.stack(chunk, axis=1), dtype=np.int32),
                    metadata={
                        "modality": "audio_codes",
                        "is_streaming": bool(payload.request.params.get("stream")),
                    },
                )
        state.completion_tokens = token_count
        state.engine_time_s = time.perf_counter() - started
        payload.data = state.to_dict()
        yield OutgoingMessage(
            request_id=payload.request_id, type="result", data=payload
        )

    return Qwen3TTSMlxScheduler(generate, max_concurrency=max_concurrency)


def create_mlx_vocoder_executor(
    model_path: str,
    *,
    gpu_id: int | None = None,
    mlx_model_path: str | None = None,
    mlx_model_revision: str | None = None,
) -> StreamingSimpleScheduler:
    """Load the speech decoder once for all active requests."""
    from sglang_omni.models.qwen3_tts.mlx.vocoder import Qwen3TTSMlxVocoder

    converted_dir = resolve_mlx_model_dir(
        model_path,
        gpu_id=gpu_id,
        mlx_model_path=mlx_model_path,
        mlx_model_revision=mlx_model_revision,
    )
    return Qwen3TTSMlxVocoder(converted_dir)
