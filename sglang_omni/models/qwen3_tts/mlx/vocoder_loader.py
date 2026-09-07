# SPDX-License-Identifier: Apache-2.0
"""Build the MLX Qwen3-TTS vocoder stage from a checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def load_mlx_speech_tokenizer(model_path: str | Path) -> Any:
    """Load the speech tokenizer from a checkpoint's ``speech_tokenizer/``."""
    from .config import TokenizerConfig
    from .speech_tokenizer import Qwen3TTSSpeechTokenizer
    from .weights import align_conv_weights

    tokenizer_dir = Path(model_path) / "speech_tokenizer"
    config_path = tokenizer_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Qwen3-TTS MLX vocoder needs {config_path}; the checkpoint has no "
            "speech tokenizer"
        )

    config = TokenizerConfig.from_dict(json.loads(config_path.read_text()))
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    weights = mx.load(str(tokenizer_dir / "model.safetensors"))
    tokenizer.load_weights(
        list(align_conv_weights(tokenizer.sanitize(weights), tokenizer).items()),
        strict=True,
    )
    tokenizer.eval()
    mx.eval(tokenizer.parameters())
    return tokenizer


def create_mlx_vocoder_scheduler(
    model_path: str,
    *,
    stream_stride: int | None = None,
    initial_chunk_frames: int | None = None,
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 0,
) -> Any:
    """Construct the MLX vocoder stage scheduler."""
    from .streaming_vocoder import (
        DEFAULT_MLX_STREAM_STRIDE,
        Qwen3TTSMlxStreamingVocoder,
    )

    tokenizer = load_mlx_speech_tokenizer(model_path)
    logger.info("Loaded MLX Qwen3-TTS speech tokenizer from %s", model_path)
    return Qwen3TTSMlxStreamingVocoder(
        tokenizer,
        sample_rate=int(tokenizer.sample_rate),
        stream_stride=(
            DEFAULT_MLX_STREAM_STRIDE if stream_stride is None else stream_stride
        ),
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
