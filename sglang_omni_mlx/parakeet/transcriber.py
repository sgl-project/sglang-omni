# SPDX-License-Identifier: Apache-2.0
"""Parakeet transcription on MLX: features, greedy decoding, detokenization and long audio."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from sglang_omni_mlx.parakeet.audio import (
    MAX_AUDIO_SECONDS,
    chunk_spans,
    is_silent,
    log_mel_features,
    mel_filter_bank,
)
from sglang_omni_mlx.parakeet.model import ParakeetModel, load_parakeet
from sglang_omni_mlx.transcription import (
    CancelCheck,
    FinishReason,
    TranscriptionCancelled,
)
from sglang_omni_mlx.wav import SAMPLE_RATE

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


@dataclass(frozen=True, kw_only=True)
class TranscriptionOptions:
    """Parakeet has no prompt, language hint or token budget to set."""


@dataclass(frozen=True, kw_only=True)
class TranscriptionResult:
    text: str
    generated_token_count: int
    chunk_count: int
    finish_reason: FinishReason = FinishReason.STOP


def require_supported_duration(samples: np.ndarray) -> None:
    if len(samples) / SAMPLE_RATE > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Parakeet accepts audio up to {MAX_AUDIO_SECONDS:.0f} seconds, "
            f"got {len(samples) / SAMPLE_RATE:.3f} seconds"
        )
    else:
        pass


def collapse_ctc(frame_ids: list[int], blank_id: int) -> list[int]:
    """One id per run of identical frames, blanks removed."""
    return [token for token, _ in itertools.groupby(frame_ids) if token != blank_id]


class ParakeetTranscriber:
    """One loaded checkpoint; callers serialize access, MLX runs on one thread."""

    def __init__(self, model_directory: Path, dtype: str = "bfloat16") -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
        else:
            pass
        self.dtype = DTYPES[dtype]
        self.model: ParakeetModel = load_parakeet(model_directory, self.dtype)
        self.tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
        self.mel_filters = mel_filter_bank(self.model.config.encoder.num_mel_bins)

    def token_ids(self, samples: np.ndarray, cancel: CancelCheck) -> list[int]:
        features = log_mel_features(samples, self.mel_filters)
        output = self.model.greedy_decode(
            mx.array(features)[None].astype(self.dtype), None, cancel
        )[0]
        config = self.model.config
        if config.is_ctc:
            return collapse_ctc(output, int(config.pad_token_id))
        else:
            return output

    def transcribe(
        self, samples: np.ndarray, options: TranscriptionOptions, cancel: CancelCheck
    ) -> TranscriptionResult:
        require_supported_duration(samples)
        texts: list[str] = []
        token_count = 0
        spans = chunk_spans(samples)
        for start, end in spans:
            if cancel.is_set():
                raise TranscriptionCancelled()
            else:
                pass
            chunk = samples[start:end]
            if is_silent(chunk):
                continue
            else:
                pass
            ids = self.token_ids(chunk, cancel)
            token_count += len(ids)
            text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            if text:
                texts.append(text)
            else:
                pass
        return TranscriptionResult(
            text=" ".join(texts),
            generated_token_count=token_count,
            chunk_count=len(spans),
        )
