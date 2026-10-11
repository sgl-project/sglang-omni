# SPDX-License-Identifier: Apache-2.0
"""MOSS-TD prompt construction, audio encoding and greedy transcription."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from sglang_omni_mlx.moss_transcribe_diarize.audio import audio_features
from sglang_omni_mlx.moss_transcribe_diarize.model import (
    MossTranscribeDiarize,
    load_moss_transcribe_diarize,
)
from sglang_omni_mlx.text_decoder import greedy_tokens
from sglang_omni_mlx.transcription import CancelCheck, FinishReason
from sglang_omni_mlx.wav import SAMPLE_RATE

AUDIO_PAD = "<|audio_pad|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
IM_END = "<|im_end|>"
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
OUTPUT_TOKENS_PER_AUDIO_SECOND = 10
MIN_DEFAULT_OUTPUT_TOKENS = 512
PREFILL_CHUNK_SIZE = 2048


@dataclass(frozen=True, kw_only=True)
class TranscriptionOptions:
    prompt: str | None = None
    max_new_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptionResult:
    text: str
    generated_token_count: int
    finish_reason: FinishReason


def default_output_tokens(audio_seconds: float) -> int:
    return max(
        MIN_DEFAULT_OUTPUT_TOKENS,
        math.ceil(audio_seconds * OUTPUT_TOKENS_PER_AUDIO_SECOND),
    )


def audio_span_ids(
    audio_token_count: int,
    audio_pad_id: int,
    digit_ids: dict[str, int],
    tokens_per_second: float,
    marker_interval_seconds: int,
) -> list[int]:
    """Audio placeholders with numeric time anchors at checkpoint intervals."""
    marker_token_count = int(tokens_per_second * marker_interval_seconds)
    duration_seconds = audio_token_count / tokens_per_second
    output: list[int] = []
    consumed = 0
    for seconds in range(
        marker_interval_seconds,
        int(duration_seconds) + 1,
        marker_interval_seconds,
    ):
        position = seconds // marker_interval_seconds * marker_token_count
        output.extend([audio_pad_id] * (position - consumed))
        consumed = position
        output.extend(digit_ids[digit] for digit in str(seconds))
    output.extend([audio_pad_id] * (audio_token_count - consumed))
    return output


class MossTranscribeDiarizeTranscriber:
    """One loaded checkpoint; callers serialize access, MLX runs on one thread."""

    def __init__(self, model_directory: Path) -> None:
        self.model: MossTranscribeDiarize = load_moss_transcribe_diarize(
            model_directory
        )
        self.tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
        self.audio_pad_id = self.tokenizer.token_to_id(AUDIO_PAD)
        self.im_end_id = self.tokenizer.token_to_id(IM_END)
        self.digit_ids = {
            digit: self.tokenizer.encode(digit, add_special_tokens=False).ids[0]
            for digit in "0123456789"
        }
        model_config = json.loads((model_directory / "config.json").read_text())
        processor_config = json.loads(
            (model_directory / "processor_config.json").read_text()
        )
        self.context_length = model_config["text_config"]["max_position_embeddings"]
        self.tokens_per_second = processor_config["audio_tokens_per_second"]
        self.marker_interval_seconds = processor_config["time_marker_every_seconds"]

    def prompt_ids(
        self, audio_token_count: int, options: TranscriptionOptions
    ) -> list[int]:
        prefix = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{AUDIO_START}"
        )
        suffix = (
            f"{AUDIO_END}\n{options.prompt or DEFAULT_PROMPT}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        return (
            self.tokenizer.encode(prefix, add_special_tokens=False).ids
            + audio_span_ids(
                audio_token_count,
                self.audio_pad_id,
                self.digit_ids,
                self.tokens_per_second,
                self.marker_interval_seconds,
            )
            + self.tokenizer.encode(suffix, add_special_tokens=False).ids
        )

    def transcribe(
        self, samples: np.ndarray, options: TranscriptionOptions, cancel: CancelCheck
    ) -> TranscriptionResult:
        features, audio_token_lengths = audio_features(samples)
        audio_token_count = int(audio_token_lengths.sum())
        prompt_ids = self.prompt_ids(audio_token_count, options)
        audio_embeddings = self.model.encode_audio(
            mx.array(features), audio_token_lengths
        )
        input_ids = mx.array([prompt_ids], dtype=mx.int32)
        embeddings = self.model.model.embed_tokens(input_ids)
        audio_positions = [
            index
            for index, token_id in enumerate(prompt_ids)
            if token_id == self.audio_pad_id
        ]
        assert len(audio_positions) == audio_embeddings.shape[0]
        embeddings[0, mx.array(audio_positions), :] = audio_embeddings.astype(
            embeddings.dtype
        )

        remaining_context = self.context_length - len(prompt_ids)
        if remaining_context < 1:
            raise ValueError("audio prompt exceeds the model context length")
        else:
            pass
        max_new_tokens = min(
            options.max_new_tokens or default_output_tokens(len(samples) / SAMPLE_RATE),
            remaining_context,
        )
        tokens = greedy_tokens(
            self.model.model,
            embeddings,
            self.model.new_caches(),
            cancel,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
        )
        output_ids: list[int] = []
        finish_reason = FinishReason.LENGTH
        for token_id in itertools.islice(tokens, max_new_tokens):
            if token_id == self.im_end_id:
                finish_reason = FinishReason.STOP
                break
            else:
                output_ids.append(token_id)
        return TranscriptionResult(
            text=self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            generated_token_count=len(output_ids),
            finish_reason=finish_reason,
        )
