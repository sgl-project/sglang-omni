# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR transcription on MLX: tokenizer, prompt, greedy decoding and stop rules."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from sglang_omni_mlx.fun_asr.audio import (
    MAX_AUDIO_SECONDS,
    SAMPLE_RATE,
    audio_features,
    token_count,
)
from sglang_omni_mlx.fun_asr.model import FunASR, load_fun_asr
from sglang_omni_mlx.text_decoder import greedy_tokens
from sglang_omni_mlx.transcription import CancelCheck, FinishReason

AUDIO_PLACEHOLDER = "<|object_ref_start|>"
IM_END = "<|im_end|>"
# Output budget: 200 tokens for a full 30 s clip, proportionally less for shorter ones.
MAX_OUTPUT_TOKENS_AT_MAX_DURATION = 200
MIN_DEFAULT_OUTPUT_TOKENS = 16
MAX_OUTPUT_TOKENS = 512
HOTWORD_PREAMBLE = (
    "请结合上下文信息，更加准确地完成语音转写任务。"
    "如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
)


@dataclass(frozen=True, kw_only=True)
class TranscriptionOptions:
    # Fun-ASR's prompt language: None transcribes as spoken, "英文" asks for English.
    language: str | None = None
    itn: bool = True
    hotwords: tuple[str, ...] = ()
    max_new_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptionResult:
    text: str
    generated_token_count: int
    finish_reason: FinishReason


def normalize_language(language: str) -> str | None:
    """The prompt language for a request's language field.

    Fun-ASR-Nano transcribes Chinese by default, so Chinese and auto both mean
    no instruction; English is named the way the model was trained.
    """
    normalized = language.strip().casefold()
    if normalized in ("", "auto", "null", "none", "zh", "cn", "chinese", "中文"):
        return None
    elif normalized in ("en", "english", "英文"):
        return "英文"
    else:
        return language.strip()


def prompt_text(options: TranscriptionOptions) -> str:
    text = ""
    if options.hotwords:
        text += HOTWORD_PREAMBLE + f"热词列表：[{', '.join(options.hotwords)}]\n"
    else:
        pass
    if options.language is None:
        text += "语音转写"
    else:
        text += f"语音转写成{options.language}"
    if not options.itn:
        text += "，不进行文本规整"
    else:
        pass
    return text + "："


def require_supported_duration(samples: np.ndarray) -> None:
    if len(samples) / SAMPLE_RATE > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Fun-ASR accepts audio up to {MAX_AUDIO_SECONDS:.0f} seconds; "
            "split longer audio before transcribing"
        )
    else:
        pass


def default_output_tokens(audio_seconds: float) -> int:
    proportional = math.ceil(
        audio_seconds / MAX_AUDIO_SECONDS * MAX_OUTPUT_TOKENS_AT_MAX_DURATION
    )
    return max(MIN_DEFAULT_OUTPUT_TOKENS, proportional)


class FunASRTranscriber:
    """One loaded checkpoint; callers serialize access, MLX runs on one thread."""

    def __init__(self, model_directory: Path) -> None:
        self.model: FunASR = load_fun_asr(model_directory)
        self.tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
        self.audio_placeholder_id = self.tokenizer.token_to_id(AUDIO_PLACEHOLDER)
        self.im_end_id = self.tokenizer.token_to_id(IM_END)

    def prompt_ids(
        self, audio_token_count: int, options: TranscriptionOptions
    ) -> list[int]:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text(options)}"
            + AUDIO_PLACEHOLDER * audio_token_count
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids

    def transcribe(
        self, samples: np.ndarray, options: TranscriptionOptions, cancel: CancelCheck
    ) -> TranscriptionResult:
        require_supported_duration(samples)
        audio_seconds = len(samples) / SAMPLE_RATE
        features = audio_features(samples)
        audio_token_count = token_count(features.shape[0])
        prompt_ids = self.prompt_ids(audio_token_count, options)
        audio_start = prompt_ids.index(self.audio_placeholder_id)
        embeddings = self.model.model.embed_tokens(
            mx.array([prompt_ids], dtype=mx.int32)
        )
        embeddings[0, audio_start : audio_start + audio_token_count, :] = (
            self.model.encode_audio(mx.array(features), audio_token_count).astype(
                embeddings.dtype
            )
        )

        max_new_tokens = options.max_new_tokens or default_output_tokens(audio_seconds)
        tokens = greedy_tokens(
            self.model.model, embeddings, self.model.new_caches(), cancel
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
            text=self.tokenizer.decode(output_ids, skip_special_tokens=True),
            generated_token_count=len(output_ids),
            finish_reason=finish_reason,
        )
