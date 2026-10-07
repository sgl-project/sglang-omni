# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR transcription on MLX: tokenizer, prompt, greedy decoding and stop rules."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
)

from sglang_omni_mlx.qwen3_asr.audio import AudioLayout, log_mel, token_count
from sglang_omni_mlx.qwen3_asr.model import KVCache, Qwen3ASR, load_qwen3_asr

# Qwen2's pre-tokenization split, as the checkpoint's tokenizer defines it.
QWEN2_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
ASR_TEXT_MARKER = "<asr_text>"
AUDIO_PAD = "<|audio_pad|>"
END_OF_TEXT = "<|endoftext|>"
IM_END = "<|im_end|>"
# Greedy loop guard: stop once the last 24 tokens use at most 3 distinct ids.
TOKEN_LOOP_WINDOW = 24
TOKEN_LOOP_MAX_DISTINCT = 3
OUTPUT_TOKENS_PER_AUDIO_SECOND = 10
MIN_DEFAULT_OUTPUT_TOKENS = 128

LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "fil": "Filipino", "fi": "Finnish", "fr": "French",
    "de": "German", "el": "Greek", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}  # fmt: skip
LANGUAGE_NAME_BY_CASEFOLD = {
    name.casefold(): name for name in LANGUAGE_CODE_TO_NAME.values()
}


class FinishReason(enum.Enum):
    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True, kw_only=True)
class TranscriptionOptions:
    language: str | None = None
    context: str | None = None
    max_new_tokens: int | None = None
    stop_at_end_of_text: bool = False
    stop_on_token_loop: bool = False
    layout: AudioLayout = AudioLayout.REFERENCE
    # Realtime refreshes continue from text already shown, as token ids and their text.
    prefix_token_ids: tuple[int, ...] = ()
    prefix_text: str = ""


@dataclass(frozen=True, kw_only=True)
class TranscriptionResult:
    text: str
    language: str | None
    generated_token_count: int
    finish_reason: FinishReason


class CancelCheck(Protocol):
    def is_set(self) -> bool: ...


class TranscriptionCancelled(Exception):
    """The caller gave up on the transcription."""


def normalize_language(language: str) -> str | None:
    """The canonical prompt name for a Qwen3-ASR language code or name.

    Like Voxt's Swift port, a name outside the table is used as given and a
    blank one means no language.
    """
    stripped = language.strip()
    normalized = stripped.casefold()
    if not stripped:
        return None
    elif normalized == "cn" or normalized.startswith(("zh-", "zh_")):
        return "Chinese"
    elif normalized in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[normalized]
    elif normalized in LANGUAGE_NAME_BY_CASEFOLD:
        return LANGUAGE_NAME_BY_CASEFOLD[normalized]
    else:
        return stripped


def load_tokenizer(model_directory: Path) -> Tokenizer:
    """The checkpoint's Qwen2 byte-level BPE tokenizer, with its added tokens."""
    tokenizer = Tokenizer(
        models.BPE.from_file(
            str(model_directory / "vocab.json"),
            str(model_directory / "merges.txt"),
            byte_fallback=False,
        )
    )
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN2_SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer_config = json.loads(
        (model_directory / "tokenizer_config.json").read_text()
    )
    added = sorted(
        tokenizer_config["added_tokens_decoder"].items(),
        key=lambda entry: int(entry[0]),
    )
    for token_id, token in added:
        added_token = AddedToken(
            token["content"], special=token["special"], normalized=False
        )
        if token["special"]:
            tokenizer.add_special_tokens([added_token])
        else:
            tokenizer.add_tokens([added_token])
        if tokenizer.token_to_id(token["content"]) != int(token_id):
            raise ValueError(
                f"Added token {token['content']!r} does not get id {token_id}"
            )
        else:
            pass
    return tokenizer


class Qwen3ASRTranscriber:
    """One loaded checkpoint; callers serialize access, MLX runs on one thread."""

    def __init__(self, model_directory: Path) -> None:
        self.model: Qwen3ASR = load_qwen3_asr(model_directory)
        self.tokenizer = load_tokenizer(model_directory)
        self.audio_pad_id = self.tokenizer.token_to_id(AUDIO_PAD)
        self.end_of_text_id = self.tokenizer.token_to_id(END_OF_TEXT)
        self.im_end_id = self.tokenizer.token_to_id(IM_END)
        self.asr_text_ids = self.tokenizer.encode(
            ASR_TEXT_MARKER, add_special_tokens=False
        ).ids

    def prompt_ids(
        self, audio_token_count: int, options: TranscriptionOptions
    ) -> list[int]:
        context = (options.context or "").strip()
        prompt = (
            f"<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>"
            + AUDIO_PAD * audio_token_count
            + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        )
        language = normalize_language(options.language or "")
        if language is not None:
            prompt += f"language {language}{ASR_TEXT_MARKER}"
        else:
            pass
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids + list(
            options.prefix_token_ids
        )

    def retained_prefix(
        self, text: str, rollback_token_count: int
    ) -> tuple[tuple[int, ...], str]:
        """Text already shown minus its last tokens, cut back to whole characters."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        retained = token_ids[: max(len(token_ids) - rollback_token_count, 0)]
        while retained:
            decoded = self.tokenizer.decode(retained, skip_special_tokens=False)
            if decoded.endswith("\ufffd"):
                retained.pop()
            else:
                return tuple(retained), decoded
        return (), ""

    def transcribe(
        self, samples: np.ndarray, options: TranscriptionOptions, cancel: CancelCheck
    ) -> TranscriptionResult:
        mel = log_mel(samples, options.layout)
        audio_token_count = token_count(mel.shape[-1], options.layout)
        prompt_ids = self.prompt_ids(audio_token_count, options)
        audio_features = self.model.audio_tower(mel, options.layout)
        input_ids = mx.array([prompt_ids], dtype=mx.int32)
        embeddings = self.model.model.embed_tokens(input_ids)
        audio_start = prompt_ids.index(self.audio_pad_id)
        # The Swift layout can reserve more placeholders than encoder rows; those
        # keep the audio_pad embedding, and surplus rows are dropped.
        filled = min(audio_token_count, audio_features.shape[0])
        embeddings[0, audio_start : audio_start + filled, :] = audio_features[
            :filled
        ].astype(embeddings.dtype)

        max_new_tokens = options.max_new_tokens or max(
            MIN_DEFAULT_OUTPUT_TOKENS,
            int(np.ceil(len(samples) / 16000 * OUTPUT_TOKENS_PER_AUDIO_SECOND)),
        )
        stop_ids = (
            {self.im_end_id, self.end_of_text_id}
            if options.stop_at_end_of_text
            else {self.im_end_id}
        )
        caches = self.model.new_caches()
        next_token = mx.argmax(self.model.model(embeddings, caches))
        mx.async_eval(next_token)
        output_ids: list[int] = []
        finish_reason = FinishReason.LENGTH
        while len(output_ids) < max_new_tokens:
            if cancel.is_set():
                raise TranscriptionCancelled()
            else:
                pass
            token = next_token
            # Queue the following step before reading this token, so the GPU
            # decodes while Python checks the stop rules.
            next_token = self.next_token(token, caches)
            mx.async_eval(next_token)
            token_id = int(token.item())
            output_ids.append(token_id)
            if token_id in stop_ids:
                finish_reason = FinishReason.STOP
                break
            elif options.stop_on_token_loop and is_token_loop(output_ids):
                finish_reason = FinishReason.STOP
                break
            else:
                pass
        text, language = self.split_output(output_ids, options)
        ended_on_stop = bool(output_ids) and output_ids[-1] in stop_ids
        return TranscriptionResult(
            text=options.prefix_text + text,
            language=language,
            generated_token_count=len(output_ids) - int(ended_on_stop),
            finish_reason=finish_reason,
        )

    def next_token(self, token: mx.array, caches: list[KVCache]) -> mx.array:
        """Greedy token after token, appending its keys and values to caches."""
        return mx.argmax(
            self.model.model(self.model.model.embed_tokens(token.reshape(1, 1)), caches)
        )

    def split_output(
        self, output_ids: list[int], options: TranscriptionOptions
    ) -> tuple[str, str | None]:
        """Transcript text and the language the model declared before it."""
        marker = find_subsequence(output_ids, self.asr_text_ids)
        detected = None
        if options.language is None and marker is not None:
            prefix = self.tokenizer.decode(
                output_ids[:marker], skip_special_tokens=True
            ).strip()
            label, separator, value = prefix.partition(" ")
            if separator and label.casefold() == "language":
                detected = value.strip() or None
                if detected is not None and detected.casefold() == "none":
                    detected = None
                else:
                    pass
            elif prefix:
                detected = prefix
            else:
                pass
        else:
            pass
        transcript_ids = (
            output_ids[marker + len(self.asr_text_ids) :]
            if marker is not None
            else output_ids
        )
        text = self.tokenizer.decode(transcript_ids, skip_special_tokens=True)
        language = normalize_language(options.language or "") or (
            normalize_language(detected) if detected is not None else None
        )
        return text, language


def is_token_loop(output_ids: list[int]) -> bool:
    return (
        len(output_ids) >= TOKEN_LOOP_WINDOW
        and len(set(output_ids[-TOKEN_LOOP_WINDOW:])) <= TOKEN_LOOP_MAX_DISTINCT
    )


def find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    for start in range(len(values) - len(pattern) + 1):
        if values[start : start + len(pattern)] == pattern:
            return start
        else:
            pass
    return None
