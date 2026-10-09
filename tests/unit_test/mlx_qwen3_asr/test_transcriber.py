# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from sglang_omni_mlx.qwen3_asr.audio import AudioLayout  # noqa: E402
from sglang_omni_mlx.qwen3_asr.transcriber import (  # noqa: E402
    FinishReason,
    Qwen3ASRTranscriber,
    TranscriptionCancelled,
    TranscriptionOptions,
    load_tokenizer,
    normalize_language,
)

SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<|endoftext|>",
    "<asr_text>",
]
SPECIAL_IDS = {token: 1000 + index for index, token in enumerate(SPECIAL_TOKENS)}
SPECIAL_SPLIT = re.compile(
    "(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")"
)
VOCAB_SIZE = 1100
IM_END_ID = SPECIAL_IDS["<|im_end|>"]
END_OF_TEXT_ID = SPECIAL_IDS["<|endoftext|>"]
ONE_SECOND = np.full(16000, 0.1, dtype=np.float32)
REAL_CHECKPOINT = os.environ.get("QWEN3_ASR_MLX_MODEL_PATH")


class CharacterTokenizer:
    """One id per character, plus the Qwen3-ASR special tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        ids: list[int] = []
        for piece in SPECIAL_SPLIT.split(text):
            if piece in SPECIAL_IDS:
                ids.append(SPECIAL_IDS[piece])
            else:
                ids.extend(ord(character) for character in piece)
        return SimpleNamespace(ids=ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        names = {token_id: token for token, token_id in SPECIAL_IDS.items()}
        return "".join(
            (
                ("" if skip_special_tokens else names[token_id])
                if token_id in names
                else chr(token_id)
            )
            for token_id in ids
        )

    def token_to_id(self, token: str) -> int:
        return SPECIAL_IDS[token]


class ScriptedDecoder:
    """Returns logits for a fixed sequence of tokens, one per forward call."""

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.call_count = 0
        self.prompts: list[int] = []

    def embed_tokens(self, ids: mx.array) -> mx.array:
        return mx.zeros((1, ids.shape[1], 4))

    def __call__(self, embeddings: mx.array, caches: list[object]) -> mx.array:
        if self.call_count == 0:
            self.prompts.append(embeddings.shape[1])
        else:
            pass
        token_id = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        return mx.array(np.eye(VOCAB_SIZE, dtype=np.float32)[token_id])


def scripted_transcriber(
    output_text: str, *tail_ids: int
) -> tuple[Qwen3ASRTranscriber, ScriptedDecoder]:
    tokenizer = CharacterTokenizer()
    decoder = ScriptedDecoder(tokenizer.encode(output_text).ids + list(tail_ids))
    transcriber = Qwen3ASRTranscriber.__new__(Qwen3ASRTranscriber)
    transcriber.model = SimpleNamespace(
        audio_tower=lambda mel, layout: mx.zeros((mel.shape[-1] // 8, 4)),
        model=decoder,
        new_caches=lambda: [],
    )
    transcriber.tokenizer = tokenizer
    transcriber.audio_pad_id = SPECIAL_IDS["<|audio_pad|>"]
    transcriber.end_of_text_id = END_OF_TEXT_ID
    transcriber.im_end_id = IM_END_ID
    transcriber.asr_text_ids = [SPECIAL_IDS["<asr_text>"]]
    return transcriber, decoder


def transcribe(transcriber: Qwen3ASRTranscriber, **options: object):
    return transcriber.transcribe(
        ONE_SECOND, TranscriptionOptions(**options), threading.Event()
    )


def test_stops_at_im_end_and_reads_the_declared_language() -> None:
    transcriber, _ = scripted_transcriber("language English<asr_text>hello", IM_END_ID)
    result = transcribe(transcriber)
    assert result.text == "hello"
    assert result.language == "English"
    assert result.finish_reason is FinishReason.STOP
    assert result.generated_token_count == len("language English") + 1 + len("hello")


def test_language_none_means_no_language() -> None:
    transcriber, _ = scripted_transcriber("language None<asr_text>", IM_END_ID)
    result = transcribe(transcriber)
    assert (result.text, result.language) == ("", None)


def test_a_forced_language_goes_in_the_prompt() -> None:
    transcriber, _ = scripted_transcriber("hello", IM_END_ID)
    automatic, _ = scripted_transcriber("hello", IM_END_ID)
    result = transcribe(transcriber, language="en")
    assert (result.text, result.language) == ("hello", "English")
    forced_prompt = transcriber.prompt_ids(5, TranscriptionOptions(language="en"))
    assert (
        CharacterTokenizer()
        .decode(forced_prompt)
        .endswith("<|im_start|>assistant\nlanguage English<asr_text>")
    )
    assert (
        len(forced_prompt)
        == len(automatic.prompt_ids(5, TranscriptionOptions()))
        + len("language English")
        + 1
    )


def test_end_of_text_stops_only_when_asked() -> None:
    transcriber, _ = scripted_transcriber("ab", END_OF_TEXT_ID, ord("c"), IM_END_ID)
    assert transcribe(transcriber, language="en").text == "abc"
    transcriber, _ = scripted_transcriber("ab", END_OF_TEXT_ID, ord("c"), IM_END_ID)
    result = transcribe(transcriber, language="en", stop_at_end_of_text=True)
    assert (result.text, result.generated_token_count, result.finish_reason) == (
        "ab",
        2,
        FinishReason.STOP,
    )


def test_token_loop_stops_only_when_asked() -> None:
    transcriber, _ = scripted_transcriber("x" * 30, IM_END_ID)
    assert transcribe(transcriber, language="en").text == "x" * 30
    transcriber, _ = scripted_transcriber("x" * 30, IM_END_ID)
    result = transcribe(transcriber, language="en", stop_on_token_loop=True)
    assert (result.text, result.finish_reason) == ("x" * 24, FinishReason.STOP)


def test_max_new_tokens_ends_with_length() -> None:
    transcriber, _ = scripted_transcriber("abcdef", IM_END_ID)
    result = transcribe(transcriber, language="en", max_new_tokens=3)
    assert (result.text, result.generated_token_count, result.finish_reason) == (
        "abc",
        3,
        FinishReason.LENGTH,
    )


def test_default_budget_is_ten_tokens_per_second_with_a_floor_of_128() -> None:
    transcriber, _ = scripted_transcriber("y" * 200)
    assert transcribe(transcriber, language="en").generated_token_count == 128
    transcriber, _ = scripted_transcriber("y" * 200)
    result = transcriber.transcribe(
        np.tile(ONE_SECOND, 15), TranscriptionOptions(language="en"), threading.Event()
    )
    assert result.generated_token_count == 150


def test_prefix_continues_the_prompt_and_the_text() -> None:
    transcriber, decoder = scripted_transcriber("world", IM_END_ID)
    prefix_ids = tuple(CharacterTokenizer().encode("Hello ").ids)
    options = TranscriptionOptions(
        language="en", prefix_token_ids=prefix_ids, prefix_text="Hello "
    )
    result = transcriber.transcribe(ONE_SECOND, options, threading.Event())
    assert result.text == "Hello world"
    assert result.generated_token_count == len("world")
    # One second is 100 mel frames, 13 audio tokens.
    prompt = transcriber.prompt_ids(13, options)
    assert tuple(prompt[-len(prefix_ids) :]) == prefix_ids
    assert decoder.prompts == [len(prompt)]


def test_cancellation_stops_decoding() -> None:
    transcriber, _ = scripted_transcriber("z" * 50, IM_END_ID)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe(ONE_SECOND, TranscriptionOptions(language="en"), cancel)


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en", "English"),
        ("English", "English"),
        ("zh-CN", "Chinese"),
        ("cn", "Chinese"),
        ("yue", "Cantonese"),
        # Like Voxt's Swift port: an unknown name is used as given, never an error.
        ("  Klingon ", "Klingon"),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_language(language: str, expected: str | None) -> None:
    assert normalize_language(language) == expected


def test_an_unknown_language_goes_into_the_prompt_as_given() -> None:
    transcriber, _ = scripted_transcriber("hello", IM_END_ID)
    prompt = transcriber.prompt_ids(3, TranscriptionOptions(language="Klingon"))
    assert CharacterTokenizer().decode(prompt).endswith("language Klingon<asr_text>")
    result = transcribe(transcriber, language="Klingon")
    assert (result.text, result.language) == ("hello", "Klingon")


def test_a_detected_language_outside_the_table_is_reported_as_given() -> None:
    transcriber, _ = scripted_transcriber("language Elvish<asr_text>hi", IM_END_ID)
    assert transcribe(transcriber).language == "Elvish"


@pytest.mark.skipif(
    REAL_CHECKPOINT is None,
    reason="set QWEN3_ASR_MLX_MODEL_PATH to a Qwen3-ASR MLX checkpoint",
)
def test_tokenizer_matches_the_checkpoint_tokenizer() -> None:
    from transformers import AutoTokenizer

    model_directory = Path(REAL_CHECKPOINT)
    tokenizer = load_tokenizer(model_directory)
    reference = AutoTokenizer.from_pretrained(model_directory)
    texts = [
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>",
        "language English<asr_text>Surely you are not thinking of going off there.",
        "我们今天去 Starbucks 买咖啡, it's 3:45pm！",
        "  spaces\tand\nnewlines  ",
    ]
    for text in texts:
        assert tokenizer.encode(text, add_special_tokens=False).ids == reference.encode(
            text, add_special_tokens=False
        )
        ids = reference.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids, skip_special_tokens=True) == reference.decode(
            ids, skip_special_tokens=True
        )


@pytest.mark.skipif(
    REAL_CHECKPOINT is None,
    reason="set QWEN3_ASR_MLX_MODEL_PATH to a Qwen3-ASR MLX checkpoint",
)
def test_retained_prefix_never_ends_inside_a_character() -> None:
    transcriber = Qwen3ASRTranscriber.__new__(Qwen3ASRTranscriber)
    transcriber.tokenizer = load_tokenizer(Path(REAL_CHECKPOINT))
    text = "今天天气很好我们去公园散步吧"
    for rollback in range(0, 12):
        ids, retained = transcriber.retained_prefix(text, rollback)
        assert "�" not in retained
        assert text.startswith(retained)
        assert transcriber.tokenizer.decode(list(ids)) == retained


@pytest.mark.skipif(
    REAL_CHECKPOINT is None,
    reason="set QWEN3_ASR_MLX_MODEL_PATH to a Qwen3-ASR MLX checkpoint",
)
def test_a_real_checkpoint_transcribes_a_tone_without_failing() -> None:
    transcriber = Qwen3ASRTranscriber(Path(REAL_CHECKPOINT))
    samples = (0.1 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000)).astype(
        np.float32
    )
    for layout in AudioLayout:
        result = transcriber.transcribe(
            samples,
            TranscriptionOptions(
                language="en", layout=layout, stop_at_end_of_text=True
            ),
            threading.Event(),
        )
        assert result.language == "English"
        assert result.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
