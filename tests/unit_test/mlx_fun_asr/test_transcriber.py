# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from sglang_omni_mlx.fun_asr.transcriber import (  # noqa: E402
    FunASRTranscriber,
    TranscriptionOptions,
    default_output_tokens,
    normalize_language,
    prompt_text,
)
from sglang_omni_mlx.transcription import (  # noqa: E402
    FinishReason,
    TranscriptionCancelled,
)

REAL_CHECKPOINT = os.environ.get("FUN_ASR_MLX_MODEL_PATH")
ENGLISH_CLIP = (
    Path(__file__).parents[3]
    / "Voxt/VoxtTests/Fixtures/Audio/qwen-official/qwen_audio_short_en.wav"
)


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("", None),
        ("auto", None),
        ("zh", None),
        ("Chinese", None),
        ("en", "英文"),
        (" English ", "英文"),
        ("日文", "日文"),
    ],
)
def test_normalize_language(language: str, expected: str | None) -> None:
    assert normalize_language(language) == expected


def test_prompt_text_matches_the_pipeline_prompt() -> None:
    assert prompt_text(TranscriptionOptions()) == "语音转写："
    assert prompt_text(TranscriptionOptions(language="英文", itn=False)) == (
        "语音转写成英文，不进行文本规整："
    )
    with_hotwords = prompt_text(TranscriptionOptions(hotwords=("SGLang", "MLX")))
    assert with_hotwords.endswith("热词列表：[SGLang, MLX]\n语音转写：")


@pytest.mark.parametrize(("seconds", "tokens"), [(0.5, 16), (6.0, 40), (30.0, 200)])
def test_default_output_tokens(seconds: float, tokens: int) -> None:
    assert default_output_tokens(seconds) == tokens


@pytest.fixture(scope="module")
def transcriber() -> FunASRTranscriber:
    if REAL_CHECKPOINT is None:
        pytest.skip("set FUN_ASR_MLX_MODEL_PATH to a Fun-ASR-Nano-2512-hf snapshot")
    else:
        pass
    return FunASRTranscriber(Path(REAL_CHECKPOINT))


def english_clip() -> np.ndarray:
    with wave.open(str(ENGLISH_CLIP)) as reader:
        assert reader.getframerate() == 16000
        return (
            np.frombuffer(reader.readframes(reader.getnframes()), "<i2").astype(
                np.float32
            )
            / 32768
        )


def test_real_checkpoint_transcribes_english(transcriber: FunASRTranscriber) -> None:
    result = transcriber.transcribe(
        english_clip(), TranscriptionOptions(language="英文"), threading.Event()
    )
    assert result.finish_reason is FinishReason.STOP
    assert result.text == (
        "Mr Quilter is the apostle of the middle classes, "
        "and we are glad to welcome his gospel."
    )


def test_real_checkpoint_stops_at_the_token_budget(
    transcriber: FunASRTranscriber,
) -> None:
    result = transcriber.transcribe(
        english_clip(), TranscriptionOptions(max_new_tokens=3), threading.Event()
    )
    assert result.finish_reason is FinishReason.LENGTH
    assert result.generated_token_count == 3


def test_real_checkpoint_honors_cancel(transcriber: FunASRTranscriber) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe(english_clip(), TranscriptionOptions(), cancel)


def test_audio_over_thirty_seconds_is_rejected(
    transcriber: FunASRTranscriber,
) -> None:
    with pytest.raises(ValueError, match="30 seconds"):
        transcriber.transcribe(
            np.zeros(16000 * 31, dtype=np.float32),
            TranscriptionOptions(),
            threading.Event(),
        )
