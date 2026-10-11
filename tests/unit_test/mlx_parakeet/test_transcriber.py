# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from sglang_omni_mlx.parakeet.transcriber import (  # noqa: E402
    ParakeetTranscriber,
    TranscriptionOptions,
    collapse_ctc,
)
from sglang_omni_mlx.transcription import TranscriptionCancelled  # noqa: E402

REAL_CHECKPOINT = os.environ.get("PARAKEET_MLX_MODEL_PATH")
ENGLISH_CLIP = (
    Path(__file__).parents[3]
    / "Voxt/VoxtTests/Fixtures/Audio/qwen-official/qwen_audio_short_en.wav"
)


def test_collapse_ctc_merges_runs_and_drops_blanks() -> None:
    assert collapse_ctc([0, 5, 5, 0, 5, 7, 7, 7, 0], blank_id=0) == [5, 5, 7]
    assert collapse_ctc([], blank_id=0) == []


class RecordingTranscriber(ParakeetTranscriber):
    """Decodes every chunk to one token id, its sample count."""

    def __init__(self) -> None:
        self.chunk_lengths: list[int] = []
        self.tokenizer = SimpleNamespace(
            decode=lambda ids, skip_special_tokens: " ".join(map(str, ids))
        )

    def token_ids(self, samples: np.ndarray, cancel: threading.Event) -> list[int]:
        self.chunk_lengths.append(len(samples))
        return [len(samples)]


def test_long_audio_joins_chunk_texts_and_skips_silent_chunks() -> None:
    samples = np.full(16000 * 300, 0.1, dtype=np.float32)
    samples[16000 * 230 :] = 0.0
    transcriber = RecordingTranscriber()
    result = transcriber.transcribe(samples, TranscriptionOptions(), threading.Event())
    assert result.chunk_count == 3
    assert len(transcriber.chunk_lengths) == 2
    assert result.text == " ".join(map(str, transcriber.chunk_lengths))


def test_audio_over_an_hour_is_rejected() -> None:
    with pytest.raises(ValueError, match="3600 seconds"):
        RecordingTranscriber().transcribe(
            np.zeros(16000 * 3601, dtype=np.float32),
            TranscriptionOptions(),
            threading.Event(),
        )


@pytest.fixture(scope="module")
def transcriber() -> ParakeetTranscriber:
    if REAL_CHECKPOINT is None:
        pytest.skip("set PARAKEET_MLX_MODEL_PATH to a Parakeet checkpoint directory")
    else:
        pass
    return ParakeetTranscriber(Path(REAL_CHECKPOINT))


def english_clip() -> np.ndarray:
    with wave.open(str(ENGLISH_CLIP)) as reader:
        assert reader.getframerate() == 16000
        return (
            np.frombuffer(reader.readframes(reader.getnframes()), "<i2").astype(
                np.float32
            )
            / 32768
        )


def test_real_checkpoint_transcribes_english(
    transcriber: ParakeetTranscriber,
) -> None:
    result = transcriber.transcribe(
        english_clip(), TranscriptionOptions(), threading.Event()
    )
    words = result.text.casefold().replace(",", "").replace(".", "").split()
    assert words[:4] == ["mister", "quilter", "is", "the"] or words[:3] == [
        "mr",
        "quilter",
        "is",
    ]
    assert words[-2:] == ["his", "gospel"]


def test_real_checkpoint_honors_cancel(transcriber: ParakeetTranscriber) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe(english_clip(), TranscriptionOptions(), cancel)
