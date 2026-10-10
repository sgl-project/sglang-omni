# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

pytest.importorskip("mlx.core")

from sglang_omni_mlx.moss_transcribe_diarize.transcriber import (  # noqa: E402
    MossTranscribeDiarizeTranscriber,
    TranscriptionOptions,
    audio_span_ids,
    default_output_tokens,
)
from sglang_omni_mlx.transcription import (  # noqa: E402
    FinishReason,
    TranscriptionCancelled,
)

REAL_CHECKPOINT = os.environ.get("MOSS_TD_MLX_MODEL_PATH")
SHORT_CLIP = Path(__file__).parents[3] / "tests/data/query_to_cars.wav"


@pytest.mark.parametrize(
    ("seconds", "tokens"), [(0.5, 512), (60.0, 600), (3600.0, 36000)]
)
def test_default_output_tokens(seconds: float, tokens: int) -> None:
    assert default_output_tokens(seconds) == tokens


def test_audio_span_inserts_time_markers_without_losing_placeholders() -> None:
    span = audio_span_ids(
        audio_token_count=130,
        audio_pad_id=99,
        digit_ids={str(index): index for index in range(10)},
        tokens_per_second=12.5,
        marker_interval_seconds=5,
    )
    assert span.count(99) == 130
    assert span[62:64] == [5, 99]
    assert span[125:128] == [1, 0, 99]


@pytest.fixture(scope="module")
def transcriber() -> MossTranscribeDiarizeTranscriber:
    if REAL_CHECKPOINT is None:
        pytest.skip("set MOSS_TD_MLX_MODEL_PATH to an official checkpoint snapshot")
    else:
        pass
    return MossTranscribeDiarizeTranscriber(Path(REAL_CHECKPOINT))


def short_clip() -> np.ndarray:
    with wave.open(str(SHORT_CLIP)) as reader:
        sample_rate = reader.getframerate()
        samples = (
            np.frombuffer(reader.readframes(reader.getnframes()), "<i2").astype(
                np.float32
            )
            / 32768.0
        )
    return resample_poly(samples, 16000, sample_rate).astype(np.float32)


def test_real_checkpoint_transcribes_short_clip(
    transcriber: MossTranscribeDiarizeTranscriber,
) -> None:
    result = transcriber.transcribe(
        short_clip(), TranscriptionOptions(), threading.Event()
    )
    assert result.finish_reason is FinishReason.STOP
    assert result.text == "[1.45][S01] How many cars are there in the picture?[4.41]"


def test_real_checkpoint_honors_token_budget(
    transcriber: MossTranscribeDiarizeTranscriber,
) -> None:
    result = transcriber.transcribe(
        short_clip(), TranscriptionOptions(max_new_tokens=1), threading.Event()
    )
    assert result.finish_reason is FinishReason.LENGTH
    assert result.generated_token_count == 1


def test_real_checkpoint_honors_cancellation(
    transcriber: MossTranscribeDiarizeTranscriber,
) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe(short_clip(), TranscriptionOptions(), cancel)
