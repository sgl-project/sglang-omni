# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_tts.delay_pattern import split_moss_audio_segments


def _delay(raw_codes: torch.Tensor, *, fill: int = 1024) -> torch.Tensor:
    frames, num_codebooks = raw_codes.shape
    delayed = torch.full(
        (frames + num_codebooks - 1, num_codebooks),
        fill,
        dtype=raw_codes.dtype,
    )
    for codebook in range(num_codebooks):
        delayed[codebook : codebook + frames, codebook] = raw_codes[:, codebook]
    return delayed


def test_split_moss_audio_segments_reverses_delay_pattern() -> None:
    raw_codes = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=torch.long,
    )

    segments = split_moss_audio_segments(_delay(raw_codes), audio_pad_code=1024)

    assert len(segments) == 1
    assert torch.equal(segments[0], raw_codes)


def test_split_moss_audio_segments_preserves_contiguous_runs() -> None:
    raw_codes = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [1024, 1024, 1024],
            [7, 8, 9],
            [10, 11, 12],
        ],
        dtype=torch.long,
    )

    segments = split_moss_audio_segments(_delay(raw_codes), audio_pad_code=1024)

    assert [segment.tolist() for segment in segments] == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]


def test_split_moss_audio_segments_trims_assistant_prefix() -> None:
    raw_codes = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype=torch.long,
    )

    segments = split_moss_audio_segments(
        _delay(raw_codes),
        audio_pad_code=1024,
        assistant_start_length=2,
    )

    assert len(segments) == 1
    assert torch.equal(segments[0], raw_codes[2:])


def test_split_moss_audio_segments_accepts_noncontiguous_input() -> None:
    raw_codes = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
    delayed = _delay(raw_codes)
    storage = torch.full(
        (delayed.shape[0], delayed.shape[1] * 2),
        -1,
        dtype=delayed.dtype,
    )
    storage[:, ::2] = delayed
    noncontiguous = storage[:, ::2]
    assert not noncontiguous.is_contiguous()

    segments = split_moss_audio_segments(noncontiguous, audio_pad_code=1024)

    assert len(segments) == 1
    assert torch.equal(segments[0], raw_codes)


def test_split_moss_audio_segments_allows_short_delay_window() -> None:
    delayed = torch.zeros((2, 3), dtype=torch.long)

    assert split_moss_audio_segments(delayed, audio_pad_code=1024) == []


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3)])
def test_split_moss_audio_segments_rejects_non_matrix(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        split_moss_audio_segments(torch.zeros(shape), audio_pad_code=1024)
