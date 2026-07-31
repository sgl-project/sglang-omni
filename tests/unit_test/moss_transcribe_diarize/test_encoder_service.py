# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import queue

import pytest

from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
    BatchedAudioEncoderService,
)


def test_drain_batch_respects_gpu_microbatch_limit() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._max_batch_size = 2
    service._queue = queue.Queue()
    entries = [
        (object(), concurrent.futures.Future())
        for _ in range(service._max_batch_size + 2)
    ]
    for entry in entries:
        service._queue.put(entry)

    assert service._drain_batch() == entries[:2]
    assert service._queue.qsize() == 2


def test_encoder_microbatch_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
        BatchedAudioEncoderService(object(), max_batch_size=0)
