# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from sglang_omni.serve.realtime.g711 import G711StreamDecoder


def pcm16(payload: bytes) -> list[int]:
    return np.frombuffer(payload, dtype=np.int16).tolist()


def test_ulaw_silence_decodes_to_zero() -> None:
    decoder = G711StreamDecoder("g711_ulaw")
    assert pcm16(decoder.decode(b"\xff")) == [0, 0]


def test_alaw_silence_decodes_to_eight() -> None:
    decoder = G711StreamDecoder("g711_alaw")
    assert pcm16(decoder.decode(b"\xd5")) == [8, 8]


def test_ulaw_minimum_decodes_to_negative_full_scale() -> None:
    decoder = G711StreamDecoder("g711_ulaw")
    assert pcm16(decoder.decode(b"\x00")) == [-32124, -32124]


def test_upsample_carries_previous_sample_across_chunks() -> None:
    decoder = G711StreamDecoder("g711_ulaw")
    assert pcm16(decoder.decode(b"\xff")) == [0, 0]
    assert pcm16(decoder.decode(b"\x00")) == [-32124, -16062]
