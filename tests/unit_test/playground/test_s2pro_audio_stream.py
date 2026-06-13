# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from playground.s2pro.audio_stream import PcmChunkAssembler, PcmStreamFormat


def test_pcm_chunk_assembler_preserves_split_frames() -> None:
    assembler = PcmChunkAssembler(PcmStreamFormat(sample_width=2))

    assert assembler.add_chunk(b"\x01") is None
    assert assembler.add_chunk(b"\x02\x03\x04") == b"\x01\x02\x03\x04"
    assembler.flush()


def test_pcm_chunk_assembler_rejects_trailing_partial_frame() -> None:
    assembler = PcmChunkAssembler(PcmStreamFormat(sample_width=2))

    assert assembler.add_chunk(b"\x01\x02\x03") == b"\x01\x02"
    with pytest.raises(ValueError, match="partial audio frame"):
        assembler.flush()
