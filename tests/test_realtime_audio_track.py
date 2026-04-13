# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from sglang_omni.realtime.audio_track import BufferedAudioStreamTrack
from sglang_omni.realtime.media import resample_linear


@pytest.mark.asyncio
async def test_buffered_audio_stream_track_preserves_partial_tail_with_zero_padding():
    input_sample_rate = 24000
    output_sample_rate = 48000
    duration_s = 0.13

    t = (
        np.arange(int(input_sample_rate * duration_s), dtype=np.float32)
        / input_sample_rate
    )
    tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    expected = (
        resample_linear(tone, input_sample_rate, output_sample_rate) * 32767.0
    ).astype(np.int16)

    track = BufferedAudioStreamTrack(
        sample_rate=output_sample_rate, frame_duration_s=0.02
    )
    await track.enqueue(tone, input_sample_rate)

    frames = []
    for _ in range(10):
        frame = await track.recv()
        frames.append(frame.to_ndarray().reshape(-1))

    output = np.concatenate(frames)

    np.testing.assert_array_equal(output[: expected.size], expected)
    assert np.count_nonzero(output[expected.size :]) == 0
    assert track.pending_samples == 0
