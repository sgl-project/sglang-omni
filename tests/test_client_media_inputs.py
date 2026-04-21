# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from sglang_omni.client.client import _extract_inputs
from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.preprocessing.audio import ensure_audio_list_async


def test_extract_inputs_preserves_media_and_preprocessing_hints():
    request = GenerateRequest(
        messages=[Message(role="user", content="describe this turn")],
        metadata={
            "audios": [np.zeros(160, dtype=np.float32)],
            "videos": ["demo.mp4"],
            "audio_target_sr": 16000,
            "video_fps": 2.0,
            "use_audio_in_video": False,
            "video_seconds_per_chunk": 4.0,
        },
    )

    inputs = _extract_inputs(request)

    assert isinstance(inputs, dict)
    assert inputs["messages"] == [{"role": "user", "content": "describe this turn"}]
    assert inputs["audio_target_sr"] == 16000
    assert inputs["video_fps"] == 2.0
    assert inputs["use_audio_in_video"] is False
    assert inputs["video_seconds_per_chunk"] == 4.0
    assert len(inputs["audios"]) == 1
    assert inputs["videos"] == ["demo.mp4"]


@pytest.mark.asyncio
async def test_ensure_audio_list_async_decodes_waveform_payload():
    waveform = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)

    audios = await ensure_audio_list_async(
        [
            {
                "audio_waveform": waveform.tobytes(),
                "audio_waveform_dtype": "float32",
                "audio_waveform_shape": [8],
            }
        ],
        target_sr=16000,
    )

    assert len(audios) == 1
    np.testing.assert_allclose(audios[0], waveform)
