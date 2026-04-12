# SPDX-License-Identifier: Apache-2.0

import numpy as np

from sglang_omni.client.client import _extract_inputs
from sglang_omni.client.types import GenerateRequest, Message


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
