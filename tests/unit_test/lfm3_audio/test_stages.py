# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the LFM3 prompt and media helpers of the preprocessing stage."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import av
import numpy as np
import pytest
from transformers.video_utils import VideoMetadata

from sglang_omni.models.lfm3_audio import stages
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    MultiModalResourceConnector,
    resource_connector_scope,
)
from sglang_omni.serve.openai_errors import is_bad_request_error


def test_tts_prompt_uses_voice_and_language() -> None:
    assert (
        stages.tts_system_prompt({})
        == "Perform TTS in American English using a female voice."
    )
    assert (
        stages.tts_system_prompt({"voice": "Male", "language": "British English"})
        == "Perform TTS in British English using a male voice."
    )


def test_tts_instructions_replace_the_prompt() -> None:
    assert (
        stages.tts_system_prompt({"instructions": " Speak slowly. "}) == "Speak slowly."
    )


def test_tts_rejects_unknown_voices() -> None:
    with pytest.raises(ValueError, match="voices"):
        stages.tts_system_prompt({"voice": "alloy"})


def test_speech_sampling_fields_of_tts_and_chat() -> None:
    """A TTS request's sampling fields, and only a chat request's talker fields, drive the speech frames."""
    params = {"temperature": 0.7, "top_k": 64, "talker_temperature": 0.0}
    assert stages.audio_sampling(params, is_tts=True) == {
        "audio_temperature": 0.7,
        "audio_top_k": 64,
        "audio_repetition_penalty": None,
    }
    assert stages.audio_sampling(params, is_tts=False) == {
        "audio_temperature": 0.0,
        "audio_top_k": None,
        "audio_repetition_penalty": None,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"temperature": "hot"},
        {"top_k": -1},
        {"top_k": 1.5},
        {"repetition_penalty": 0},
        {"repetition_penalty": float("inf")},
    ],
)
def test_speech_sampling_values_the_sampler_rejects_are_request_errors(
    params: dict[str, object],
) -> None:
    """The frame sampler runs on the whole batch: a repetition penalty of 0 or a negative temperature raised inside it and failed every request of the batch. The request alone is now answered with a 400."""
    with pytest.raises(ValueError) as tts_error:
        stages.audio_sampling(params, is_tts=True)
    with pytest.raises(ValueError, match="talker_") as chat_error:
        stages.audio_sampling(
            {f"talker_{name}": value for name, value in params.items()}, is_tts=False
        )
    assert is_bad_request_error(tts_error.value)
    assert is_bad_request_error(chat_error.value)


@pytest.mark.parametrize(
    ("part", "kind", "source"),
    [
        (
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}},
            "audio",
            "data:audio/mp3;base64,AAAA",
        ),
        ({"type": "image_url", "image_url": {"url": "a.png"}}, "image", "a.png"),
        ({"type": "video_url", "video_url": "b.mp4"}, "video", "b.mp4"),
        ({"type": "image", "image": "c.png"}, "image", "c.png"),
    ],
)
def test_content_part_sources(part: dict[str, object], kind: str, source: str) -> None:
    assert stages.part_source(part, kind) == source


def test_content_part_without_media_is_a_request_error() -> None:
    with pytest.raises(ValueError, match="needs its video"):
        stages.part_source({"type": "video_url"}, "video")


def test_token_id_prompts_are_a_request_error() -> None:
    with pytest.raises(ValueError, match="token ids"):
        stages.as_messages([1, 2, 3])


def test_local_video_paths_follow_the_media_policy(tmp_path: Path) -> None:
    """A bare video path outside allowed_local_media_path is refused before it is read."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"not a video")
    connector = MultiModalResourceConnector(allowed_local_media_path=str(allowed))
    with resource_connector_scope(connector), pytest.raises(MediaPolicyError):
        asyncio.run(stages.load_video_bytes(str(outside)))


def test_video_frames_are_the_sampled_ones_in_order() -> None:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(10):
            image = np.full((32, 32, 3), index * 25, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class EveryThirdFrame:
        def sample_frames(self, metadata: VideoMetadata) -> list[int]:
            return list(range(0, metadata.total_num_frames, 3))

    frames, metadata = stages.load_video_frames(buffer.getvalue(), EveryThirdFrame())
    assert metadata.total_num_frames == 10 and metadata.fps == 8
    assert metadata.frames_indices == [0, 3, 6, 9]
    brightness = [int(frame.mean().round()) for frame in frames]
    assert brightness == pytest.approx([0, 75, 150, 225], abs=4)
