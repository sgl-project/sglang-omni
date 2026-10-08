# SPDX-License-Identifier: Apache-2.0
"""OpenAI chat content parts become placeholders in place, with their media in order."""

import pytest

from sglang_omni.preprocessing.chat_content import split_content_parts
from sglang_omni.serve.openai_errors import is_bad_request_error


def test_media_parts_become_placeholders_where_they_stood() -> None:
    messages = [
        {"role": "system", "content": "Answer briefly."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "first.png", "detail": "low"},
                },
                {"type": "text", "text": "and"},
                {"type": "input_image", "image_url": "second.png"},
                {"type": "video_url", "video_url": {"url": "clip.mp4"}},
            ],
        },
        {"role": "assistant", "content": None},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "speech.wav"}},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "UklG", "format": "wav"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": "SUQz", "format": "mp3"},
                },
                {"type": "input_text", "text": "What is said?"},
            ],
        },
    ]

    template_messages, media = split_content_parts(messages)

    assert template_messages == [
        {"role": "system", "content": "Answer briefly."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "and"},
                {"type": "image"},
                {"type": "video"},
            ],
        },
        {"role": "assistant", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "audio"},
                {"type": "audio"},
                {"type": "text", "text": "What is said?"},
            ],
        },
    ]
    assert media.images == ["first.png", "second.png"]
    assert media.videos == ["clip.mp4"]
    assert media.audios == [
        "speech.wav",
        "data:audio/wav;base64,UklG",
        "data:audio/mpeg;base64,SUQz",
    ]


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ([{"type": "model_private", "value": 1}], "Unsupported chat content part type"),
        ([7], "chat content part must be an object"),
        ([{"type": "image_url"}], "image_url chat content part requires a url"),
        ([{"type": "image_url", "image_url": {"url": ""}}], "requires a url"),
        (
            [{"type": "video_url", "video_url": {}}],
            "video_url chat content part requires",
        ),
        ([{"type": "text", "text": 3}], "text chat content part requires a string"),
        (
            [{"type": "input_audio", "input_audio": {"data": "AA", "format": "flac"}}],
            "input_audio chat content part format must be one of",
        ),
        ([{"type": "input_audio", "input_audio": {}}], "requires base64 data"),
        ({"type": "text", "text": "a dict"}, "a list of chat content parts"),
    ],
)
def test_malformed_content_is_a_bad_request(content: object, error: str) -> None:
    with pytest.raises(ValueError, match=error) as excinfo:
        split_content_parts([{"role": "user", "content": content}])

    assert is_bad_request_error(excinfo.value)
