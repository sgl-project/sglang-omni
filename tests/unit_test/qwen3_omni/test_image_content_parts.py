# SPDX-License-Identifier: Apache-2.0
"""Image content parts must retain their place in the chat template."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from sglang_omni.client.client import _extract_inputs
from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.models.qwen3_omni.components import preprocessor as preprocessor_mod
from sglang_omni.preprocessing import normalize_messages
from sglang_omni.proto import OmniRequest, StagePayload


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


@pytest.mark.parametrize("top_level", [None, "extra.png", ["extra.png"]])
def test_image_parts_reach_processor_in_conversation_order(monkeypatch, top_level):
    messages = [
        {"role": "user", "content": [_image("first.png")]},
        {"role": "assistant", "content": "The first image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare it with "},
                _image("second.png"),
                {"type": "text", "text": " and this image."},
                _image("third.png"),
            ],
        },
    ]
    original = deepcopy(messages)
    # The common client must leave inline media for each model's own adapter.
    request = GenerateRequest(
        messages=[Message(**message) for message in messages],
        metadata={"images": top_level} if top_level is not None else {},
    )
    inputs = _extract_inputs(request)
    if top_level is None:
        assert inputs == messages
    else:
        assert inputs == {"messages": messages, "images": top_level}

    loader = AsyncMock(side_effect=lambda images, **kwargs: images or [])
    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", loader)
    monkeypatch.setattr(
        preprocessor_mod, "ensure_audio_list_async", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        preprocessor_mod,
        "ensure_video_list_async",
        AsyncMock(return_value=([], None, [])),
    )
    processor = Mock(return_value={"input_ids": torch.tensor([[1, 2]])})
    processor.apply_chat_template.return_value = "chat prompt"
    pre = object.__new__(preprocessor_mod.Qwen3OmniPreprocessor)
    pre.processor = processor
    pre.max_seq_len = None
    for field in (
        "default_video_fps",
        "default_video_max_frames",
        "default_video_min_pixels",
        "default_video_max_pixels",
        "default_video_total_pixels",
    ):
        setattr(pre, field, None)
    payload = StagePayload(
        request_id="image-parts",
        request=OmniRequest(inputs=inputs, params={"max_new_tokens": 2}),
        data={},
    )

    asyncio.run(pre._call_impl(payload))

    expected_images = ["first.png", "second.png", "third.png"]
    if top_level is not None:
        expected_images.append("extra.png")
    loader.assert_awaited_once()
    assert loader.await_args.args == (expected_images,)
    assert isinstance(
        loader.await_args.kwargs["media_connector"],
        preprocessor_mod.MultiModalResourceConnector,
    )
    assert processor.call_args.kwargs["images"] == expected_images
    templated = processor.apply_chat_template.call_args.args[0]
    assert templated[0] == {"role": "user", "content": [{"type": "image"}]}
    assert templated[1] == messages[1]
    expected_parts = [
        {"type": "text", "text": "Compare it with "},
        {"type": "image"},
        {"type": "text", "text": " and this image."},
        {"type": "image"},
    ]
    if top_level is not None:
        expected_parts.append({"type": "image"})
    assert templated[2] == {"role": "user", "content": expected_parts}
    assert messages == original


@pytest.mark.parametrize(
    "url",
    ["/tmp/image.png", "https://example.com/image.png", "data:image/png;base64,YQ=="],
)
def test_extracts_image_url_forms(url):
    messages = [{"role": "user", "content": [_image(url)]}]
    normalized, images = preprocessor_mod._extract_image_content_parts(messages)
    assert normalized == [{"role": "user", "content": [{"type": "image"}]}]
    assert images == [url]


@pytest.mark.parametrize("unknown", [{"type": "model_private", "value": "keep"}, 7])
@pytest.mark.parametrize("unknown_first", [False, True])
def test_unknown_parts_retain_legacy_handling_atomically(unknown, unknown_first):
    parts = (
        [unknown, _image("one.png")] if unknown_first else [_image("one.png"), unknown]
    )
    messages = [{"role": "user", "content": parts}]
    normalized, images = preprocessor_mod._extract_image_content_parts(messages)
    assert normalized == normalize_messages(messages)
    assert images == []


@pytest.mark.parametrize(
    "part, error",
    [
        ({"type": "image_url"}, "image_url.*non-empty url"),
        ({"type": "image_url", "image_url": {}}, "image_url.*non-empty url"),
        (_image(""), "image_url.*non-empty url"),
        (_image(None), "image_url.*non-empty url"),
        (_image(3), "image_url.*non-empty url"),
        ({"type": "text"}, "text.*string"),
        ({"type": "text", "text": 3}, "text.*string"),
    ],
)
@pytest.mark.parametrize("unknown_first", [False, True])
def test_malformed_known_parts_are_rejected_regardless_of_order(
    part, error, unknown_first
):
    parts = [{"type": "private"}, part] if unknown_first else [part]
    with pytest.raises(ValueError, match=error):
        preprocessor_mod._extract_image_content_parts(
            [{"role": "user", "content": parts}]
        )


def test_existing_top_level_media_precedes_plain_text():
    pre = object.__new__(preprocessor_mod.Qwen3OmniPreprocessor)
    messages, images = preprocessor_mod._extract_image_content_parts(
        [{"role": "user", "content": "Describe the media."}]
    )
    assert images == []
    assert pre._build_multimodal_messages(
        messages, num_images=1, num_audios=1, num_videos=1
    ) == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "video"},
                {"type": "audio"},
                {"type": "text", "text": "Describe the media."},
            ],
        }
    ]
