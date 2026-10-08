# SPDX-License-Identifier: Apache-2.0
"""Chat content parts must retain their place in the chat template."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from sglang_omni.client.client import extract_inputs
from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.models.qwen3_omni.components import preprocessor as preprocessor_mod
from sglang_omni.preprocessing.chat_content import split_content_parts
from sglang_omni.proto import OmniRequest, StagePayload


def image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


@pytest.mark.parametrize("top_level", [None, "extra.png", ["extra.png"]])
def test_image_parts_reach_processor_in_conversation_order(monkeypatch, top_level):
    messages = [
        {"role": "user", "content": [image_part("first.png")]},
        {"role": "assistant", "content": "The first image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare it with "},
                image_part("second.png"),
                {"type": "text", "text": " and this image."},
                image_part("third.png"),
            ],
        },
    ]
    original = deepcopy(messages)
    # The common client must leave inline media for each model's own adapter.
    request = GenerateRequest(
        messages=[Message(**message) for message in messages],
        metadata={"images": top_level} if top_level is not None else {},
    )
    inputs = extract_inputs(request)
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

    asyncio.run(pre.call_impl(payload))

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


class ProcessorCalled(Exception):
    pass


def bare_preprocessor(processor: Mock) -> preprocessor_mod.Qwen3OmniPreprocessor:
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
    return pre


def test_audio_and_video_parts_reach_processor_in_placeholder_order(monkeypatch):
    """With use_audio_in_video each video's audio is read where its placeholder stands."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "question.wav"}},
                {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                {"type": "text", "text": "Answer the question about the clip."},
            ],
        }
    ]
    question, clip_track, top_level = (
        np.full(4, value, dtype=np.float32) for value in (1.0, 2.0, 3.0)
    )
    video_loader = AsyncMock(return_value=(["clip frames"], None, [clip_track]))
    audio_loader = AsyncMock(return_value=[question, top_level])
    monkeypatch.setattr(
        preprocessor_mod, "ensure_image_list_async", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", video_loader)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", audio_loader)
    for name in (
        "compute_audio_cache_key",
        "compute_image_cache_key",
        "compute_video_cache_key",
    ):
        monkeypatch.setattr(preprocessor_mod, name, lambda media: None)
    processor = Mock(side_effect=ProcessorCalled)
    processor.apply_chat_template.return_value = "chat prompt"
    payload = StagePayload(
        request_id="audio-video-parts",
        request=OmniRequest(
            inputs={
                "messages": messages,
                "audios": ["top.wav"],
                "use_audio_in_video": True,
            },
            params={"max_new_tokens": 2},
        ),
        data={},
    )

    with pytest.raises(ProcessorCalled):
        asyncio.run(bare_preprocessor(processor).call_impl(payload))

    assert video_loader.await_args.args == (["clip.mp4"],)
    assert audio_loader.await_args.args == (["question.wav", "top.wav"],)
    assert processor.apply_chat_template.call_args.args[0] == [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "video"},
                {"type": "text", "text": "Answer the question about the clip."},
                {"type": "audio"},
            ],
        }
    ]
    assert [audio[0] for audio in processor.call_args.kwargs["audio"]] == [
        1.0,
        2.0,
        3.0,
    ]


def test_unknown_parts_are_rejected_before_any_media_loads(monkeypatch):
    loader = AsyncMock(return_value=[])
    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", loader)
    payload = StagePayload(
        request_id="unknown-part",
        request=OmniRequest(
            inputs=[
                {
                    "role": "user",
                    "content": [image_part("one.png"), {"type": "model_private"}],
                }
            ],
            params={},
        ),
        data={},
    )

    with pytest.raises(ValueError, match="Unsupported chat content part type"):
        asyncio.run(bare_preprocessor(Mock()).call_impl(payload))
    loader.assert_not_awaited()


def test_existing_top_level_media_precedes_plain_text():
    pre = object.__new__(preprocessor_mod.Qwen3OmniPreprocessor)
    messages, media = split_content_parts(
        [{"role": "user", "content": "Describe the media."}]
    )
    assert media.images == []
    assert pre.build_multimodal_messages(
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
