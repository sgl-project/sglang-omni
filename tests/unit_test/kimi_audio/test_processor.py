# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from types import MethodType

import pytest
import torch

from sglang_omni.models.kimi_audio.processor import (
    KimiAudioProcessor,
    KimiPrompt,
    KimiSpecialTokens,
)


class _Tokenizer:
    def encode(self, text: str, **kwargs):
        del kwargs
        return {"hello": [40, 41], "old answer": [50, 51]}[text]


def _processor() -> KimiAudioProcessor:
    processor = object.__new__(KimiAudioProcessor)
    processor.tokenizer = _Tokenizer()
    processor.special = KimiSpecialTokens(
        msg_end=10,
        media_begin=11,
        media_end=12,
        text_blank=13,
        text_eos=14,
        user_start=15,
        assistant_start=16,
        speech_continue_text=17,
    )

    def audio_fragment(self, source):
        assert source == "clip.wav"
        return KimiPrompt(
            audio_ids=[11, 152071, 12],
            text_ids=[13, 13, 13],
            continuous_mask=[False, True, False],
            continuous_features=[torch.ones((1, 5120))],
        )

    processor._audio_fragment = MethodType(audio_fragment, processor)
    return processor


def test_top_level_audio_builds_official_parallel_stream_order() -> None:
    prompt = _processor().build_prompt(
        [{"role": "user", "content": "hello"}], ["clip.wav"]
    )

    assert prompt.audio_ids == [15, 13, 13, 11, 152071, 12, 17, 10, 16]
    assert prompt.text_ids == [13, 40, 41, 13, 13, 13, 13, 13, 13]
    assert prompt.continuous_mask == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert prompt.continuous_features[0].shape == (1, 5120)


def test_assistant_history_gets_text_eos_before_message_end() -> None:
    prompt = _processor().build_prompt(
        [
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "clip.wav"}}],
            },
            {"role": "assistant", "content": "old answer"},
        ]
    )

    assistant_start = prompt.audio_ids.index(16)
    assert prompt.text_ids[assistant_start + 1 : assistant_start + 5] == [
        50,
        51,
        14,
        13,
    ]
    assert prompt.audio_ids[assistant_start + 3] == 13
    assert prompt.audio_ids[assistant_start + 4] == 10


def test_assistant_content_parts_get_one_text_eos_per_message() -> None:
    prompt = _processor().build_prompt(
        [
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "clip.wav"}}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "old answer"},
                ],
            },
        ]
    )

    assert prompt.text_ids.count(14) == 1
    assert prompt.text_ids[-3:] == [14, 13, 13]


def test_same_role_messages_share_one_role_boundary() -> None:
    prompt = _processor().build_prompt(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "clip.wav"}}],
            },
        ]
    )

    assert prompt.audio_ids == [15, 13, 13, 11, 152071, 12, 17, 10, 16]


def test_assistant_audio_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="assistant audio history"):
        _processor().build_prompt(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "clip.wav"}}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "clip.wav"}}
                    ],
                },
            ]
        )


def test_input_audio_decodes_openai_base64_payload() -> None:
    encoded = base64.b64encode(b"RIFF-test-audio").decode("ascii")

    parts = KimiAudioProcessor._content_parts(
        [
            {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            }
        ]
    )

    assert parts == [("audio", b"RIFF-test-audio")]


def test_input_audio_rejects_invalid_base64_payload() -> None:
    with pytest.raises(ValueError, match="not valid base64"):
        KimiAudioProcessor._content_parts(
            [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "not base64!", "format": "wav"},
                }
            ]
        )
