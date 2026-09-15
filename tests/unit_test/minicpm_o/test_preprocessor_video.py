from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import (
    MiniCPMOPreprocessor,
    _video_to_images,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(inputs: dict) -> StagePayload:
    return StagePayload(
        request_id="video-test",
        request=OmniRequest(inputs=inputs),
        data=None,
    )


class _FakeProcessor:
    def __init__(self) -> None:
        self.images = None
        self.audios = None

    def __call__(self, prompt_text, *, images, audios, return_tensors):
        self.images = images
        self.audios = audios
        image_count = len(images[0]) if images else 0
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "image_bound": [[torch.tensor([0, 1])] * image_count],
            "pixel_values": [[torch.zeros(1, 2) for _ in range(image_count)]],
            "tgt_sizes": [[torch.tensor([1, 1]) for _ in range(image_count)]],
            "audio_bounds": [[]],
            "audio_feature_lens": [[]],
            "audio_features": [],
        }


async def _empty_images(_images):
    return []


async def _explicit_audios(_audios, *, target_sr):
    return [np.array([0.25, 0.5], dtype=np.float32)] if _audios else []


async def _unexpected_video_loader(*_args, **_kwargs):
    raise AssertionError("video loader called without a video input")


def test_video_to_images_preserves_frame_order_and_rgb() -> None:
    video = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    video[0, 0] = 255
    video[1, 1] = 128

    images = _video_to_images(video)

    assert [image.getpixel((0, 0)) for image in images] == [
        (255, 0, 0),
        (0, 128, 0),
    ]
    assert all(image.mode == "RGB" for image in images)


def test_minicpm_normalizes_openai_text_content_parts() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"},
            ],
        }
    ]

    normalized = MiniCPMOPreprocessor._normalize_message_contents(messages)

    assert normalized == [{"role": "user", "content": "Hello world"}]


def test_minicpm_preprocessor_consumes_video_frames_and_audio(
    monkeypatch,
) -> None:
    fake_processor = _FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = fake_processor
    preprocessor._speech_enabled = False
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "_render_chat_template",
        lambda messages, **_: str(messages),
    )

    video = torch.stack(
        [
            torch.zeros((3, 2, 2)),
            torch.ones((3, 2, 2)) * 0.5,
        ]
    )
    captured_video_kwargs = {}

    async def _videos(_videos, **kwargs):
        captured_video_kwargs.update(kwargs)
        return [video], [2.0], [np.array([1.0, 2.0], dtype=np.float32)]

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", _empty_images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", _explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", _videos)
    monkeypatch.setattr(
        preprocessor_mod,
        "compute_video_cache_key",
        lambda *_args, **_kwargs: "video-cache",
    )

    payload = _payload(
        {
            "messages": [{"role": "user", "content": "What happens?"}],
            "videos": ["clip.mp4"],
            "audios": ["question.wav"],
            "video_fps": 2,
            "video_max_frames": 8,
            "video_min_pixels": 128,
            "video_max_pixels": 4096,
            "video_total_pixels": 8192,
        }
    )

    result = asyncio.run(preprocessor(payload))

    assert captured_video_kwargs == {
        "fps": 2,
        "max_frames": 8,
        "min_pixels": 128,
        "max_pixels": 4096,
        "total_pixels": 8192,
        "extract_audio": True,
        "audio_target_sr": 16000,
    }
    assert len(fake_processor.images[0]) == 2
    assert len(fake_processor.audios[0]) == 2
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == 2
    assert prompt_text.count("<audio>./</audio>") == 2
    assert payload.request.inputs is None


def test_minicpm_preprocessor_does_not_load_video_for_audio_only(
    monkeypatch,
) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = _FakeProcessor()
    preprocessor._speech_enabled = False
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "_render_chat_template",
        lambda messages, **_: str(messages),
    )
    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", _empty_images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", _explicit_audios)
    monkeypatch.setattr(
        preprocessor_mod, "ensure_video_list_async", _unexpected_video_loader
    )

    result = asyncio.run(
        preprocessor(
            _payload(
                {
                    "messages": [{"role": "user", "content": "Listen"}],
                    "audios": ["question.wav"],
                }
            )
        )
    )

    assert result.data["encoder_inputs"]["audio_encoder"]
