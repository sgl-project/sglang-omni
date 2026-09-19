from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
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


def test_minicpm_preprocessor_consumes_video_frames_and_audio(
    monkeypatch,
) -> None:
    fake_processor = _FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = fake_processor
    preprocessor.speech_enabled = False
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "render_chat_template",
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
