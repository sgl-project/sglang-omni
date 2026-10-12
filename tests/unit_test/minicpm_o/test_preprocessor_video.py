from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.image_processing import CUDAImageProcessor
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(inputs: dict) -> StagePayload:
    return StagePayload(
        request_id="video-test",
        request=OmniRequest(inputs=inputs),
        data=None,
    )


class FakeProcessor:
    def __init__(self) -> None:
        self.images = None
        self.audios = None
        self.options = None

    def __call__(self, prompt_text, *, images, audios, return_tensors, **options):
        self.images = images
        self.audios = audios
        self.options = options
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


async def empty_images(images):
    return []


async def explicit_audios(_audios, *, target_sr):
    return [np.array([0.25, 0.5], dtype=np.float32)] if _audios else []


@pytest.mark.parametrize("use_audio_in_video", [None, False, True])
@pytest.mark.parametrize("explicit_audio", [False, True])
def test_minicpm_preprocessor_uses_only_requested_video_audio(
    monkeypatch,
    use_audio_in_video,
    explicit_audio,
) -> None:
    fake_processor = FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        fake_processor  # noqa: leading-underscore  # production name
    )
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

    async def videos(videos, **kwargs):
        captured_video_kwargs.update(kwargs)
        audio = (
            [np.array([1.0, 2.0], dtype=np.float32)]
            if kwargs["extract_audio"]
            else None
        )
        return [video], [2.0], audio

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", empty_images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)
    monkeypatch.setattr(
        preprocessor_mod,
        "compute_video_cache_key",
        lambda *args, **_kwargs: "video-cache",
    )

    payload = make_payload(
        {
            "messages": [{"role": "user", "content": "What happens?"}],
            "videos": ["clip.mp4"],
            **({"audios": ["question.wav"]} if explicit_audio else {}),
            **(
                {"use_audio_in_video": use_audio_in_video}
                if use_audio_in_video is not None
                else {}
            ),
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
        "extract_audio": bool(use_audio_in_video),
        "audio_target_sr": 16000,
    }
    assert len(fake_processor.images[0]) == 2
    assert fake_processor.options == {"max_slice_nums": 1, "use_image_id": False}
    expected_audio_count = int(explicit_audio) + int(bool(use_audio_in_video))
    if expected_audio_count:
        assert len(fake_processor.audios[0]) == expected_audio_count
        if explicit_audio:
            np.testing.assert_array_equal(
                fake_processor.audios[0][0],
                np.array([0.25, 0.5], dtype=np.float32),
            )
        if use_audio_in_video:
            np.testing.assert_array_equal(
                fake_processor.audios[0][-1],
                np.array([1.0, 2.0], dtype=np.float32),
            )
    else:
        assert fake_processor.audios is None
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == 2
    assert prompt_text.count("<audio>./</audio>") == expected_audio_count
    assert payload.request.inputs is None


@pytest.mark.parametrize(
    ("with_image", "with_audio", "with_video"),
    [(True, False, False), (False, True, False), (True, True, True)],
)
def test_minicpm_video_options_preserve_other_media(
    monkeypatch, with_image, with_audio, with_video
) -> None:
    fake_processor = FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        fake_processor  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    image = Image.new("RGB", (2, 2), color="red")
    frame = Image.new("RGB", (2, 2), color="blue")

    async def images(raw_images):
        return [image] if raw_images else []

    async def videos(raw_videos, **kwargs):
        return [[frame]], [1.0], None

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)
    result = asyncio.run(
        preprocessor(
            make_payload(
                {
                    "messages": [{"role": "user", "content": "Describe this."}],
                    "images": [image] if with_image else None,
                    "audios": ["question.wav"] if with_audio else None,
                    "videos": ["clip.mp4"] if with_video else None,
                }
            )
        )
    )

    # The processor has one policy for the whole image list, including mixed inputs.
    assert fake_processor.options == (
        {"max_slice_nums": 1, "use_image_id": False} if with_video else {}
    )
    expected_images = ([image] if with_image else []) + ([frame] if with_video else [])
    if expected_images:
        assert len(fake_processor.images[0]) == len(expected_images)
        for actual, expected in zip(fake_processor.images[0], expected_images):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        assert fake_processor.images is None
    if with_audio:
        np.testing.assert_array_equal(
            fake_processor.audios[0][0], np.array([0.25, 0.5], dtype=np.float32)
        )
    else:
        assert fake_processor.audios is None
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == len(expected_images)
    assert prompt_text.count("<audio>./</audio>") == int(with_audio)


@pytest.mark.parametrize("changed", ["image", "video"])
def test_minicpm_visual_cache_key_tracks_decoded_content(monkeypatch, changed) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        FakeProcessor()  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    media = {"image": Image.new("RGB", (2, 2), "red"), "video": torch.zeros(2, 3, 2, 2)}

    async def images(raw_images):
        return [media["image"]]

    async def videos(raw_videos, **kwargs):
        return [media["video"]], [1.0], None

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)

    def cache_key(name: str = "same") -> str:
        inputs = {
            "messages": [{"role": "user", "content": "Describe this."}],
            "images": [f"https://media.invalid/{name}.png"],
            "videos": [f"https://media.invalid/{name}.mp4"],
        }
        data = asyncio.run(preprocessor(make_payload(inputs))).data
        key = data["encoder_inputs"]["image_encoder"]["cache_key"]
        assert data["mm_inputs"]["image"]["cache_key"] == key
        return key

    before = cache_key()
    assert cache_key() == before
    # New content behind the same URL must not reuse the previous entry.
    media[changed] = (
        Image.new("RGB", (2, 2), "blue")
        if changed == "image"
        else torch.ones(2, 3, 2, 2)
    )
    after = cache_key()
    assert after != before
    # Identical content at another address shares the entry.
    assert cache_key("other") == after


@pytest.fixture(scope="module")
def checkpoint_processor() -> ProcessorMixin:
    checkpoint = Path(os.environ.get("MINICPMO_CHECKPOINT", "MiniCPM-o-4_5"))
    if not (checkpoint / "processing_minicpmo.py").exists():
        pytest.skip("Set MINICPMO_CHECKPOINT to a MiniCPM-o processor checkpoint")
    return AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("as_video", [False, True])
@pytest.mark.parametrize("do_pad", [False, True])
def test_real_processor_visual_packing(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_processor: ProcessorMixin,
    as_video: bool,
    do_pad: bool,
) -> None:
    reference_processor = checkpoint_processor
    pixels = np.random.default_rng(17).integers(0, 256, (672, 1120, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    frames = (
        preprocessor_mod.video_to_images(
            torch.from_numpy(
                np.stack([pixels, np.flip(pixels, axis=1).copy()])
            ).permute(0, 3, 1, 2)
        )
        if as_video
        else [image]
    )
    options = (
        {"max_slice_nums": 1, "use_image_id": False}
        if as_video
        else {"max_slice_nums": 4}
    )
    prompt = "Describe " + " ".join(["<image>./</image>"] * len(frames))
    expected = reference_processor(
        prompt,
        images=[frames],
        audios=None,
        do_pad=do_pad,
        return_tensors="pt",
        **options,
    )
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = None  # noqa: leading-underscore  # production contract
    preprocessor.model_dir = "checkpoint-fixture"
    preprocessor.device = torch.device("cuda", torch.cuda.device_count() - 1)
    monkeypatch.setattr(
        preprocessor_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: reference_processor,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            reference_processor, "process_image", reference_processor.process_image
        )
        actual = preprocessor.processor(
            prompt,
            images=[frames],
            audios=None,
            do_pad=do_pad,
            return_tensors="pt",
            **options,
        )
        assert isinstance(reference_processor.process_image, CUDAImageProcessor)
    assert len(actual["pixel_values"][0]) == len(actual["image_bound"][0])
    assert len(actual["pixel_values"][0]) == (len(frames) if as_video else 5)
    torch.testing.assert_close(
        actual["input_ids"], expected["input_ids"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["image_bound"][0], expected["image_bound"][0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["tgt_sizes"][0], expected["tgt_sizes"][0], rtol=0, atol=0
    )
    for actual_slice, expected_slice in zip(
        actual["pixel_values"][0], expected["pixel_values"][0]
    ):
        assert actual_slice.device == preprocessor.device
        torch.testing.assert_close(actual_slice.cpu(), expected_slice, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_slice.cpu().bfloat16(), expected_slice.bfloat16(), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [(713, 439), (23, 41), (448, 448), (1500, 20)])
def test_cuda_image_packing_matches_reference(
    checkpoint_processor: ProcessorMixin, size: tuple[int, int]
) -> None:
    geometry = checkpoint_processor.image_processor
    image = Image.fromarray(
        np.random.default_rng(5).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    )
    processor = CUDAImageProcessor(geometry, torch.device("cuda:0"))
    expected = geometry([[image]], return_tensors="pt")
    for _ in range(2):
        actual = processor([[image]], max_slice_nums=None)
        assert actual["image_sizes"] == [[size]]
        assert expected["image_sizes"][0][0].tolist() == list(size)
        assert len(actual["pixel_values"][0]) == len(expected["pixel_values"][0])
        for actual_slice, expected_slice in zip(
            actual["pixel_values"][0], expected["pixel_values"][0]
        ):
            torch.testing.assert_close(
                actual_slice.cpu(), expected_slice, rtol=0, atol=0
            )


def test_cpu_processor_preserves_reference(
    monkeypatch: pytest.MonkeyPatch, checkpoint_processor: ProcessorMixin
) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = None  # noqa: leading-underscore  # production contract
    preprocessor.model_dir = "checkpoint-fixture"
    preprocessor.device = torch.device("cpu")
    monkeypatch.setattr(
        preprocessor_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: checkpoint_processor,
    )
    reference_process_image = checkpoint_processor.process_image
    assert preprocessor.processor.process_image == reference_process_image
