"""Verify video decoding and media order through processor inputs."""

from __future__ import annotations

import asyncio
import os
from fractions import Fraction
from pathlib import Path
from typing import Literal

import audioread
import av
import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.image_processing import CUDAImageProcessor
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.prompt_frontend import video_to_images
from sglang_omni.models.minicpm_o.video_frontend import MiniCPMVideoIO, TimedVideo
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.minicpm_o.test_preprocessor_prompt import (
    CHAT_TEMPLATE,
    ProcessorOutput,
    RecordingProcessor,
)


class VideoRecordingProcessor(RecordingProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.max_slice_nums: int | None = None
        self.use_image_id: bool | None = None

    def __call__(
        self,
        prompt_text: str,
        *,
        images: list[list[Image.Image]] | None,
        audios: list[list[npt.NDArray[np.float32]]] | None,
        return_tensors: Literal["pt"],
        audio_parts: list[list[int]] | None = None,
        max_slice_nums: int | None = None,
        use_image_id: bool | None = None,
    ) -> ProcessorOutput:
        self.max_slice_nums = max_slice_nums
        self.use_image_id = use_image_id
        return super().__call__(
            prompt_text,
            images=images,
            audios=audios,
            return_tensors=return_tensors,
            audio_parts=audio_parts,
        )


@pytest.fixture
def video_processor() -> VideoRecordingProcessor:
    return VideoRecordingProcessor()


@pytest.fixture
def video_preprocessor(
    video_processor: VideoRecordingProcessor,
) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = video_processor  # noqa: leading-underscore  # Existing processor injection contract.
    preprocessor.speech_enabled = False
    preprocessor.tokenizer = PreTrainedTokenizerBase(chat_template=CHAT_TEMPLATE)
    return preprocessor


@pytest.fixture
def decoded_video(monkeypatch: pytest.MonkeyPatch) -> TimedVideo:
    video = TimedVideo(
        frames=[Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")],
        audio_segments=[
            np.zeros(16000, dtype=np.float32),
            np.ones(16000, dtype=np.float32),
        ],
        timestamps_seconds=[0.0, 1.0],
        duration_seconds=2.0,
    )

    async def load_video(
        source: str,
        *,
        use_audio: bool,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
    ) -> TimedVideo:
        return TimedVideo(
            frames=video.frames,
            audio_segments=video.audio_segments if use_audio else [],
            timestamps_seconds=video.timestamps_seconds,
            duration_seconds=video.duration_seconds,
        )

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.prompt_frontend.load_timed_video", load_video
    )
    return video


@pytest.mark.parametrize("use_audio", [False, True])
def test_inline_video_preserves_frame_audio_order_and_surrounding_text(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    decoded_video: TimedVideo,
    use_audio: bool,
) -> None:
    request_payload = StagePayload(
        request_id="inline-video",
        request=OmniRequest(
            inputs=[
                {
                    "role": "user",
                    "content": [
                        "Before",
                        {
                            "type": "video_url",
                            "video_url": {"url": "clip.mp4", "use_audio": use_audio},
                        },
                        "After",
                    ],
                }
            ]
        ),
        data=None,
    )
    result = asyncio.run(video_preprocessor(request_payload))
    expected_content = (
        "Before<image>./</image><audio>./</audio><image>./</image><audio>./</audio>After"
        if use_audio
        else "Before\n<image>./</image>\n<image>./</image>\nAfter"
    )
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\n{expected_content}<|im_end|>\n"
    )
    assert video_processor.images == [decoded_video.frames]
    if use_audio:
        for actual_waveform, expected_waveform in zip(
            video_processor.audios[0], decoded_video.audio_segments, strict=True
        ):
            np.testing.assert_array_equal(actual_waveform, expected_waveform)
        assert video_processor.audio_parts == [[0, 0]]
    else:
        assert video_processor.audios is None
    assert video_processor.max_slice_nums == 1
    assert video_processor.use_image_id is False


@pytest.mark.parametrize("include_explicit_audio", [False, True])
def test_top_level_video_audio_follows_explicit_media(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    decoded_video: TimedVideo,
    include_explicit_audio: bool,
) -> None:
    explicit_image = Image.new("RGB", (4, 4), "green")
    explicit_waveform = np.full(1600, 0.25, dtype=np.float32)
    request_payload = StagePayload(
        request_id="top-level-video",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "Describe."}],
                "images": [explicit_image],
                "audios": [explicit_waveform] if include_explicit_audio else [],
                "videos": ["clip.mp4"],
                "use_audio_in_video": True,
            }
        ),
        data=None,
    )
    result = asyncio.run(video_preprocessor(request_payload))
    expected_content = "<image>./</image>" + (
        "<audio>./</audio>" if include_explicit_audio else ""
    )
    expected_content += (
        "<image>./</image><audio>./</audio><image>./</image><audio>./</audio>Describe."
    )
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\n{expected_content}<|im_end|>\n"
    )
    assert video_processor.images == [[explicit_image, *decoded_video.frames]]
    expected_waveforms = (
        [explicit_waveform] if include_explicit_audio else []
    ) + decoded_video.audio_segments
    for actual_waveform, expected_waveform in zip(
        video_processor.audios[0], expected_waveforms, strict=True
    ):
        np.testing.assert_array_equal(actual_waveform, expected_waveform)
    assert video_processor.audio_parts == [[0] * len(expected_waveforms)]


def test_silent_video_keeps_chat_separators_and_omits_audio(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    decoded_video: TimedVideo,
) -> None:
    decoded_video.audio_segments = []
    request_payload = StagePayload(
        request_id="silent-video",
        request=OmniRequest(
            inputs=[
                {"role": "user", "content": ["First", "Second"]},
                {"role": "assistant", "content": "OK."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "silent.mp4", "use_audio": True},
                        },
                        "Describe.",
                    ],
                },
            ]
        ),
        data=None,
    )
    result = asyncio.run(video_preprocessor(request_payload))
    assert result.data["prompt"]["prompt_text"].startswith(
        "<|im_start|>user\nFirst\nSecond<|im_end|>\n"
    )
    assert video_processor.audios is None


@pytest.fixture
def encoded_video_path(tmp_path: Path) -> Path:
    video_path = tmp_path / "timeline.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for frame_index in range(23):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), frame_index * 10, dtype=np.uint8), format="rgb24"
            )
            frame.pts = frame_index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return video_path


@pytest.mark.parametrize(
    "fps,max_frames,expected_timestamps",
    [
        (None, None, [0.0, 1.0, 2.0]),
        (2.0, 3, [0.0, 1.0, 2.0]),
        (1.0, 2, [0.0, 2.0]),
    ],
)
def test_video_decoder_samples_matching_frames_and_audio_intervals(
    encoded_video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fps: float | None,
    max_frames: int | None,
    expected_timestamps: list[float],
) -> None:
    waveform = np.arange(36800, dtype=np.float32)

    def load_waveform(
        path: str, *, sr: int, mono: bool
    ) -> tuple[npt.NDArray[np.float32], int]:
        return waveform, sr

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.video_frontend.librosa.load", load_waveform
    )
    video = MiniCPMVideoIO(use_audio=True, fps=fps, max_frames=max_frames).load_file(
        encoded_video_path
    )
    assert video.timestamps_seconds == expected_timestamps
    for frame, timestamp_seconds in zip(video.frames, expected_timestamps, strict=True):
        assert abs(np.asarray(frame).mean() - timestamp_seconds * 100) < 5
    for segment_index, start_seconds in enumerate(expected_timestamps):
        end_seconds = (
            expected_timestamps[segment_index + 1]
            if segment_index + 1 < len(expected_timestamps)
            else 2.3
        )
        np.testing.assert_array_equal(
            video.audio_segments[segment_index],
            waveform[int(start_seconds * 16000) : int(end_seconds * 16000)],
        )


def test_video_decoder_pads_short_tail_audio(
    encoded_video_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waveform = np.ones(32800, dtype=np.float32)

    def load_waveform(
        path: str, *, sr: int, mono: bool
    ) -> tuple[npt.NDArray[np.float32], int]:
        return waveform, sr

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.video_frontend.librosa.load", load_waveform
    )
    video = MiniCPMVideoIO(use_audio=True).load_file(encoded_video_path)
    np.testing.assert_array_equal(
        video.audio_segments[-1],
        np.concatenate(
            [np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)]
        ),
    )


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
def test_video_decoder_rejects_invalid_requested_frame_rate(fps: float) -> None:
    with pytest.raises(ValueError, match="video_fps"):
        MiniCPMVideoIO(use_audio=False, fps=fps)


def test_inline_video_rejects_unsupported_frame_stacking(
    video_preprocessor: MiniCPMOPreprocessor,
) -> None:
    request_payload = StagePayload(
        request_id="stacked-video",
        request=OmniRequest(
            inputs=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "clip.mp4", "stack_frames": 2},
                        }
                    ],
                }
            ]
        ),
        data=None,
    )
    with pytest.raises(ValueError, match="stack_frames"):
        asyncio.run(video_preprocessor(request_payload))


@pytest.mark.parametrize(
    "minimum_pixels,maximum_pixels,total_pixels,expected_size",
    [
        (None, 28 * 28, None, (28, 28)),
        (None, None, 3 * 28 * 28, (28, 28)),
        (112 * 112, 112 * 112, None, (112, 112)),
    ],
)
def test_video_pixel_budgets_resize_decoded_frames(
    encoded_video_path: Path,
    minimum_pixels: int | None,
    maximum_pixels: int | None,
    total_pixels: int | None,
    expected_size: tuple[int, int],
) -> None:
    video = MiniCPMVideoIO(
        use_audio=False,
        min_pixels=minimum_pixels,
        max_pixels=maximum_pixels,
        total_pixels=total_pixels,
    ).load_file(encoded_video_path)
    assert [frame.size for frame in video.frames] == [expected_size] * 3
    assert video.audio_segments == []


def test_video_audio_decoder_uses_fallback_when_backend_is_unavailable(
    encoded_video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.ones(36800, dtype=np.float32)

    def unavailable_audio_backend(
        path: str, *, sr: int, mono: bool
    ) -> tuple[npt.NDArray[np.float32], int]:
        raise audioread.NoBackendError()

    def fallback_waveform(video_path: Path, target_sr: int) -> npt.NDArray[np.float32]:
        return waveform

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.video_frontend.librosa.load",
        unavailable_audio_backend,
    )
    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.video_frontend.extract_audio_from_path",
        fallback_waveform,
    )
    video = MiniCPMVideoIO(use_audio=True).load_file(encoded_video_path)
    assert [len(segment) for segment in video.audio_segments] == [16000, 16000, 4800]
    assert all(np.all(segment == 1) for segment in video.audio_segments)


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
        video_to_images(
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
