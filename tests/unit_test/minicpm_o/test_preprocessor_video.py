"""Verify video decoding and media order through processor inputs."""

from __future__ import annotations

import asyncio
import threading
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
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o import video_frontend
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.video_frontend import MiniCPMVideoIO, TimedVideo
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.minicpm_o.test_preprocessor_prompt import (
    CHAT_TEMPLATE,
    ProcessorOutput,
    RecordingProcessor,
)


def test_timed_video_reports_missing_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_frontend, "VideoReader", None)
    with pytest.raises(RuntimeError, match="decord==0.6.0.*x86_64 only"):
        MiniCPMVideoIO(use_audio=False).load_file(Path("unused.mp4"))


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


@pytest.mark.parametrize("video_alias", ["video", "videos"])
@pytest.mark.parametrize("use_audio", [False, True])
@pytest.mark.parametrize("include_text", [False, True])
def test_top_level_video_matches_inline_reference_policy(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    decoded_video: TimedVideo,
    video_alias: str,
    use_audio: bool,
    include_text: bool,
) -> None:
    messages = [
        {"role": "user", "content": ["Hello", "World"]},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": ["Describe.", "Details."] if include_text else ""},
    ]
    top_level_result = asyncio.run(
        video_preprocessor(
            StagePayload(
                request_id="top-level-policy",
                request=OmniRequest(
                    inputs={
                        "messages": messages,
                        video_alias: "clip.mp4",
                        "use_audio_in_video": use_audio,
                    }
                ),
                data=None,
            )
        )
    )
    inline_result = asyncio.run(
        video_preprocessor(
            StagePayload(
                request_id="inline-policy",
                request=OmniRequest(
                    inputs=[
                        *messages[:-1],
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": "clip.mp4",
                                        "use_audio": use_audio,
                                    },
                                },
                                *(messages[-1]["content"] if include_text else []),
                            ],
                        },
                    ]
                ),
                data=None,
            )
        )
    )
    assert (
        top_level_result.data["prompt"]["prompt_text"]
        == inline_result.data["prompt"]["prompt_text"]
    )
    for modality_name, top_level_modality in top_level_result.data["mm_inputs"].items():
        inline_modality = inline_result.data["mm_inputs"][modality_name]
        assert top_level_modality["cache_key"] == inline_modality["cache_key"]
        torch.testing.assert_close(
            top_level_modality["bounds"], inline_modality["bounds"]
        )
    assert video_processor.images == [decoded_video.frames]
    if use_audio:
        assert video_processor.audio_parts == [[2, 2]]
    else:
        assert video_processor.audios is None
    assert video_processor.max_slice_nums == 1


@pytest.mark.parametrize("include_video_file", [False, True])
def test_top_level_video_preserves_predecoded_tensor_support(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    encoded_video_path: Path,
    include_video_file: bool,
) -> None:
    frames = torch.zeros((2, 3, 64, 64), dtype=torch.uint8)
    frames[0, 0] = 255
    frames[1, 2] = 255
    result = asyncio.run(
        video_preprocessor(
            StagePayload(
                request_id="predecoded-video",
                request=OmniRequest(
                    inputs={
                        "messages": [{"role": "user", "content": "Describe."}],
                        "videos": (
                            [encoded_video_path, frames]
                            if include_video_file
                            else [frames]
                        ),
                    }
                ),
                data=None,
            )
        )
    )
    frame_count = 5 if include_video_file else 2
    expected_content = "\n".join(["<image>./</image>"] * frame_count + ["Describe."])
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\n{expected_content}<|im_end|>"
    )
    assert len(video_processor.images[0]) == frame_count
    assert video_processor.images[0][-2].getpixel((0, 0)) == (255, 0, 0)
    assert video_processor.images[0][-1].getpixel((0, 0)) == (0, 0, 255)
    assert video_processor.audios is None


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


def test_omni_video_joins_text_parts_in_preceding_turns(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    decoded_video: TimedVideo,
) -> None:
    payload = StagePayload(
        request_id="omni-video-turns",
        request=OmniRequest(
            inputs=[
                {"role": "user", "content": ["Hello", "World"]},
                {"role": "assistant", "content": "OK"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "clip.mp4", "use_audio": True},
                        }
                    ],
                },
            ]
        ),
        data=None,
    )
    result = asyncio.run(video_preprocessor(payload))
    assert result.data["prompt"]["prompt_text"].startswith(
        "<|im_start|>user\nHelloWorld<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>\n"
    )
    assert video_processor.audio_parts == [[2, 2]]


@pytest.mark.parametrize("use_audio", [False, True])
@pytest.mark.parametrize("include_image", [False, True])
def test_image_cache_distinguishes_video_processing_policy(
    video_preprocessor: MiniCPMOPreprocessor,
    decoded_video: TimedVideo,
    use_audio: bool,
    include_image: bool,
) -> None:
    image = Image.new("RGB", (896, 896), "red")
    decoded_video.frames = [image]
    decoded_video.audio_segments = [np.zeros(16000, dtype=np.float32)]
    image_count = 2 if include_image else 1

    def preprocess(as_video: bool) -> StagePayload:
        content = (
            ([image] if include_image else [])
            + [
                {
                    "type": "video_url",
                    "video_url": {"url": "clip.mp4", "use_audio": use_audio},
                }
            ]
            if as_video
            else [image] * image_count
        )
        return asyncio.run(
            video_preprocessor(
                StagePayload(
                    request_id="image-policy",
                    request=OmniRequest(inputs=[{"role": "user", "content": content}]),
                    data=None,
                )
            )
        )

    image_result = preprocess(False)
    video_result = preprocess(True)
    image_cache_key = image_result.data["encoder_inputs"]["image_encoder"]["cache_key"]
    video_cache_key = video_result.data["encoder_inputs"]["image_encoder"]["cache_key"]
    assert image_cache_key is not None and video_cache_key is not None
    assert image_cache_key != video_cache_key
    assert image_result.data["mm_inputs"]["image"]["cache_key"] == image_cache_key
    assert video_result.data["mm_inputs"]["image"]["cache_key"] == video_cache_key
    assert (
        preprocess(False).data["encoder_inputs"]["image_encoder"]["cache_key"]
        == image_cache_key
    )
    assert (
        preprocess(True).data["encoder_inputs"]["image_encoder"]["cache_key"]
        == video_cache_key
    )


@pytest.mark.asyncio
async def test_cancelled_local_timed_video_drains_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    released = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    def decode(self: MiniCPMVideoIO, video_path: Path) -> TimedVideo:
        loop.call_soon_threadsafe(started.set)
        try:
            if not released.wait(timeout=5):
                raise TimeoutError("decoder was not released")
            else:
                return TimedVideo()
        finally:
            finished.set()

    monkeypatch.setattr(MiniCPMVideoIO, "load_file", decode)
    monkeypatch.setattr(
        video_frontend, "get_global_resource_connector", MultiModalResourceConnector
    )
    task = asyncio.create_task(
        video_frontend.load_timed_video("clip.mp4", use_audio=True)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert finished.is_set()
    finally:
        released.set()


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
