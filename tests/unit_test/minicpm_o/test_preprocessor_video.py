"""Verify video decoding and media order through processor inputs."""

from __future__ import annotations

import asyncio
import functools
import http.server
import os
import threading
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, Mock

import audioread
import av
import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin

from sglang_omni.models.minicpm_o import stages, video_frontend
from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.audio_encoder import (
    feature_lens_after_pooling,
)
from sglang_omni.models.minicpm_o.components.image_processing import CUDAImageProcessor
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.prompt_frontend import video_to_images
from sglang_omni.models.minicpm_o.request_builders import apply_mm_pad_values
from sglang_omni.models.minicpm_o.video_frontend import MiniCPMVideoIO, TimedVideo
from sglang_omni.preprocessing import resource_connector
from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    ResourceHTTPConnection,
)
from sglang_omni.preprocessing.video import load_video_path
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
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
    preprocessor.video_decode_threads = 16
    return preprocessor


def test_threaded_preprocessing_loads_repeated_remote_images(
    video_preprocessor: MiniCPMOPreprocessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color=(12, 34, 56)).save(image_bytes, format="PNG")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            response_bytes = image_bytes.getvalue()
            self.send_response(200)
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        def log_message(self, format: str, *arguments: object) -> None:
            pass

    monkeypatch.setattr(
        stages, "MiniCPMOPreprocessor", Mock(return_value=video_preprocessor)
    )
    # note (wirybeaver): isolate the global client so loop reuse stays observable.
    monkeypatch.setattr(
        resource_connector,
        "_global_connector",
        MultiModalResourceConnector(connection=ResourceHTTPConnection()),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http_worker = threading.Thread(target=server.serve_forever, daemon=True)
    http_worker.start()
    scheduler = stages.create_preprocessing_executor(
        "model", device="cpu", max_concurrency=2
    )
    scheduler_worker = threading.Thread(target=scheduler.start, daemon=True)
    scheduler_worker.start()
    try:
        for index in range(3):
            request_id = f"remote-{index}"
            payload = StagePayload(
                request_id=request_id,
                request=OmniRequest(
                    inputs=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"http://127.0.0.1:{server.server_port}/image.png"
                                    },
                                },
                                "Describe.",
                            ],
                        }
                    ]
                ),
                data=None,
            )
            scheduler.inbox.put(
                IncomingMessage(request_id=request_id, type="new_request", data=payload)
            )
            result = scheduler.outbox.get(timeout=10)
            assert result.type == "result", repr(result.data)
            assert result.request_id == request_id
    finally:
        scheduler.stop()
        scheduler_worker.join(timeout=3)
        server.shutdown()
        server.server_close()
        http_worker.join(timeout=3)
    assert not scheduler_worker.is_alive()


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
        assert video_processor.audio_parts is None
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
    assert video_processor.audio_parts is None


@pytest.mark.parametrize("video_alias", ["videos"])
@pytest.mark.parametrize("include_text", [False, True])
def test_top_level_video_audio_matches_inline_reference_policy(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    decoded_video: TimedVideo,
    video_alias: str,
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
                        "use_audio_in_video": True,
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
                                        "use_audio": True,
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
    assert video_processor.audio_parts is None
    assert video_processor.max_slice_nums == 1


@pytest.mark.parametrize(
    ("include_video_file", "video_alias", "bare_tensor"),
    [
        (False, "video", False),
        (False, "video", True),
        (False, "videos", False),
        (False, "videos", True),
        (True, "video", False),
        (True, "videos", False),
    ],
)
def test_top_level_video_preserves_predecoded_tensor_support(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    encoded_video_path: Path,
    include_video_file: bool,
    video_alias: str,
    bare_tensor: bool,
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
                        video_alias: (
                            [encoded_video_path, frames]
                            if include_video_file
                            else frames if bare_tensor else [frames]
                        ),
                        "video_fps": 2,
                        "video_min_pixels": 3136,
                        "video_max_pixels": 3136,
                    }
                ),
                data=None,
            )
        )
    )
    frame_count = 6 if include_video_file else 2
    expected_content = "\n".join(["<image>./</image>"] * frame_count + ["Describe."])
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\n{expected_content}<|im_end|>"
    )
    assert len(video_processor.images[0]) == frame_count
    assert video_processor.images[0][-2].getpixel((0, 0)) == (255, 0, 0)
    assert video_processor.images[0][-1].getpixel((0, 0)) == (0, 0, 255)
    assert video_processor.audios is None
    assert video_processor.max_slice_nums == 1
    assert video_processor.use_image_id is False


@pytest.mark.parametrize("video_alias", ["videos"])
@pytest.mark.parametrize("source_kind", ["path", "file_uri", "url"])
def test_top_level_video_question_audio_retains_legacy_inputs(
    video_preprocessor: MiniCPMOPreprocessor,
    video_processor: VideoRecordingProcessor,
    encoded_video_path: Path,
    video_alias: str,
    source_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit_image = Image.new("RGB", (4, 4), "green")
    question_audio = np.full(1600, 0.25, dtype=np.float32)
    expected_video, _ = load_video_path(
        encoded_video_path,
        fps=2,
        max_frames=128,
        min_pixels=3136,
        max_pixels=3136,
        decode_threads=16,
    )
    expected_frames = video_to_images(expected_video)
    assert len(expected_frames) == 4
    assert all(frame.size == (56, 56) for frame in expected_frames)
    assert [round(np.asarray(frame).mean() / 10) for frame in expected_frames] == [
        0,
        7,
        15,
        22,
    ]

    async def reject_timed_video(
        source: str,
        *,
        use_audio: bool,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
    ) -> TimedVideo:
        raise AssertionError(
            "Top-level video without audio must retain legacy decoding"
        )

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.prompt_frontend.load_timed_video",
        reject_timed_video,
    )
    connector = MultiModalResourceConnector(
        allowed_local_media_path=str(encoded_video_path.parent)
    )
    monkeypatch.setattr(
        "sglang_omni.preprocessing.resource_connector.get_global_resource_connector",
        lambda: connector,
    )
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(encoded_video_path.parent),
        ),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        if source_kind == "url":
            video_source = (
                f"http://127.0.0.1:{server.server_port}/{encoded_video_path.name}"
            )
        elif source_kind == "file_uri":
            video_source = encoded_video_path.as_uri()
        else:
            video_source = str(encoded_video_path)
        result = asyncio.run(
            video_preprocessor(
                StagePayload(
                    request_id="legacy-video-question",
                    request=OmniRequest(
                        inputs={
                            "messages": [{"role": "user", "content": "Answer."}],
                            "images": [explicit_image],
                            "audios": [question_audio],
                            video_alias: video_source,
                            "use_audio_in_video": False,
                            "video_fps": 2,
                            "video_max_frames": 128,
                            "video_min_pixels": 3136,
                            "video_max_pixels": 3136,
                        },
                        metadata={"videos": [video_source], "audios": [question_audio]},
                    ),
                    data=None,
                )
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    expected_content = "\n".join(
        ["<image>./</image>"] * 5 + ["<audio>./</audio>", "Answer."]
    )
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\n{expected_content}<|im_end|>"
    )
    assert video_processor.images[0][0] == explicit_image
    for actual_frame, expected_frame in zip(
        video_processor.images[0][1:], expected_frames, strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_frame), np.asarray(expected_frame)
        )
    np.testing.assert_array_equal(video_processor.audios[0][0], question_audio)
    assert video_processor.audio_parts is None
    assert video_processor.max_slice_nums == 1
    assert video_processor.use_image_id is False
    image_cache_key = result.data["mm_inputs"]["image"]["cache_key"]
    assert image_cache_key.endswith("|policy=video-v1")
    assert (
        result.data["encoder_inputs"]["image_encoder"]["cache_key"] == image_cache_key
    )
    assert result.request.inputs is None
    assert "videos" not in result.request.metadata
    assert "audios" not in result.request.metadata


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
    assert video_processor.audio_parts is None


@pytest.mark.parametrize("use_audio", [False])
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


def test_top_level_video_real_processor_bounds_match_embedding_rows(
    encoded_video_path: Path,
) -> None:
    checkpoint = os.environ.get("MINICPMO_CHECKPOINT")
    if not checkpoint:
        pytest.skip("MINICPMO_CHECKPOINT is required for real processor validation")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        processor  # noqa: leading-underscore  # Existing processor injection contract.
    )
    preprocessor.tokenizer = processor.tokenizer
    preprocessor.speech_enabled = False
    explicit_image = Image.new("RGB", (56, 56), "green")
    question_audio = np.sin(np.arange(16000, dtype=np.float32) / 20)
    result = asyncio.run(
        preprocessor(
            StagePayload(
                request_id="real-video-bounds",
                request=OmniRequest(
                    inputs={
                        "messages": [{"role": "user", "content": "Answer."}],
                        "images": [explicit_image],
                        "videos": [str(encoded_video_path)],
                        "audios": [question_audio],
                        "video_fps": 2,
                        "video_min_pixels": 3136,
                        "video_max_pixels": 3136,
                    }
                ),
                data=None,
            )
        )
    )
    image_bounds = result.data["mm_inputs"]["image"]["bounds"]
    audio_bounds = result.data["mm_inputs"]["audio"]["bounds"]
    image_inputs = result.data["encoder_inputs"]["image_encoder"]
    audio_inputs = result.data["encoder_inputs"]["audio_encoder"]
    assert image_bounds.shape == (5, 2)
    assert len(image_inputs["pixel_values"]) == 5
    assert image_inputs["tgt_sizes"].shape[0] == 5
    image_row_count = int((image_bounds[:, 1] - image_bounds[:, 0]).sum())
    audio_row_count = int((audio_bounds[:, 1] - audio_bounds[:, 0]).sum())
    assert image_row_count == 5 * processor.image_processor.image_feature_size
    assert audio_row_count == int(
        feature_lens_after_pooling(
            audio_inputs["audio_feature_lens"], processor.pool_step
        ).sum()
    )
    assert int(image_bounds[-1, 1]) < int(audio_bounds[0, 0])
    embeddings = {
        "image_embeds": torch.full((image_row_count, 2), 1.0),
        "audio_embeds": torch.full((audio_row_count, 2), 2.0),
    }
    input_ids, positions = apply_mm_pad_values(
        result.data["prompt"]["input_ids"],
        mm_inputs=result.data["mm_inputs"],
        model_inputs=embeddings,
        vocab_size=processor.tokenizer.vocab_size,
    )
    for modality in ("image", "audio"):
        assert len(positions[modality]) == embeddings[f"{modality}_embeds"].shape[0]
        assert torch.all(
            input_ids[positions[modality]] == embeddings["pad_values"][modality]
        )


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


@pytest.mark.parametrize("max_frames", [1, 3])
def test_video_decoder_caps_timestamp_allocation(
    encoded_video_path: Path, monkeypatch: pytest.MonkeyPatch, max_frames: int
) -> None:
    original_arange = np.arange

    def bounded_arange(
        start: float, stop: float, step: float
    ) -> npt.NDArray[np.float64]:
        assert (stop - start) / step <= max_frames
        return original_arange(start, stop, step)

    monkeypatch.setattr(video_frontend.np, "arange", bounded_arange)
    video = MiniCPMVideoIO(use_audio=False, fps=1e12, max_frames=max_frames).load_file(
        encoded_video_path
    )

    assert len(video.frames) == max_frames
    assert video.timestamps_seconds[0] == 0.0
    if max_frames > 1:
        assert video.timestamps_seconds[-1] == pytest.approx(2.3 - 1e-12)


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


@pytest.mark.parametrize("fps", [0.0, float("inf"), float("nan")])
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


@pytest.mark.parametrize(
    "as_video", [False, True], ids=["two-clips", "three-fps-video"]
)
def test_real_processor_fractional_audio_segments_match_embedding_rows(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_processor: ProcessorMixin,
    as_video: bool,
) -> None:
    if as_video:
        waveform = np.sin(np.arange(16000, dtype=np.float32) / 20)
        segments = list(np.split(waveform, [16000 // 3, 32000 // 3]))
        video = TimedVideo(
            frames=[Image.new("RGB", (56, 56)) for _ in segments],
            audio_segments=segments,
            timestamps_seconds=[0.0, 1.0 / 3, 2.0 / 3],
            duration_seconds=1.0,
        )
        monkeypatch.setattr(
            "sglang_omni.models.minicpm_o.prompt_frontend.load_timed_video",
            AsyncMock(return_value=video),
        )
        content = [
            {"type": "video_url", "video_url": {"url": "clip.mp4", "use_audio": True}}
        ]
    else:
        segments = [np.sin(np.arange(16800, dtype=np.float32) / 20) for _ in range(2)]
        content = segments

    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = checkpoint_processor  # noqa: leading-underscore  # Existing processor injection contract.
    preprocessor.tokenizer = checkpoint_processor.tokenizer
    preprocessor.speech_enabled = False
    result = asyncio.run(
        preprocessor(
            StagePayload(
                request_id="fractional-audio-segments",
                request=OmniRequest(inputs=[{"role": "user", "content": content}]),
                data=None,
            )
        )
    )
    audio_bounds = result.data["mm_inputs"]["audio"]["bounds"]
    audio_inputs = result.data["encoder_inputs"]["audio_encoder"]
    embedding_lengths = feature_lens_after_pooling(
        audio_inputs["audio_feature_lens"], checkpoint_processor.pool_step
    )
    assert len(embedding_lengths) == len(segments)
    torch.testing.assert_close(
        audio_bounds[:, 1] - audio_bounds[:, 0], embedding_lengths, check_dtype=False
    )
    embedding_row_count = int(embedding_lengths.sum())
    assert embedding_row_count == (9 if as_video else 20)
    embeddings = {"audio_embeds": torch.zeros(embedding_row_count, 2)}
    _, positions = apply_mm_pad_values(
        result.data["prompt"]["input_ids"],
        mm_inputs=result.data["mm_inputs"],
        model_inputs=embeddings,
        vocab_size=checkpoint_processor.tokenizer.vocab_size,
    )
    assert len(positions["audio"]) == embedding_row_count


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
    preprocessor._processor_lock = (  # noqa: leading-underscore  # production contract
        threading.Lock()
    )
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
    preprocessor._processor_lock = (  # noqa: leading-underscore  # production contract
        threading.Lock()
    )
    preprocessor.model_dir = "checkpoint-fixture"
    preprocessor.device = torch.device("cpu")
    monkeypatch.setattr(
        preprocessor_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: checkpoint_processor,
    )
    reference_process_image = checkpoint_processor.process_image
    assert preprocessor.processor.process_image == reference_process_image
