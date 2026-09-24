# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64

import numpy as np
import pytest

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")
pytest.importorskip("qwen_vl_utils")
pytest.importorskip("librosa")

from sglang_omni.preprocessing.video import ensure_video_list_async, load_video_path


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 56
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            pixels = np.full((56, 56, 3), i * 25, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.mark.parametrize("tensor_index", [0, 1, 2])
@pytest.mark.parametrize("source", ["path", "data_url"])
@pytest.mark.parametrize("extract_audio", [False, True])
def test_mixed_video_inputs_keep_order(video_file, tensor_index, source, extract_audio):
    decoded, _ = load_video_path(video_file, fps=2)
    loaded_input = video_file
    if source == "data_url":
        data = base64.b64encode(video_file.read_bytes()).decode("ascii")
        loaded_input = f"data:video/mp4;base64,{data}"
    processed = torch.zeros_like(decoded)
    inputs = [loaded_input, loaded_input]
    inputs.insert(tensor_index, processed)

    videos, fps, audios = asyncio.run(
        ensure_video_list_async(inputs, fps=2, extract_audio=extract_audio)
    )

    assert len(videos) == 3
    for index, video in enumerate(videos):
        if index == tensor_index:
            assert video is processed
        else:
            assert torch.equal(video, decoded)
    assert fps is None
    assert audios == ([None, None, None] if extract_audio else None)


def test_video_files_keep_sample_rates(video_file):
    decoded, sampled_fps = load_video_path(video_file, fps=2)

    videos, fps, audios = asyncio.run(
        ensure_video_list_async([video_file, video_file], fps=2)
    )

    assert len(videos) == 2
    assert all(torch.equal(video, decoded) for video in videos)
    assert fps == [sampled_fps, sampled_fps]
    assert audios is None


@pytest.mark.parametrize("inputs, expected_fps", [(None, None), ([], [])])
def test_empty_video_inputs(inputs, expected_fps):
    assert asyncio.run(ensure_video_list_async(inputs)) == ([], expected_fps, None)
