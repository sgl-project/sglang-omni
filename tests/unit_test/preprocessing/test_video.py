# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_omni.preprocessing import video as video_utils  # noqa: E402


def test_unpack_video_reader_result_accepts_new_qwen_vl_tuple() -> None:
    frames = torch.zeros(2, 3, 4, 4)

    video, sample_fps = video_utils._unpack_video_reader_result(
        (frames, {"video_backend": "decord"}, 2.5)
    )

    assert video is frames
    assert sample_fps == 2.5


def test_load_video_path_falls_back_to_decord_tuple(
    monkeypatch,
    tmp_path,
) -> None:
    video_utils.clear_video_decode_cache()
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"not decoded by the mocked reader")
    frames = torch.zeros(2, 3, 4, 4)
    calls: list[str] = []

    def fail_reader(ele):
        calls.append("torchcodec")
        raise ValueError("reader api mismatch")

    def decord_reader(ele):
        calls.append("decord")
        return frames, {"video_backend": "decord"}, 2.0

    monkeypatch.setattr(
        video_utils.qwen_vision,
        "get_video_reader_backend",
        lambda: "torchcodec",
    )
    monkeypatch.setitem(
        video_utils.qwen_vision.VIDEO_READER_BACKENDS,
        "torchcodec",
        fail_reader,
    )
    monkeypatch.setitem(
        video_utils.qwen_vision.VIDEO_READER_BACKENDS,
        "decord",
        decord_reader,
    )
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "smart_resize",
        lambda height, width, **kwargs: (height, width),
    )
    monkeypatch.setattr(
        video_utils.tv_f,
        "resize",
        lambda video, size, interpolation, antialias: video,
    )

    video, sample_fps = video_utils.load_video_path(path, fps=2)

    assert calls == ["torchcodec", "decord"]
    assert video is frames
    assert sample_fps == 2.0
    video_utils.clear_video_decode_cache()


def test_load_video_path_reuses_local_decode_cache(
    monkeypatch,
    tmp_path,
) -> None:
    video_utils.clear_video_decode_cache()
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"stable local video")
    frames = torch.zeros(2, 3, 4, 4)
    calls = 0

    def decord_reader(ele):
        nonlocal calls
        calls += 1
        return frames, {"video_backend": "decord"}, 2.0

    monkeypatch.setattr(
        video_utils.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "different-temp-root"),
    )
    monkeypatch.setenv("SGLANG_OMNI_VIDEO_DECODE_CACHE", "1")
    monkeypatch.setenv("SGLANG_OMNI_VIDEO_DECODE_CACHE_MAX_BYTES", str(1024 * 1024))
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "get_video_reader_backend",
        lambda: "decord",
    )
    monkeypatch.setitem(
        video_utils.qwen_vision.VIDEO_READER_BACKENDS,
        "decord",
        decord_reader,
    )
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "smart_resize",
        lambda height, width, **kwargs: (height, width),
    )
    monkeypatch.setattr(
        video_utils.tv_f,
        "resize",
        lambda video, size, interpolation, antialias: video,
    )

    first, first_fps = video_utils.load_video_path(path, fps=2)
    second, second_fps = video_utils.load_video_path(path, fps=2)

    assert calls == 1
    assert first is second
    assert first_fps == second_fps == 2.0
    video_utils.clear_video_decode_cache()


def test_load_video_path_cache_is_opt_in_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    video_utils.clear_video_decode_cache()
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"stable local video")
    frames = torch.zeros(2, 3, 4, 4)
    calls = 0

    def decord_reader(ele):
        nonlocal calls
        calls += 1
        return frames.clone(), {"video_backend": "decord"}, 2.0

    monkeypatch.delenv("SGLANG_OMNI_VIDEO_DECODE_CACHE", raising=False)
    monkeypatch.setattr(
        video_utils.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "different-temp-root"),
    )
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "get_video_reader_backend",
        lambda: "decord",
    )
    monkeypatch.setitem(
        video_utils.qwen_vision.VIDEO_READER_BACKENDS,
        "decord",
        decord_reader,
    )
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "smart_resize",
        lambda height, width, **kwargs: (height, width),
    )
    monkeypatch.setattr(
        video_utils.tv_f,
        "resize",
        lambda video, size, interpolation, antialias: video,
    )

    first, first_fps = video_utils.load_video_path(path, fps=2)
    second, second_fps = video_utils.load_video_path(path, fps=2)

    assert calls == 2
    assert first is not second
    assert first_fps == second_fps == 2.0
    video_utils.clear_video_decode_cache()


def test_load_video_path_does_not_cache_temp_files(
    monkeypatch,
    tmp_path,
) -> None:
    video_utils.clear_video_decode_cache()
    temp_root = tmp_path / "temp-root"
    path = temp_root / "request" / "sample.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"request scoped video")
    frames = torch.zeros(2, 3, 4, 4)
    calls = 0

    def decord_reader(ele):
        nonlocal calls
        calls += 1
        return frames.clone(), {"video_backend": "decord"}, 2.0

    monkeypatch.setattr(video_utils.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setenv("SGLANG_OMNI_VIDEO_DECODE_CACHE", "1")
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "get_video_reader_backend",
        lambda: "decord",
    )
    monkeypatch.setitem(
        video_utils.qwen_vision.VIDEO_READER_BACKENDS,
        "decord",
        decord_reader,
    )
    monkeypatch.setattr(
        video_utils.qwen_vision,
        "smart_resize",
        lambda height, width, **kwargs: (height, width),
    )
    monkeypatch.setattr(
        video_utils.tv_f,
        "resize",
        lambda video, size, interpolation, antialias: video,
    )

    first, first_fps = video_utils.load_video_path(path, fps=2)
    second, second_fps = video_utils.load_video_path(path, fps=2)

    assert calls == 2
    assert first is not second
    assert first_fps == second_fps == 2.0
    video_utils.clear_video_decode_cache()
