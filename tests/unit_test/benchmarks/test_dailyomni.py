from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks.dataset import dailyomni
from benchmarks.dataset.videomme import VideoAMMESample
from benchmarks.tasks.video_understanding import make_video_send_fn


def _write_dailyomni_fixture(tmp_path: Path, **overrides: object) -> Path:
    row = {
        "Question": "What happened at the same time?",
        "Choice": ["A. First", "B. Second", "C. Third", "D. Fourth"],
        "Answer": "B",
        "video_id": "video-id-01",
        "Type": "AV Event Alignment",
        "content_fine_category": "Daily Routines",
        "video_category": "People & Blogs",
        "video_duration": "30s",
    }
    row.update(overrides)

    dataset_dir = tmp_path / "dailyomni"
    media_dir = dataset_dir / "Videos"
    media_dir.mkdir(parents=True)
    (dataset_dir / "qa.json").write_text(json.dumps([row]), encoding="utf-8")

    video_id = str(row["video_id"])
    if not video_id.startswith("."):
        video_dir = media_dir / video_id
        video_dir.mkdir()
        (video_dir / f"{video_id}_video.mp4").write_bytes(b"video")
        (video_dir / f"{video_id}_audio.wav").write_bytes(b"audio")
    return dataset_dir


def test_load_dailyomni_samples_maps_official_schema(tmp_path: Path) -> None:
    dataset_dir = _write_dailyomni_fixture(tmp_path)

    [sample] = dailyomni.load_dailyomni_samples(repo_id=str(dataset_dir))

    assert sample.sample_id == "dailyomni:0"
    assert sample.video_id == "video-id-01"
    assert sample.options == ["First", "Second", "Third", "Fourth"]
    assert sample.answer == "B"
    assert sample.duration == "30s"
    assert sample.domain == "People & Blogs"
    assert sample.task_type == "AV Event Alignment"
    assert sample.sub_category == "Daily Routines"
    assert sample.video_path.endswith("video-id-01_video.mp4")
    assert sample.audio_path.endswith("video-id-01_audio.wav")


def test_load_dailyomni_samples_rejects_escaping_video_id(tmp_path: Path) -> None:
    dataset_dir = _write_dailyomni_fixture(tmp_path, video_id="../escape")

    with pytest.raises(ValueError, match="escapes media root"):
        dailyomni.load_dailyomni_samples(repo_id=str(dataset_dir))


def test_dailyomni_archive_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "Videos.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe path"):
        dailyomni._safe_extract(archive_path, tmp_path / "extracted")

    assert not (tmp_path / "escape.txt").exists()


class _FakeResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "B"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }


class _FakeSession:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def post(self, _url: str, *, json: dict) -> _FakeResponse:
        self.payload = json
        return _FakeResponse()


@pytest.mark.parametrize(
    ("enable_video", "enable_audio", "expected_media_keys"),
    [
        (True, True, {"videos", "audios"}),
        (True, False, {"videos"}),
        (False, True, {"audios"}),
    ],
)
def test_video_send_fn_supports_dailyomni_input_modes(
    enable_video: bool,
    enable_audio: bool,
    expected_media_keys: set[str],
) -> None:
    sample = VideoAMMESample(
        sample_id="dailyomni:0",
        video_path="/tmp/video.mp4",
        audio_path="/tmp/audio.wav",
        question="Question",
        options=["First", "Second", "Third", "Fourth"],
        answer="B",
        prompt="Prompt",
        all_choices=["A", "B", "C", "D"],
        index2ans={"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
    )
    session = _FakeSession()
    send_fn = make_video_send_fn(
        "test-multimodal-model",
        "http://localhost/v1/chat/completions",
        enable_video_input=enable_video,
        enable_audio_input=enable_audio,
    )

    result = asyncio.run(send_fn(session, sample))

    assert result.is_success
    assert session.payload is not None
    assert {key for key in ("videos", "audios") if key in session.payload} == (
        expected_media_keys
    )
