# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import av
import pytest

from benchmarks.dataset import prepare, socialomni
from benchmarks.dataset.socialomni import (
    build_ffmpeg_prefix_command,
    create_video_prefix,
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
    parse_socialomni_timestamp,
    resolve_ffmpeg_executable,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _level1(sample_id: str, video: str, consistency: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "video_path": video,
        "question": "Who is speaking?",
        "options": ["A. one", "B. two", "C. three", "D. four"],
        "correct_answer": "A",
        "metadata": {"consistency": consistency},
    }


def _level2(sample_id: str, video: str, answer: str) -> dict[str, object]:
    return {
        "video_id": sample_id,
        "video_file": video,
        "full_asr": "Reference text for the judge only.",
        "question_1": {
            "question": "Should Alex speak now?",
            "timestamp": "00:03:25",
            "correct_answer": "A" if answer == "YES" else "B",
            "option_A": "YES",
            "option_B": "NO",
        },
        "question_2": {
            "question": "What should Alex say?",
            "answer": "Hello" if answer == "YES" else "",
        },
        "metadata": {},
    }


def test_loaders_preserve_nested_paths_and_mini_groups(tmp_path: Path) -> None:
    level1 = tmp_path / "data" / "level_1"
    level2 = tmp_path / "data" / "level_2"
    for path in (
        level1 / "videos" / "nested" / "visible.mp4",
        level1 / "videos" / "mismatch.mp4",
        level2 / "videos" / "yes.mp4",
        level2 / "videos" / "nested" / "no.mp4",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write(
        level1 / "dataset.json",
        [
            _level1("visible", "nested/visible.mp4", "consistent"),
            _level1("mismatch", "mismatch.mp4", "inconsistent"),
        ],
    )
    _write(
        level2 / "annotations.json",
        {
            "total_samples": 2,
            "data": [
                _level2("yes", "yes.mp4", "YES"),
                _level2("no", "nested/no.mp4", "NO"),
            ],
        },
    )

    first = load_socialomni_level1_samples(tmp_path, mini=True)
    second = load_socialomni_level2_samples(tmp_path, mini=True)

    assert [sample.sample_id for sample in first] == ["visible", "mismatch"]
    assert (
        Path(first[0].video_path)
        .relative_to(tmp_path)
        .as_posix()
        .endswith("videos/nested/visible.mp4")
    )
    assert [sample.gold_when for sample in second] == ["YES", "NO"]
    assert second[0].timestamp_s == 3.25


def test_inspect_dataset_matches_expected_metadata_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    level1 = tmp_path / "data" / "level_1" / "dataset.json"
    level2 = tmp_path / "data" / "level_2" / "annotations.json"
    _write(level1, [])
    _write(level2, {"total_samples": 0, "data": []})
    expected_metadata = {
        "level1": socialomni._sha256(level1),
        "level2": socialomni._sha256(level2),
    }
    monkeypatch.setattr(socialomni, "SOCIALOMNI_METADATA_SHA256", expected_metadata)

    identity = inspect_socialomni_dataset(tmp_path, ("level1", "level2"))

    assert identity["metadata_sha256"] == expected_metadata
    assert identity["verification_scope"] == "metadata_only"
    assert identity["metadata_matches_expected_revision"] is True


def test_prepare_downloads_pinned_public_snapshot(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def fake_snapshot_download(**kwargs) -> None:
        observed.update(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    prepare.download_dataset(
        socialomni.SOCIALOMNI_DATASET_ID,
        local_dir=str(tmp_path),
        quiet=True,
    )

    assert observed == {
        "repo_id": socialomni.SOCIALOMNI_DATASET_ID,
        "repo_type": "dataset",
        "local_dir": str(tmp_path),
        "allow_patterns": ["README.md", "data/level_1/**", "data/level_2/**"],
        "revision": socialomni.SOCIALOMNI_DATASET_REVISION,
    }


def test_prepare_rejects_unpinned_socialomni_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned dataset revision"):
        prepare.download_dataset(
            socialomni.SOCIALOMNI_DATASET_ID,
            revision="main",
            local_dir=str(tmp_path),
            quiet=True,
        )


def test_inspect_dataset_rejects_modified_metadata_as_expected_revision(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = tmp_path / "data" / "level_1" / "dataset.json"
    _write(metadata, [])
    monkeypatch.setattr(socialomni, "SOCIALOMNI_METADATA_SHA256", {"level1": "0" * 64})

    identity = inspect_socialomni_dataset(tmp_path, ("level1",))

    assert identity["metadata_matches_expected_revision"] is False


def test_inspect_dataset_checks_only_requested_levels(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "level_1" / "dataset.json", [])
    _write(tmp_path / "data" / "level_2" / "annotations.json", "invalid")

    identity = inspect_socialomni_dataset(tmp_path, ("level1",))

    assert set(identity["metadata_sha256"]) == {"level1"}


@pytest.mark.parametrize("video", ["../escape.mp4", "/tmp/escape.mp4", "level_2/x.mp4"])
def test_level1_rejects_path_escape(tmp_path: Path, video: str) -> None:
    level = tmp_path / "data" / "level_1"
    _write(level / "dataset.json", [_level1("bad", video, "consistent")])
    with pytest.raises(ValueError, match="unsafe|wrong"):
        load_socialomni_level1_samples(tmp_path)


@pytest.mark.parametrize(
    "level,filename,loader",
    [
        ("level_1", "dataset.json", socialomni.load_socialomni_level1_samples),
        ("level_2", "annotations.json", socialomni.load_socialomni_level2_samples),
    ],
)
@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_metadata_symlink_containment(
    tmp_path, level, filename, loader, outside, direct
):
    root = tmp_path / level if direct else tmp_path / "dataset"
    directory = root if direct else root / "data" / level
    directory.mkdir(parents=True)
    target = tmp_path / "outside.json" if outside else root / "actual.json"
    target.write_text("[]")
    (directory / filename).symlink_to(target)
    if outside:
        with pytest.raises(ValueError, match="metadata escapes dataset root"):
            loader(root)
        with pytest.raises(ValueError, match="metadata escapes dataset root"):
            inspect_socialomni_dataset(root, [level.replace("_", "")])
    else:
        assert socialomni._level_dir(root, level, filename) == directory


def test_direct_level_name_preserves_nested_candidates(tmp_path):
    root = tmp_path / "level_1"
    directory = root / "data" / "level_1"
    _write(directory / "dataset.json", [])
    assert socialomni._level_dir(root, "level_1", "dataset.json") == directory


def test_level1_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.touch()
    level = tmp_path / "data" / "level_1"
    videos = level / "videos"
    videos.mkdir(parents=True)
    (videos / "escape.mp4").symlink_to(outside)
    _write(level / "dataset.json", [_level1("bad", "escape.mp4", "consistent")])
    with pytest.raises(ValueError, match="escapes"):
        load_socialomni_level1_samples(tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(3, 3.0), ("3.5", 3.5), ("01:03.5", 63.5), ("00:17:25", 17.25)],
)
def test_parse_timestamp(raw: object, expected: float) -> None:
    assert parse_socialomni_timestamp(raw) == expected


@pytest.mark.parametrize(
    "raw", [True, "", "bad", 0, -1, float("nan"), float("inf"), float("-inf")]
)
def test_parse_timestamp_rejects_invalid_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_socialomni_timestamp(raw)


@pytest.mark.parametrize("replace_file", [False, True])
def test_source_digest_cache_invalidates_changed_media(
    tmp_path, monkeypatch, replace_file
):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"first")
    original_stat = path.stat()
    time.sleep(1.1)
    original_hash = socialomni._sha256
    reads = []

    def counted_hash(source):
        reads.append(source)
        return original_hash(source)

    monkeypatch.setattr(socialomni, "_sha256", counted_hash)
    first = socialomni._source_digest(path)
    assert socialomni._source_digest(path) == first
    assert len(reads) == 1
    if replace_file:
        replacement = tmp_path / "replacement.mp4"
        replacement.write_bytes(b"other")
        replacement.replace(path)
    else:
        path.write_bytes(b"other")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert socialomni._source_digest(path) != first
    assert len(reads) == 2


def test_recent_source_changes_bypass_digest_cache(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"first")
    original_stat = path.stat()
    first = socialomni._source_digest(path)
    path.write_bytes(b"other")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert socialomni._source_digest(path) != first


def test_source_digest_rejects_changes_during_hashing(tmp_path, monkeypatch):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"first")
    original_hash = socialomni._sha256

    def changing_hash(source):
        digest = original_hash(source)
        source.write_bytes(b"changed")
        return digest

    monkeypatch.setattr(socialomni, "_sha256", changing_hash)
    with pytest.raises(RuntimeError, match="changed while computing"):
        socialomni._source_digest(path)
    monkeypatch.setattr(socialomni, "_sha256", original_hash)
    assert socialomni._source_digest(path) == original_hash(path)


def test_prefix_command_reencodes_video_and_audio(tmp_path: Path) -> None:
    command = build_ffmpeg_prefix_command(
        "ffmpeg", tmp_path / "source.mp4", 1.25, tmp_path / "prefix.mp4"
    )
    assert "-c:v" in command and "libx264" in command
    assert "-c:a" in command and "aac" in command
    assert "copy" not in command
    assert command[command.index("-t") + 1] == "1.250000"


def test_prepare_import_does_not_require_imageio_ffmpeg() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['imageio_ffmpeg'] = None; "
            "from benchmarks.dataset import prepare",
        ],
        check=True,
        timeout=10,
        cwd=Path(__file__).resolve().parents[3],
    )


def test_system_ffmpeg_does_not_require_imageio_ffmpeg(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(socialomni.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    assert socialomni.resolve_ffmpeg_executable() == "/usr/bin/ffmpeg"


@pytest.mark.asyncio
async def test_missing_encoder_is_a_prefix_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(socialomni.shutil, "which", lambda _: None)
    assert socialomni.resolve_ffmpeg_executable() is None
    source = tmp_path / "video.mp4"
    source.touch()
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        await socialomni.create_video_prefix(source, 1, tmp_path / "cache")


@pytest.mark.asyncio
async def test_cancel_prefix_stops_process_and_removes_temporary(
    tmp_path: Path, monkeypatch
) -> None:
    """Cancellation must leave neither an encoder process nor a partial cache file."""
    source = tmp_path / "source.mp4"
    source.touch()
    cache = tmp_path / "cache"
    started = asyncio.Event()
    processes = []
    create_subprocess = asyncio.create_subprocess_exec

    async def start_process(*command, **kwargs):
        process = await create_subprocess(*command, **kwargs)
        processes.append(process)
        assert await process.stdout.readline() == b"ready\n"
        started.set()
        return process

    monkeypatch.setattr(socialomni, "resolve_ffmpeg_executable", lambda: sys.executable)
    monkeypatch.setattr(
        socialomni,
        "build_ffmpeg_prefix_command",
        lambda ffmpeg, source, timestamp, output: [
            ffmpeg,
            "-c",
            "import pathlib, sys, time; "
            "pathlib.Path(sys.argv[1]).write_bytes(b'partial'); "
            "print('ready', flush=True); time.sleep(60)",
            str(output),
        ],
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start_process)

    task = asyncio.create_task(socialomni.create_video_prefix(source, 1, cache))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        assert list(cache.glob("*.tmp.mp4"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

        assert processes[0].returncode is not None
        assert not list(cache.iterdir())
    finally:
        task.cancel()
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.communicate()


@pytest.mark.asyncio
async def test_prefix_rename_failure_removes_temporary(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    cache = tmp_path / "cache"
    cache.mkdir()
    retained = cache / "existing.mp4"
    retained.write_bytes(b"keep")

    class Encoder:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def encode(*command, **kwargs):
        Path(command[-1]).write_bytes(b"encoded")
        return Encoder()

    def fail_replace(path, output):
        assert path.is_file()
        raise PermissionError("rename denied")

    monkeypatch.setattr(socialomni, "resolve_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", encode)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="rename denied"):
        await create_video_prefix(source, 1, cache)
    assert list(cache.iterdir()) == [retained]
    assert retained.read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_prefix_media_ends_at_query_time(tmp_path: Path, monkeypatch) -> None:
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        pytest.skip("ffmpeg is unavailable")
    source = tmp_path / "source.mp4"
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=10:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        str(source),
    )
    assert await process.wait() == 0
    await asyncio.sleep(1.1)
    monkeypatch.chdir(tmp_path)
    prefix = await create_video_prefix(source, 0.75, "cache")
    assert prefix.is_absolute()

    def unexpected_hash(path):
        pytest.fail("Cached source must not be read again")

    monkeypatch.setattr(socialomni, "_sha256", unexpected_hash)
    monkeypatch.setattr(socialomni, "resolve_ffmpeg_executable", lambda: None)
    assert await socialomni.create_video_prefix(source, 0.75, "cache") == prefix
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    monkeypatch.chdir(server_dir)
    with av.open(str(prefix)) as container:
        assert container.duration is not None
        assert container.duration / av.time_base <= 0.85
