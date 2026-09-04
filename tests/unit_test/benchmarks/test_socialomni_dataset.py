# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.dataset import prepare, socialomni
from benchmarks.dataset.socialomni import (
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
    parse_socialomni_timestamp,
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


@pytest.mark.parametrize("raw", [True, "", "bad", 0, -1])
def test_parse_timestamp_rejects_invalid_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_socialomni_timestamp(raw)
