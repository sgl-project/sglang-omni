# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bare local paths in MultiModalResourceConnector.

Bare paths are a trusted-local convenience: they stay accepted when no
allowlist is configured, but once allowed_local_media_path is set they must
resolve inside it — the same policy file:// references follow.

The security invariant under test is that a rejected reference is never
passed to MediaIO. Rejection must happen before any I/O, not after, so every
reject case asserts both the raised error and that the recording MediaIO saw
zero load_file calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector


class _RecordingMediaIO(MediaIO[Path]):
    """Fake MediaIO that records the file paths it is asked to load."""

    def __init__(self) -> None:
        self.loaded_paths: list[Path] = []

    def load_bytes(self, data: bytes) -> Path:
        return Path("bytes")

    def load_base64(self, media_type: str, data: str) -> Path:
        return Path("base64")

    def load_file(self, filepath: Path) -> Path:
        self.loaded_paths.append(filepath)
        return filepath


def test_load_local_path_allowed_without_allowlist(tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector()
    media_io = _RecordingMediaIO()

    result = connector.load_local_path(audio, media_io)

    assert result == audio.resolve()
    assert media_io.loaded_paths == [audio.resolve()]


def test_load_local_path_inside_allowlist(tmp_path: Path) -> None:
    media_dir = tmp_path / "refs"
    media_dir.mkdir()
    audio = media_dir / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=media_dir)
    media_io = _RecordingMediaIO()

    result = connector.load_local_path(audio, media_io)

    assert result == audio.resolve()
    assert media_io.loaded_paths == [audio.resolve()]


def test_load_local_path_rejects_outside_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(outside, media_io)

    assert media_io.loaded_paths == []


def test_load_local_path_rejects_traversal_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(allowed / ".." / "outside.wav", media_io)

    assert media_io.loaded_paths == []


def test_load_local_path_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    link = allowed / "escape.wav"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # Windows without developer mode cannot create symlinks.
        pytest.skip(f"symlink creation unsupported: {exc}")

    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(link, media_io)

    assert media_io.loaded_paths == []


def test_file_url_rejection_never_reaches_media_io(tmp_path: Path) -> None:
    """A rejected file:// reference must be refused before any I/O.

    The reference carries a ".." segment, so this also pins that the path is
    normalized before containment is checked: is_relative_to() alone would accept
    <allowed>/../outside.wav.
    """
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "outside.wav").write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()
    traversal_url = (allowed / ".." / "outside.wav").as_uri()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_resource(traversal_url, media_io)

    assert media_io.loaded_paths == []


def test_file_url_passes_resolved_path_to_media_io(tmp_path: Path) -> None:
    """The connector resolves/normalizes the path; existence is media_io's job."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()
    missing = allowed / "ghost.wav"

    result = connector.load_resource(missing.as_uri(), media_io)

    assert media_io.loaded_paths == [missing.resolve()]
    assert result == missing.resolve()
