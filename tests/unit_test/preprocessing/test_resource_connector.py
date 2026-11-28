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

import asyncio
import base64
import io
import sys
import threading
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

from sglang_omni.client.audio import encode_wav
from sglang_omni.preprocessing import resource_connector
from sglang_omni.preprocessing.audio import ensure_audio_list_async
from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.preprocessing.video import ensure_video_list_async


class BytesMediaIO(MediaIO[bytes]):
    def load_bytes(self, data: bytes) -> bytes:
        return data

    def load_base64(self, media_type: str, data: str) -> bytes:
        return base64.b64decode(data, validate=True)

    def load_file(self, filepath: Path) -> bytes:
        return filepath.read_bytes()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("%2Bw%3D%3D", b"\xfb"),
        ("%2Fw%3D%3D", b"\xff"),
        ("%2b%2f8%3d", b"\xfb\xff"),
        ("+/8=", b"\xfb\xff"),
        ("+%2F8%3D", b"\xfb\xff"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_data_url_percent_encoding_reaches_base64_decoder(
    payload, expected, asynchronous
):
    connector = MultiModalResourceConnector()
    url = f"data:application/octet-stream;base64,{payload}"
    if asynchronous:
        result = asyncio.run(connector.load_resource_async(url, BytesMediaIO()))
    else:
        result = connector.load_resource(url, BytesMediaIO())
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("escaped", [False, True])
async def test_real_image_and_audio_data_urls_keep_decoded_content(escaped):
    image = Image.new("RGB", (3, 2), (251, 127, 63))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    audio = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    wav = encode_wav(audio, 16000)

    def data_url(media_type, content):
        encoded = base64.b64encode(content).decode("ascii")
        if escaped:
            encoded = quote(encoded, safe="")
        else:
            pass
        return f"data:{media_type};base64,{encoded}"

    images = await ensure_image_list_async(
        [data_url("image/png", image_bytes.getvalue())]
    )
    audios = await ensure_audio_list_async([data_url("audio/wav", wav)])
    assert images[0].size == image.size
    assert images[0].tobytes() == image.tobytes()
    np.testing.assert_allclose(audios[0], audio, atol=1 / 32768)


@pytest.mark.skipif(sys.platform != "linux", reason="comm names live in /proc")
def test_media_decode_workers_have_their_python_name_in_proc_comm() -> None:
    worker = resource_connector.global_thread_pool.submit(
        threading.current_thread
    ).result(timeout=5.0)

    native_name = Path(f"/proc/self/task/{worker.native_id}/comm").read_text().strip()

    assert worker.name.startswith("media-decode_")
    assert native_name == worker.name[:15]


class RecordingMediaIO(MediaIO[Path]):
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


def test_load_local_path_inside_allowlist(tmp_path: Path) -> None:
    media_dir = tmp_path / "refs"
    media_dir.mkdir()
    audio = media_dir / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=media_dir)
    media_io = RecordingMediaIO()

    result = connector.load_local_path(audio, media_io)

    assert result == audio.resolve()
    assert media_io.loaded_paths == [audio.resolve()]


def test_load_local_path_rejects_outside_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(outside, media_io)

    assert media_io.loaded_paths == []


def test_chat_media_follows_the_server_media_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    missing = str(tmp_path / "missing.mp4")
    monkeypatch.setenv(resource_connector.ALLOWED_LOCAL_MEDIA_PATH_ENV, "")
    monkeypatch.setenv(resource_connector.ALLOWED_MEDIA_DOMAINS_ENV, "")
    resource_connector.export_media_policy(str(allowed), ["example.com"])
    connector = MultiModalResourceConnector()

    for load in (
        ensure_image_list_async([str(outside)], media_connector=connector),
        ensure_audio_list_async([str(outside)], resource_connector=connector),
        ensure_video_list_async(
            [str(outside)], resource_connector=connector, video_decode_threads=16
        ),
        ensure_video_list_async(
            [missing], resource_connector=connector, video_decode_threads=16
        ),
    ):
        with pytest.raises(ValueError, match="not within allowed directory"):
            asyncio.run(load)
    with pytest.raises(ValueError, match="is not allowed"):
        connector.assert_url_allowed("http://127.0.0.1/image.png")


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
    media_io = RecordingMediaIO()

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
    media_io = RecordingMediaIO()
    traversal_url = (allowed / ".." / "outside.wav").as_uri()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_resource(traversal_url, media_io)

    assert media_io.loaded_paths == []
