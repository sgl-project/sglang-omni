# SPDX-License-Identifier: Apache-2.0
"""Artifact storage helpers for the S2-Pro TTS playground."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path


class ArtifactStore:
    """Manage temporary demo artifacts under a dedicated directory."""

    def __init__(self, *, prefix: str = "sglang-omni-tts-") -> None:
        self._root = Path(tempfile.mkdtemp(prefix=prefix))
        atexit.register(self.cleanup_all)

    def write_bytes(self, data: bytes, *, suffix: str = ".wav") -> str:
        fd, path = tempfile.mkstemp(dir=self._root, suffix=suffix)
        with os.fdopen(fd, "wb") as file_obj:
            file_obj.write(data)
        return path

    def cleanup_paths(self, paths: list[str] | None) -> None:
        for raw_path in paths or []:
            try:
                path = Path(raw_path).resolve()
            except OSError:
                continue
            try:
                path.relative_to(self._root)
            except ValueError:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def cleanup_all(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
