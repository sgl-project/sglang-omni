# SPDX-License-Identifier: Apache-2.0
"""Base MediaIO interface for loading media from different sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar
from urllib.parse import urlparse

_M = TypeVar("_M")


def _is_url(path: str | Path) -> bool:
    """Check if a string is a URL (HTTP, data, or file URL).

    This is a shared utility function used by audio, image, and video frontends.
    """
    if not isinstance(path, str):
        return False
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.scheme in ("http", "https", "data", "file"))


class MediaIO(ABC, Generic[_M]):
    """Base class for media I/O operations.

    Subclasses should implement methods to load media from bytes, base64 data,
    or file paths.
    """

    @abstractmethod
    def load_bytes(self, data: bytes) -> _M:
        """Load media from raw bytes.

        Args:
            data: Raw bytes of the media file.

        Returns:
            Loaded media object (type depends on subclass).
        """
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _M:
        """Load media from base64-encoded data URL.

        Args:
            media_type: MIME type of the media (e.g., "audio/wav").
            data: Base64-encoded data string.

        Returns:
            Loaded media object (type depends on subclass).
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _M:
        """Load media from a local file path.

        Args:
            filepath: Path to the media file.

        Returns:
            Loaded media object (type depends on subclass).
        """
        raise NotImplementedError
