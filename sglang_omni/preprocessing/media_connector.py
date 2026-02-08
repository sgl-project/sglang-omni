# SPDX-License-Identifier: Apache-2.0
"""Media connector for loading media from URLs (HTTP, data, file)."""

from __future__ import annotations

import asyncio
import atexit
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import httpx
import numpy.typing as npt
from urllib.parse import urlparse
from urllib.request import url2pathname

from .base import MediaIO

_M = TypeVar("_M")

# Global thread pool for media loading operations
global_thread_pool = ThreadPoolExecutor(max_workers=4)
atexit.register(global_thread_pool.shutdown)

# Global HTTP client
_global_http_client: httpx.Client | None = None


def get_global_http_client() -> httpx.Client:
    """Get or create the global HTTP client."""
    global _global_http_client
    if _global_http_client is None:
        _global_http_client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
        )
    return _global_http_client


class MediaConnector:
    """Connector for loading media from URLs (HTTP, data, file)."""

    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the media connector.

        Args:
            media_io_kwargs: Additional args passed to process media inputs,
                             keyed by modalities. For example, to set target_sr
                             for audio, set `{"audio": {"target_sr": 16000}}`.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: If set, only media URLs that belong to this
                                   domain can be used for multi-modal inputs.
            http_client: HTTP client to use for downloads. If None, uses a
                        global client.
        """
        self.media_io_kwargs: dict[str, dict[str, Any]] = (
            media_io_kwargs if media_io_kwargs else {}
        )
        self.http_client = http_client or get_global_http_client()

        if allowed_local_media_path:
            allowed_local_media_path_ = Path(allowed_local_media_path)

            if not allowed_local_media_path_.exists():
                raise ValueError(
                    "Invalid `allowed_local_media_path`: The path "
                    f"{allowed_local_media_path_} does not exist."
                )
            if not allowed_local_media_path_.is_dir():
                raise ValueError(
                    "Invalid `allowed_local_media_path`: The path "
                    f"{allowed_local_media_path_} must be a directory."
                )
        else:
            allowed_local_media_path_ = None

        self.allowed_local_media_path = allowed_local_media_path_
        if allowed_media_domains is None:
            allowed_media_domains = []
        self.allowed_media_domains = allowed_media_domains

    def _load_data_url(
        self,
        url_spec: urlparse,
        media_io: MediaIO[_M],
    ) -> _M:
        """Load media from a data URL (base64 encoded)."""
        url_path = url_spec.path or ""
        if "," not in url_path:
            raise ValueError("Invalid data URL format: missing comma separator")

        data_spec, data = url_path.split(",", 1)
        parts = data_spec.split(";", 1)
        if len(parts) == 2:
            media_type, data_type = parts
        else:
            media_type = parts[0]
            data_type = "base64"

        # media_type may start with a leading "/" (e.g., "/audio/wav")
        media_type = media_type.lstrip("/")

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        return media_io.load_base64(media_type, data)

    def _load_file_url(
        self,
        url_spec: urlparse,
        media_io: MediaIO[_M],
    ) -> _M:
        """Load media from a file URL."""
        allowed_local_media_path = self.allowed_local_media_path
        if allowed_local_media_path is None:
            raise RuntimeError(
                "Cannot load local files without `allowed_local_media_path`."
            )

        url_spec_path = url_spec.path or ""
        url_spec_netloc = url_spec.netloc or ""
        filepath = Path(url2pathname(url_spec_netloc + url_spec_path))
        resolved_path = filepath.resolve()

        # Check if the file is within the allowed directory
        try:
            resolved_path.relative_to(allowed_local_media_path.resolve())
        except ValueError:
            raise ValueError(
                f"The file path {filepath} must be a subpath "
                f"of `allowed_local_media_path {allowed_local_media_path}`."
            )

        return media_io.load_file(resolved_path)

    def _assert_url_in_allowed_media_domains(self, url_spec: urlparse) -> None:
        """Check if URL hostname is in allowed domains."""
        if (
            self.allowed_media_domains
            and url_spec.hostname not in self.allowed_media_domains
        ):
            raise ValueError(
                f"The URL must be from one of the allowed domains: "
                f"{self.allowed_media_domains}. Input URL domain: "
                f"{url_spec.hostname}"
            )

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: float | None = None,
    ) -> _M:
        """Load media from a URL (HTTP, data, or file).

        Args:
            url: URL to load from (HTTP/HTTPS, data, or file).
            media_io: MediaIO instance to use for loading.
            fetch_timeout: Timeout for HTTP requests in seconds.

        Returns:
            Loaded media object.
        """
        url_spec = urlparse(url)

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            timeout = fetch_timeout if fetch_timeout is not None else 30.0
            response = self.http_client.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.content

            return media_io.load_bytes(data)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: float | None = None,
    ) -> _M:
        """Asynchronously load media from a URL.

        Args:
            url: URL to load from (HTTP/HTTPS, data, or file).
            media_io: MediaIO instance to use for loading.
            fetch_timeout: Timeout for HTTP requests in seconds.

        Returns:
            Loaded media object.
        """
        url_spec = urlparse(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            timeout = fetch_timeout if fetch_timeout is not None else 30.0
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.content

            future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
            return await future

        if url_spec.scheme == "data":
            future = loop.run_in_executor(
                global_thread_pool, self._load_data_url, url_spec, media_io
            )
            return await future

        if url_spec.scheme == "file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_file_url, url_spec, media_io
            )
            return await future

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    def fetch_audio(
        self,
        audio_url: str,
        *,
        target_sr: int = 16000,
    ) -> tuple[npt.NDArray, float]:
        """Load audio from a URL.

        Args:
            audio_url: URL to the audio file.
            target_sr: Target sample rate for resampling.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        from .audio import AudioMediaIO

        audio_io = AudioMediaIO(
            target_sr=target_sr, **self.media_io_kwargs.get("audio", {})
        )

        return self.load_from_url(audio_url, audio_io)

    async def fetch_audio_async(
        self,
        audio_url: str,
        *,
        target_sr: int = 16000,
    ) -> tuple[npt.NDArray, float]:
        """Asynchronously fetch audio from a URL.

        Args:
            audio_url: URL to the audio file.
            target_sr: Target sample rate for resampling.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        from .audio import AudioMediaIO

        audio_io = AudioMediaIO(
            target_sr=target_sr, **self.media_io_kwargs.get("audio", {})
        )

        return await self.load_from_url_async(audio_url, audio_io)

    def fetch_image(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Any:
        """Load image from a URL.

        Args:
            image_url: URL to the image file.
            image_mode: Target image mode (default: "RGB").

        Returns:
            PIL Image object.
        """
        from PIL import Image, UnidentifiedImageError
        from .image import ImageMediaIO

        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        try:
            return self.load_from_url(image_url, image_io)
        except UnidentifiedImageError as e:
            # Convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Any:
        """Asynchronously load image from a URL.

        Args:
            image_url: URL to the image file.
            image_mode: Target image mode (default: "RGB").

        Returns:
            PIL Image object.
        """
        from PIL import Image, UnidentifiedImageError
        from .image import ImageMediaIO

        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        try:
            return await self.load_from_url_async(image_url, image_io)
        except UnidentifiedImageError as e:
            # Convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    def fetch_video(
        self,
        video_url: str,
        *,
        fps: float | None = None,
        image_mode: str = "RGB",
    ) -> tuple[Any, float]:
        """Load video from a URL.

        Args:
            video_url: URL to the video file.
            fps: Target FPS for video loading.
            image_mode: Target image mode (default: "RGB").

        Returns:
            Tuple of (video_tensor, sample_fps).
        """
        from .video import VideoMediaIO

        video_io = VideoMediaIO(
            fps=fps, image_mode=image_mode, **self.media_io_kwargs.get("video", {})
        )

        return self.load_from_url(video_url, video_io)

    async def fetch_video_async(
        self,
        video_url: str,
        *,
        fps: float | None = None,
        image_mode: str = "RGB",
    ) -> tuple[Any, float]:
        """Asynchronously load video from a URL.

        Args:
            video_url: URL to the video file.
            fps: Target FPS for video loading.
            image_mode: Target image mode (default: "RGB").

        Returns:
            Tuple of (video_tensor, sample_fps).
        """
        from .video import VideoMediaIO

        video_io = VideoMediaIO(
            fps=fps, image_mode=image_mode, **self.media_io_kwargs.get("video", {})
        )

        return await self.load_from_url_async(video_url, video_io)


# Global media connector instance
_global_media_connector: MediaConnector | None = None


def get_global_media_connector() -> MediaConnector:
    """Get or create the global media connector."""
    global _global_media_connector
    if _global_media_connector is None:
        _global_media_connector = MediaConnector()
    return _global_media_connector

