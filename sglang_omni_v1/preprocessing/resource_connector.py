# SPDX-License-Identifier: Apache-2.0
"""Media connector for loading media from URLs (HTTP, data, file)."""

from __future__ import annotations

import asyncio
import atexit
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx
import numpy.typing as npt

from .base import MediaIO

_M = TypeVar("_M")

# Global thread pool for CPU-bound tasks (decoding/resampling)
global_thread_pool = ThreadPoolExecutor(max_workers=8)
atexit.register(global_thread_pool.shutdown)


class ResourceHTTPConnection:
    """Manages persistent HTTP clients for connection pooling."""

    def __init__(self, timeout: float = 30.0):
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._timeout = timeout

    def get_sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout, follow_redirects=True)
        return self._client

    async def get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            timeout_config = httpx.Timeout(
                connect=30.0,
                read=self._timeout,
                write=30.0,
                pool=30.0,
            )
            self._async_client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True
            )
        return self._async_client

    async def close(self):
        if self._async_client:
            await self._async_client.aclose()
        if self._client:
            self._client.close()


global_http_connection = ResourceHTTPConnection()


class MultiModalResourceConnector:
    """Connector for optimized multi-modal data loading."""

    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        *,
        connection: ResourceHTTPConnection = global_http_connection,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None:
        """Initialize the media connector.

        Args:
            media_io_kwargs: Additional args passed to process media inputs, keyed by modalities.
            connection: ResourceHTTPConnection instance for HTTP clients.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: If set, only media URLs from these domains are allowed.
        """
        self.media_io_kwargs = media_io_kwargs or {}
        self.connection = connection

        self.allowed_local_media_path = None
        if allowed_local_media_path:
            p = Path(allowed_local_media_path)
            if not p.exists() or not p.is_dir():
                raise ValueError(f"Invalid path: {allowed_local_media_path}")
            self.allowed_local_media_path = p.resolve()

        self.allowed_media_domains = allowed_media_domains or []

    def _assert_url_allowed(self, url_spec: Any) -> None:
        """Check if URL hostname is in allowed domains."""
        if (
            self.allowed_media_domains
            and url_spec.hostname not in self.allowed_media_domains
        ):
            raise ValueError(f"Domain {url_spec.hostname} is not allowed.")

    def _load_data_url(self, url_spec: Any, media_io: MediaIO[_M]) -> _M:
        """Load media from a data URL (base64 encoded)."""
        path = url_spec.path or ""
        if "," not in path:
            raise ValueError("Invalid data URL format")
        spec, data = path.split(",", 1)
        media_type = spec.split(";")[0].lstrip("/")
        return media_io.load_base64(media_type, data)

    def _load_file_url(self, url_spec: Any, media_io: MediaIO[_M]) -> _M:
        """Load media from a file URL."""
        if not self.allowed_local_media_path:
            raise RuntimeError("Local file loading is disabled.")

        netloc = url_spec.netloc or ""
        filepath = Path(url2pathname(netloc + url_spec.path)).resolve()

        if self.allowed_local_media_path not in filepath.parents:
            raise ValueError(f"File path {filepath} is not within allowed directory.")
        return media_io.load_file(filepath)

    def load_resource(
        self,
        url: str,
        media_io: MediaIO[_M],
        timeout: float = 30.0,
    ) -> _M:
        """Load media from a URL.

        Args:
            url: URL to load from (HTTP/HTTPS, data, or file).
            media_io: MediaIO instance to use for loading.
            timeout: Timeout for HTTP requests in seconds.

        Returns:
            Loaded media object.
        """
        url_spec = urlparse(url)

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_allowed(url_spec)
            client = self.connection.get_sync_client()
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return media_io.load_bytes(response.content)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        raise ValueError(f"Unsupported URL scheme: {url_spec.scheme}")

    async def load_resource_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        timeout: float = 30.0,
    ) -> _M:
        """Asynchronously load media from a URL.

        Args:
            url: URL to load from (HTTP/HTTPS, data, or file).
            media_io: MediaIO instance to use for loading.
            timeout: Timeout for HTTP requests in seconds.

        Returns:
            Loaded media object.
        """
        url_spec = urlparse(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_allowed(url_spec)
            client = await self.connection.get_async_client()

            download_start = time.time()
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.content
            download_time = time.time() - download_start

            if len(data) > 1024 * 1024:
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Downloaded {len(data) / 1024 / 1024:.2f}MB in "
                    f"{download_time:.2f}s"
                )

            decode_start = time.time()
            result = await loop.run_in_executor(
                global_thread_pool, media_io.load_bytes, data
            )
            decode_time = time.time() - decode_start

            if len(data) > 1024 * 1024:
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Decoded in {decode_time:.2f}s "
                    f"(total: {download_time + decode_time:.2f}s)"
                )

            return result

        if url_spec.scheme in ["data", "file"]:
            method = (
                self._load_data_url
                if url_spec.scheme == "data"
                else self._load_file_url
            )
            return await loop.run_in_executor(
                global_thread_pool, method, url_spec, media_io
            )

        raise ValueError(f"Unsupported URL scheme: {url_spec.scheme}")

    async def fetch_audio_async(
        self,
        audio_url: str,
        *,
        target_sr: int = 16000,
        timeout: float = 30.0,
    ) -> tuple[npt.NDArray, float]:
        """Asynchronously fetch audio from a URL.

        Args:
            audio_url: URL to the audio file.
            target_sr: Target sample rate for resampling.
            timeout: Timeout for HTTP requests in seconds.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        from .audio import AudioMediaIO

        audio_io = AudioMediaIO(
            target_sr=target_sr, **self.media_io_kwargs.get("audio", {})
        )

        return await self.load_resource_async(audio_url, audio_io, timeout=timeout)

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
        timeout: float = 30.0,
    ) -> Any:
        """Asynchronously load image from a URL.

        Args:
            image_url: URL to the image file.
            image_mode: Target image mode (default: "RGB").
            timeout: Timeout for HTTP requests in seconds.

        Returns:
            PIL Image object.
        """
        from .image import ImageMediaIO

        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        return await self.load_resource_async(image_url, image_io, timeout=timeout)

    async def fetch_video_async(
        self,
        video_url: str,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        image_mode: str = "RGB",
        timeout: float = 30.0,
        extract_audio: bool = False,
        audio_target_sr: int = 16000,
    ) -> tuple[Any, float, Any | None]:
        """Asynchronously load video from a URL.

        Args:
            video_url: URL to the video file.
            fps: Target FPS for video loading.
            max_frames: Optional frame cap passed to the video reader backend.
            min_pixels: Optional lower resize budget per frame.
            max_pixels: Optional upper resize budget per frame.
            total_pixels: Optional total video pixel budget.
            image_mode: Target image mode (default: "RGB").
            timeout: Timeout for HTTP requests in seconds.
            extract_audio: If True, extract audio from video and return as third element.
            audio_target_sr: Target sample rate for audio extraction (default: 16000).

        Returns:
            Tuple of (video_tensor, sample_fps, audio_or_None).
        """
        from .video import VideoMediaIO

        video_io = VideoMediaIO(
            fps=fps,
            max_frames=max_frames,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            image_mode=image_mode,
            extract_audio=extract_audio,
            audio_target_sr=audio_target_sr,
            **self.media_io_kwargs.get("video", {}),
        )

        return await self.load_resource_async(video_url, video_io, timeout=timeout)


_global_connector: MultiModalResourceConnector | None = None


def get_global_resource_connector() -> MultiModalResourceConnector:
    """Get or create the global resource connector."""
    global _global_connector
    if _global_connector is None:
        _global_connector = MultiModalResourceConnector()
    return _global_connector
