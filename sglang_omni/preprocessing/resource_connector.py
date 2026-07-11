# SPDX-License-Identifier: Apache-2.0
"""Media connector for loading media from URLs (HTTP, data, file)."""

from __future__ import annotations

import asyncio
import atexit
import ipaddress
import logging
import socket
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
_MAX_HTTP_REDIRECTS = 5

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


def resolve_allowed_local_media_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"allowed local media path must be a directory: {path}")
    return resolved


def _next_redirect_url(response: httpx.Response) -> str:
    location = response.headers.get("location")
    if not location:
        raise ValueError("Redirect response is missing a Location header.")
    return str(response.url.join(location))


def _response_media_type(response: httpx.Response) -> str | None:
    content_type = response.headers.get("content-type")
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


def _validate_response_length(
    response: httpx.Response, *, max_bytes: int | None
) -> None:
    if max_bytes is None:
        return
    content_length = response.headers.get("content-length")
    if content_length is None:
        return
    try:
        size = int(content_length)
    except ValueError:
        return
    if size > max_bytes:
        raise ValueError(f"Media URL response exceeds {max_bytes} bytes.")


def _validate_downloaded_size(size: int, *, max_bytes: int | None) -> None:
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"Media URL response exceeds {max_bytes} bytes.")


def _resolve_remote_addresses(
    hostname: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return (literal,)

    try:
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve media URL hostname: {hostname}") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for addr_info in addr_infos:
        sockaddr = addr_info[4]
        if sockaddr:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
    if not addresses:
        raise ValueError(f"Could not resolve media URL hostname: {hostname}")
    return tuple(addresses)


def _unsafe_remote_address_category(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str | None:
    if address.is_loopback:
        return "loopback"
    if address.is_private:
        return "private"
    if address.is_link_local:
        return "link-local"
    if address.is_reserved:
        return "reserved"
    if address.is_multicast:
        return "multicast"
    if address.is_unspecified:
        return "unspecified"
    return None


def _is_allowed_remote_domain(hostname: str, allowed_domain: str) -> bool:
    allowed_domain = allowed_domain.rstrip(".").lower()
    if allowed_domain.startswith("."):
        suffix = allowed_domain[1:]
        return hostname == suffix or hostname.endswith(allowed_domain)
    return hostname == allowed_domain


def _read_limited_response_bytes(
    response: httpx.Response, *, max_bytes: int | None
) -> bytes:
    _validate_response_length(response, max_bytes=max_bytes)
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        total += len(chunk)
        _validate_downloaded_size(total, max_bytes=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _media_http_error(exc: httpx.HTTPError, url: str) -> ValueError:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return ValueError(f"Media URL returned HTTP {status_code}: {url}")
    if isinstance(exc, httpx.TimeoutException):
        return ValueError(f"Timed out loading media URL: {url}")
    return ValueError(f"Failed to load media URL {url}: {exc}")


async def _read_limited_response_bytes_async(
    response: httpx.Response, *, max_bytes: int | None
) -> bytes:
    _validate_response_length(response, max_bytes=max_bytes)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        total += len(chunk)
        _validate_downloaded_size(total, max_bytes=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


class MultiModalResourceConnector:
    """Connector for optimized multi-modal data loading."""

    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        *,
        connection: ResourceHTTPConnection = global_http_connection,
        allowed_local_media_path: str | Path | None = None,
        allowed_media_domains: list[str] | None = None,
        allow_remote_media_without_domains: bool = True,
        reject_unsafe_remote_addresses: bool = False,
    ) -> None:
        """Initialize the media connector.

        Args:
            media_io_kwargs: Additional args passed to process media inputs, keyed by modalities.
            connection: ResourceHTTPConnection instance for HTTP clients.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: Domains allowed for remote media URLs.
            allow_remote_media_without_domains: Whether remote HTTP(S) URLs are
                allowed when no domain allowlist is configured.
            reject_unsafe_remote_addresses: Whether resolved loopback, private,
                link-local, reserved, multicast, and unspecified addresses are rejected.
        """
        self.media_io_kwargs = media_io_kwargs or {}
        self.connection = connection

        self.allowed_local_media_path = None
        if allowed_local_media_path:
            self.allowed_local_media_path = resolve_allowed_local_media_path(
                allowed_local_media_path
            )

        self.allowed_media_domains = [
            domain.strip().rstrip(".").lower()
            for domain in allowed_media_domains or []
            if domain.strip()
        ]
        self.allow_remote_media_without_domains = allow_remote_media_without_domains
        self.reject_unsafe_remote_addresses = reject_unsafe_remote_addresses

    def _assert_url_allowed(self, url_spec: Any) -> None:
        """Check whether a remote media URL is allowed to be fetched."""
        hostname = url_spec.hostname
        if not hostname:
            raise ValueError("Remote media URL must include a hostname.")
        normalized_hostname = hostname.rstrip(".").lower()
        if (
            not self.allowed_media_domains
            and not self.allow_remote_media_without_domains
        ):
            raise ValueError(
                "Remote media URLs require --allowed-media-domain to be configured."
            )
        if self.allowed_media_domains and not any(
            _is_allowed_remote_domain(normalized_hostname, domain)
            for domain in self.allowed_media_domains
        ):
            raise ValueError(f"Domain {hostname} is not allowed.")

        if self.reject_unsafe_remote_addresses:
            for address in _resolve_remote_addresses(normalized_hostname):
                category = _unsafe_remote_address_category(address)
                if category is not None:
                    raise ValueError(
                        f"Remote media URL resolves to a {category} address: {address}"
                    )

    def assert_url_allowed(self, url: str) -> None:
        """Validate URL policy without loading the resource."""
        self._assert_url_allowed(urlparse(url))

    async def _assert_url_allowed_async(self, url_spec: Any) -> None:
        await asyncio.to_thread(self._assert_url_allowed, url_spec)

    def _load_data_url(self, url_spec: Any, media_io: MediaIO[_M]) -> _M:
        """Load media from a data URL (base64 encoded)."""
        path = url_spec.path or ""
        if "," not in path:
            raise ValueError("Invalid data URL format")
        spec, data = path.split(",", 1)
        if ";base64" not in spec.lower():
            raise ValueError("Data URL must use base64 encoding")
        media_type = spec.split(";")[0].lstrip("/")
        return media_io.load_base64(media_type, data)

    def _load_file_url(self, url_spec: Any, media_io: MediaIO[_M]) -> _M:
        """Load media from a file URL."""
        if not self.allowed_local_media_path:
            raise RuntimeError("Local file loading is disabled.")

        netloc = url_spec.netloc or ""
        if netloc and netloc != "localhost":
            raise ValueError(f"File URL netloc is not supported: {netloc}")
        filepath = Path(url2pathname(url_spec.path)).resolve()

        try:
            filepath.relative_to(self.allowed_local_media_path)
        except ValueError:
            raise ValueError(f"File path {filepath} is not within allowed directory.")
        return media_io.load_file(filepath)

    def load_resource(
        self,
        url: str,
        media_io: MediaIO[_M],
        timeout: float = 30.0,
        max_bytes: int | None = None,
    ) -> _M:
        """Load media from a URL.

        Args:
            url: URL to load from (HTTP/HTTPS, data, or file).
            media_io: MediaIO instance to use for loading.
            timeout: Timeout for HTTP requests in seconds.
            max_bytes: Optional HTTP response byte cap.

        Returns:
            Loaded media object.
        """
        url_spec = urlparse(url)

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            data, media_type = self._load_http_bytes(
                url, timeout=timeout, max_bytes=max_bytes
            )
            return media_io.load_http_bytes(data, media_type)

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
        max_bytes: int | None = None,
    ) -> _M:
        """Asynchronously load media from a URL.

        Args:
            url: URL to load from (HTTP/HTTPS, data, or file).
            media_io: MediaIO instance to use for loading.
            timeout: Timeout for HTTP requests in seconds.
            max_bytes: Optional HTTP response byte cap.

        Returns:
            Loaded media object.
        """
        url_spec = urlparse(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            download_start = time.time()
            data, media_type = await self._load_http_bytes_async(
                url, timeout=timeout, max_bytes=max_bytes
            )
            download_time = time.time() - download_start

            if len(data) > 1024 * 1024:
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Downloaded {len(data) / 1024 / 1024:.2f}MB in "
                    f"{download_time:.2f}s"
                )

            decode_start = time.time()
            result = await loop.run_in_executor(
                global_thread_pool, media_io.load_http_bytes, data, media_type
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

    def _load_http_bytes(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int | None,
    ) -> tuple[bytes, str | None]:
        client = self.connection.get_sync_client()
        current_url = url
        for _ in range(_MAX_HTTP_REDIRECTS + 1):
            self._assert_url_allowed(urlparse(current_url))
            try:
                with client.stream(
                    "GET",
                    current_url,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        current_url = _next_redirect_url(response)
                        continue
                    response.raise_for_status()
                    return (
                        _read_limited_response_bytes(response, max_bytes=max_bytes),
                        _response_media_type(response),
                    )
            except httpx.HTTPError as exc:
                raise _media_http_error(exc, current_url) from exc
        raise ValueError(f"Too many redirects while loading media URL: {url}")

    async def _load_http_bytes_async(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int | None,
    ) -> tuple[bytes, str | None]:
        client = await self.connection.get_async_client()
        current_url = url
        for _ in range(_MAX_HTTP_REDIRECTS + 1):
            await self._assert_url_allowed_async(urlparse(current_url))
            try:
                async with client.stream(
                    "GET",
                    current_url,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        current_url = _next_redirect_url(response)
                        continue
                    response.raise_for_status()
                    return (
                        await _read_limited_response_bytes_async(
                            response, max_bytes=max_bytes
                        ),
                        _response_media_type(response),
                    )
            except httpx.HTTPError as exc:
                raise _media_http_error(exc, current_url) from exc
        raise ValueError(f"Too many redirects while loading media URL: {url}")

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
