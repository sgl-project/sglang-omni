# SPDX-License-Identifier: Apache-2.0
"""Shared utilities: WAV duration, SSE parsing, service health check."""

from __future__ import annotations

import base64
import json
import logging
import struct
import time

logger = logging.getLogger(__name__)

WAV_HEADER_SIZE = 44
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return duration in seconds from a WAV byte buffer."""
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    pcm_size = len(wav_bytes) - WAV_HEADER_SIZE
    return pcm_size / (sample_rate * bytes_per_sample)


def parse_sse_event(line: str) -> dict | None:
    """Parse an SSE data line into a JSON dict, or None."""
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None
    return json.loads(line[len(SSE_DATA_PREFIX) :])


def process_sse_line(
    line: str,
    total_duration: float,
    usage: dict | None,
) -> tuple[float, dict | None]:
    """Process one SSE line: accumulate audio duration and capture usage."""
    event = parse_sse_event(line)
    if event is None:
        return total_duration, usage
    audio = event.get("audio")
    if isinstance(audio, dict) and audio.get("data"):
        total_duration += get_wav_duration(base64.b64decode(audio["data"]))
    event_usage = event.get("usage")
    if isinstance(event_usage, dict):
        usage = event_usage
    return total_duration, usage


def wait_for_service(base_url: str, timeout: int = 1200) -> None:
    """Block until the server health endpoint returns 200."""
    import requests as requests_lib

    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException as exc:
            logger.debug(
                "Health check request failed while waiting for service: %s", exc
            )
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(1)
