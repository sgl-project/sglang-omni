# SPDX-License-Identifier: Apache-2.0
"""Shared utilities: WAV duration, SSE parsing, service health check."""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
import subprocess
import time

import requests as requests_lib

logger = logging.getLogger(__name__)

WAV_HEADER_SIZE = 44
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return PCM playback length in seconds for raw WAV bytes."""
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
    """Parse one Server-Sent Event (SSE) JSON line."""
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None
    return json.loads(line[len(SSE_DATA_PREFIX) :])


def process_sse_line(
    line: str,
    total_duration: float,
    usage: dict | None,
) -> tuple[float, dict | None]:
    """Merge one TTS-style Server-Sent Event (SSE) JSON event with duration and latest usage."""
    event = parse_sse_event(line)
    if event is None:
        return total_duration, usage
    audio = event.get("audio")
    if audio is not None:
        chunk_b64 = audio.get("data")
        if chunk_b64:
            total_duration += get_wav_duration(base64.b64decode(chunk_b64))
    event_usage = event.get("usage")
    if event_usage is not None:
        usage = event_usage
    return total_duration, usage


def wait_for_service(
    base_url: str,
    timeout: int = 1200,
    *,
    server_process: subprocess.Popen | None = None,
    server_log_file: str | os.PathLike[str] | None = None,
    health_body_contains: str | None = None,
) -> None:
    """Wait for SGLang Omni Server to be ready."""
    logger.info(f"Waiting for service at {base_url} ...")
    start = time.time()
    while True:
        if server_process is not None:
            exit_code = server_process.poll()
            if exit_code is not None:
                log_text = ""
                if server_log_file is not None:
                    log_path = os.fspath(server_log_file)
                    if os.path.isfile(log_path):
                        with open(log_path) as f:
                            log_text = f.read()
                raise RuntimeError(f"Server exited with code {exit_code}.\n{log_text}")
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200 and (
                health_body_contains is None or health_body_contains in resp.text
            ):
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException as exc:
            logger.debug(f"Health check failed for {base_url}: {exc}")
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(1)


def save_json_results(results: dict, output_dir: str, filename: str) -> str:
    """Write results as JSON to output_dir/filename and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {path}")
    return path
