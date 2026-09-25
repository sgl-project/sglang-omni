# SPDX-License-Identifier: Apache-2.0
"""Video prefix generation for the SocialOmni benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path

from .socialomni import PREFIX_ENCODING, _source_digest


def resolve_ffmpeg_executable() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    return str(bundled) if bundled.is_file() and os.access(bundled, os.X_OK) else None


def build_ffmpeg_prefix_command(
    ffmpeg: str, source: Path, timestamp_s: float, output: Path
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{timestamp_s:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        PREFIX_ENCODING["video_codec"],
        "-preset",
        PREFIX_ENCODING["preset"],
        "-crf",
        PREFIX_ENCODING["crf"],
        "-c:a",
        PREFIX_ENCODING["audio_codec"],
        "-b:a",
        PREFIX_ENCODING["audio_bitrate"],
        "-movflags",
        "+faststart",
        "-y",
        str(output),
    ]


async def create_video_prefix(
    input_path: str | Path, timestamp_s: float, cache_dir: str | Path
) -> Path:
    """Re-encode video and audio up to the query time into an atomic cache entry."""
    if not math.isfinite(timestamp_s) or timestamp_s <= 0:
        raise ValueError("timestamp_s must be finite and positive")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    key = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": await asyncio.to_thread(_source_digest, source),
                "timestamp_s": f"{timestamp_s:.6f}",
                "encoding": PREFIX_ENCODING,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{key}.mp4"
    if output.is_file() and output.stat().st_size:
        return output
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for SocialOmni Level 2")
    temporary = cache / f".{key}.{uuid.uuid4().hex}.tmp.mp4"
    process = await asyncio.create_subprocess_exec(
        *build_ffmpeg_prefix_command(ffmpeg, source, timestamp_s, temporary),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await process.communicate()
    except asyncio.CancelledError:
        try:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
        finally:
            temporary.unlink(missing_ok=True)
        raise
    if process.returncode or not temporary.is_file() or not temporary.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg prefix generation failed for {source}: "
            f"{stderr.decode(errors='replace')[:2000]}"
        )
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
