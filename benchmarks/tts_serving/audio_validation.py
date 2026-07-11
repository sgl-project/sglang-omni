# SPDX-License-Identifier: Apache-2.0
"""Audio response validators for the TTS serving benchmark."""

from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass
from shutil import which

from benchmarks.tts_serving.metrics import (
    MIN_AUDIO_FRAME_PREFIX_BYTES,
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH,
    WAV_CHUNK_HEADER_BYTES,
    WAV_FORMAT_END,
    WAV_FORMAT_OFFSET,
    WAV_HEADER_BYTES,
    WAV_RIFF_MARKER,
    WAV_WAVE_MARKER,
    duration_from_audio_bytes,
)

MIN_GENERATED_AUDIO_DURATION_S = 0.05
MIN_PCM_AUDIO_BYTES = int(
    PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH * MIN_GENERATED_AUDIO_DURATION_S
)
MIN_COMPRESSED_AUDIO_BYTES = 32
MAX_PRINTABLE_ASCII_RATIO = 0.95
COMPRESSED_DECODE_TIMEOUT_S = 10
FFMPEG_DECODE_CHANNELS = 1
MAX_DECODED_PCM_BYTES = 64 * 1024 * 1024
COMPRESSED_DECODE_MAX_DURATION_S = (MAX_DECODED_PCM_BYTES + PCM_SAMPLE_WIDTH) / (
    PCM_SAMPLE_RATE * FFMPEG_DECODE_CHANNELS * PCM_SAMPLE_WIDTH
)

PCM_CONTENT_TYPES = frozenset(
    {"application/octet-stream", "audio/pcm", "audio/raw", ""}
)
EXPECTED_AUDIO_CONTENT_TYPES = {
    "wav": {"audio/wav", "audio/x-wav", "application/octet-stream", ""},
    "pcm": PCM_CONTENT_TYPES,
    "mp3": {"audio/mpeg", "audio/mp3"},
    "flac": {"audio/flac"},
    "aac": {"audio/aac", "audio/aacp"},
    "opus": {"audio/opus", "audio/ogg"},
}


@dataclass(frozen=True)
class AudioValidation:
    ok: bool
    error: str | None = None
    duration_s: float = 0.0


def validate_audio_response(
    body: bytes,
    *,
    response_format: str,
    content_type: str | None = None,
    sample_rate: int = PCM_SAMPLE_RATE,
    require_min_audio: bool = True,
    require_content_type: bool = True,
) -> AudioValidation:
    fmt = response_format.lower()
    normalized_content_type = _normalized_content_type(content_type)
    if not body:
        return AudioValidation(False, "audio body is empty")
    if require_content_type and not _content_type_matches(normalized_content_type, fmt):
        return AudioValidation(
            False,
            "content type does not match requested audio format "
            f"(format={fmt!r}, content_type={content_type!r})",
        )
    if fmt == "wav":
        return _validate_wav(body, require_min_audio=require_min_audio)
    if fmt == "pcm":
        return _validate_pcm(
            body,
            sample_rate=sample_rate,
            require_min_audio=require_min_audio,
            require_signal=True,
        )
    if fmt == "mp3":
        return _validate_compressed(
            body, fmt, _has_mp3_prefix(body), require_min_audio=require_min_audio
        )
    if fmt == "flac":
        return _validate_compressed(
            body, fmt, body.startswith(b"fLaC"), require_min_audio=require_min_audio
        )
    if fmt == "opus":
        return _validate_compressed(
            body, fmt, body.startswith(b"OggS"), require_min_audio=require_min_audio
        )
    if fmt == "aac":
        return _validate_compressed(
            body,
            fmt,
            _has_aac_adts_prefix(body),
            require_min_audio=require_min_audio,
        )
    return AudioValidation(False, f"unsupported response format: {response_format!r}")


def validate_pcm_chunk(
    body: bytes,
    *,
    sample_rate: int = PCM_SAMPLE_RATE,
) -> AudioValidation:
    return _validate_pcm(
        body,
        sample_rate=sample_rate,
        require_min_audio=False,
        require_signal=False,
    )


def _validate_pcm(
    body: bytes,
    *,
    sample_rate: int,
    require_min_audio: bool,
    require_signal: bool,
) -> AudioValidation:
    if not body:
        return AudioValidation(False, "PCM bytes are empty")
    if len(body) % PCM_SAMPLE_WIDTH:
        return AudioValidation(False, "PCM bytes are not aligned to 16-bit samples")
    if require_min_audio and len(body) < MIN_PCM_AUDIO_BYTES:
        return AudioValidation(
            False,
            "PCM audio is shorter than the minimum generated-audio duration "
            f"(bytes={len(body)}, minimum={MIN_PCM_AUDIO_BYTES})",
        )
    if _printable_ascii_ratio(body) >= MAX_PRINTABLE_ASCII_RATIO:
        return AudioValidation(False, "PCM bytes look like printable placeholder text")
    if require_signal and not _has_nonzero_int16_sample(body):
        return AudioValidation(False, "PCM bytes contain only zero-amplitude samples")
    return AudioValidation(
        True,
        duration_s=duration_from_audio_bytes(
            body,
            response_format="pcm",
            sample_rate=sample_rate,
        ),
    )


def _validate_wav(body: bytes, *, require_min_audio: bool) -> AudioValidation:
    if (
        len(body) <= WAV_HEADER_BYTES
        or not body.startswith(WAV_RIFF_MARKER)
        or body[WAV_FORMAT_OFFSET:WAV_FORMAT_END] != WAV_WAVE_MARKER
    ):
        return AudioValidation(False, "WAV body is missing RIFF/WAVE markers")
    wav = _wav_info(body)
    if wav is None:
        return AudioValidation(False, "WAV body is missing valid fmt/data chunks")
    sample_rate, channels, bits_per_sample, data = wav
    if bits_per_sample != 16:
        return AudioValidation(
            False,
            f"WAV validator expects 16-bit PCM, observed {bits_per_sample} bits",
        )
    if require_min_audio and len(data) < MIN_PCM_AUDIO_BYTES:
        return AudioValidation(
            False,
            "WAV data chunk is shorter than the minimum generated-audio duration "
            f"(bytes={len(data)}, minimum={MIN_PCM_AUDIO_BYTES})",
        )
    if len(data) % PCM_SAMPLE_WIDTH:
        return AudioValidation(False, "WAV data chunk is not 16-bit aligned")
    if not _has_nonzero_int16_sample(data):
        return AudioValidation(False, "WAV data chunk contains only zero samples")
    duration = len(data) / float(sample_rate * channels * (bits_per_sample // 8))
    return AudioValidation(True, duration_s=duration)


def _validate_compressed(
    body: bytes,
    response_format: str,
    has_prefix: bool,
    *,
    require_min_audio: bool,
) -> AudioValidation:
    if len(body) < MIN_COMPRESSED_AUDIO_BYTES:
        return AudioValidation(
            False,
            f"{response_format} audio is too small to be a generated response",
        )
    if not has_prefix:
        return AudioValidation(
            False,
            f"{response_format} audio is missing the expected container prefix",
        )
    decoded = _decode_compressed_to_pcm(body, response_format)
    if not decoded.ok:
        return AudioValidation(False, decoded.error)
    return _validate_pcm(
        decoded.pcm,
        sample_rate=PCM_SAMPLE_RATE,
        require_min_audio=require_min_audio,
        require_signal=True,
    )


@dataclass(frozen=True)
class _DecodedPcm:
    ok: bool
    pcm: bytes = b""
    error: str | None = None


def _decode_compressed_to_pcm(body: bytes, response_format: str) -> _DecodedPcm:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        return _DecodedPcm(
            False,
            error=(
                "ffmpeg is required to validate compressed TTS audio responses "
                f"({response_format})"
            ),
        )
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                str(FFMPEG_DECODE_CHANNELS),
                "-ar",
                str(PCM_SAMPLE_RATE),
                "-t",
                f"{COMPRESSED_DECODE_MAX_DURATION_S:.6f}",
                "pipe:1",
            ],
            input=body,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=COMPRESSED_DECODE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return _DecodedPcm(False, error=f"{response_format} decode timed out")
    if completed.returncode != 0:
        return _DecodedPcm(
            False,
            error=(
                f"{response_format} decode failed: "
                f"{_decoder_stderr(completed.stderr)}"
            ),
        )
    if not completed.stdout:
        return _DecodedPcm(False, error=f"{response_format} decoded to empty PCM")
    if len(completed.stdout) >= MAX_DECODED_PCM_BYTES:
        return _DecodedPcm(
            False,
            error=(
                f"{response_format} decoded PCM exceeds benchmark validation cap "
                f"(bytes={len(completed.stdout)}, max_bytes={MAX_DECODED_PCM_BYTES})"
            ),
        )
    return _DecodedPcm(True, pcm=completed.stdout)


def _decoder_stderr(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return "<empty stderr>"
    return text[:500]


def _wav_info(data: bytes) -> tuple[int, int, int, bytes] | None:
    sample_rate = 0
    channels = 0
    bits_per_sample = 0
    data_chunk: bytes | None = None
    pos = 12
    while pos + WAV_CHUNK_HEADER_BYTES <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        chunk_start = pos + WAV_CHUNK_HEADER_BYTES
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            return None
        if chunk_id == b"fmt " and chunk_size >= 16:
            try:
                audio_format = struct.unpack_from("<H", data, chunk_start)[0]
                channels = struct.unpack_from("<H", data, chunk_start + 2)[0]
                sample_rate = struct.unpack_from("<I", data, chunk_start + 4)[0]
                bits_per_sample = struct.unpack_from("<H", data, chunk_start + 14)[0]
            except struct.error:
                return None
            if audio_format != 1 or channels <= 0 or sample_rate <= 0:
                return None
        elif chunk_id == b"data":
            data_chunk = data[chunk_start:chunk_end]
            break
        pos = chunk_end + (chunk_size % 2)
    if not sample_rate or not channels or not bits_per_sample or data_chunk is None:
        return None
    return sample_rate, channels, bits_per_sample, data_chunk


def _content_type_matches(content_type: str, response_format: str) -> bool:
    return content_type in EXPECTED_AUDIO_CONTENT_TYPES.get(response_format, set())


def _normalized_content_type(content_type: str | None) -> str:
    return str(content_type or "").lower().split(";", 1)[0]


def _has_mp3_prefix(body: bytes) -> bool:
    return body.startswith(b"ID3") or (
        len(body) >= MIN_AUDIO_FRAME_PREFIX_BYTES
        and body[0] == 0xFF
        and (body[1] & 0xE0) == 0xE0
    )


def _has_aac_adts_prefix(body: bytes) -> bool:
    return (
        len(body) >= MIN_AUDIO_FRAME_PREFIX_BYTES
        and body[0] == 0xFF
        and (body[1] & 0xF0) == 0xF0
    )


def _has_nonzero_int16_sample(body: bytes) -> bool:
    aligned_bytes = len(body) - (len(body) % PCM_SAMPLE_WIDTH)
    if aligned_bytes <= 0:
        return False
    return body[:aligned_bytes].count(0) != aligned_bytes


def _printable_ascii_ratio(body: bytes) -> float:
    if not body:
        return 0.0
    printable = sum(1 for byte in body if byte in b"\t\n\r" or 32 <= byte <= 126)
    return printable / len(body)
