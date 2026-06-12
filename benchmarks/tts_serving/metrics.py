# SPDX-License-Identifier: Apache-2.0
"""Metrics and result types for the TTS serving benchmark."""

from __future__ import annotations

import base64
import json
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
MIN_AUDIO_FRAME_PREFIX_BYTES = 2
WAV_HEADER_BYTES = 44
WAV_CHUNK_HEADER_BYTES = 8
WAV_FORMAT_OFFSET = 8
WAV_FORMAT_END = 12
WAV_RIFF_MARKER = b"RIFF"
WAV_WAVE_MARKER = b"WAVE"
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


@dataclass
class ScenarioResult:
    scenario_id: str
    endpoint: str
    category: str
    capability_key: str | None = None
    stage_id: str | None = None
    load_mode: str | None = None
    load_concurrency: int | None = None
    configured_max_concurrency: int | None = None
    peak_inflight: int | None = None
    peak_pending_tasks: int | None = None
    scheduled_task_count: int | None = None
    load_generator_lagged: bool = False
    load_generator_saturated: bool = False
    status: str = "error"
    success: bool = False
    expected_success: bool = True
    http_status: int | None = None
    http_status_class: str | None = None
    latency_s: float = 0.0
    planned_start_s: float | None = None
    actual_start_s: float | None = None
    completed_s: float | None = None
    queue_wait_s: float | None = None
    generator_lag_s: float | None = None
    ttfa_s: float | None = None
    inter_chunk_s: list[float] = field(default_factory=list)
    audio_bytes: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    response_format: str | None = None
    batch_size: int | None = None
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    error_type: str | None = None
    error_class: str | None = None
    error: str | None = None
    capability: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    ws_event_counts: dict[str, int] = field(default_factory=dict)
    ws_active_sentence_index: int | None = None
    ws_active_sample_rate: int | None = None
    ws_active_sentence_bytes: int = 0
    ws_completed_sentences: int = 0
    ws_close_reason: str | None = None
    was_cancelled: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def duration_from_audio_bytes(
    data: bytes,
    *,
    content_type: str | None = None,
    response_format: str | None = None,
    sample_rate: int = PCM_SAMPLE_RATE,
) -> float:
    fmt = (response_format or "").lower()
    ctype = (content_type or "").lower()
    if len(data) > WAV_HEADER_BYTES and data[:4] == WAV_RIFF_MARKER:
        return _wav_duration(data)
    if fmt == "pcm" or "audio/pcm" in ctype or "audio/raw" in ctype:
        return len(data) / float(sample_rate * PCM_CHANNELS * PCM_SAMPLE_WIDTH)
    return 0.0


def _wav_duration(data: bytes) -> float:
    if len(data) < 12 or data[:4] != WAV_RIFF_MARKER or data[8:12] != WAV_WAVE_MARKER:
        return 0.0
    sample_rate = 0
    channels = 0
    bits = 0
    data_size = 0
    pos = 12
    while pos + WAV_CHUNK_HEADER_BYTES <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        chunk_start = pos + WAV_CHUNK_HEADER_BYTES
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            break
        if chunk_id == b"fmt " and chunk_size >= 16:
            try:
                channels = struct.unpack_from("<H", data, chunk_start + 2)[0]
                sample_rate = struct.unpack_from("<I", data, chunk_start + 4)[0]
                bits = struct.unpack_from("<H", data, chunk_start + 14)[0]
            except struct.error:
                return 0.0
        elif chunk_id == b"data":
            data_size = chunk_size
            break
        pos = chunk_end + (chunk_size % 2)
    if sample_rate <= 0 or channels <= 0 or bits <= 0 or data_size <= 0:
        return 0.0
    return data_size / float(sample_rate * channels * bits // 8)


def parse_sse_audio_event(line: str) -> tuple[bytes | None, dict[str, Any] | None]:
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None, None
    event = json.loads(line[len(SSE_DATA_PREFIX) :])
    if not isinstance(event, dict):
        raise ValueError("SSE data payload must be a JSON object")
    audio = event.get("audio")
    if audio is None:
        return None, event
    if not isinstance(audio, dict):
        raise ValueError("SSE audio field must be a JSON object")
    audio_data = audio.get("data")
    if not audio_data:
        return None, event
    if not isinstance(audio_data, str):
        raise ValueError("SSE audio.data field must be a base64 string")
    return base64.b64decode(audio_data, validate=True), event


def finish_timing(result: ScenarioResult, start: float) -> None:
    if result.completed_s is None:
        now = time.perf_counter()
        result.latency_s = now - start
        result.completed_s = now
    if result.audio_duration_s > 0 and result.rtf <= 0:
        result.rtf = result.latency_s / result.audio_duration_s


def classify_http_status(status: int | None) -> str | None:
    if status is None:
        return None
    return f"{status // 100}xx"
