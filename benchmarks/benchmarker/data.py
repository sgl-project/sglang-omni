# SPDX-License-Identifier: Apache-2.0
"""Shared data structures for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypedDict

RequestClass = Literal["non_streaming", "streaming"]


class FinishReason(str, Enum):
    """Why a generation ended; UNKNOWN when the server reported no reason."""

    UNKNOWN = "unknown"
    STOP = "stop"
    LENGTH = "length"


def finish_reason_from_server(value: str | None) -> FinishReason:
    try:
        return FinishReason(value)
    except ValueError:
        # note (Yucheng Hu): absent from servers without the field; other
        # values are engine states this metric does not classify.
        return FinishReason.UNKNOWN


@dataclass
class RequestResult:
    request_id: str = ""
    text: str = ""
    is_success: bool = False
    latency_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    tok_per_s: float = 0.0
    finish_reason: FinishReason = FinishReason.UNKNOWN
    speech_outcome_id: str = ""
    server_worker_id: str = ""
    wav_path: str = ""
    error: str = ""
    audio_ttfp_s: float | None = None
    inter_chunk_s: list[float] = field(default_factory=list)
    chunk_audio_duration_s: list[float] = field(default_factory=list)
    max_playback_underrun_s: float | None = None
    text_ttft_s: float | None = None
    audio_chunk_count: int = 0
    first_audio_payload_bytes: int = 0
    # note (luojiaxuan): the client-side slot cap was busy when this request
    # arrived, so its clock started late and it was not open loop.
    waited_for_slot: bool = False
    # note (luojiaxuan): open-loop runs only; how long after its planned
    # arrival the request was actually sent.
    dispatch_lateness_s: float | None = None
    # note (db-ol): fixed duration runs only. Both times are seconds from the start of the measured window, so settle requests have negative times.
    request_class: RequestClass | None = None
    sent_at_s: float | None = None
    finished_at_s: float | None = None


class MixtureWindow(TypedDict):
    """The mixture block of a fixed duration run's speed results."""

    load: str
    non_streaming_clients: int
    streaming_clients: int
    settle_s: float
    duration_s: float
    pool_size: int
    issued_requests: dict[RequestClass, int]
    pool_cycles: dict[RequestClass, float]
    outside_window_requests: dict[RequestClass, int]
    outside_window_failed_requests: dict[RequestClass, int]
    drain_s: float
