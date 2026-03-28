# SPDX-License-Identifier: Apache-2.0
"""Shared data structures for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass


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
    wav_path: str = ""
    error: str = ""
