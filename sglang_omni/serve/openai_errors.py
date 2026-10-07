# SPDX-License-Identifier: Apache-2.0
"""Shared OpenAI-compatible API error classification helpers."""

from __future__ import annotations

import re

from sglang_omni.admission import QueueFullError

_BAD_REQUEST_MARKERS = (
    "Unsupported language:",
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
    "Request has no prompt tokens",
    "Request requires more tokens than the thinker KV cache can hold",
    "out-of-vocabulary token id",
    "accepts audio up to",
    "could not decode the uploaded audio",
    "use_audio_in_video requires every video in a multi-video request",
    "Embedded audio stream decoded no samples:",
    "Invalid media data while extracting embedded audio from",
    "Invalid media data while decoding video path=",
    "Qwen3-Omni requires all videos in a request to have the same sampled FPS",
    "max_new_tokens must be",
    "Fun-ASR Apple currently requires temperature=0",
    "exceeds the maximum allowed length",
    "sequence exceeds max_length",
    "multimodal_train_inputs",
    "disallowed special token",
    "stop strings are allowed",
    "stop_regex patterns are allowed",
    "AuK speech requires",
    "AuK gen_seconds must be",
    "AuK seed must be",
    "AuK requires a natural-language",
    "AuK accepts at most one",
    "AuK expected a",
    "AuK references must be",
    "PersonaPlex request needs",
    "PersonaPlex needs at least one 80 ms frame",
    "PersonaPlex seed must be",
    "PersonaPlex audio_temperature must be",
    "PersonaPlex audio_top_k must be",
    "PersonaPlex stop must be",
    "PersonaPlex stop_token_ids must be",
    "PersonaPlex sampling parameters must be",
    "PersonaPlex takes one caller recording",
    "Unsupported PersonaPlex audio input",
    "; packaged voices:",
)
_BAD_REQUEST_PATTERNS = (
    re.compile(
        r"\bAuK (?:nfe|cfg_strength|sway_sampling_coef|max_seconds) is a server-level setting"
    ),
    re.compile(r"^Could not decode .+ audio input$"),
    re.compile(r"^Request\s+\S+\s+exceeds the maximum number of tokens:"),
    re.compile(r"^Request\s+\S+\s+requires too many SWA KV tokens for"),
    re.compile(r"^Request .+ already exists$", re.DOTALL),
    re.compile(r"^stop_regex is \d+ bytes, over the \d+-byte limit"),
    re.compile(r"^Media URL returned HTTP 404: "),
    re.compile(
        r"^Qwen3-TTS (?:Base|CustomVoice|VoiceDesign) "
        r"(?:checkpoint does not support|does not accept) "
    ),
    re.compile(
        r"^Qwen3-TTS (?:Base|VoiceDesign) requires "
        r"(?:ref_audio|reference audio|non-empty ref_text|instructions)\b"
    ),
    re.compile(r"^Qwen3-TTS task_type must be one of "),
    re.compile(r"^Unsupported Qwen3-TTS CustomVoice speaker "),
)


def is_bad_request_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS) or any(
        pattern.search(message) is not None for pattern in _BAD_REQUEST_PATTERNS
    )


def generation_error_status_code(exc: BaseException) -> int:
    """Map a failure to 503 for a full queue, 400 for a caller error, else 500."""
    if QueueFullError.matches(exc):
        return 503
    elif is_bad_request_error(exc):
        return 400
    else:
        return 500
