from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamSignal
from sglang_omni.proto import StagePayload

TokenCountFn = Callable[[str], int]


_SEGMENT_END_PUNCTUATION = (".", "!", "?", "。", "！", "？", "；", ";")


@dataclass
class CompletedResult:
    request_id: str
    payload: StagePayload | None = None
    error: BaseException | None = None


def is_done_signal(item: Any) -> bool:
    if item is None:
        return True
    return isinstance(item, StreamSignal) and item.is_done and item.error is None


def text_to_uint8_tensor(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def uint8_tensor_to_text(tensor: torch.Tensor) -> str:
    if tensor.dtype != torch.uint8:
        raise TypeError("uint8_tensor_to_text expects a torch.uint8 tensor")
    values = tensor.detach().cpu().flatten().tolist()
    return bytes(values).decode("utf-8", errors="ignore")


def split_whitespace_tokens(text: str, max_tokens: int) -> tuple[str, str]:
    parts = text.split()
    if len(parts) <= max_tokens:
        return text, ""
    return " ".join(parts[:max_tokens]), " ".join(parts[max_tokens:])


@dataclass(frozen=True)
class SegmenterConfig:
    segment_min_tokens: int = 8
    segment_max_tokens: int = 40
    first_segment_min_tokens: int = 4
    first_segment_max_wait_ms: int = 450

    def __post_init__(self) -> None:
        if self.segment_min_tokens <= 0:
            raise ValueError("segment_min_tokens must be positive")
        if self.segment_max_tokens <= 0:
            raise ValueError("segment_max_tokens must be positive")
        if self.first_segment_min_tokens <= 0:
            raise ValueError("first_segment_min_tokens must be positive")
        if self.segment_min_tokens > self.segment_max_tokens:
            raise ValueError("segment_min_tokens must be <= segment_max_tokens")
        if self.first_segment_min_tokens > self.segment_max_tokens:
            raise ValueError("first_segment_min_tokens must be <= segment_max_tokens")
        if self.first_segment_max_wait_ms < 0:
            raise ValueError("first_segment_max_wait_ms must be non-negative")


@dataclass(frozen=True)
class TextSegment:
    segment_id: int
    text: str
    is_final_segment: bool = False


class SegmenterState:
    def __init__(
        self,
        config: SegmenterConfig,
        token_count_fn: TokenCountFn,
    ) -> None:
        self.config = config
        self.token_count_fn = token_count_fn
        self._buffer = ""
        self._segment_id = 0
        self._first_text_ms: int | None = None

    def push(self, text: str, *, now_ms: int) -> list[TextSegment]:
        if text and self._first_text_ms is None:
            self._first_text_ms = now_ms
        self._buffer += text
        if not self._buffer:
            return []

        tokens = self.token_count_fn(self._buffer)
        if tokens >= self.config.segment_max_tokens:
            return [self._emit_max_window(now_ms=now_ms)]

        first_timeout_ready = (
            self._segment_id == 0
            and self._first_text_ms is not None
            and tokens >= self.config.first_segment_min_tokens
            and now_ms - self._first_text_ms >= self.config.first_segment_max_wait_ms
        )
        should_emit = (
            tokens >= self.config.segment_min_tokens
            and self._has_segment_end_punctuation()
        ) or first_timeout_ready
        if not should_emit:
            return []

        return [self._emit(is_final_segment=False)]

    def buffer_token_count(self) -> int:
        return self.token_count_fn(self._buffer) if self._buffer else 0

    def flush(self) -> list[TextSegment]:
        if not self._buffer:
            return []
        return [self._emit(is_final_segment=True)]

    def _has_segment_end_punctuation(self) -> bool:
        return self._buffer.rstrip().endswith(_SEGMENT_END_PUNCTUATION)

    def _emit_max_window(self, *, now_ms: int) -> TextSegment:
        text, remainder = split_whitespace_tokens(
            self._buffer, self.config.segment_max_tokens
        )
        return self._emit_text(
            text=text,
            remainder=remainder,
            is_final_segment=False,
            remainder_start_ms=now_ms if remainder else None,
        )

    def _emit(self, *, is_final_segment: bool) -> TextSegment:
        return self._emit_text(
            text=self._buffer,
            remainder="",
            is_final_segment=is_final_segment,
            remainder_start_ms=None,
        )

    def _emit_text(
        self,
        *,
        text: str,
        remainder: str,
        is_final_segment: bool,
        remainder_start_ms: int | None,
    ) -> TextSegment:
        segment = TextSegment(
            segment_id=self._segment_id,
            text=text,
            is_final_segment=is_final_segment,
        )
        self._segment_id += 1
        self._buffer = remainder
        self._first_text_ms = remainder_start_ms
        return segment
