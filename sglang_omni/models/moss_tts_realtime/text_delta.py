# SPDX-License-Identifier: Apache-2.0
"""BPE-safe text-delta tokenization for MOSS-TTS-Realtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Any

DEFAULT_MOSS_TTS_REALTIME_TOKEN_HOLDBACK = 3

# One MOSS-TTS-Realtime tokenizer is loaded per server process.
_TOKENIZER_VOCAB_SIZE: int | None = None


def moss_tts_realtime_tokenizer_size(tokenizer: Any) -> int | None:
    """Return the full tokenizer size, including added tokens when available."""

    try:
        size = len(tokenizer)
    except (AttributeError, TypeError, NotImplementedError):
        size = getattr(tokenizer, "vocab_size", None)
    if isinstance(size, bool) or not isinstance(size, Integral) or int(size) < 1:
        return None
    return int(size)


def initialize_moss_tts_realtime_tokenizer_vocab_size(tokenizer: Any) -> int:
    """Record the process-wide tokenizer size once during service startup."""

    global _TOKENIZER_VOCAB_SIZE
    if _TOKENIZER_VOCAB_SIZE is None:
        vocab_size = moss_tts_realtime_tokenizer_size(tokenizer)
        if vocab_size is None:
            raise ValueError("tokenizer must expose a positive vocabulary size")
        _TOKENIZER_VOCAB_SIZE = vocab_size
    return _TOKENIZER_VOCAB_SIZE


def get_moss_tts_realtime_tokenizer_vocab_size() -> int:
    """Return the vocabulary size initialized for this server process."""

    if _TOKENIZER_VOCAB_SIZE is None:
        raise RuntimeError("MOSS-TTS-Realtime tokenizer size is not initialized")
    return _TOKENIZER_VOCAB_SIZE


def validate_moss_tts_realtime_text_token_ids(
    token_ids: Sequence[Any],
    *,
    allow_empty: bool = False,
    name: str = "token_ids",
) -> tuple[int, ...]:
    """Validate public text-token IDs without decoding or retokenizing them."""

    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(
        token_ids, Sequence
    ):
        raise TypeError(f"{name} must be a sequence of integers")
    normalized = tuple(token_ids)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty")

    vocab_size = get_moss_tts_realtime_tokenizer_vocab_size()
    for index, token_id in enumerate(normalized):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{name}[{index}] must be an integer")
        if token_id < 0:
            raise ValueError(f"{name}[{index}] must be non-negative")
        if token_id >= vocab_size:
            raise ValueError(
                f"{name}[{index}]={token_id} exceeds tokenizer size {vocab_size}"
            )
    return normalized


@dataclass(frozen=True, slots=True)
class MossTTSRealtimeTextDeltaResult:
    """Newly stable IDs and the raw bytes assigned to this token update."""

    token_ids: tuple[int, ...] = ()
    byte_count: int = 0


@dataclass(frozen=True, slots=True)
class MossTTSRealtimeTextDeltaSnapshot:
    """Restorable state used when downstream control admission fails."""

    text: str
    all_ids: tuple[int, ...]
    emitted_ids: tuple[int, ...]
    pending_bytes: int
    finished: bool


class MossTTSRealtimeTextDeltaTokenizer:
    """Convert text deltas into a stable token-id stream.

    The full accumulated text is re-encoded after every delta. The last few
    IDs remain un-emitted because appending text can change a BPE boundary.
    Already-emitted IDs are verified on every update because they cannot be
    retracted from an active autoregressive request.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        hold_back: int = DEFAULT_MOSS_TTS_REALTIME_TOKEN_HOLDBACK,
        max_text_bytes: int,
        max_token_ids: int,
    ) -> None:
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must implement encode()")
        if isinstance(hold_back, bool) or not isinstance(hold_back, int):
            raise TypeError("hold_back must be an integer")
        if hold_back < 0:
            raise ValueError("hold_back must be non-negative")
        for name, value in (
            ("max_text_bytes", max_text_bytes),
            ("max_token_ids", max_token_ids),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")

        self.tokenizer = tokenizer
        initialize_moss_tts_realtime_tokenizer_vocab_size(tokenizer)
        self.hold_back = hold_back
        self.max_text_bytes = max_text_bytes
        self.max_token_ids = max_token_ids
        self._text = ""
        self._all_ids: tuple[int, ...] = ()
        self._emitted_ids: tuple[int, ...] = ()
        self._pending_bytes = 0
        self._finished = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self._all_ids

    @property
    def emitted_token_ids(self) -> tuple[int, ...]:
        return self._emitted_ids

    @property
    def total_text_bytes(self) -> int:
        return len(self._text.encode("utf-8"))

    @property
    def finished(self) -> bool:
        return self._finished

    def snapshot(self) -> MossTTSRealtimeTextDeltaSnapshot:
        return MossTTSRealtimeTextDeltaSnapshot(
            text=self._text,
            all_ids=self._all_ids,
            emitted_ids=self._emitted_ids,
            pending_bytes=self._pending_bytes,
            finished=self._finished,
        )

    def restore(self, snapshot: MossTTSRealtimeTextDeltaSnapshot) -> None:
        if not isinstance(snapshot, MossTTSRealtimeTextDeltaSnapshot):
            raise TypeError("snapshot must be MossTTSRealtimeTextDeltaSnapshot")
        self._text = snapshot.text
        self._all_ids = snapshot.all_ids
        self._emitted_ids = snapshot.emitted_ids
        self._pending_bytes = snapshot.pending_bytes
        self._finished = snapshot.finished

    def push_delta(self, delta: str) -> MossTTSRealtimeTextDeltaResult:
        """Append one non-empty delta and return newly stable token IDs."""

        if self._finished:
            raise RuntimeError("text delta tokenizer is already flushed")
        if not isinstance(delta, str):
            raise TypeError("text delta must be a string")
        if not delta:
            raise ValueError("text delta must not be empty")

        encoded_delta = delta.encode("utf-8")
        text = self._text + delta
        if len(text.encode("utf-8")) > self.max_text_bytes:
            raise ValueError(
                "realtime text byte limit exceeded: "
                f"{len(text.encode('utf-8'))} > {self.max_text_bytes}"
            )
        all_ids = self._encode(text)
        self._verify_emitted_prefix(all_ids)

        stable_count = max(len(self._emitted_ids), len(all_ids) - self.hold_back)
        new_ids = all_ids[len(self._emitted_ids) : stable_count]
        pending_bytes = self._pending_bytes + len(encoded_delta)

        self._text = text
        self._all_ids = all_ids
        self._emitted_ids += new_ids
        if new_ids:
            self._pending_bytes = 0
            return MossTTSRealtimeTextDeltaResult(new_ids, pending_bytes)
        self._pending_bytes = pending_bytes
        return MossTTSRealtimeTextDeltaResult()

    def flush(self) -> MossTTSRealtimeTextDeltaResult:
        """Emit the complete remaining tail and close this turn tokenizer."""

        if self._finished:
            return MossTTSRealtimeTextDeltaResult()
        all_ids = self._encode(self._text)
        self._verify_emitted_prefix(all_ids)
        new_ids = all_ids[len(self._emitted_ids) :]
        byte_count = self._pending_bytes if new_ids else 0

        self._all_ids = all_ids
        self._emitted_ids += new_ids
        self._pending_bytes = 0
        self._finished = True
        return MossTTSRealtimeTextDeltaResult(new_ids, byte_count)

    def _encode(self, text: str) -> tuple[int, ...]:
        raw_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = validate_moss_tts_realtime_text_token_ids(
            raw_ids,
            allow_empty=True,
            name="encoded token_ids",
        )
        if len(token_ids) > self.max_token_ids:
            raise ValueError(
                "realtime text token limit exceeded: "
                f"{len(token_ids)} > {self.max_token_ids}"
            )
        return token_ids

    def _verify_emitted_prefix(self, all_ids: tuple[int, ...]) -> None:
        emitted_count = len(self._emitted_ids)
        if len(all_ids) < emitted_count or all_ids[:emitted_count] != self._emitted_ids:
            raise RuntimeError(
                "tokenizer changed an already-emitted prefix; increase holdback or "
                "use input.tokens"
            )


@lru_cache(maxsize=4)
def load_moss_tts_realtime_text_tokenizer(model_path: str) -> Any:
    """Load and share the model text tokenizer in the API process."""

    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a non-empty string")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
