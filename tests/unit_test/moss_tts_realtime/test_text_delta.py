# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import pytest

from sglang_omni.models.moss_tts_realtime import text_delta
from sglang_omni.models.moss_tts_realtime.text_delta import (
    MossTTSRealtimeTextDeltaTokenizer,
    validate_moss_tts_realtime_text_token_ids,
)


class BoundaryTokenizer:
    """Greedy tokenizer whose final character can merge with future text."""

    vocab_size = 10000

    def __init__(self) -> None:
        self.len_calls = 0

    def __len__(self) -> int:
        self.len_calls += 1
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids: list[int] = []
        index = 0
        while index < len(text):
            pair = text[index : index + 2]
            if len(pair) == 2 and pair in {"ab", "世界", "!?", "，。"}:
                ids.append(5000 + sum(ord(char) for char in pair) % 4000)
                index += 2
                continue
            ids.append(1 + ord(text[index]) % 4999)
            index += 1
        return ids


@pytest.fixture(autouse=True)
def _clear_tokenizer_vocab_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_delta, "_TOKENIZER_VOCAB_SIZE", None)


def _stream_ids(text: str, chunks: list[str]) -> tuple[int, ...]:
    tokenizer = BoundaryTokenizer()
    delta = MossTTSRealtimeTextDeltaTokenizer(
        tokenizer,
        max_text_bytes=4096,
        max_token_ids=4096,
    )
    emitted: list[int] = []
    for chunk in chunks:
        emitted.extend(delta.push_delta(chunk).token_ids)
    emitted.extend(delta.flush().token_ids)
    assert delta.text == text
    return tuple(emitted)


def test_arbitrary_chunkings_match_one_shot_tokenization() -> None:
    tokenizer = BoundaryTokenizer()
    corpus = "ab你好世界，。 punctuation!? café"
    expected = tuple(tokenizer.encode(corpus, add_special_tokens=False))

    boundaries = range(1, len(corpus))
    chunkings = [[corpus]]
    chunkings.extend([corpus[:split], corpus[split:]] for split in boundaries)
    chunkings.extend(
        [corpus[:first], corpus[first:second], corpus[second:]]
        for first, second in itertools.combinations(boundaries, 2)
    )

    for chunks in chunkings:
        assert _stream_ids(corpus, chunks) == expected


def test_last_three_ids_are_held_until_flush() -> None:
    tokenizer = BoundaryTokenizer()
    delta = MossTTSRealtimeTextDeltaTokenizer(
        tokenizer,
        hold_back=3,
        max_text_bytes=100,
        max_token_ids=100,
    )

    first = delta.push_delta("abcdef")
    expected = tuple(tokenizer.encode("abcdef", add_special_tokens=False))

    assert first.token_ids == expected[:-3]
    assert first.byte_count == len("abcdef".encode())
    assert delta.emitted_token_ids == expected[:-3]
    assert delta.flush().token_ids == expected[-3:]
    assert delta.finished is True


def test_bytes_wait_for_the_next_stable_token_update() -> None:
    delta = MossTTSRealtimeTextDeltaTokenizer(
        BoundaryTokenizer(),
        hold_back=3,
        max_text_bytes=100,
        max_token_ids=100,
    )

    assert delta.push_delta("a").byte_count == 0
    assert delta.push_delta("b").byte_count == 0
    result = delta.push_delta("cdef")

    assert result.token_ids
    assert result.byte_count == len("abcdef".encode())


def test_delta_tokenizer_reuses_cached_vocab_size() -> None:
    tokenizer = BoundaryTokenizer()
    delta = MossTTSRealtimeTextDeltaTokenizer(
        tokenizer,
        max_text_bytes=100,
        max_token_ids=100,
    )

    delta.push_delta("ab")
    delta.push_delta("cd")
    delta.flush()

    assert tokenizer.len_calls == 1


def test_snapshot_restores_tokenizer_state_after_failed_admission() -> None:
    delta = MossTTSRealtimeTextDeltaTokenizer(
        BoundaryTokenizer(),
        max_text_bytes=100,
        max_token_ids=100,
    )
    snapshot = delta.snapshot()

    delta.push_delta("temporary")
    delta.restore(snapshot)

    assert delta.text == ""
    assert delta.token_ids == ()
    assert delta.push_delta("actual").token_ids


def test_prefix_drift_outside_holdback_is_rejected() -> None:
    class UnstableTokenizer(BoundaryTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if len(text) >= 8 and ids:
                ids[0] += 1
            return ids

    delta = MossTTSRealtimeTextDeltaTokenizer(
        UnstableTokenizer(),
        hold_back=3,
        max_text_bytes=100,
        max_token_ids=100,
    )
    delta.push_delta("abcdefg")

    with pytest.raises(RuntimeError, match="already-emitted prefix"):
        delta.push_delta("h")


def test_delta_limits_are_transactional() -> None:
    delta = MossTTSRealtimeTextDeltaTokenizer(
        BoundaryTokenizer(),
        max_text_bytes=4,
        max_token_ids=100,
    )
    delta.push_delta("ab")
    snapshot = delta.snapshot()

    with pytest.raises(ValueError, match="byte limit"):
        delta.push_delta("世界")

    assert delta.snapshot() == snapshot


@pytest.mark.parametrize(
    "token_ids, error",
    [
        ([], "must not be empty"),
        ([True], "must be an integer"),
        ([-1], "must be non-negative"),
        ([10000], "exceeds tokenizer size"),
    ],
)
def test_direct_token_validation_is_strict(token_ids: list[int], error: str) -> None:
    text_delta.initialize_moss_tts_realtime_tokenizer_vocab_size(BoundaryTokenizer())
    with pytest.raises((TypeError, ValueError), match=error):
        validate_moss_tts_realtime_text_token_ids(token_ids)


def test_direct_token_validation_preserves_ids_exactly() -> None:
    token_ids = [1, 9999, 42]
    text_delta.initialize_moss_tts_realtime_tokenizer_vocab_size(BoundaryTokenizer())

    assert validate_moss_tts_realtime_text_token_ids(token_ids) == tuple(token_ids)
