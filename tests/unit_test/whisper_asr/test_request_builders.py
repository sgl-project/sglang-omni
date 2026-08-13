# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.whisper_asr import request_builders as whisper_request_builders
from sglang_omni.models.whisper_asr.request_builders import (
    _build_prev_context_tokens,
    _decoder_token_budgets,
    make_whisper_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload

_SOT_PREV = 50361
_PREFIX = [50258, 50259, 50359, 50363]
_ENCODER_TOKEN_COUNT = 4


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 3
    vocab_size = 51865

    def set_prefix_tokens(
        self, *, language: str, task: str, predict_timestamps: bool
    ) -> None:
        assert predict_timestamps is False
        self.prefix_language = language
        self.prefix_task = task

    @property
    def prefix_tokens(self) -> list[int]:
        return list(_PREFIX)

    def get_prompt_ids(self, text: str, return_tensors=None) -> list[int]:
        assert return_tensors is None
        # note (jiannan-17): One token per character keeps truncation arithmetic transparent.
        return [_SOT_PREV] + [1000 + i for i in range(len(text))]


def _make_request_builder(tokenizer: _FakeTokenizer | None = None):
    fake_processor = SimpleNamespace(
        feature_extractor=lambda audio, *, sampling_rate, return_tensors: (
            SimpleNamespace(input_features=torch.zeros((1, 128, 3000)))
        ),
    )
    request_builder, _ = make_whisper_scheduler_adapters(
        processor=fake_processor,
        tokenizer=tokenizer if tokenizer is not None else _FakeTokenizer(),
        generation_config=SimpleNamespace(suppress_tokens=None),
        encoder_token_count=_ENCODER_TOKEN_COUNT,
        max_new_tokens=32,
    )
    return request_builder


def _make_payload(
    params: dict | None = None, *, request_id: str = "req-whisper"
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params or {}),
        data={},
    )


def _build(monkeypatch, params: dict | None = None):
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    return _make_request_builder()(_make_payload(params))


def test_request_builder_rejects_audio_past_the_mel_window(monkeypatch) -> None:
    # Anything past the 30s window is rejected. Without the guard the feature
    # extractor would silently drop everything past 30 seconds instead.
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(int(30.1 * 16000), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="accepts audio up to"):
        _make_request_builder()(_make_payload())


def test_request_builder_accepts_a_full_window_chunk(monkeypatch) -> None:
    # Exactly 30.0s must build: the serve-layer chunker cuts spans of up to
    # exactly max_audio_clip_s samples, so the guard has to be a strict
    # greater-than or every full-length chunk of a long upload would 400.
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(30 * 16000, dtype=np.float32),
    )

    data = _make_request_builder()(_make_payload())

    assert data.audio_duration_s == pytest.approx(30.0)


def test_request_builder_without_prompt_keeps_prefix_only(monkeypatch) -> None:
    data = _build(monkeypatch)

    pad_id = _FakeTokenizer.pad_token_id
    expected = [pad_id] * _ENCODER_TOKEN_COUNT + _PREFIX
    assert data.input_ids.tolist() == expected
    assert data.prompt_token_ids == _PREFIX


def test_request_builder_maps_prompt_to_prev_context(monkeypatch) -> None:
    data = _build(monkeypatch, {"prompt": "hello"})

    pad_id = _FakeTokenizer.pad_token_id
    prev_context = [_SOT_PREV, 1000, 1001, 1002, 1003, 1004]
    expected = [pad_id] * _ENCODER_TOKEN_COUNT + prev_context + _PREFIX
    assert data.input_ids.tolist() == expected
    assert data.prompt_token_ids == prev_context + _PREFIX
    assert list(data.req.origin_input_ids) == expected


def test_request_builder_truncates_prompt_to_tail(monkeypatch) -> None:
    data = _build(monkeypatch, {"prompt": "x" * 300})

    prev_context = data.prompt_token_ids[: -len(_PREFIX)]
    assert len(prev_context) == 224
    assert prev_context[0] == _SOT_PREV
    # note (jiannan-17): The last 223 of the 300 prompt tokens survive (ids 1077..1299).
    assert prev_context[1:] == [1000 + i for i in range(77, 300)]
    assert data.prompt_token_ids[-len(_PREFIX) :] == _PREFIX


def test_request_builder_ignores_blank_prompt(monkeypatch) -> None:
    data = _build(monkeypatch, {"prompt": "   "})

    assert data.prompt_token_ids == _PREFIX


def test_request_builder_shrinks_prompt_budget_for_large_max_new_tokens(
    monkeypatch,
) -> None:
    """prev context + prefix + generated tokens must fit the 448-position table."""
    data = _build(monkeypatch, {"prompt": "x" * 300, "max_new_tokens": 300})

    prev_context = data.prompt_token_ids[: -len(_PREFIX)]
    # note (jiannan-17): 448 total - 4 prefix - 300 generated = 144 previous-context positions.
    assert len(prev_context) == 144
    assert prev_context[0] == _SOT_PREV
    assert prev_context[1:] == [1000 + i for i in range(157, 300)]
    assert len(prev_context) + len(_PREFIX) + 300 <= 448


def test_request_builder_drops_prompt_when_generation_fills_context(
    monkeypatch,
) -> None:
    data = _build(monkeypatch, {"prompt": "hello", "max_new_tokens": 500})

    assert data.prompt_token_ids == _PREFIX
    # note (jiannan-17): The four-token prefix leaves 444 decoder positions for generation.
    assert data.max_new_tokens == 448 - len(_PREFIX)


def test_request_builder_clamps_max_new_tokens_without_prompt(monkeypatch) -> None:
    """A promptless max_new_tokens=448 request must not outgrow the table."""
    data = _build(monkeypatch, {"max_new_tokens": 448})

    assert data.prompt_token_ids == _PREFIX
    assert data.max_new_tokens == 448 - len(_PREFIX)
    assert len(data.prompt_token_ids) + data.max_new_tokens <= 448


def test_decoder_budget_invariant_exhaustive() -> None:
    """prev block + prefix + max_new never exceed the 448-position table.

    Exhausts the full request space (max_new_tokens 1..600 x prompt lengths
    0..300) against the pure budget arithmetic, without paying for feature
    extraction per combination.
    """
    tokenizer = _FakeTokenizer()
    prefix_len = len(_PREFIX)
    for requested_max_new in range(1, 601):
        max_new, max_prev = _decoder_token_budgets(
            decoder_context_len=448,
            prefix_len=prefix_len,
            requested_max_new_tokens=requested_max_new,
        )
        assert 1 <= max_new <= 448 - prefix_len
        for prompt_len in range(0, 301):
            prev_block = _build_prev_context_tokens(
                tokenizer,
                "x" * prompt_len if prompt_len else None,
                max_prev_tokens=max_prev,
            )
            budget = len(prev_block) + prefix_len + max_new
            assert budget <= 448, (
                f"decoder budget {budget} > 448 for "
                f"max_new={requested_max_new} prompt_len={prompt_len}"
            )


def test_request_builder_guard_trips_on_over_budget_prev_context(
    monkeypatch,
) -> None:
    """If the budget arithmetic ever regresses, the builder must fail before
    an out-of-range decoder position can reach the GPU."""
    monkeypatch.setattr(
        whisper_request_builders,
        "_build_prev_context_tokens",
        lambda tokenizer, prompt, *, max_prev_tokens: [_SOT_PREV] + [1] * 500,
    )

    with pytest.raises(ValueError, match="decoder budget exceeded"):
        _build(monkeypatch, {"prompt": "hello"})


def test_concurrent_request_builds_serialize_mutable_tokenizer_state(
    monkeypatch,
) -> None:
    class _ConcurrentTokenizer(_FakeTokenizer):
        def __init__(self) -> None:
            self._state_lock = threading.Lock()
            self.active_calls = 0
            self.max_active_calls = 0
            self.prefix_language = ""

        def set_prefix_tokens(
            self, *, language: str, task: str, predict_timestamps: bool
        ) -> None:
            with self._state_lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            time.sleep(0.02)
            super().set_prefix_tokens(
                language=language,
                task=task,
                predict_timestamps=predict_timestamps,
            )
            with self._state_lock:
                self.active_calls -= 1

        @property
        def prefix_tokens(self) -> list[int]:
            return [1 if self.prefix_language == "english" else 2]

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    tokenizer = _ConcurrentTokenizer()
    request_builder = _make_request_builder(tokenizer)
    start = threading.Barrier(2)

    def _build_concurrently(language: str):
        start.wait()
        return request_builder(
            _make_payload({"language": language}, request_id=f"req-{language}")
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_build_concurrently, language) for language in ("en", "fr")
        ]
        results = [future.result() for future in futures]

    assert tokenizer.max_active_calls == 1
    assert {result.language for result in results} == {"english", "fr"}
    assert {result.language: result.prompt_token_ids for result in results} == {
        "english": [1],
        "fr": [2],
    }
