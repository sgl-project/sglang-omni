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
    build_prev_context_tokens,
    decoder_token_budgets,
    make_whisper_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload

SOT_PREV = 50361
PREFIX = [50258, 50259, 50359, 50363]
ENCODER_TOKEN_COUNT = 4


def generation_config() -> SimpleNamespace:
    return SimpleNamespace(
        suppress_tokens=None,
        max_length=None,
        no_timestamps_token_id=50363,
        max_initial_timestamp_index=50,
    )


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 3
    vocab_size = 51865

    def __len__(self) -> int:
        return 51866

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|startoftranscript|>": 50258}[token]

    def set_prefix_tokens(
        self, *, language: str, task: str, predict_timestamps: bool
    ) -> None:
        self.prefix_language = language
        self.prefix_task = task
        self.predict_timestamps = predict_timestamps

    @property
    def prefix_tokens(self) -> list[int]:
        prefix = list(PREFIX)
        if getattr(self, "predict_timestamps", False):
            prefix = prefix[:-1]
        return prefix

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in ids if token_id < 50000)

    def get_prompt_ids(self, text: str, return_tensors=None) -> list[int]:
        assert return_tensors is None
        # note (jiannan-17): One token per character keeps truncation arithmetic transparent.
        return [SOT_PREV] + [1000 + i for i in range(len(text))]


def make_request_builder(tokenizer: FakeTokenizer | None = None):
    fake_processor = SimpleNamespace(
        feature_extractor=lambda audio, *, sampling_rate, return_tensors: (
            SimpleNamespace(input_features=torch.zeros((1, 128, 3000)))
        ),
    )
    request_builder, _ = make_whisper_scheduler_adapters(
        processor=fake_processor,
        tokenizer=tokenizer if tokenizer is not None else FakeTokenizer(),
        generation_config=generation_config(),
        encoder_token_count=ENCODER_TOKEN_COUNT,
        max_new_tokens=32,
    )
    return request_builder


def make_payload(
    params: dict | None = None, *, request_id: str = "req-whisper"
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params or {}),
        data={},
    )


def build(monkeypatch, params: dict | None = None):
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    return make_request_builder()(make_payload(params))


def test_request_builder_rejects_audio_past_the_mel_window(monkeypatch) -> None:
    # Anything past the 30s window is rejected. Without the guard the feature
    # extractor would silently drop everything past 30 seconds instead.
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(int(30.1 * 16000), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="accepts audio up to"):
        make_request_builder()(make_payload())


def test_request_builder_accepts_a_full_window_chunk(monkeypatch) -> None:
    # Exactly 30.0s must build: the serve-layer chunker cuts spans of up to
    # exactly max_audio_clip_s samples, so the guard has to be a strict
    # greater-than or every full-length chunk of a long upload would 400.
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(30 * 16000, dtype=np.float32),
    )

    data = make_request_builder()(make_payload())

    assert data.audio_duration_s == pytest.approx(30.0)


def test_request_builder_without_prompt_keeps_prefix_only(monkeypatch) -> None:
    data = build(monkeypatch)

    pad_id = FakeTokenizer.pad_token_id
    expected = [pad_id] * ENCODER_TOKEN_COUNT + PREFIX
    assert data.input_ids.tolist() == expected
    assert data.prompt_token_ids == PREFIX


def test_request_builder_maps_prompt_to_prev_context(monkeypatch) -> None:
    data = build(monkeypatch, {"prompt": "hello"})

    pad_id = FakeTokenizer.pad_token_id
    prev_context = [SOT_PREV, 1000, 1001, 1002, 1003, 1004]
    expected = [pad_id] * ENCODER_TOKEN_COUNT + prev_context + PREFIX
    assert data.input_ids.tolist() == expected
    assert data.prompt_token_ids == prev_context + PREFIX
    assert list(data.req.origin_input_ids) == expected


def test_request_builder_truncates_prompt_to_tail(monkeypatch) -> None:
    data = build(monkeypatch, {"prompt": "x" * 300})

    prev_context = data.prompt_token_ids[: -len(PREFIX)]
    assert len(prev_context) == 224
    assert prev_context[0] == SOT_PREV
    # note (jiannan-17): The last 223 of the 300 prompt tokens survive (ids 1077..1299).
    assert prev_context[1:] == [1000 + i for i in range(77, 300)]
    assert data.prompt_token_ids[-len(PREFIX) :] == PREFIX


def test_request_builder_ignores_blank_prompt(monkeypatch) -> None:
    data = build(monkeypatch, {"prompt": "   "})

    assert data.prompt_token_ids == PREFIX


def test_request_builder_shrinks_prompt_budget_for_large_max_new_tokens(
    monkeypatch,
) -> None:
    """prev context + prefix + generated tokens must fit the 448-position table."""
    data = build(monkeypatch, {"prompt": "x" * 300, "max_new_tokens": 300})

    prev_context = data.prompt_token_ids[: -len(PREFIX)]
    # note (jiannan-17): 448 total - 4 prefix - 300 generated = 144 previous-context positions.
    assert len(prev_context) == 144
    assert prev_context[0] == SOT_PREV
    assert prev_context[1:] == [1000 + i for i in range(157, 300)]
    assert len(prev_context) + len(PREFIX) + 300 <= 448


def test_request_builder_drops_prompt_when_generation_fills_context(
    monkeypatch,
) -> None:
    data = build(monkeypatch, {"prompt": "hello", "max_new_tokens": 500})

    assert data.prompt_token_ids == PREFIX
    # note (jiannan-17): The four-token prefix leaves 444 decoder positions for generation.
    assert data.max_new_tokens == 448 - len(PREFIX)


def test_request_builder_clamps_max_new_tokens_without_prompt(monkeypatch) -> None:
    """A promptless max_new_tokens=448 request must not outgrow the table."""
    data = build(monkeypatch, {"max_new_tokens": 448})

    assert data.prompt_token_ids == PREFIX
    assert data.max_new_tokens == 448 - len(PREFIX)
    assert len(data.prompt_token_ids) + data.max_new_tokens <= 448


def test_decoder_budget_invariant_exhaustive() -> None:
    """prev block + prefix + max_new never exceed the 448-position table.

    Exhausts the full request space (max_new_tokens 1..600 x prompt lengths
    0..300) against the pure budget arithmetic, without paying for feature
    extraction per combination.
    """
    tokenizer = FakeTokenizer()
    prefix_len = len(PREFIX)
    for requested_max_new in range(1, 601):
        max_new, max_prev = decoder_token_budgets(
            decoder_context_len=448,
            prefix_len=prefix_len,
            requested_max_new_tokens=requested_max_new,
        )
        assert 1 <= max_new <= 448 - prefix_len
        for prompt_len in range(0, 301):
            prev_block = build_prev_context_tokens(
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
        "build_prev_context_tokens",
        lambda tokenizer, prompt, *, max_prev_tokens: [SOT_PREV] + [1] * 500,
    )

    with pytest.raises(ValueError, match="decoder budget exceeded"):
        build(monkeypatch, {"prompt": "hello"})


def test_concurrent_request_builds_serialize_mutable_tokenizer_state(
    monkeypatch,
) -> None:
    class ConcurrentTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            self.state_lock = threading.Lock()
            self.active_calls = 0
            self.max_active_calls = 0
            self.prefix_language = ""

        def set_prefix_tokens(
            self, *, language: str, task: str, predict_timestamps: bool
        ) -> None:
            with self.state_lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            time.sleep(0.02)
            super().set_prefix_tokens(
                language=language,
                task=task,
                predict_timestamps=predict_timestamps,
            )
            with self.state_lock:
                self.active_calls -= 1

        @property
        def prefix_tokens(self) -> list[int]:
            return [1 if self.prefix_language == "english" else 2]

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    tokenizer = ConcurrentTokenizer()
    request_builder = make_request_builder(tokenizer)
    start = threading.Barrier(2)

    def build_concurrently(language: str):
        start.wait()
        return request_builder(
            make_payload({"language": language}, request_id=f"req-{language}")
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(build_concurrently, language) for language in ("en", "fr")
        ]
        results = [future.result() for future in futures]

    assert tokenizer.max_active_calls == 1
    assert {result.language for result in results} == {"english", "fr"}
    assert {result.language: result.prompt_token_ids for result in results} == {
        "english": [1],
        "fr": [2],
    }


def test_request_builder_uses_generated_timestamps_for_segment_formats(
    monkeypatch,
) -> None:
    data = build(monkeypatch, {"segment_timestamps": True})

    assert data.prompt_token_ids == PREFIX[:-1]
    assert 50364 not in data.prompt_token_ids
    assert data.req.custom_logit_processor is not None
    custom_params = data.req.sampling_params.custom_params
    assert custom_params["segment_timestamps"] is True
    assert custom_params["timestamp_begin_id"] == 50364
    assert custom_params["no_timestamps_token_id"] == 50363
    assert custom_params["eos_token_id"] == FakeTokenizer.eos_token_id
    assert custom_params["max_initial_timestamp_index"] == 50


def test_request_builder_keeps_timestamp_off_prefix_by_default(monkeypatch) -> None:
    data = build(monkeypatch)

    assert data.prompt_token_ids == list(PREFIX)
    assert data.req.custom_logit_processor is None
    assert data.req.sampling_params.custom_params is None


def test_timestamped_text_renders_markers_from_token_ids() -> None:
    tokenizer = FakeTokenizer()

    text = whisper_request_builders.render_timestamped_text(
        tokenizer, [50364, 7, 8, 50414, 50439, 9, 50464], timestamp_begin_id=50364
    )

    assert text == "<|0.00|>7 8<|1.00|><|1.50|>9<|2.00|>"


def test_request_builder_pre_lm_encode_attaches_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    calls: list[object] = []

    class Service:
        def lookup_cached_embedding(self, fingerprint, expected_tokens):
            return None

        def encode_item(self, item):
            calls.append(item)
            item.precomputed_embeddings = torch.ones(
                (ENCODER_TOKEN_COUNT, 2), dtype=torch.float32
            )
            item.feature = None

        def attach_embedding(self, item, embedding):
            item.precomputed_embeddings = embedding
            item.feature = None

    fake_processor = SimpleNamespace(
        feature_extractor=lambda audio, *, sampling_rate, return_tensors: (
            SimpleNamespace(input_features=torch.zeros((1, 128, 3000)))
        ),
    )
    request_builder, _ = make_whisper_scheduler_adapters(
        processor=fake_processor,
        tokenizer=FakeTokenizer(),
        generation_config=generation_config(),
        encoder_token_count=ENCODER_TOKEN_COUNT,
        max_new_tokens=32,
        audio_encoder_service=Service(),
    )
    data = request_builder(make_payload())
    assert len(calls) == 1
    item = data.req.multimodal_inputs.mm_items[0]
    assert item.feature is None
    assert item.precomputed_embeddings is not None
    assert item.num_audio_tokens == ENCODER_TOKEN_COUNT


def test_request_builder_pre_lm_cache_miss_extracts_and_encodes(monkeypatch) -> None:
    mel_calls = {"n": 0}
    lookups: list[tuple[str | None, int]] = []

    def feature_extractor(audio, *, sampling_rate, return_tensors):
        mel_calls["n"] += 1
        return SimpleNamespace(input_features=torch.zeros((1, 128, 3000)))

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )

    class Service:
        def lookup_cached_embedding(self, fingerprint, expected_tokens):
            lookups.append((fingerprint, expected_tokens))
            return None

        def encode_item(self, item):
            assert item.feature is not None
            item.precomputed_embeddings = torch.ones(
                (ENCODER_TOKEN_COUNT, 2), dtype=torch.float32
            )
            item.feature = None

        def attach_embedding(self, item, embedding):
            raise AssertionError("cache miss must encode, not attach")

    fake_processor = SimpleNamespace(feature_extractor=feature_extractor)
    request_builder, _ = make_whisper_scheduler_adapters(
        processor=fake_processor,
        tokenizer=FakeTokenizer(),
        generation_config=generation_config(),
        encoder_token_count=ENCODER_TOKEN_COUNT,
        max_new_tokens=32,
        audio_encoder_service=Service(),
    )
    data = request_builder(make_payload())
    assert mel_calls["n"] == 1
    assert lookups and lookups[0][1] == ENCODER_TOKEN_COUNT
    assert data.req.extra_key == lookups[0][0]
    item = data.req.multimodal_inputs.mm_items[0]
    assert item.feature is None
    assert item.precomputed_embeddings is not None


def test_request_builder_pre_lm_cache_hit_skips_mel(monkeypatch) -> None:
    mel_calls = {"n": 0}

    def feature_extractor(audio, *, sampling_rate, return_tensors):
        mel_calls["n"] += 1
        return SimpleNamespace(input_features=torch.zeros((1, 128, 3000)))

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )

    class Service:
        def lookup_cached_embedding(self, fingerprint, expected_tokens):
            return torch.full((ENCODER_TOKEN_COUNT, 2), 7.0)

        def encode_item(self, item):
            raise AssertionError("cache hit must not encode")

        def attach_embedding(self, item, embedding):
            item.precomputed_embeddings = embedding
            item.feature = None

    fake_processor = SimpleNamespace(feature_extractor=feature_extractor)
    request_builder, _ = make_whisper_scheduler_adapters(
        processor=fake_processor,
        tokenizer=FakeTokenizer(),
        generation_config=generation_config(),
        encoder_token_count=ENCODER_TOKEN_COUNT,
        max_new_tokens=32,
        audio_encoder_service=Service(),
    )
    data = request_builder(make_payload())
    assert mel_calls["n"] == 0
    item = data.req.multimodal_inputs.mm_items[0]
    assert item.feature is None
    assert torch.equal(
        item.precomputed_embeddings, torch.full((ENCODER_TOKEN_COUNT, 2), 7.0)
    )
