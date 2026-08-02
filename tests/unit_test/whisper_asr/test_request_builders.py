# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.models.whisper_asr.request_builders as request_builders
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    vocab_size = 100

    def __init__(self) -> None:
        self.language = "original"
        self.task = "original"
        self.predict_timestamps = True

    @property
    def prefix_tokens(self) -> list[int]:
        tokens = [10]
        if self.language is not None:
            tokens.append({"english": 11, "chinese": 12}[self.language])
        tokens.append({"transcribe": 20, "translate": 21}[self.task])
        if not self.predict_timestamps:
            tokens.append(30)
        return tokens

    def decode(self, token_ids, *, skip_special_tokens=False) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _SlowFakeTokenizer(_FakeTokenizer):
    @property
    def prefix_tokens(self) -> list[int]:
        language = self.language
        time.sleep(0.001)
        task = self.task
        predict_timestamps = self.predict_timestamps
        tokens = [10]
        if language is not None:
            tokens.append({"english": 11, "chinese": 12}[language])
        tokens.append({"transcribe": 20, "translate": 21}[task])
        if not predict_timestamps:
            tokens.append(30)
        return tokens


class _FailingPrefixTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = True

    @property
    def prefix_tokens(self) -> list[int]:
        if self.fail_next:
            self.fail_next = False
            raise ValueError("unsupported language")
        return super().prefix_tokens


def _processor() -> SimpleNamespace:
    def _feature_extractor(audio, *, sampling_rate, return_tensors):
        assert len(audio) > 0
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        return SimpleNamespace(input_features=torch.zeros((1, 80, 3000)))

    return SimpleNamespace(feature_extractor=_feature_extractor)


def _payload(params: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id="whisper-request",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params=params or {},
        ),
        data={},
    )


def _adapters(tokenizer: _FakeTokenizer | None = None):
    tokenizer = tokenizer or _FakeTokenizer()
    return request_builders.make_whisper_scheduler_adapters(
        processor=_processor(),
        tokenizer=tokenizer,
        generation_config=SimpleNamespace(suppress_tokens=[]),
        encoder_token_count=1500,
        max_new_tokens=64,
    )


def test_whisper_transcription_defaults_to_english(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    tokenizer = _FakeTokenizer()
    request_builder, _ = _adapters(tokenizer)

    data = request_builder(_payload())

    assert data.prompt_token_ids == [10, 11, 20, 30]
    assert data.language == "english"
    assert data.task == "transcribe"
    assert data.req.sampling_params.max_new_tokens == 64
    assert tokenizer.language == "original"
    assert tokenizer.task == "original"
    assert tokenizer.predict_timestamps is True


def test_whisper_prefix_state_is_request_local_under_concurrency(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    tokenizer = _SlowFakeTokenizer()
    request_builder, _ = _adapters(tokenizer)
    params = [
        {"task": "transcribe", "language": "english"},
        {"task": "translate", "language": "chinese"},
    ] * 16

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(request_builder, map(_payload, params)))

    assert [result.prompt_token_ids for result in results] == [
        [10, 11, 20, 30],
        [10, 12, 21, 30],
    ] * 16
    assert tokenizer.language == "original"
    assert tokenizer.task == "original"
    assert tokenizer.predict_timestamps is True


def test_whisper_prefix_restores_missing_optional_state() -> None:
    tokenizer = _FakeTokenizer()
    del tokenizer.language
    del tokenizer.task
    del tokenizer.predict_timestamps

    assert request_builders._build_prefix_tokens(
        tokenizer,
        language="english",
        task="transcribe",
    ) == [10, 11, 20, 30]
    with pytest.raises(AttributeError):
        _ = tokenizer.language
    with pytest.raises(AttributeError):
        _ = tokenizer.task
    with pytest.raises(AttributeError):
        _ = tokenizer.predict_timestamps


def test_whisper_prefix_restores_state_after_tokenizer_error() -> None:
    tokenizer = _FailingPrefixTokenizer()

    with pytest.raises(ValueError, match="unsupported language"):
        request_builders._build_prefix_tokens(
            tokenizer,
            language="invalid",
            task="translate",
        )

    assert tokenizer.language == "original"
    assert tokenizer.task == "original"
    assert tokenizer.predict_timestamps is True
    assert request_builders._build_prefix_tokens(
        tokenizer,
        language="english",
        task="transcribe",
    ) == [10, 11, 20, 30]


def test_whisper_logit_bias_treats_missing_suppress_tokens_as_optional() -> None:
    assert request_builders._build_logit_bias(SimpleNamespace()) is None


def test_whisper_translation_requires_source_language_before_audio_load(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: (_ for _ in ()).throw(AssertionError("must not load audio")),
    )
    request_builder, _ = _adapters()

    with pytest.raises(ValueError, match="requires an explicit source language"):
        request_builder(_payload({"task": "translate"}))


def test_whisper_translation_accepts_explicit_source_language(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    request_builder, result_adapter = _adapters()

    data = request_builder(_payload({"task": "translate", "language": "chinese"}))

    assert data.prompt_token_ids == [10, 12, 21, 30]
    assert data.language == "english"
    assert data.task == "translate"

    data.output_ids = [40, 41]
    result = result_adapter(data)
    assert result.data["text"] == "40 41"
    assert result.data["language"] == "english"
    assert result.data["task"] == "translate"


def test_whisper_request_rejects_unknown_task(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = _adapters()

    with pytest.raises(ValueError, match="Whisper task must be one of"):
        request_builder(_payload({"task": "summarize"}))


def test_whisper_request_rejects_excessive_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = _adapters()

    with pytest.raises(ValueError, match="max_new_tokens must be between 1 and 64"):
        request_builder(_payload({"max_new_tokens": 65}))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"task": "summarize"}, "Whisper task must be one of"),
        ({"max_new_tokens": 65}, "max_new_tokens must be between 1 and 64"),
        ({"task": "translate"}, "requires an explicit source language"),
        ({"max_new_tokens": True}, "max_new_tokens must be an integer"),
        ({"max_new_tokens": 3.9}, "max_new_tokens must be an integer"),
    ],
)
def test_whisper_rejects_invalid_params_before_loading_audio(
    monkeypatch, params, message
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: (_ for _ in ()).throw(AssertionError("audio load must not run")),
    )
    request_builder, _ = _adapters()

    with pytest.raises(ValueError, match=message):
        request_builder(_payload(params))


def test_whisper_request_rejects_audio_over_30_seconds(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(30 * 16000 + 1, dtype=np.float32),
    )
    request_builder, _ = _adapters()

    with pytest.raises(ValueError, match="accepts audio up to 30.0 seconds"):
        request_builder(_payload())


def test_whisper_request_reports_audio_decode_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    request_builder, _ = _adapters()

    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        request_builder(_payload())
