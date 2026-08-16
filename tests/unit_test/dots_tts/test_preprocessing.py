# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from dots_tts.utils.tokenizer import AUDIO_COMP_SPAN_TOKEN, AUDIO_GEN_SPAN_TOKEN

from sglang_omni.models.dots_tts import stages
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.stages import preprocess_dots_tts_payload
from sglang_omni.proto import OmniRequest, StagePayload


class _RecordingTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        from dots_tts.utils.tokenizer import (
            AUDIO_COMP_SPAN_TOKEN,
            AUDIO_GEN_SPAN_TOKEN,
            AUDIO_GEN_START_TOKEN,
        )

        self.encoded_text: list[str] = []
        self._tokens = {
            AUDIO_GEN_START_TOKEN: 101,
            AUDIO_GEN_SPAN_TOKEN: 102,
            AUDIO_COMP_SPAN_TOKEN: 103,
        }
        self.len_calls = 0
        self.converted_tokens: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_text.append(text)
        return [10] if text else []

    def decode(self, token_ids: list[int], **_kwargs) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        self.converted_tokens.append(token)
        return self._tokens[token]

    def __len__(self) -> int:
        self.len_calls += 1
        return 256


def _payload(
    *,
    tts_params: dict | None = None,
    params: dict | None = None,
    references: list[dict] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs={
                "input": "hello",
                "references": references
                or [
                    {
                        "audio_path": "data:audio/wav;base64,UklGRg==",
                        "text": "reference",
                    }
                ],
            },
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def _preprocess(payload: StagePayload, tokenizer: _RecordingTokenizer) -> DotsTTSState:
    result = preprocess_dots_tts_payload(
        payload,
        tokenizer=tokenizer,
        model_config=SimpleNamespace(
            patch_size=4,
            vocoder=SimpleNamespace(sample_rate=48000),
        ),
        max_generate_length=20,
        max_sequence_length=128,
    )
    return DotsTTSState.from_dict(result.data)


def test_public_base_auto_and_generation_budget_reach_native_state(monkeypatch) -> None:
    monkeypatch.setattr("dots_tts.utils.text.detect", lambda _text: "en")
    tokenizer = _RecordingTokenizer()

    state = _preprocess(
        _payload(
            tts_params={"task_type": "Base", "language": "Auto"},
            params={"max_new_tokens": 3},
        ),
        tokenizer,
    )

    assert state.max_new_tokens == 3
    assert state.prompt_audio_path == "data:audio/wav;base64,UklGRg=="
    assert state.use_prompt_prefill is True
    assert any("[EN]reference" in text for text in tokenizer.encoded_text)


def test_preprocessing_rejects_unconsumed_extra_references() -> None:
    payload = _payload(
        references=[
            {"audio_path": "first.wav", "text": "first"},
            {"audio_path": "second.wav", "text": "second"},
        ]
    )

    with pytest.raises(ValueError, match="at most one reference"):
        _preprocess(payload, _RecordingTokenizer())


def test_dots_executor_resolves_tokenizer_invariants_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _RecordingTokenizer()
    monkeypatch.setattr(
        stages,
        "_load_model_metadata",
        lambda _path: (
            "model",
            SimpleNamespace(
                patch_size=4,
                vocoder=SimpleNamespace(sample_rate=48000),
            ),
            tokenizer,
            128,
        ),
    )
    monkeypatch.setattr(
        "dots_tts.data.pipelines.tokenizing.build_generation_schedule",
        lambda **_kwargs: {"schedule_ids": [10, 102, 103]},
    )

    preprocess = stages.create_preprocessing_executor("model")._fn

    assert tokenizer.len_calls == 1
    assert tokenizer.converted_tokens == [
        AUDIO_GEN_SPAN_TOKEN,
        AUDIO_COMP_SPAN_TOKEN,
    ]
    first = DotsTTSState.from_dict(preprocess(_payload()).data)
    second = DotsTTSState.from_dict(preprocess(_payload()).data)
    assert tokenizer.len_calls == 1
    assert tokenizer.converted_tokens == [
        AUDIO_GEN_SPAN_TOKEN,
        AUDIO_COMP_SPAN_TOKEN,
    ]
    assert first.audio_span_token_ids == second.audio_span_token_ids == [102, 103]
    assert first.audio_span_token_ids is not second.audio_span_token_ids
    assert first.vocab_size == second.vocab_size == 256


def test_dots_direct_preprocessor_resolves_tokenizer_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dots_tts.data.pipelines.tokenizing.build_generation_schedule",
        lambda **_kwargs: {"schedule_ids": [10, 102, 103]},
    )
    tokenizer = _RecordingTokenizer()

    state = _preprocess(_payload(), tokenizer)

    assert tokenizer.len_calls == 1
    assert tokenizer.converted_tokens == [
        AUDIO_GEN_SPAN_TOKEN,
        AUDIO_COMP_SPAN_TOKEN,
    ]
    assert state.audio_span_token_ids == [102, 103]
    assert state.vocab_size == 256
