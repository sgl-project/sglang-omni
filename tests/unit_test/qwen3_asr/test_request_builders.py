# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.qwen3_asr.audio_lengths import (
    qwen3_asr_audio_token_lengths,
    qwen3_asr_num_audio_tokens,
)
from sglang_omni.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRProcessor
from sglang_omni.models.qwen3_asr.languages import (
    LANGUAGE_CODE_TO_NAME,
    resolve_language,
)
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 1000

    def __init__(self) -> None:
        self.call_texts: list[str] = []
        self.encode_calls: list[str] = []
        self.decode_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|audio_pad|>"
        return 42

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        self.encode_calls.append(text)
        assert text == "<asr_text>"
        return [100, 101]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        self.call_texts.append(text)
        audio_pad_count = text.count("<|audio_pad|>")
        return SimpleNamespace(input_ids=[11] + [42] * audio_pad_count + [12, 13, 14])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        pieces = {
            10: "language English",
            100: "<asr_text>",
            101: "",
            20: " leading",
            21: "\u00a0middle",
            22: "  ",
            99: "<|endoftext|>",
        }
        text = "".join(pieces[token_id] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<|endoftext|>", "")
        return text


def test_qwen3_asr_audio_token_length_formula_is_shared() -> None:
    lengths = torch.tensor([0, 1, 99, 100, 101, 3000], dtype=torch.long)
    expected = torch.tensor([0, 1, 13, 13, 14, 390], dtype=torch.long)

    processor = object.__new__(Qwen3ASRProcessor)

    assert torch.equal(qwen3_asr_audio_token_lengths(lengths), expected)
    assert torch.equal(processor._get_feat_extract_output_lengths(lengths), expected)
    assert qwen3_asr_num_audio_tokens(3000) == 390


@pytest.mark.parametrize(
    ("language_code", "expected_name"), sorted(LANGUAGE_CODE_TO_NAME.items())
)
def test_qwen3_asr_resolves_every_supported_language_code(
    language_code: str,
    expected_name: str,
) -> None:
    assert resolve_language(language_code) == expected_name
    assert resolve_language(language_code.upper()) == expected_name


@pytest.mark.parametrize("language_name", sorted(LANGUAGE_CODE_TO_NAME.values()))
def test_qwen3_asr_resolves_canonical_names_case_insensitively(
    language_name: str,
) -> None:
    assert resolve_language(f"  {language_name.swapcase()}  ") == language_name


@pytest.mark.parametrize("language", ["cn", "zh-CN", "zh_Hant"])
def test_qwen3_asr_preserves_chinese_compatibility_aliases(language: str) -> None:
    assert resolve_language(language) == "Chinese"


@pytest.mark.parametrize(
    ("language", "expected_name", "expected_language"),
    [
        (None, "English", "en"),
        ("en", "English", "en"),
        ("es", "Spanish", "es"),
        ("fReNcH", "French", "fReNcH"),
    ],
)
def test_qwen3_asr_request_builder_uses_canonical_language_prompt(
    monkeypatch,
    language: str | None,
    expected_name: str,
    expected_language: str,
) -> None:
    tokenizer = _FakeTokenizer()
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 100)),
        attention_mask=torch.ones((1, 100), dtype=torch.long),
    )
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-language",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={} if language is None else {"language": language},
        ),
        data={},
    )

    data = request_builder(payload)

    assert tokenizer.call_texts[-1].endswith(f"language {expected_name}<asr_text>")
    assert data.language == expected_language


@pytest.mark.parametrize(
    ("language", "error_match"),
    [
        ("", "language hint is empty.*supported language code"),
        ("   ", "language hint is empty.*supported language code"),
        ("Klingon", "Unsupported language: 'Klingon'.*supported language code"),
    ],
)
def test_qwen3_asr_rejects_explicit_unsupported_language_before_loading_audio(
    monkeypatch,
    language: str,
    error_match: str,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: pytest.fail(
            "invalid language must fail before audio loading"
        ),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-unsupported-language",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"language": language},
        ),
        data={},
    )

    with pytest.raises(ValueError, match=error_match):
        request_builder(payload)


def test_qwen3_asr_request_builder_records_inclusive_audio_offsets(monkeypatch) -> None:
    num_mel_frames = 101
    num_audio_tokens = qwen3_asr_num_audio_tokens(num_mel_frames)
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 3000)),
        attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
    )
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    start, end = audio_item.offsets[0]
    assert audio_item.feature_attention_mask.shape == (1, num_mel_frames)
    assert end - start + 1 == num_audio_tokens
    assert data.prompt_token_ids[start : end + 1] == (
        [audio_item.pad_value] * num_audio_tokens
    )


def test_qwen3_asr_request_builder_preserves_audio_beyond_30_seconds(
    monkeypatch,
) -> None:
    """The request builder must not apply Whisper's default 30-second truncation."""
    sample_rate = 16000
    audio_duration_s = 31
    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=sample_rate,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(
            sample_rate * audio_duration_s, dtype=np.float32
        ),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-long-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    assert audio_item.feature.shape == (1, 128, 3100)
    assert int(audio_item.feature_attention_mask.sum().item()) == 3100
    assert data.audio_duration_s == audio_duration_s


def test_qwen3_asr_result_adapter_decodes_without_text_round_trip() -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = Qwen3ASRRequestData(
        output_ids=[10, 100, 101, 20, 21, 22, 99],
        stage_payload=payload,
        language="en",
        audio_duration_s=1.25,
    )

    result = result_adapter(data)

    assert result.data["text"] == " leading\u00a0middle  "
    assert tokenizer.encode_calls == ["<asr_text>"]
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 22, 99],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_qwen3_asr_request_builder_encodes_after_offsets_are_final(
    monkeypatch,
) -> None:
    num_mel_frames = 101
    num_audio_tokens = qwen3_asr_num_audio_tokens(num_mel_frames)
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 3000)),
        attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
    )
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    observed: dict[str, object] = {}

    class _EncoderService:
        def encode_item(self, item) -> None:
            observed["offsets"] = item.offsets
            observed["num_audio_tokens"] = item.num_audio_tokens
            observed["audio_fingerprint"] = item.audio_fingerprint
            item.precomputed_embeddings = torch.zeros(item.num_audio_tokens, 4)
            item.feature = None

    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=_EncoderService(),
    )
    payload = StagePayload(
        request_id="req-asr-pre-lm",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    item = data.req.multimodal_inputs.mm_items[0]
    assert observed["offsets"] == item.offsets
    assert observed["num_audio_tokens"] == num_audio_tokens
    assert isinstance(observed["audio_fingerprint"], str)
    assert observed["audio_fingerprint"] == data.req.extra_key
    assert item.feature is None
    assert item.precomputed_embeddings.shape[0] == num_audio_tokens
