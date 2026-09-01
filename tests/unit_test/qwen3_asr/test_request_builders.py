# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
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
from sglang_omni.scheduling.types import DeferredAdmission
from sglang_omni.utils import audio as audio_utils
from sglang_omni.utils.audio import AudioDecodeError

_EXPECTED_QWEN3_ASR_PROMPT_PREFIX = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _unwrap_built(
    result: Qwen3ASRRequestData | DeferredAdmission,
) -> Qwen3ASRRequestData:
    if isinstance(result, DeferredAdmission):
        result.ready.result(timeout=5)
        return result.value
    return result


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 100

    def __init__(self) -> None:
        self.call_texts: list[str] = []
        self.encode_calls: list[str] = []
        self.decode_calls: list[dict] = []

    def __len__(self) -> int:
        return 102

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
            30: "language None",
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


def test_qwen3_asr_max_audio_tokens_covers_the_native_limit() -> None:
    from sglang_omni.models.qwen3_asr.audio_lengths import (
        QWEN3_ASR_MAX_INPUT_SECONDS,
        qwen3_asr_max_audio_tokens,
        qwen3_asr_max_output_tokens,
    )

    # The official wrapper accepts up to 1,200s natively; the engine context
    # is sized from this figure (13 audio tokens per second) plus the
    # duration-scaled output budget that clip would get.
    assert QWEN3_ASR_MAX_INPUT_SECONDS == 1200
    assert qwen3_asr_max_audio_tokens() == 15_600
    assert qwen3_asr_max_output_tokens() == 12_000


def _budget_test_builder(monkeypatch, num_samples: int):
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 100)),
        attention_mask=torch.ones((1, 100), dtype=torch.long),
    )
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(num_samples, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=128,
        feature_extractor=feature_extractor,
    )
    return request_builder


def test_qwen3_asr_short_audio_keeps_the_floor_budget(monkeypatch) -> None:
    # 0.1s of audio scales to ~1 token; the stage default is the floor, so a
    # short clip reserves no more scheduler budget than before.
    request_builder = _budget_test_builder(monkeypatch, num_samples=1600)
    payload = StagePayload(
        request_id="req-floor",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
        data={},
    )

    data = request_builder(payload)

    assert data.max_new_tokens == 128


def test_qwen3_asr_request_builder_enforces_scheduler_request_limits(
    monkeypatch,
) -> None:
    request_builder = _budget_test_builder(monkeypatch, num_samples=1600)
    payload = StagePayload(
        request_id="req-request-limits",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
        data={},
    )

    data = request_builder(payload)

    assert data.enforce_request_limits is True


def test_qwen3_asr_long_audio_scales_the_budget(monkeypatch) -> None:
    # 60s of audio needs ~300 output tokens; the scaled default (10/s with
    # margin) covers it without a large flat default.
    request_builder = _budget_test_builder(monkeypatch, num_samples=60 * 16000)
    payload = StagePayload(
        request_id="req-scaled",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
        data={},
    )

    data = request_builder(payload)

    assert data.max_new_tokens == 600


def test_qwen3_asr_explicit_budget_overrides_scaling(monkeypatch) -> None:
    request_builder = _budget_test_builder(monkeypatch, num_samples=60 * 16000)
    payload = StagePayload(
        request_id="req-explicit",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"max_new_tokens": 32},
        ),
        data={},
    )

    data = request_builder(payload)

    assert data.max_new_tokens == 32


@pytest.mark.parametrize("num_mel_frames", range(0, 401))
def test_qwen3_asr_scalar_audio_token_length_matches_tensor_formula(
    num_mel_frames: int,
) -> None:
    expected = int(qwen3_asr_audio_token_lengths(num_mel_frames).item())

    assert qwen3_asr_num_audio_tokens(num_mel_frames) == expected


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
        ("en", "English", "en"),
        ("es", "Spanish", "es"),
        ("fReNcH", "French", "fReNcH"),
    ],
)
def test_qwen3_asr_request_builder_uses_canonical_language_prompt(
    monkeypatch,
    language: str,
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
            params={"language": language},
        ),
        data={},
    )

    data = request_builder(payload)

    assert tokenizer.call_texts[-1] == (
        _EXPECTED_QWEN3_ASR_PROMPT_PREFIX + f"language {expected_name}<asr_text>"
    )
    assert data.language == expected_language


def test_qwen3_asr_request_builder_omits_language_prompt_for_auto_detection(
    monkeypatch,
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
        request_id="req-auto-language",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    assert tokenizer.call_texts[-1] == _EXPECTED_QWEN3_ASR_PROMPT_PREFIX
    assert "<asr_text>" not in tokenizer.call_texts[-1]
    assert data.language is None
    assert data.req.vocab_size == len(tokenizer)
    assert set(data.req.sampling_params.stop_token_ids) == {2}


def test_qwen3_asr_request_builder_caches_prompt_template_per_language(
    monkeypatch,
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

    for request_id in ("req-first", "req-second"):
        request_builder(
            StagePayload(
                request_id=request_id,
                request=OmniRequest(
                    inputs={"audio_bytes": b"wav"},
                    params={"language": "en"},
                ),
                data={},
            )
        )

    assert len(tokenizer.call_texts) == 1
    assert tokenizer.call_texts[0].count("<|audio_pad|>") == 1


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


@pytest.mark.parametrize(
    (
        "params",
        "expected_temperature",
        "expected_sampling_temperature",
        "expected_top_k",
    ),
    [
        ({}, 0.0, 1.0, 1),
        ({"temperature": 0.1}, 0.1, 0.1, 1 << 30),
    ],
)
def test_qwen3_asr_request_builder_preserves_sampling_mode(
    monkeypatch,
    params: dict[str, float],
    expected_temperature: float,
    expected_sampling_temperature: float,
    expected_top_k: int,
) -> None:
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 3000)),
        attention_mask=torch.ones((1, 101), dtype=torch.long),
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
        request_id="req-asr-sampling",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params),
        data={},
    )

    data = request_builder(payload)

    assert data.temperature == expected_temperature
    assert data.req.sampling_params.temperature == expected_sampling_temperature
    assert data.req.sampling_params.top_k == expected_top_k


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
        context_length=512,
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


def test_qwen3_asr_request_builder_rejects_undecodable_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: (_ for _ in ()).throw(
            AudioDecodeError("decode failed")
        ),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-invalid",
        request=OmniRequest(inputs={"audio_bytes": b"not-audio"}),
        data={},
    )

    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        request_builder(payload)


def test_qwen3_asr_request_builder_rejects_corrupt_local_audio_path(
    monkeypatch,
    tmp_path,
) -> None:
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"not-audio")

    def raise_decode_error(source):
        assert source == str(corrupt_path)
        raise RuntimeError("invalid audio data")

    monkeypatch.setattr(audio_utils, "_ensure_torchaudio_decoder_ready", lambda: None)
    monkeypatch.setattr(audio_utils.torchaudio, "load", raise_decode_error)
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-invalid-path",
        request=OmniRequest(inputs={"audio_path": str(corrupt_path)}),
        data={},
    )

    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        request_builder(payload)


def test_qwen3_asr_request_builder_preserves_operational_audio_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: (_ for _ in ()).throw(
            RuntimeError("decoder backend unavailable")
        ),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-loader-failure",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    with pytest.raises(RuntimeError, match="decoder backend unavailable"):
        request_builder(payload)


def test_qwen3_asr_rejects_full_context_before_mel_extraction(
    monkeypatch,
) -> None:
    class _UnexpectedFeatureExtractor:
        hop_length = 160

        def __call__(self, *args, **kwargs):
            raise AssertionError("feature extractor should not be called")

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=5,
        feature_extractor=_UnexpectedFeatureExtractor(),
        # 100 mel frames produce 13 audio tokens, so the fake prompt has
        # 17 input tokens and exactly fills context with 5 output tokens.
        context_length=22,
    )
    payload = StagePayload(
        request_id="req-over-context",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    with pytest.raises(ValueError, match="longer than the model's context length"):
        request_builder(payload)


def test_qwen3_asr_embedding_cache_hit_skips_mel_extraction(monkeypatch) -> None:
    class _UnexpectedFeatureExtractor:
        hop_length = 160

        def __call__(self, *args, **kwargs):
            raise AssertionError("feature extractor should not be called")

    class _EncoderService:
        def __init__(self) -> None:
            self.lookup: tuple[str, int] | None = None
            self.embedding = torch.zeros((13, 4))

        def lookup_cached_embedding(
            self, audio_fingerprint: str, expected_tokens: int
        ) -> torch.Tensor | None:
            self.lookup = (audio_fingerprint, expected_tokens)
            return self.embedding

        def attach_embedding(self, item, embedding: torch.Tensor) -> None:
            item.precomputed_embeddings = embedding
            item.feature = None

        def encode_item(self, item) -> None:
            raise AssertionError("encoder should not be called on a cache hit")

        def submit_item(self, item):
            raise AssertionError("encoder should not be called on a cache hit")

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    encoder_service = _EncoderService()
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=_UnexpectedFeatureExtractor(),
        audio_encoder_service=encoder_service,
    )
    payload = StagePayload(
        request_id="req-asr-cache-hit",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    assert isinstance(data, Qwen3ASRRequestData)
    item = data.req.multimodal_inputs.mm_items[0]
    assert encoder_service.lookup == (data.req.extra_key, 13)
    assert item.feature is None
    assert item.precomputed_embeddings is encoder_service.embedding
    assert item.num_audio_tokens == 13


def test_qwen3_asr_embedding_cache_miss_extracts_and_encodes(monkeypatch) -> None:
    class _FeatureExtractor:
        hop_length = 160

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                input_features=torch.zeros((1, 128, 100)),
                attention_mask=torch.ones((1, 100), dtype=torch.long),
            )

    class _EncoderService:
        def __init__(self) -> None:
            self.lookup: tuple[str, int] | None = None
            self.encoded_feature: torch.Tensor | None = None

        def lookup_cached_embedding(
            self, audio_fingerprint: str, expected_tokens: int
        ) -> None:
            self.lookup = (audio_fingerprint, expected_tokens)
            return None

        def attach_embedding(self, item, embedding: torch.Tensor) -> None:
            raise AssertionError("no cached embedding should be attached")

        def encode_item(self, item) -> None:
            raise AssertionError("cache miss should submit, not block on encode")

        def submit_item(self, item):
            self.encoded_feature = item.feature
            item.precomputed_embeddings = torch.zeros((item.num_audio_tokens, 4))
            item.feature = None
            future: concurrent.futures.Future[torch.Tensor] = (
                concurrent.futures.Future()
            )
            future.set_result(item.precomputed_embeddings)
            return future

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    feature_extractor = _FeatureExtractor()
    encoder_service = _EncoderService()
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=encoder_service,
    )
    payload = StagePayload(
        request_id="req-asr-cache-miss",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    result = request_builder(payload)

    assert isinstance(result, DeferredAdmission)
    data = _unwrap_built(result)
    assert encoder_service.lookup == (data.req.extra_key, 13)
    assert feature_extractor.calls == 1
    assert encoder_service.encoded_feature is not None
    assert data.req.multimodal_inputs.mm_items[0].feature is None


def test_qwen3_asr_cache_miss_waits_in_builder_when_asked(monkeypatch) -> None:
    class _FeatureExtractor:
        hop_length = 160

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                input_features=torch.zeros((1, 128, 100)),
                attention_mask=torch.ones((1, 100), dtype=torch.long),
            )

    class _EncoderService:
        def lookup_cached_embedding(self, audio_fingerprint, expected_tokens):
            del audio_fingerprint, expected_tokens
            return None

        def encode_item(self, item) -> None:
            item.precomputed_embeddings = torch.zeros((item.num_audio_tokens, 4))
            item.feature = None

        def submit_item(self, item):
            raise AssertionError("builder should wait on encode_item, not submit_item")

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=_FeatureExtractor(),
        audio_encoder_service=_EncoderService(),
        should_wait_for_encode=lambda: True,
    )
    payload = StagePayload(
        request_id="req-asr-wait",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    result = request_builder(payload)

    assert isinstance(result, Qwen3ASRRequestData)
    assert result.req.multimodal_inputs.mm_items[0].feature is None


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
        audio_duration_s=1.25,
    )

    result = result_adapter(data)

    assert result.data["text"] == " leading\u00a0middle  "
    assert result.data["language"] == "English"
    assert tokenizer.encode_calls == ["<asr_text>"]
    assert len(tokenizer.decode_calls) == 2
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 22, 99],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


@pytest.mark.parametrize(
    ("output_ids", "expected_text"),
    [
        ([30, 100, 101], ""),
        ([30, 100, 101, 20], " leading"),
    ],
)
def test_qwen3_asr_result_adapter_normalizes_none_language(
    output_ids: list[int], expected_text: str
) -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-asr-none-language",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = Qwen3ASRRequestData(
        output_ids=output_ids,
        stage_payload=payload,
    )

    result = result_adapter(data)

    assert result.data["text"] == expected_text
    assert result.data["language"] is None


def test_qwen3_asr_request_builder_encodes_after_offsets_are_final(
    monkeypatch,
) -> None:
    num_mel_frames = 101
    num_audio_tokens = qwen3_asr_num_audio_tokens(num_mel_frames)
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 3000)),
        attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
    )
    feature_extractor.hop_length = 160
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    observed: dict[str, object] = {}

    class _EncoderService:
        def lookup_cached_embedding(
            self, audio_fingerprint: str, expected_tokens: int
        ) -> None:
            return None

        def submit_item(self, item):
            observed["offsets"] = item.offsets
            observed["num_audio_tokens"] = item.num_audio_tokens
            observed["audio_fingerprint"] = item.audio_fingerprint
            item.precomputed_embeddings = torch.zeros(item.num_audio_tokens, 4)
            item.feature = None
            future: concurrent.futures.Future[torch.Tensor] = (
                concurrent.futures.Future()
            )
            future.set_result(item.precomputed_embeddings)
            return future

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

    data = _unwrap_built(request_builder(payload))

    item = data.req.multimodal_inputs.mm_items[0]
    assert observed["offsets"] == item.offsets
    assert observed["num_audio_tokens"] == num_audio_tokens
    assert isinstance(observed["audio_fingerprint"], str)
    assert observed["audio_fingerprint"] == data.req.extra_key
    assert item.feature is None
    assert item.precomputed_embeddings.shape[0] == num_audio_tokens
