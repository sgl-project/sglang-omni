# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalDataItem

import sglang_omni.models.fun_asr.request_builders as request_builders
import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoFeatureExtractor
from sglang_omni.models.fun_asr.encoder_service import FunASRPreLMEncoderService
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.audio import audio_fingerprint

AUDIO_PAD = "<|object_ref_start|>"
AUDIO_PAD_ID = 42  # arbitrary sentinel distinct from vocabulary ids below


class UnexpectedEncoderService:
    def get_cached_embedding(
        self, audio_fingerprint: str, expected_tokens: int
    ) -> None:
        return None

    def encode_item(self, item: object) -> None:
        pytest.fail("invalid requests must not be encoded")


class FakeTokenizer:
    eos_token_id = 151645
    vocab_size = 151936

    def __init__(self) -> None:
        self.decode_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == AUDIO_PAD
        return AUDIO_PAD_ID

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        # Mirror the real ChatML prompt shape: a fixed head/tail with N audio
        # placeholders in the middle. The request builder only inspects the
        # placeholder span, so the surrounding text need not be real tokens.
        audio_pad_count = text.count(AUDIO_PAD)
        # system(3) + user-open(2) + [pad]*N + user-close/assistant(4)
        input_ids = (
            [10, 11, 12, 13, 14] + [AUDIO_PAD_ID] * audio_pad_count + [15, 16, 17, 18]
        )
        return SimpleNamespace(input_ids=input_ids)

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
        pieces = {20: "你好", 21: "世界", 30: "<|im_end|>"}
        text = "".join(pieces.get(t, "") for t in token_ids)
        if skip_special_tokens:
            text = text.replace("<|im_end|>", "")
        return text


def feature_extractor(num_lfr_frames: int):
    """Stand-in for FunAsrNanoFeatureExtractor: returns [1, 560, T_lfr]."""

    def call(
        audio,
        sampling_rate=None,
        return_tensors=None,
        return_attention_mask=True,
        padding="longest",
    ):
        return {
            "input_features": torch.zeros((1, 560, num_lfr_frames)),
            "attention_mask": torch.ones((1, num_lfr_frames), dtype=torch.long),
        }

    call.num_lfr_frames = lambda num_samples: num_lfr_frames
    return call


def test_fun_asr_request_builder_records_inclusive_audio_offsets(
    monkeypatch,
) -> None:
    num_lfr_frames = 17
    num_audio_tokens = 3
    extractor = feature_extractor(num_lfr_frames)

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600 * 3, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=extractor,
    )
    payload = StagePayload(
        request_id="req-fun-asr",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"}, params={"max_new_tokens": 32}
        ),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    start, end = audio_item.offsets[0]
    assert audio_item.feature_attention_mask.shape == (1, num_lfr_frames)
    assert end - start + 1 == num_audio_tokens
    assert (
        data.prompt_token_ids[start : end + 1]
        == [audio_item.pad_value] * num_audio_tokens
    )
    # pad_value replaces the placeholder span (general_mm_embed_routine matches it
    # by pad_value, not by the original <|object_ref_start|> token id)
    assert audio_item.pad_value != AUDIO_PAD_ID
    # greedy by default (Fun-ASR reference uses no sampling args): temperature=0.0
    # is normalized by sglang to top_k=1. The original intent is on FunASRRequestData.
    assert data.temperature == 0.0
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.max_new_tokens == 32
    # mrope positions broadcast as [3, seq] degenerate (plain 1-D positions)
    seq_len = len(data.prompt_token_ids)
    assert data.req.multimodal_inputs.mrope_positions.shape == (3, seq_len)
    assert torch.equal(
        data.req.multimodal_inputs.mrope_positions[0],
        torch.arange(seq_len, dtype=torch.long),
    )


def test_fun_asr_request_builder_encodes_after_offsets_are_final(monkeypatch) -> None:
    audio = np.zeros(1600, dtype=np.float32)
    monkeypatch.setattr(transcription, "load_audio", lambda source, **kwargs: audio)
    observed: dict[str, object] = {}

    class EncoderService:
        def get_cached_embedding(
            self, audio_fingerprint: str, expected_tokens: int
        ) -> None:
            return None

        def encode_item(self, item) -> None:
            observed["offsets"] = item.offsets
            observed["num_audio_tokens"] = item.num_audio_tokens
            observed["audio_fingerprint"] = item.audio_fingerprint
            item.precomputed_embeddings = torch.zeros(item.num_audio_tokens, 4)
            item.feature = None

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor(17),
        audio_encoder_service=EncoderService(),
    )
    payload = StagePayload(
        request_id="req-fun-asr-pre-lm",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    item = data.req.multimodal_inputs.mm_items[0]
    assert observed["offsets"] == item.offsets
    assert observed["num_audio_tokens"] == 3
    assert observed["audio_fingerprint"] == audio_fingerprint(audio)
    assert item.feature is None
    assert item.precomputed_embeddings.shape[0] == 3


def test_fun_asr_request_builder_language_prompt(monkeypatch) -> None:

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    captured = {}

    class CapturingTokenizer(FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            captured["prompt_text"] = text
            return super().__call__(text, add_special_tokens=add_special_tokens)

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=CapturingTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor(11),
    )
    payload = StagePayload(
        request_id="req-fun-asr-en",
        request=OmniRequest(inputs={"audio_path": "x.wav"}, params={"language": "en"}),
        data={},
    )

    data = request_builder(payload)
    assert "语音转写成英文" in captured["prompt_text"]
    assert data.language == "en"


def test_fun_asr_result_adapter_decodes_transcript_directly() -> None:
    tokenizer = FakeTokenizer()
    _, result_adapter = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-fun-asr",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = request_builders.FunASRRequestData(
        output_ids=[20, 21, 30],  # 你好 世界 <|im_end|>
        prompt_token_ids=[10, 11, 12, 13],
        num_audio_tokens=3,
        finish_reason="stop",
        stage_payload=payload,
        language="zh",
        audio_duration_s=2.5,
    )

    result = result_adapter(data)

    # Fun-ASR emits the transcript directly after <|im_start|>assistant\n — no
    # forced prefix marker to strip. skip_special_tokens=True drops <|im_end|>.
    assert result.data["text"] == "你好世界"
    assert result.data["language"] == "zh"
    assert result.data["duration_s"] == 2.5
    assert result.data["prompt_tokens"] == 4
    assert result.data["audio_tokens"] == 3
    assert result.data["completion_tokens"] == 3
    assert result.data["finish_reason"] == "stop"
    assert result.data["modality"] == "text"
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 30],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_fun_asr_result_adapter_returns_the_speculative_acceptance_details() -> None:
    _, result_adapter = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=object(),
    )
    data = request_builders.FunASRRequestData(
        output_ids=[20, 30],
        stage_payload=StagePayload(
            request_id="req-fun-asr", request=OmniRequest(inputs={}), data={}
        ),
    )
    data.spec_tokens_details = {
        "spec_accept_rate": 0.5,
        "spec_accept_length": 2.0,
        "spec_num_correct_drafts": 1,
        "spec_num_proposed_drafts": 2,
        "spec_verify_ct": 1,
        "spec_correct_drafts_histogram": [0, 1],
    }

    assert result_adapter(data).data["spec_tokens_details"] == data.spec_tokens_details


def test_fun_asr_load_audio_uses_shared_audio_utility(monkeypatch) -> None:
    calls = []
    expected = np.zeros(1600, dtype=np.float32)

    def shared(source, **kwargs):
        calls.append((source, kwargs))
        return expected

    monkeypatch.setattr(transcription, "load_audio", shared)

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-shared-load",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    request_builder(payload)

    assert calls == [
        (
            b"wav",
            {"source_name": "Fun-ASR", "target_sample_rate": 16000},
        )
    ]


def test_fun_asr_request_builder_scales_default_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000 * 3, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-budget",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    assert data.req.sampling_params.max_new_tokens == 20
    assert data.max_new_tokens == 20


def test_fun_asr_request_builder_rejects_audio_over_vad_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000 * 30 + 1, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-too-long",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    with pytest.raises(ValueError, match=r"30(?:\.0)? seconds.*VAD"):
        request_builder(payload)


def test_fun_asr_request_builder_enforces_scheduler_request_limits(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor(17),
    )
    payload = StagePayload(
        request_id="req-fun-asr-request-limits",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
        data={},
    )

    data = request_builder(payload)

    assert data.enforce_request_limits is True


def test_fun_asr_request_builder_rejects_explicit_token_budget_over_cap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=feature_extractor(17),
        audio_encoder_service=UnexpectedEncoderService(),
    )
    payload = StagePayload(
        request_id="req-fun-asr-token-cap",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"max_new_tokens": 201},
        ),
        data={},
    )

    with pytest.raises(ValueError, match=r"max_new_tokens.*200"):
        request_builder(payload)


def test_fun_asr_request_builder_wraps_string_hotword_as_single_entry(
    monkeypatch,
) -> None:
    # A bare string must stay one hotword, not be split into characters.
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    captured = {}

    class CapturingTokenizer(FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            captured["prompt_text"] = text
            return super().__call__(text, add_special_tokens=add_special_tokens)

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=CapturingTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor(11),
    )
    payload = StagePayload(
        request_id="req-fun-asr-hotwords-str",
        request=OmniRequest(
            inputs={"audio_path": "x.wav"},
            params={"hotwords": "人工智能"},
        ),
        data={},
    )

    request_builder(payload)
    assert "热词列表：[人工智能]" in captured["prompt_text"]


def capture_prompt_text(monkeypatch, params):
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    captured = {}

    class CapturingTokenizer(FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            captured["prompt_text"] = text
            return super().__call__(text, add_special_tokens=add_special_tokens)

    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=CapturingTokenizer(),
        max_new_tokens=16,
        feature_extractor=feature_extractor(11),
    )
    request_builder(
        StagePayload(
            request_id="req-fun-asr-prompt",
            request=OmniRequest(inputs={"audio_path": "x.wav"}, params=params),
            data={},
        )
    )
    return captured["prompt_text"]


def test_fun_asr_prompt_field_feeds_the_hotword_list(monkeypatch) -> None:
    """The endpoint sends vocabulary hints as ``prompt``, not ``hotwords``.

    Without the fallback the hint is silently dropped and biasing is
    unreachable over HTTP.
    """
    text = capture_prompt_text(monkeypatch, {"prompt": "PyTorch, nginx, Kubernetes"})

    assert "热词列表：[PyTorch, nginx, Kubernetes]" in text


def test_fun_asr_explicit_hotwords_take_precedence_over_prompt(monkeypatch) -> None:
    text = capture_prompt_text(
        monkeypatch, {"hotwords": ["人工智能"], "prompt": "PyTorch"}
    )

    assert "热词列表：[人工智能]" in text
    assert "PyTorch" not in text


def test_fun_asr_request_builder_rejects_prompt_overrun_of_context_length(
    monkeypatch,
) -> None:
    # 17 LFR frames -> 3 audio tokens; FakeTokenizer yields 9 + 3 = 12 input ids.
    # With context_length=10 and max_new_tokens=5, 12 + 5 > 10 must raise a
    # clear bad-request error (caught by _BAD_REQUEST_MARKERS -> HTTP 400).
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    request_builder, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=200,
        feature_extractor=feature_extractor(17),
        context_length=10,
        audio_encoder_service=UnexpectedEncoderService(),
    )
    payload = StagePayload(
        request_id="req-fun-asr-overflow",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"max_new_tokens": 5},
        ),
        data={},
    )

    with pytest.raises(ValueError, match=r"longer than the model's context length"):
        request_builder(payload)


def test_apple_rejects_sampling_before_preprocessing(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("sampling rejection must precede audio loading")

    monkeypatch.setattr(request_builders, "prepare_audio", unexpected)
    build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor(17),
        greedy_only=True,
    )
    payload = StagePayload(
        request_id="sampling",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"}, params={"temperature": 0.5}
        ),
        data={},
    )
    with pytest.raises(ValueError, match="greedy"):
        build(payload)


def test_apple_sampling_error_is_an_http_bad_request():
    from sglang_omni.serve.openai_errors import is_bad_request_error

    assert is_bad_request_error(
        RuntimeError("Fun-ASR Apple currently requires temperature=0 (greedy decoding)")
    )
    assert not is_bad_request_error(RuntimeError("Fun-ASR model execution failed"))


@pytest.mark.parametrize("model", ["Fun-ASR", "Qwen3-ASR"])
def test_audio_decode_error_survives_pipeline_error_serialization(model):
    from sglang_omni.serve.openai_errors import is_bad_request_error

    assert is_bad_request_error(RuntimeError(f"Could not decode {model} audio input"))
    assert not is_bad_request_error(RuntimeError("Could not decode model weights"))


class FrontendTestEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(1, 1)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return torch.cat(
            [
                item.feature.mean(dim=(0, 2))[:4].repeat(item.num_audio_tokens, 1)
                for item in items
            ]
        )


class CacheTestTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> SimpleNamespace:
        self.prompts.append(text)
        if return_offsets_mapping or AUDIO_PAD not in text:
            return SimpleNamespace(
                input_ids=list(range(100, 100 + len(text))),
                offset_mapping=[(i, i + 1) for i in range(len(text))],
            )
        else:
            return super().__call__(text, add_special_tokens=add_special_tokens)


@pytest.fixture
def cached_frontend(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[tuple[Mock, FunASRPreLMEncoderService, CacheTestTokenizer, np.ndarray]]:
    num_samples, frame_length, frame_shift = getattr(request, "param", (8080, 25, 10))
    audio = np.random.default_rng(7).normal(0, 0.01, num_samples).astype(np.float32)
    monkeypatch.setattr(transcription, "load_audio", lambda source, **kwargs: audio)
    extractor = Mock(
        wraps=FunAsrNanoFeatureExtractor(
            frame_length=frame_length, frame_shift=frame_shift
        )
    )
    service = FunASRPreLMEncoderService(
        FrontendTestEncoder(), cache_namespace="frontend-test", cache_max_entries=1
    )
    try:
        yield extractor, service, CacheTestTokenizer(), audio
    finally:
        service.close()


@pytest.mark.parametrize(
    "cached_frontend",
    [(8080, 25, 10), (6161, 25.1, 7.55)],
    ids=["default", "fractional-kaldi-window"],
    indirect=True,
)
@pytest.mark.parametrize(
    "params",
    [
        {},
        {
            "language": "en",
            "hotwords": ["SGLang"],
            "itn": False,
            "temperature": 0.7,
            "max_new_tokens": 24,
        },
        {
            "language": "zh",
            "_asr_streaming": True,
            "_asr_streaming_prefix_text": "你好世界",
            "_asr_streaming_rollback_chars": 1,
        },
    ],
)
def test_encoder_cache_hit_skips_frontend_and_preserves_request(
    cached_frontend: tuple[
        Mock, FunASRPreLMEncoderService, CacheTestTokenizer, np.ndarray
    ],
    params: dict[str, str | bool | float | int | list[str]],
) -> None:
    """Warm requests used to recompute fbank/LFR despite an embedding hit."""
    extractor, service, tokenizer, audio = cached_frontend
    build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=extractor,
        audio_encoder_service=service,
    )
    payload = StagePayload(
        request_id="cold",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params),
        data={},
    )
    cold = build(payload)
    cold_prompts = list(tokenizer.prompts)
    tokenizer.prompts.clear()
    extractor.side_effect = AssertionError("cache hit recomputed audio features")
    warm = build(
        StagePayload(
            request_id="warm",
            request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params),
            data={},
        )
    )
    assert extractor.call_count == 1
    assert tokenizer.prompts == cold_prompts
    assert torch.equal(cold.input_ids, warm.input_ids)
    assert cold.prompt_token_ids == warm.prompt_token_ids
    assert cold.num_audio_tokens == warm.num_audio_tokens == 2
    assert cold.audio_duration_s == warm.audio_duration_s
    assert cold.max_new_tokens == warm.max_new_tokens
    assert cold.temperature == warm.temperature
    assert cold.streaming_prefix_text == warm.streaming_prefix_text
    assert cold.req.sampling_params == warm.req.sampling_params
    assert cold.req.extra_key == warm.req.extra_key == audio_fingerprint(audio)
    for name in ("mrope_positions", "mrope_position_delta"):
        assert torch.equal(
            getattr(cold.req.multimodal_inputs, name),
            getattr(warm.req.multimodal_inputs, name),
        )
    cold_item = cold.req.multimodal_inputs.mm_items[0]
    warm_item = warm.req.multimodal_inputs.mm_items[0]
    assert cold_item.offsets == warm_item.offsets
    assert cold_item.pad_value == warm_item.pad_value
    assert cold_item.format == warm_item.format
    assert torch.equal(
        cold_item.feature_attention_mask, warm_item.feature_attention_mask
    )
    assert torch.equal(
        cold_item.precomputed_embeddings, warm_item.precomputed_embeddings
    )
    assert cold_item.feature is warm_item.feature is None
    assert service.stats()["hits"] == 1
    assert service.stats()["misses"] == 1


@pytest.mark.parametrize(
    "invalid", ["tokens", "hidden", "dtype", "namespace", "evicted", "disabled"]
)
def test_early_cache_miss_runs_original_frontend(
    cached_frontend: tuple[
        Mock, FunASRPreLMEncoderService, CacheTestTokenizer, np.ndarray
    ],
    invalid: str,
) -> None:
    """Invalid or unavailable embeddings must recompute the original features."""
    extractor, service, tokenizer, audio = cached_frontend
    build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=extractor,
        audio_encoder_service=service,
    )
    payload = StagePayload(
        request_id="first", request=OmniRequest(inputs={"audio_bytes": b"wav"}), data={}
    )
    cold = build(payload)
    original = cold.req.multimodal_inputs.mm_items[0].precomputed_embeddings.clone()
    cache_key = f"{service.namespace}:{audio_fingerprint(audio)}"
    if invalid == "tokens":
        service.cache.put(cache_key, torch.zeros(cold.num_audio_tokens + 1, 4))
    elif invalid == "hidden":
        service.cache.put(cache_key, torch.zeros(cold.num_audio_tokens, 5))
    elif invalid == "dtype":
        service.cache.put(cache_key, original.double())
    elif invalid == "namespace":
        service.namespace = "changed-model"
    else:
        service.cache.put("other-audio", original)
        if invalid == "disabled":
            service.cache.max_size = 0
        else:
            pass
    rebuilt = build(payload)
    assert extractor.call_count == 2
    assert torch.equal(
        rebuilt.req.multimodal_inputs.mm_items[0].precomputed_embeddings, original
    )
    assert service.stats()["misses"] == 2
    assert service.stats()["hits"] == 0


@pytest.mark.parametrize(
    "params,context_length,error",
    [
        ({"max_new_tokens": 33}, None, "max_new_tokens"),
        ({"max_new_tokens": 32}, 10, "context length"),
    ],
)
def test_warm_cache_preserves_request_validation(
    cached_frontend: tuple[
        Mock, FunASRPreLMEncoderService, CacheTestTokenizer, np.ndarray
    ],
    params: dict[str, int],
    context_length: int | None,
    error: str,
) -> None:
    """A cache hit must not admit an invalid generation request or count a consumed hit."""
    extractor, service, tokenizer, audio = cached_frontend
    build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=extractor,
        audio_encoder_service=service,
    )
    build(
        StagePayload(
            request_id="prime",
            request=OmniRequest(inputs={"audio_bytes": b"wav"}),
            data={},
        )
    )
    checked_build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        context_length=context_length,
        feature_extractor=extractor,
        audio_encoder_service=service,
    )
    extractor.side_effect = AssertionError("warm validation recomputed frontend")
    with pytest.raises(ValueError, match=error):
        checked_build(
            StagePayload(
                request_id="invalid",
                request=OmniRequest(inputs={"audio_bytes": b"wav"}, params=params),
                data={},
            )
        )
    assert service.stats()["hits"] == 0
    assert service.stats()["misses"] == 1


def test_warm_audio_does_not_reuse_request_specific_prompt(
    cached_frontend: tuple[
        Mock, FunASRPreLMEncoderService, CacheTestTokenizer, np.ndarray
    ],
) -> None:
    """Embedding reuse must not reuse another request's language or vocabulary."""
    extractor, service, tokenizer, audio = cached_frontend
    build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=extractor,
        audio_encoder_service=service,
    )
    first = build(
        StagePayload(
            request_id="zh",
            request=OmniRequest(inputs={"audio_bytes": b"wav"}),
            data={},
        )
    )
    original_prompt = tokenizer.prompts[-1]
    extractor.side_effect = AssertionError("warm audio recomputed features")
    second = build(
        StagePayload(
            request_id="en",
            request=OmniRequest(
                inputs={"audio_bytes": b"wav"},
                params={"language": "en", "hotwords": ["SGLang"], "max_new_tokens": 24},
            ),
            data={},
        )
    )
    assert tokenizer.prompts[-1] != original_prompt
    assert "语音转写成英文" in tokenizer.prompts[-1]
    assert "SGLang" in tokenizer.prompts[-1]
    assert second.language == "en"
    assert second.max_new_tokens == 24
    assert first.req.rid != second.req.rid
    assert torch.equal(
        first.req.multimodal_inputs.mm_items[0].precomputed_embeddings,
        second.req.multimodal_inputs.mm_items[0].precomputed_embeddings,
    )
    assert service.stats()["hits"] == 1


@pytest.mark.parametrize("num_samples", [0, 1, 480001])
def test_invalid_audio_still_fails_before_cache_can_bypass_it(
    cached_frontend: tuple[
        Mock, FunASRPreLMEncoderService, CacheTestTokenizer, np.ndarray
    ],
    monkeypatch: pytest.MonkeyPatch,
    num_samples: int,
) -> None:
    """Cached embeddings cannot turn invalid or overlong audio into a request."""
    extractor, service, tokenizer, audio = cached_frontend
    invalid_audio = np.zeros(num_samples, dtype=np.float32)
    monkeypatch.setattr(
        transcription, "load_audio", lambda source, **kwargs: invalid_audio
    )
    service.cache.put(
        f"{service.namespace}:{audio_fingerprint(invalid_audio)}", torch.ones(1, 4)
    )
    build, _ = request_builders.make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=extractor,
        audio_encoder_service=service,
    )
    with pytest.raises((AssertionError, ValueError)):
        build(
            StagePayload(
                request_id="invalid",
                request=OmniRequest(inputs={"audio_bytes": b"wav"}),
                data={},
            )
        )
    assert service.stats()["hits"] == 0
