# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3PreparedRequest,
    CosyVoice3SGLangRequestData,
    _CosyVoice3ReferenceEncodeHook,
    apply_sglang_cosyvoice3_result,
    build_cosyvoice3_state,
    build_generation_kwargs,
    build_sglang_cosyvoice3_request,
    cleanup_prepared_cosyvoice3_request,
    clear_cosyvoice3_preprocessing_context,
    pop_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
    set_cosyvoice3_preprocessing_context,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


@pytest.fixture(autouse=True)
def stable_sampling_params(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep scheduler construction independent of the installed SGLang tokenizer."""
    monkeypatch.setattr(
        request_builders.SamplingParams, "normalize", lambda self, tokenizer: None
    )
    monkeypatch.setattr(
        request_builders.SamplingParams, "verify", lambda self, vocab_size: None
    )
    clear_cosyvoice3_preprocessing_context()
    yield
    clear_cosyvoice3_preprocessing_context()


def _payload(
    inputs: Any,
    *,
    params: dict[str, Any] | None = None,
    tts_params: dict[str, Any] | None = None,
    request_id: str = "req-cosy",
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_build_cosyvoice3_state_merges_input_params_and_tts_metadata() -> None:
    state = build_cosyvoice3_state(
        _payload(
            {"text": "你好", "ref_audio": "reference.wav", "ref_text": "原文"},
            params={"language": "en", "stream": True, "speed": 1.1, "seed": 4},
            tts_params={"language": "zh", "max_new_tokens": 64},
        )
    )

    assert state.text == "你好"
    assert state.ref_audio == "reference.wav"
    assert state.ref_text == "原文"
    assert state.language == "zh"
    assert state.instructions is None
    assert state.stream is True
    assert state.speed == 1.1
    assert state.seed == 4
    assert state.generation_kwargs == {"max_new_tokens": 2048}


def test_build_cosyvoice3_state_accepts_plain_text_input() -> None:
    state = build_cosyvoice3_state(_payload("hello", params={"language": ""}))
    assert state.text == "hello"
    assert state.language == "auto"
    assert state.ref_audio is None
    assert state.ref_text is None


def test_build_cosyvoice3_state_accepts_structured_reference() -> None:
    state = build_cosyvoice3_state(
        _payload(
            {
                "text": "hello",
                "references": [{"audio_path": "reference.wav", "text": "reference"}],
            }
        )
    )

    assert state.ref_audio == "reference.wav"
    assert state.ref_text == "reference"


def test_build_cosyvoice3_state_rejects_multiple_references() -> None:
    with pytest.raises(ValueError, match="exactly one reference"):
        build_cosyvoice3_state(
            _payload(
                {
                    "text": "hello",
                    "references": [{"audio_path": "a.wav"}, {"audio_path": "b.wav"}],
                }
            )
        )


def test_build_cosyvoice3_state_rejects_reference_text_and_instructions() -> None:
    with pytest.raises(ValueError, match="either ref_text or instructions"):
        build_cosyvoice3_state(
            _payload(
                {
                    "text": "hello",
                    "ref_audio": "reference.wav",
                    "ref_text": "reference",
                },
                tts_params={"instructions": "speak warmly"},
            )
        )


@pytest.mark.parametrize(
    ("public_seed", "normalized_seed"),
    [(7, 7), (-1, 0x7FFFFFFF), (0x80000000, 0)],
)
def test_build_cosyvoice3_state_normalizes_sampling_seed(
    public_seed: int, normalized_seed: int
) -> None:
    state = build_cosyvoice3_state(_payload("hello", params={"seed": public_seed}))
    assert state.seed == normalized_seed


def test_build_cosyvoice3_state_rejects_non_integer_sampling_seed() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        build_cosyvoice3_state(_payload("hello", params={"seed": 1.5}))


def test_cosyvoice3_reference_cache_reuses_audio_derived_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"audio_16k": 0, "audio_24k": 0, "speech": 0, "speaker": 0}

    def load_16k(source: Any) -> np.ndarray:
        calls["audio_16k"] += 1
        return np.zeros(1600, dtype=np.float32)

    def load_24k(source: Any) -> np.ndarray:
        calls["audio_24k"] += 1
        return np.zeros(2400, dtype=np.float32)

    class _CountingSpeechTokenizer(_FakeSpeechTokenizer):
        def extract_speech_token(self, audio, sample_rate):
            calls["speech"] += 1
            return super().extract_speech_token(audio, sample_rate)

    class _CountingSpeakerEncoder(_FakeSpeakerEncoder):
        def extract_embedding(self, audio, sample_rate):
            calls["speaker"] += 1
            return super().extract_embedding(audio, sample_rate)

    monkeypatch.setattr(request_builders, "_load_prompt_audio", load_16k)
    monkeypatch.setattr(request_builders, "_load_prompt_audio_24k", load_24k)
    monkeypatch.setattr(
        request_builders,
        "extract_prompt_speech_feat",
        lambda audio, sample_rate: torch.ones(1, 4, 80),
    )

    hook = _CosyVoice3ReferenceEncodeHook(model=_FakeModel(), model_revision="test")
    hook.bind_encoders(
        speech_tokenizer=_CountingSpeechTokenizer(),
        speaker_encoder=_CountingSpeakerEncoder(),
    )
    service = ReferenceEncodeService(hook, max_items=4, max_bytes=1 << 20)

    first = service.get_or_encode(b"same-reference")
    second = service.get_or_encode(b"same-reference")

    assert calls == {"audio_16k": 1, "audio_24k": 1, "speech": 1, "speaker": 1}
    assert service.stats()["hits"] == 1
    assert torch.equal(first.flow_prompt_speech_feat, second.flow_prompt_speech_feat)
    first.flow_prompt_speech_feat[0, 0, 0] = 99.0
    assert second.flow_prompt_speech_feat[0, 0, 0] == 1.0


def test_cosyvoice3_reference_cache_normalizes_equivalent_data_uris() -> None:
    first = "data:audio/wav;base64,QUJD"
    second = "data:audio/x-wav;base64,QUJD"

    assert request_builders._cosyvoice3_reference_input_key(first) == (
        request_builders._cosyvoice3_reference_input_key(second)
    )
    assert (
        request_builders._cosyvoice3_reference_input_key(
            "https://example.com/reference.wav"
        )
        is None
    )


def test_cosyvoice3_reference_cache_separates_audio_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio_24k",
        lambda source: np.zeros(2400, dtype=np.float32),
    )
    monkeypatch.setattr(
        request_builders,
        "extract_prompt_speech_feat",
        lambda audio, sample_rate: torch.ones(1, 4, 80),
    )

    class _SpeechTokenizer(_FakeSpeechTokenizer):
        def extract_speech_token(self, audio, sample_rate):
            nonlocal calls
            calls += 1
            return super().extract_speech_token(audio, sample_rate)

    hook = _CosyVoice3ReferenceEncodeHook(model=_FakeModel(), model_revision="test")
    hook.bind_encoders(
        speech_tokenizer=_SpeechTokenizer(),
        speaker_encoder=_FakeSpeakerEncoder(),
    )
    service = ReferenceEncodeService(hook, max_items=4, max_bytes=1 << 20)

    service.get_or_encode(b"reference-a")
    service.get_or_encode(b"reference-b")

    assert calls == 2
    assert service.stats()["hits"] == 0
    assert service.stats()["entries"] == 2


def test_cosyvoice3_reference_cache_retries_after_encode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio_24k",
        lambda source: np.zeros(2400, dtype=np.float32),
    )
    monkeypatch.setattr(
        request_builders,
        "extract_prompt_speech_feat",
        lambda audio, sample_rate: torch.ones(1, 4, 80),
    )

    class _RetrySpeechTokenizer(_FakeSpeechTokenizer):
        def extract_speech_token(self, audio, sample_rate):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary tokenizer failure")
            return super().extract_speech_token(audio, sample_rate)

    hook = _CosyVoice3ReferenceEncodeHook(model=_FakeModel(), model_revision="test")
    hook.bind_encoders(
        speech_tokenizer=_RetrySpeechTokenizer(),
        speaker_encoder=_FakeSpeakerEncoder(),
    )
    service = ReferenceEncodeService(hook, max_items=4, max_bytes=1 << 20)

    with pytest.raises(RuntimeError, match="temporary tokenizer failure"):
        service.get_or_encode(b"retry-reference")
    artifact = service.get_or_encode(b"retry-reference")

    assert artifact.llm_prompt_speech_token.tolist() == [[40, 41]]
    assert service.stats()["failed"] == 1
    assert service.stats()["entries"] == 1


def test_generation_kwargs_omit_implicit_sampling_defaults_but_keep_explicit_values() -> (
    None
):
    params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "do_sample": False,
        "max_new_tokens": 12,
    }
    assert build_generation_kwargs(params, tts_params={}) == {
        "do_sample": False,
        "max_new_tokens": 12,
    }
    assert build_generation_kwargs(
        params, tts_params={"explicit_generation_params": ["temperature", "top_p"]}
    ) == {
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_new_tokens": 12,
    }


class _FakeTokenizer:
    def __call__(self, text: str) -> torch.Tensor:
        if text == "hello":
            return torch.tensor([[1, 2]], dtype=torch.int32)
        if text == "You are a helpful assistant.<|endofprompt|>":
            return torch.tensor([[3]], dtype=torch.int32)
        assert text == "You are a helpful assistant.<|endofprompt|>reference"
        return torch.tensor([[3, 4, 5]], dtype=torch.int32)


class _FakeSpeechTokenizer:
    def extract_speech_token(self, audio, sample_rate):
        assert sample_rate == 16000
        assert audio.shape == (1600,)
        return torch.tensor([[40, 41]], dtype=torch.int32)


class _FakeSpeakerEncoder:
    def extract_embedding(self, audio, sample_rate):
        assert sample_rate == 16000
        return torch.full((1, 192), 2.0)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = SimpleNamespace(hidden_size=4)
        self.text_embed_tokens = (
            lambda tokens: tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
        )
        self.speech_embedding = (
            lambda tokens: tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
        )


def test_preprocess_and_build_request_share_prepared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio",
        lambda source: torch.zeros(1600).numpy(),
    )
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio_24k",
        lambda source: torch.zeros(2400).numpy(),
    )
    monkeypatch.setattr(
        request_builders,
        "extract_prompt_speech_feat",
        lambda audio, sample_rate: torch.ones(1, 2, 80),
    )
    model = _FakeModel()
    set_cosyvoice3_preprocessing_context(
        model=model,
        tokenizer=_FakeTokenizer(),
        speech_tokenizer=_FakeSpeechTokenizer(),
        speaker_encoder=_FakeSpeakerEncoder(),
    )
    payload = _payload(
        {"text": "hello", "ref_audio": "reference.wav"},
        params={"max_new_tokens": 5, "do_sample": False, "seed": 7},
    )

    prepared_payload = preprocess_cosyvoice3_payload(payload)
    assert (
        prepared_payload.data[request_builders._COSYVOICE3_PREPARED_MARKER]
        == "req-cosy"
    )
    assert prepared_payload.data["flow_prompt_speech_token"] == [[40]]
    assert prepared_payload.data["flow_prompt_speech_feat"] == [[[1.0] * 80] * 2]
    assert prepared_payload.data["flow_embedding"] == [[2.0] * 192]

    prepared = request_builders.pop_prepared_cosyvoice3_request(prepared_payload)
    assert prepared is not None
    # [SOS] + [cross-lingual prefix: 1] + [target text: 2] + [TASK].
    assert prepared.input_ids.numel() == 5
    assert prepared.llm_prompt_speech_token.shape == (1, 0)
    assert prepared.flow_prompt_speech_token.tolist() == [[40]]

    request_builders._PREPARED_REQUESTS["req-cosy"] = prepared
    request_data = build_sglang_cosyvoice3_request(prepared_payload, model=model)
    assert request_data.max_new_tokens == 5
    assert request_data.temperature == 0.0
    assert request_data.req.sampling_params.sampling_seed == 7
    assert request_data.req.sampling_params.stop_token_ids == set(
        range(request_builders.VOCAB_SIZE, request_builders.TOTAL_VOCAB_SIZE)
    )
    assert request_data.req._input_embeds_are_projected is True
    with pytest.raises(RuntimeError, match="state is missing"):
        build_sglang_cosyvoice3_request(prepared_payload, model=model)


def test_preprocess_includes_reference_text_before_target_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio",
        lambda source: torch.zeros(1600).numpy(),
    )
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio_24k",
        lambda source: torch.zeros(2400).numpy(),
    )
    monkeypatch.setattr(
        request_builders,
        "extract_prompt_speech_feat",
        lambda audio, sample_rate: torch.ones(1, 2, 80),
    )
    model = _FakeModel()
    set_cosyvoice3_preprocessing_context(
        model=model,
        tokenizer=_FakeTokenizer(),
        speech_tokenizer=_FakeSpeechTokenizer(),
        speaker_encoder=_FakeSpeakerEncoder(),
    )
    payload = _payload(
        {
            "text": "hello",
            "ref_audio": "reference.wav",
            "ref_text": "reference",
        },
        params={"do_sample": False},
        request_id="req-ref-text",
    )

    prepared_payload = preprocess_cosyvoice3_payload(payload)
    prepared = request_builders.pop_prepared_cosyvoice3_request(prepared_payload)
    assert prepared is not None
    # [SOS] + [prompt text: 3] + [target text: 2] + [TASK] + [speech prompt: 2].
    assert prepared.input_ids.numel() == 9
    assert prepared.llm_prompt_speech_token.tolist() == [[40, 41]]


def test_preprocess_excludes_reference_speech_from_instruct_llm_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio",
        lambda source: torch.zeros(1600).numpy(),
    )
    monkeypatch.setattr(
        request_builders,
        "_load_prompt_audio_24k",
        lambda source: torch.zeros(2400).numpy(),
    )
    monkeypatch.setattr(
        request_builders,
        "extract_prompt_speech_feat",
        lambda audio, sample_rate: torch.ones(1, 2, 80),
    )

    class _InstructTokenizer(_FakeTokenizer):
        def __call__(self, text: str) -> torch.Tensor:
            if text == "You are a helpful assistant. speak warmly<|endofprompt|>":
                return torch.tensor([[3, 4, 5]], dtype=torch.int32)
            return super().__call__(text)

    model = _FakeModel()
    set_cosyvoice3_preprocessing_context(
        model=model,
        tokenizer=_InstructTokenizer(),
        speech_tokenizer=_FakeSpeechTokenizer(),
        speaker_encoder=_FakeSpeakerEncoder(),
    )
    payload = _payload(
        {"text": "hello", "ref_audio": "reference.wav"},
        tts_params={"instructions": "speak warmly"},
        request_id="req-instruct",
    )

    prepared_payload = preprocess_cosyvoice3_payload(payload)
    prepared = request_builders.pop_prepared_cosyvoice3_request(prepared_payload)
    assert prepared is not None
    # [SOS] + [instruction: 3] + [target text: 2] + [TASK].
    assert prepared.input_ids.numel() == 7
    assert prepared.llm_prompt_speech_token.shape == (1, 0)
    assert prepared.flow_prompt_speech_token.tolist() == [[40]]


def test_result_adapter_preserves_reference_conditioning_for_vocoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FunCosyVoice3State(
        text="hello",
        speed=1.25,
        flow_embedding=torch.ones(1, 192),
        flow_prompt_speech_token=torch.tensor([[40, 41]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
    )
    payload = StagePayload(
        request_id="req-result",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    data = CosyVoice3SGLangRequestData(
        input_ids=torch.zeros(7, dtype=torch.long),
        output_codes=[torch.tensor([50]), torch.tensor([51])],
        stage_payload=payload,
        engine_start_s=10.0,
    )
    monkeypatch.setattr(request_builders.time, "perf_counter", lambda: 10.5)

    result = apply_sglang_cosyvoice3_result(payload, data)
    restored = FunCosyVoice3State.from_dict(result.data)

    assert restored.speed == 1.25
    assert restored.flow_embedding == [[1.0] * 192]
    assert restored.flow_prompt_speech_token == [[40, 41]]
    assert restored.flow_prompt_speech_feat[0][0] == [1.0] * 80
    assert restored.audio_codes == [[50], [51]]
    assert restored.prompt_tokens == 7
    assert restored.completion_tokens == 2
    assert restored.sample_rate == 24000
    assert restored.engine_time_s == pytest.approx(0.5)


def test_result_adapter_serializes_empty_generation_without_losing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FunCosyVoice3State(text="hello", flow_embedding=torch.ones(1, 192))
    payload = StagePayload(
        request_id="req-empty-result",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    data = CosyVoice3SGLangRequestData(
        output_codes=[], stage_payload=payload, engine_start_s=20.0
    )
    monkeypatch.setattr(request_builders.time, "perf_counter", lambda: 20.25)

    result = apply_sglang_cosyvoice3_result(payload, data)
    restored = FunCosyVoice3State.from_dict(result.data)

    assert restored.audio_codes == []
    assert restored.flow_embedding == [[1.0] * 192]
    assert restored.completion_tokens == 0
    assert restored.engine_time_s == pytest.approx(0.25)


def test_preprocessing_abort_cleans_prepared_request() -> None:
    from sglang_omni.models.fun_cosyvoice3 import stages

    request_id = "req-preprocess-abort"
    with request_builders._PREPARED_REQUESTS_LOCK:
        request_builders._PREPARED_REQUESTS[request_id] = object()

    scheduler = stages.create_preprocessing_executor("model")
    scheduler.abort(request_id)

    with request_builders._PREPARED_REQUESTS_LOCK:
        assert request_id not in request_builders._PREPARED_REQUESTS


def test_preprocessing_abort_race_cleans_late_prepared_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.fun_cosyvoice3 import stages

    request_id = "req-preprocess-race"
    started = threading.Event()
    release = threading.Event()

    def delayed_preprocess(payload: StagePayload) -> StagePayload:
        started.set()
        assert release.wait(timeout=2.0)
        with request_builders._PREPARED_REQUESTS_LOCK:
            request_builders._PREPARED_REQUESTS[payload.request_id] = object()
        return payload

    monkeypatch.setattr(stages, "preprocess_cosyvoice3_payload", delayed_preprocess)
    scheduler = stages.create_preprocessing_executor("model")
    payload = _payload("hello", request_id=request_id)
    loop = asyncio.new_event_loop()
    errors: list[Exception] = []

    def run_preprocessing() -> None:
        try:
            scheduler._run_single(
                IncomingMessage(
                    request_id=request_id, type="new_request", data=payload
                ),
                loop,
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_preprocessing)
    try:
        thread.start()
        assert started.wait(timeout=2.0)
        scheduler.abort(request_id)
        release.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert errors == []
        assert scheduler.outbox.empty()
        with request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in request_builders._PREPARED_REQUESTS
    finally:
        release.set()
        thread.join(timeout=2.0)
        loop.close()
        cleanup_prepared_cosyvoice3_request(request_id)


def test_prepared_request_cleanup_and_missing_marker_are_explicit() -> None:
    payload = _payload("hello", request_id="req-cleanup")
    request_builders._PREPARED_REQUESTS["req-cleanup"] = CosyVoice3PreparedRequest(
        state=FunCosyVoice3State(text="hello"),
        input_ids_list=[1],
        input_ids=torch.tensor([1]),
        prompt_input_embeds=torch.ones(1, 4),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 0, 80),
        flow_embedding=torch.zeros(0, 192),
        gen_kwargs={"max_new_tokens": 1},
    )
    cleanup_prepared_cosyvoice3_request("req-cleanup")
    assert pop_prepared_cosyvoice3_request(payload) is None

    marked = StagePayload(
        payload.request_id, payload.request, {"_cosyvoice3_prepared_request": "missing"}
    )
    with pytest.raises(RuntimeError, match="state is missing"):
        pop_prepared_cosyvoice3_request(marked)
