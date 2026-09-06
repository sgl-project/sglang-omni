# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import sglang_omni.models.voxtral_asr.request_builders as request_builders
from sglang_omni.models.voxtral_asr.request_builders import (
    VoxtralASRRequestData,
    make_voxtral_asr_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int

_SAMPLE_RATE = 16000


class _FakeAudioConfig:
    def __init__(self, tokens_per_second: int = 12) -> None:
        self.tokens_per_second = tokens_per_second

    def num_audio_tokens(self, num_samples: int) -> int:
        return max(int(num_samples / _SAMPLE_RATE * self.tokens_per_second), 1)


class _FakeInstructTokenizer:
    """Mimics mistral_common's InstructTokenizer surface used by the builder."""

    def __init__(self, prompt_tokens: list[int], padded_samples: int) -> None:
        self.audio_encoder = SimpleNamespace(
            audio_config=_FakeAudioConfig(),
            audio_token=10,
        )
        self.tokenizer = SimpleNamespace(eos_id=2, n_words=32000)
        self._prompt_tokens = prompt_tokens
        self._padded = np.zeros(padded_samples, dtype=np.float32)
        self.transcription_requests: list[object] = []

    def encode_transcription(self, transcription_request):
        self.transcription_requests.append(transcription_request)
        return SimpleNamespace(
            tokens=list(self._prompt_tokens),
            audios=[SimpleNamespace(audio_array=self._padded)],
        )


class _FakeTokenizer:
    def __init__(self, prompt_tokens: list[int], padded_samples: int) -> None:
        self.instruct_tokenizer = _FakeInstructTokenizer(prompt_tokens, padded_samples)
        self.decode_calls: list[list[int]] = []

    def decode(self, token_ids, special_token_policy=None) -> str:
        self.decode_calls.append(list(token_ids))
        return "decoded transcript"


def _make_builder(
    prompt_tokens: list[int] | None = None,
    padded_samples: int = _SAMPLE_RATE * 4,
    max_new_tokens: int = 4096,
):
    prompt_tokens = [1] + [10] * 38 if prompt_tokens is None else prompt_tokens
    tokenizer = _FakeTokenizer(prompt_tokens, padded_samples)
    request_builder, result_adapter = make_voxtral_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer, request_builder, result_adapter


def _make_payload(**inputs) -> StagePayload:
    if not inputs:
        inputs = {"audio_bytes": b"wav-bytes"}
    return StagePayload(
        request_id="req-voxtral-asr",
        request=OmniRequest(inputs=inputs),
        data={},
    )


@pytest.fixture(autouse=True)
def _fake_audio_io(monkeypatch):
    waveform = np.arange(_SAMPLE_RATE * 2, dtype=np.float32) / _SAMPLE_RATE
    monkeypatch.setattr(request_builders, "_load_audio", lambda source: waveform)
    return waveform


def test_audio_fingerprint_is_stable_and_content_sensitive() -> None:
    a = np.random.RandomState(0).randn(_SAMPLE_RATE).astype(np.float32)
    b = a.copy()
    c = a.copy()
    c[0] += 1e-3

    fp_a, fp_b, fp_c = map(audio_fingerprint, (a, b, c))

    assert fp_a == fp_b
    assert audio_fingerprint_int(fp_a) == audio_fingerprint_int(fp_b)
    assert fp_a != fp_c
    # Non-contiguous / non-float32 inputs hash identically to their float32 form.
    assert audio_fingerprint(a[::2]) == audio_fingerprint(
        np.ascontiguousarray(a[::2], dtype=np.float32)
    )


def test_request_builder_fuses_audio_over_whole_prompt(monkeypatch) -> None:
    waveform = np.arange(_SAMPLE_RATE * 2, dtype=np.float32) / _SAMPLE_RATE
    monkeypatch.setattr(request_builders, "_load_audio", lambda source: waveform)
    prompt_tokens = [1] + [10] * 38
    _, request_builder, _ = _make_builder(prompt_tokens=prompt_tokens)

    data = request_builder(_make_payload())

    assert data.prompt_token_ids == prompt_tokens
    # Realtime fusion covers the entire prompt span (no placeholder tokens).
    assert data.audio_offsets == [(0, len(prompt_tokens) - 1)]
    assert data.audio_duration_s == pytest.approx(2.0)

    mm_items = data.req.multimodal_inputs.mm_items
    assert len(mm_items) == 1
    item = mm_items[0]
    expected_fp = audio_fingerprint(waveform)
    # The radix-cache key must be derived from the audio content so that two
    # requests sharing the same prompt never alias to each other's audio.
    assert item.hash == audio_fingerprint_int(expected_fp)
    assert item.offsets == [(0, len(prompt_tokens) - 1)]


def test_request_builder_bounds_max_new_tokens_by_audio_length() -> None:
    prompt_tokens = [1] + [10] * 38
    _, request_builder, _ = _make_builder(
        prompt_tokens=prompt_tokens,
        padded_samples=_SAMPLE_RATE * 4,  # 48 audio tokens via fake config
        max_new_tokens=4096,
    )

    data = request_builder(_make_payload())

    audio_bounded_max = 48 - len(prompt_tokens) - 1
    assert data.max_new_tokens == audio_bounded_max
    assert data.req.sampling_params.max_new_tokens == audio_bounded_max
    # temperature=0 is canonicalized by SamplingParams into top_k=1 (greedy).
    assert data.req.sampling_params.top_k == 1


def test_request_builder_preserves_explicit_sampling_params() -> None:
    _, request_builder, _ = _make_builder()

    payload = _make_payload()
    payload.request.params = {"temperature": 0.7, "max_new_tokens": 5}

    data = request_builder(payload)

    assert data.req.sampling_params.temperature == pytest.approx(0.7)
    # Caller-provided max_new_tokens wins over the stage default but is still
    # bounded by the audio length.
    assert data.max_new_tokens == 5


def test_request_builder_max_new_tokens_never_below_one() -> None:
    prompt_tokens = [1] + [10] * 200  # longer than the audio token budget
    _, request_builder, _ = _make_builder(
        prompt_tokens=prompt_tokens,
        padded_samples=_SAMPLE_RATE,  # 12 audio tokens via fake config
        max_new_tokens=4096,
    )

    data = request_builder(_make_payload())

    assert data.max_new_tokens == 1


def test_request_builder_rejects_empty_prompt() -> None:
    _, request_builder, _ = _make_builder(prompt_tokens=[])

    with pytest.raises(ValueError, match="Empty prompt"):
        request_builder(_make_payload())


def test_request_builder_prefers_params_language_then_metadata() -> None:
    _, request_builder, _ = _make_builder()

    payload = _make_payload()
    payload.request.params = {"language": "en"}
    payload.request.metadata = {"asr_params": {"language": "fr"}}
    assert request_builder(payload).language == "en"

    payload = _make_payload()
    payload.request.params = {}
    payload.request.metadata = {"asr_params": {"language": "fr"}}
    assert request_builder(payload).language == "fr"


def test_result_adapter_returns_text_and_usage() -> None:
    tokenizer, _, result_adapter = _make_builder()
    payload = _make_payload()
    data = VoxtralASRRequestData(
        output_ids=[5, 6, 7],
        audio_duration_s=2.5,
        language="en",
    )
    data.stage_payload = payload

    result = result_adapter(data)

    assert result.request_id == payload.request_id
    assert result.data["text"] == "decoded transcript"
    assert result.data["language"] == "en"
    assert result.data["duration_s"] == 2.5
    assert result.data["usage"] == {"output_tokens": 3}
    assert result.data["modality"] == "text"
    assert tokenizer.decode_calls == [[5, 6, 7]]
