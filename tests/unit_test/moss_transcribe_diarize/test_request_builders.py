# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
import torch

import sglang_omni.models.moss_transcribe_diarize.request_builders as request_builders
import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.moss_transcribe_diarize import request_builders
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TRANSCRIBE_DIARIZE_PROMPT,
    _audio_feature_lengths_from_waveform,
    make_moss_transcribe_diarize_scheduler_adapters,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, OmniRequest, StagePayload


def _wav_bytes(num_samples: int = 1600, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples)
    return buffer.getvalue()


class FakeTokenizer:
    vocab_size = 200000
    eos_token_id = 151645

    def __init__(self) -> None:
        self._ids = {
            "<|audio_start|>": 151669,
            "<|audio_pad|>": 151671,
            "<|audio_end|>": 151670,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return "".join(str(token_id) for token_id in token_ids)


class FastFakeFeatureExtractor:
    n_samples = 16000
    hop_length = 160
    sampling_rate = 16000

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        chunks,
        *,
        sampling_rate: int,
        padding: str,
        return_tensors: str,
    ):
        assert sampling_rate == self.sampling_rate
        assert padding == "max_length"
        assert return_tensors == "pt"
        self.calls += 1
        return {
            "input_features": torch.zeros(
                (len(chunks), 80, 100),
                dtype=torch.float32,
            )
        }


class FastFakeTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.encode_calls: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encode_calls.append(text)
        if text.endswith("<|audio_start|>"):
            return [10, self._ids["<|audio_start|>"]]
        if text.startswith("<|audio_end|>"):
            return [self._ids["<|audio_end|>"], 11]
        raise AssertionError(f"Unexpected prompt fragment: {text!r}")


class FakeProcessor:
    audio_token = "<|audio_pad|>"
    audio_token_id = 151671
    audio_merge_size = 4
    audio_tokens_per_second = 12.5
    time_marker_every_seconds = 5
    enable_time_marker = True
    digit_token_ids = {str(digit): 1000 + digit for digit in range(10)}

    def __init__(self) -> None:
        self.tokenizer = FastFakeTokenizer()
        self.feature_extractor = FastFakeFeatureExtractor()
        self.messages = None
        self.chat_template_calls = 0
        self.processor_calls = 0

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        self.chat_template_calls += 1
        prompt = messages[0]["content"][1]["text"]
        return (
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
            f"{prompt}<|im_start|>assistant\n"
        )

    def _audio_span_ids(self, audio_seq_len: int) -> list[int]:
        tokens_per_marker = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        output: list[int] = []
        consumed = 0
        duration = audio_seq_len / self.audio_tokens_per_second
        for sec in range(
            self.time_marker_every_seconds,
            int(duration) + 1,
            self.time_marker_every_seconds,
        ):
            pos = (sec // self.time_marker_every_seconds) * tokens_per_marker
            output.extend([self.audio_token_id] * (pos - consumed))
            consumed = pos
            output.extend(self.digit_token_ids[digit] for digit in str(sec))
        output.extend([self.audio_token_id] * (audio_seq_len - consumed))
        return output

    def __call__(self, **kwargs):
        del kwargs
        self.processor_calls += 1
        raise AssertionError(
            "The streamlined request path must bypass Processor.__call__"
        )


def _payload(
    prompt: str | None = None,
    params: dict | None = None,
    metadata: dict | None = None,
) -> StagePayload:
    request_params = dict(params or {})
    if prompt is not None:
        request_params["prompt"] = prompt
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"audio_bytes": _wav_bytes()},
            params=request_params,
            metadata=metadata or {"model": "moss-transcribe-diarize"},
        ),
        data={},
    )


def _payload_with_inputs(
    inputs, *, params: dict | None = None, metadata: dict | None = None
) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata=metadata or {"model": "moss-transcribe-diarize"},
        ),
        data={},
    )


TEST_CONTEXT_LENGTH = 131072


def _request_builder(
    processor: FakeProcessor | None = None,
    *,
    context_length: int = TEST_CONTEXT_LENGTH,
    audio_encoder_service=None,
):
    processor = processor or FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=32,
        context_length=context_length,
        audio_encoder_service=audio_encoder_service,
    )
    return request_builder


def test_streamlined_request_builder_preserves_exact_input_ids() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    data = request_builder(
        _payload_with_inputs({"audio_bytes": _wav_bytes(num_samples=80000)})
    )

    audio_item = data.req.multimodal_inputs.mm_items[0]
    expected_input_ids = [
        10,
        processor.tokenizer.convert_tokens_to_ids("<|audio_start|>"),
        *([audio_item.pad_value] * 62),
        processor.digit_token_ids["5"],
        *([audio_item.pad_value] * 3),
        processor.tokenizer.convert_tokens_to_ids("<|audio_end|>"),
        11,
    ]
    assert data.input_ids.tolist() == expected_input_ids
    assert data.prompt_token_ids == expected_input_ids
    assert data.req.origin_input_ids == expected_input_ids
    assert audio_item.offsets == [(2, 63), (65, 67)]
    assert audio_item.model_specific_data["audio_feature_lengths"].tolist() == [
        13,
        13,
        13,
        13,
        13,
    ]
    assert data.req.multimodal_inputs.num_image_tokens == 65
    assert processor.processor_calls == 0


def test_streamlined_request_builder_caches_default_prompt_parts() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    request_builder(_payload())
    request_builder(_payload())

    assert processor.chat_template_calls == 1
    assert len(processor.tokenizer.encode_calls) == 2
    assert processor.feature_extractor.calls == 2


def test_streamlined_request_builder_preserves_custom_messages() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {"type": "text", "text": "custom prompt"},
            ],
        }
    ]

    request_builder(
        _payload_with_inputs(
            {
                "audio_bytes": _wav_bytes(),
                "messages": messages,
            }
        )
    )

    assert processor.messages is messages
    assert processor.chat_template_calls == 2
    assert len(processor.tokenizer.encode_calls) == 4


def test_streamlined_request_builder_preserves_direct_prompt() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)
    direct_prompt = (
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "direct prompt<|im_start|>assistant\n"
    )

    request_builder(_payload(prompt=direct_prompt))

    assert processor.chat_template_calls == 1
    assert len(processor.tokenizer.encode_calls) == 4


def test_streamlined_request_builder_enforces_context_limit() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor, context_length=5)

    with pytest.raises(
        ValueError,
        match="Prompt/audio sequence exceeds max_length=5",
    ):
        request_builder(_payload())


def test_embedding_cache_hit_skips_feature_extraction_and_encode() -> None:
    class _EncoderService:
        def __init__(self) -> None:
            self.lookup: tuple[str, int] | None = None
            self.embedding = torch.zeros((2, 4))

        def lookup_cached_embedding(
            self,
            audio_fingerprint: str,
            expected_tokens: int,
        ) -> torch.Tensor:
            self.lookup = (audio_fingerprint, expected_tokens)
            return self.embedding

        def attach_embedding(self, item, embedding: torch.Tensor) -> None:
            item.precomputed_embeddings = embedding
            item.feature = None

        def encode_item(self, item) -> None:
            raise AssertionError("encoder should not run on a cache hit")

    processor = FakeProcessor()
    encoder_service = _EncoderService()
    request_builder = _request_builder(
        processor,
        audio_encoder_service=encoder_service,
    )

    data = request_builder(_payload())

    item = data.req.multimodal_inputs.mm_items[0]
    assert encoder_service.lookup == (data.req.extra_key, 2)
    assert processor.processor_calls == 0
    assert processor.feature_extractor.calls == 0
    assert item.feature is None
    assert item.precomputed_embeddings is encoder_service.embedding
    assert item.audio_feature_lengths.tolist() == [2]
    assert item.audio_chunk_mapping.tolist() == [0]


def test_embedding_cache_miss_preserves_processor_and_encoder_path() -> None:
    class _EncoderService:
        def __init__(self) -> None:
            self.lookup: tuple[str, int] | None = None
            self.encoded_feature: torch.Tensor | None = None

        def lookup_cached_embedding(
            self,
            audio_fingerprint: str,
            expected_tokens: int,
        ) -> None:
            self.lookup = (audio_fingerprint, expected_tokens)
            return None

        def attach_embedding(self, item, embedding: torch.Tensor) -> None:
            raise AssertionError("no cached embedding should be attached")

        def encode_item(self, item) -> None:
            self.encoded_feature = item.feature
            item.precomputed_embeddings = torch.zeros((3, 4))
            item.feature = None

    processor = FakeProcessor()
    encoder_service = _EncoderService()
    request_builder = _request_builder(
        processor,
        audio_encoder_service=encoder_service,
    )

    data = request_builder(_payload())

    assert encoder_service.lookup == (data.req.extra_key, 2)
    assert processor.processor_calls == 0
    assert processor.feature_extractor.calls == 1
    assert encoder_service.encoded_feature is not None
    assert data.req.multimodal_inputs.mm_items[0].feature is None


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [
        (1, [1]),
        (1280, [1]),
        (1281, [2]),
        (480000, [375]),
        (480001, [375, 1]),
        (960123, [375, 375, 1]),
    ],
)
def test_audio_feature_lengths_match_processor_chunk_boundaries(
    num_samples: int,
    expected: list[int],
) -> None:
    processor = FakeProcessor()
    processor.feature_extractor.n_samples = 480000

    assert _audio_feature_lengths_from_waveform(processor, num_samples).tolist() == (
        expected
    )


def test_request_builder_replaces_audio_tokens_with_item_pad_value() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    data = request_builder(_payload())

    input_ids = data.input_ids.tolist()
    audio_token_id = processor.audio_token_id
    assert audio_token_id not in input_ids
    audio_item = data.req.multimodal_inputs.mm_items[0]
    assert audio_item.offsets == [(2, 3)]
    assert input_ids[2] == audio_item.pad_value
    assert input_ids[3] == audio_item.pad_value
    assert data.req.sampling_params.max_new_tokens == 32


def test_request_builder_enforces_scheduler_request_limits() -> None:
    request_builder = _request_builder()

    data = request_builder(_payload())

    assert data.enforce_request_limits is True


def test_request_builder_respects_explicit_max_length_param() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="max_length=5"):
        request_builder(_payload(params={"max_length": 5}))


def test_request_builder_caps_explicit_max_length_at_context() -> None:
    request_builder = _request_builder(context_length=5)

    with pytest.raises(ValueError, match="max_length=5"):
        request_builder(_payload(params={"max_length": TEST_CONTEXT_LENGTH + 4096}))


def test_request_builder_rejects_non_positive_max_new_tokens() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="max_new_tokens must be at least 1"):
        request_builder(_payload(params={"max_new_tokens": 0}))


def test_request_builder_scales_default_output_budget_with_duration() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload_with_inputs({"audios": [_wav_bytes(num_samples=16000 * 60)]})
    )

    expected = 60 * request_builders._OUTPUT_TOKENS_PER_AUDIO_SECOND
    assert data.req.sampling_params.max_new_tokens == expected


def test_request_builder_explicit_max_new_tokens_beats_duration_budget() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload_with_inputs(
            {"audios": [_wav_bytes(num_samples=16000 * 60)]},
            params={"max_new_tokens": 64},
        )
    )

    assert data.req.sampling_params.max_new_tokens == 64


def test_request_builder_operator_default_disables_duration_budget() -> None:
    processor = FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=32,
        context_length=TEST_CONTEXT_LENGTH,
        duration_scaled_default=False,
    )

    data = request_builder(
        _payload_with_inputs({"audios": [_wav_bytes(num_samples=16000 * 60)]})
    )

    assert data.req.sampling_params.max_new_tokens == 32


def test_request_builder_caps_default_output_budget_for_short_audio() -> None:
    # Regression test for #975: with a production sized default (5120), a
    # short non-speech clip must not inherit the full fixed budget, or
    # greedy decoding can loop on it for thousands of tokens.
    processor = FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=5120,
        context_length=TEST_CONTEXT_LENGTH,
    )

    data = request_builder(
        _payload_with_inputs({"audios": [_wav_bytes(num_samples=16000 * 6)]})
    )

    assert (
        data.req.sampling_params.max_new_tokens
        == request_builders._MIN_SCALED_OUTPUT_TOKENS
    )


def test_request_builder_scaled_budget_applies_below_fixed_default() -> None:
    processor = FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=5120,
        context_length=TEST_CONTEXT_LENGTH,
    )

    data = request_builder(
        _payload_with_inputs({"audios": [_wav_bytes(num_samples=16000 * 60)]})
    )

    expected = 60 * request_builders._OUTPUT_TOKENS_PER_AUDIO_SECOND
    assert data.req.sampling_params.max_new_tokens == expected


def test_request_builder_repetition_penalty_passthrough() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={"repetition_penalty": 1.3},
            metadata={
                "model": "moss-transcribe-diarize",
                EXPLICIT_GENERATION_PARAMS_KEY: ["repetition_penalty"],
            },
        )
    )

    assert data.repetition_penalty == 1.3
    assert data.req.sampling_params.repetition_penalty == 1.3


def test_request_builder_rejects_out_of_range_repetition_penalty() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="repetition_penalty"):
        request_builder(
            _payload(
                params={"repetition_penalty": 2.5},
                metadata={
                    "model": "moss-transcribe-diarize",
                    EXPLICIT_GENERATION_PARAMS_KEY: ["repetition_penalty"],
                },
            )
        )


def test_request_builder_scaled_budget_floor_boundary() -> None:
    # 51.2s * 10 tokens/s lands exactly on the 512 floor; 52s clears it.
    processor = FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=5120,
        context_length=TEST_CONTEXT_LENGTH,
    )

    at_floor = request_builder(
        _payload_with_inputs({"audios": [_wav_bytes(num_samples=int(16000 * 51.2))]})
    )
    above_floor = request_builder(
        _payload_with_inputs({"audios": [_wav_bytes(num_samples=16000 * 52)]})
    )

    assert at_floor.req.sampling_params.max_new_tokens == 512
    assert above_floor.req.sampling_params.max_new_tokens == 520


def test_request_builder_empty_audio_budget_keeps_floor(monkeypatch) -> None:
    import dataclasses

    real_prepare = request_builders.prepare_audio

    def zero_duration(*args, **kwargs):
        return dataclasses.replace(real_prepare(*args, **kwargs), duration_s=0.0)

    monkeypatch.setattr(request_builders, "prepare_audio", zero_duration)
    processor = FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=5120,
        context_length=TEST_CONTEXT_LENGTH,
    )

    data = request_builder(_payload())

    assert data.req.sampling_params.max_new_tokens == 128


def test_request_builder_uses_moss_sampling_defaults() -> None:
    request_builder = _request_builder()

    data = request_builder(_payload())
    sampling_params = data.req.sampling_params

    assert data.temperature == DEFAULT_TEMPERATURE
    assert data.top_p == DEFAULT_TOP_P
    assert data.top_k == DEFAULT_TOP_K
    assert data.repetition_penalty == 1.0
    assert sampling_params.repetition_penalty == 1.0
    if DEFAULT_TEMPERATURE == 0.0:
        # SGLang encodes greedy sampling as temperature=1.0 with top_k=1.
        assert sampling_params.temperature == 1.0
        assert sampling_params.top_k == 1
    else:
        assert sampling_params.temperature == DEFAULT_TEMPERATURE
        assert sampling_params.top_p == DEFAULT_TOP_P
        assert sampling_params.top_k == DEFAULT_TOP_K


def test_request_builder_preserves_sampling_overrides() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 25,
            },
            metadata={
                "model": "moss-transcribe-diarize",
                EXPLICIT_GENERATION_PARAMS_KEY: ["temperature", "top_p", "top_k"],
            },
        )
    )

    assert data.temperature == 0.0
    assert data.top_p == 0.9
    assert data.top_k == 25


def test_request_builder_ignores_implicit_client_sampling_defaults() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        )
    )

    assert data.temperature == DEFAULT_TEMPERATURE
    assert data.top_p == DEFAULT_TOP_P
    assert data.top_k == DEFAULT_TOP_K


def test_request_builder_preserves_explicit_default_valued_overrides() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            metadata={
                "model": "moss-transcribe-diarize",
                EXPLICIT_GENERATION_PARAMS_KEY: ["temperature", "top_p", "top_k"],
            },
        )
    )

    assert data.temperature == 1.0
    assert data.top_p == 1.0
    assert data.top_k == -1


def test_request_builder_ignores_openai_transcription_temperature_default() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={"temperature": 0.0},
            metadata={"model": "moss-transcribe-diarize"},
        )
    )

    assert data.temperature == DEFAULT_TEMPERATURE


def test_request_builder_uses_default_prompt_for_empty_transcription_prompt() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    request_builder(_payload(prompt=""))

    assert processor.messages is not None
    assert (
        processor.messages[0]["content"][1]["text"] == DEFAULT_TRANSCRIBE_DIARIZE_PROMPT
    )


def test_request_builder_rejects_direct_waveform_list() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="Unsupported MOSS-Transcribe-Diarize"):
        request_builder(_payload_with_inputs({"audio_data": [0.0, 0.1, -0.1, 0.0]}))


def test_request_builder_accepts_single_audio_from_audios_list() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    data = request_builder(_payload_with_inputs({"audios": [_wav_bytes()]}))

    assert data.audio_duration_s == 0.1
    assert len(data.req.multimodal_inputs.mm_items) == 1


def test_request_builder_rejects_multiple_audios() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="exactly one audio"):
        request_builder(_payload_with_inputs({"audios": [_wav_bytes(), _wav_bytes()]}))


def test_request_builder_uses_default_prompt_for_bare_string_audio_source(
    monkeypatch,
) -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )

    request_builder(_payload_with_inputs("/tmp/audio.wav"))

    assert processor.messages is not None
    assert (
        processor.messages[0]["content"][1]["text"] == DEFAULT_TRANSCRIBE_DIARIZE_PROMPT
    )


def test_request_builder_uses_string_prompt_when_audio_is_supplied_separately() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    request_builder(
        _payload_with_inputs(
            "custom diarization prompt",
            metadata={
                "model": "moss-transcribe-diarize",
                "audios": [_wav_bytes()],
            },
        )
    )

    assert processor.messages is not None
    assert processor.messages[0]["content"][1]["text"] == "custom diarization prompt"
