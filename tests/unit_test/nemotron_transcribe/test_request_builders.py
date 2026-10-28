# SPDX-License-Identifier: Apache-2.0
"""Request and response contracts without checkpoint tensors."""

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BatchEncoding

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.nemotron_transcribe.hf_config import (
    AUDIO_TOKEN_ID,
    EOS_TOKEN_ID,
)
from sglang_omni.models.nemotron_transcribe.request_builders import (
    DEFAULT_PROMPT,
    make_scheduler_adapters,
    make_stream_output_builder,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import DeferredAdmission
from sglang_omni.serve.openai_errors import generation_error_status_code


class FakeTokenizer:
    def __init__(self) -> None:
        self.prompt_token_ids = [101, AUDIO_TOKEN_ID, 102]
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt and not enable_thinking
        self.messages = messages
        return "rendered prompt"

    def __call__(self, text: str, *, add_special_tokens: bool) -> BatchEncoding:
        assert text == "rendered prompt" and not add_special_tokens
        return BatchEncoding({"input_ids": list(self.prompt_token_ids)})

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        vocabulary = {7: "Hello", 8: " world", EOS_TOKEN_ID: "<eos>"}
        return "".join(vocabulary[token_id] for token_id in token_ids)


def load_test_audio(source: object, **decode_options: object) -> np.ndarray:
    return np.zeros(16000, dtype=np.float32)


def make_payload(parameters: dict[str, object] | None = None) -> StagePayload:
    return StagePayload(
        request_id="nemotron-test",
        request=OmniRequest(
            inputs={"audio_bytes": b"waveform"},
            params=parameters or {},
        ),
        data={},
    )


@pytest.fixture(autouse=True)
def patch_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription, "load_audio", load_test_audio)


def test_audio_placeholders_and_offsets_match_encoded_duration() -> None:
    tokenizer = FakeTokenizer()
    build_request, _ = make_scheduler_adapters(tokenizer=tokenizer, max_new_tokens=32)
    request = build_request(make_payload())
    assert request.req is not None
    audio = request.req.multimodal_inputs.mm_items[0]
    assert audio.offsets == [(1, 13)]
    assert request.input_ids.numel() == 15
    assert request.req.origin_input_ids[1:14] == [audio.pad_value] * 13
    assert request.audio_duration_s == 1.0
    assert audio.feature is not None
    assert audio.num_audio_tokens == 13
    assert tokenizer.messages == [
        {"role": "user", "content": f"<|audio|>\n{DEFAULT_PROMPT}"}
    ]


def test_request_overrides_prompt_and_generation_budget() -> None:
    tokenizer = FakeTokenizer()
    build_request, _ = make_scheduler_adapters(tokenizer=tokenizer, max_new_tokens=32)
    request = build_request(
        make_payload(
            {
                "prompt": "Transcribe exactly.",
                "system_prompt": "Be concise.",
                "max_new_tokens": 64,
            }
        )
    )
    assert request.max_new_tokens == 64
    assert request.req.sampling_params.max_new_tokens == 64
    assert tokenizer.messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "<|audio|>\nTranscribe exactly."},
    ]


@pytest.mark.parametrize("token_ids", [[101, 102], [AUDIO_TOKEN_ID, AUDIO_TOKEN_ID]])
def test_prompt_requires_one_audio_marker(token_ids: list[int]) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.prompt_token_ids = token_ids
    build_request, _ = make_scheduler_adapters(tokenizer=tokenizer, max_new_tokens=32)
    with pytest.raises(ValueError, match="exactly one") as raised:
        build_request(make_payload())
    assert generation_error_status_code(raised.value) == 400


def test_audio_duration_limit_is_enforced_before_scheduling() -> None:
    build_request, _ = make_scheduler_adapters(
        tokenizer=FakeTokenizer(), max_new_tokens=32, max_audio_seconds=0.5
    )
    with pytest.raises(ValueError, match="accepts audio up to"):
        build_request(make_payload())


def test_audio_too_short_for_normalization_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_short_audio(source: object, **decode_options: object) -> np.ndarray:
        return np.zeros(319, dtype=np.float32)

    monkeypatch.setattr(transcription, "load_audio", load_short_audio)
    build_request, _ = make_scheduler_adapters(
        tokenizer=FakeTokenizer(), max_new_tokens=32
    )
    with pytest.raises(ValueError, match="two valid mel frames") as raised:
        build_request(make_payload())
    assert generation_error_status_code(raised.value) == 400


def test_result_strips_eos_and_preserves_usage() -> None:
    build_request, adapt_result = make_scheduler_adapters(
        tokenizer=FakeTokenizer(), max_new_tokens=32
    )
    request = build_request(make_payload({"language": "en"}))
    request.output_ids = [7, 8, EOS_TOKEN_ID]
    request.finish_reason = "length"
    result = adapt_result(request)
    assert result.request_id == "nemotron-test"
    assert result.data["text"] == "Hello world"
    assert result.data["language"] == "en"
    assert result.data["duration_s"] == 1.0
    assert result.data["finish_reason"] == "length"
    assert result.data["usage"]["completion_tokens"] == 2
    assert result.data["usage"]["prompt_tokens"] == 15


def test_streamed_text_matches_final_response() -> None:
    tokenizer = FakeTokenizer()
    build_request, adapt_result = make_scheduler_adapters(
        tokenizer=tokenizer, max_new_tokens=32
    )
    request = build_request(make_payload({"stream": True}))
    stream = make_stream_output_builder(
        tokenizer, min_emit_interval_s=0.0, eos_token_id=EOS_TOKEN_ID
    )
    messages = []
    for token_id in [7, 8, EOS_TOKEN_ID]:
        messages.extend(
            stream(
                request.stage_payload.request_id,
                request,
                SimpleNamespace(data=token_id),
            )
        )
    request.output_ids = [7, 8, EOS_TOKEN_ID]
    assert (
        "".join(message.data["text"] for message in messages)
        == adapt_result(request).data["text"]
    )


class RecordingEncoder:
    def __init__(self, embedding: torch.Tensor | None = None) -> None:
        self.embedding = embedding
        self.lookup: tuple[str, int] | None = None
        self.encoded: object | None = None
        self.waited = False

    def lookup_cached_embedding(
        self, audio_fingerprint: str, expected_tokens: int
    ) -> torch.Tensor | None:
        self.lookup = (audio_fingerprint, expected_tokens)
        return self.embedding

    def attach_embedding(self, item: object, embedding: torch.Tensor) -> None:
        item.precomputed_embeddings = embedding
        item.feature = None

    def encode_item(self, item: object) -> None:
        self.waited = True
        self.encoded = item.feature
        item.precomputed_embeddings = torch.zeros((item.num_audio_tokens, 4))
        item.feature = None

    def submit_item(self, item: object) -> concurrent.futures.Future[torch.Tensor]:
        self.encoded = item.feature
        item.precomputed_embeddings = torch.zeros((item.num_audio_tokens, 4))
        item.feature = None
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        future.set_result(item.precomputed_embeddings)
        return future


def test_cached_embedding_is_attached_before_admission() -> None:
    encoder = RecordingEncoder(torch.ones((13, 4)))
    build_request, _ = make_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        audio_encoder_service=encoder,
    )
    request = build_request(make_payload())
    audio = request.req.multimodal_inputs.mm_items[0]
    assert encoder.lookup == (request.req.extra_key, 13)
    assert audio.feature is None
    assert audio.precomputed_embeddings is encoder.embedding
    assert encoder.encoded is None


def test_cache_miss_defers_admission_until_encode_finishes() -> None:
    encoder = RecordingEncoder()
    build_request, _ = make_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        audio_encoder_service=encoder,
    )
    result = build_request(make_payload())
    assert isinstance(result, DeferredAdmission)
    assert result.ready.done()
    assert encoder.encoded is not None
    assert result.value.req.multimodal_inputs.mm_items[0].feature is None
    assert result.value.req.multimodal_inputs.mm_items[
        0
    ].precomputed_embeddings.shape == (
        13,
        4,
    )


def test_cache_miss_waits_in_the_builder_when_the_pool_has_room() -> None:
    encoder = RecordingEncoder()

    def submit_item(item: object) -> concurrent.futures.Future[torch.Tensor]:
        raise AssertionError("the builder should wait on encode_item")

    encoder.submit_item = submit_item
    build_request, _ = make_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        audio_encoder_service=encoder,
        should_wait_for_encode=lambda: True,
    )
    request = build_request(make_payload())
    assert encoder.waited is True
    assert request.req.multimodal_inputs.mm_items[0].feature is None
