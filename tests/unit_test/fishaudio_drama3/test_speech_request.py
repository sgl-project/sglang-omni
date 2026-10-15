# SPDX-License-Identifier: Apache-2.0
"""Drama 3 speech requests: serving's lowering in, prompt rows and sampling out."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE, encode_wav
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.request_builders import (
    Drama3GenerationInput,
    Drama3RequestBuilder,
)
from sglang_omni.models.fishaudio_drama3.speech_request import Drama3SpeechPreprocessor
from sglang_omni.models.fishaudio_s2_pro.stages import (
    FishReferenceEncodeHook,
    FishReferenceInput,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from sglang_omni.serve.openai_api import build_rollout_generate_request
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import RolloutGenerateRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.speech_voices import SpeakerSampleStore
from tests.unit_test.fishaudio_drama3.test_hf_config import (
    audio_decoder_config_dict,
    fish_config_dict,
    fish_text_config_dict,
)
from tests.unit_test.fishaudio_drama3.test_mtp_integration import (
    SERVING_PROMPT_CAP_ROWS,
)
from tests.unit_test.fishaudio_drama3.test_prompt import (
    CODEBOOK_SIZE,
    DEFAULT_SYSTEM_TURN,
    END,
    NUM_CODEBOOKS,
    REFERENCE_INSTRUCTION,
    SYSTEM,
    USER,
    VOICE_TURN,
    build_toy_tokenizer,
    clip_codes,
    expected_rows,
    reference_header,
)

CONTEXT_LENGTH = 1024
MAX_NEW_TOKENS_CAP = 16
# The voice store takes samples of at least one second.
REFERENCE_WAV = encode_wav(
    np.full(DEFAULT_SAMPLE_RATE * 6 // 5, 0.1, dtype=np.float32), DEFAULT_SAMPLE_RATE
)
GENERATION_FIELDS = (
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "seed",
)


class FixedCodesEncodeHook(FishReferenceEncodeHook):
    """The shared Fish encode hook over a codec that turns any audio into audio_codes;
    encoded records what it was asked to encode."""

    def __init__(self, audio_codes: torch.Tensor) -> None:
        super().__init__(
            codec=SimpleNamespace(sample_rate=DEFAULT_SAMPLE_RATE),
            checkpoint_id="codec.pth",
        )
        self.audio_codes = audio_codes
        self.encoded: list[FishReferenceInput] = []

    def encode_one(self, item: FishReferenceInput) -> torch.Tensor:
        self.encoded.append(item)
        return self.audio_codes


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return build_toy_tokenizer()


def toy_config(tokenizer: Tokenizer) -> FishOmniConfig:
    """The small test model resized to the toy tokenizer: ten codebooks of 4096 codes
    on its semantic tokens, eos at <|im_end|>."""
    semantic_begin_id = tokenizer.token_to_id("<|semantic:0|>")
    return FishOmniConfig(
        **{
            **fish_config_dict(),
            "text_config": {
                **fish_text_config_dict(),
                "vocab_size": tokenizer.get_vocab_size(),
            },
            "audio_decoder_config": {
                **audio_decoder_config_dict(),
                "num_codebooks": NUM_CODEBOOKS,
                "vocab_size": CODEBOOK_SIZE,
                "max_seq_len": NUM_CODEBOOKS + 1,
            },
            "semantic_start_token_id": semantic_begin_id,
            "semantic_end_token_id": semantic_begin_id + CODEBOOK_SIZE - 1,
            "eos_token_id": tokenizer.token_to_id("<|im_end|>"),
        }
    )


def speech_preprocessor(
    tokenizer: Tokenizer,
    *,
    max_prompt_rows: int = SERVING_PROMPT_CAP_ROWS,
    encode_hook: FishReferenceEncodeHook | None = None,
) -> Drama3SpeechPreprocessor:
    return Drama3SpeechPreprocessor(
        tokenizer=tokenizer,
        config=toy_config(tokenizer),
        max_prompt_rows=max_prompt_rows,
        reference_encoder=ReferenceEncodeService(
            FixedCodesEncodeHook(clip_codes(3, 7))
            if encode_hook is None
            else encode_hook
        ),
    )


def first_stage_payload(request: OmniRequest) -> StagePayload:
    return StagePayload(
        request_id="request-a", request=request, data={"raw_inputs": request.inputs}
    )


def lowered(
    payload: dict[str, object], voice_store: SpeakerSampleStore | None = None
) -> StagePayload:
    """A raw /v1/audio/speech body as serving hands it to the first stage."""
    service = SpeechRequestValidator(default_model="drama3", voice_store=voice_store)
    return first_stage_payload(
        Client.build_omni_request(
            service.build_generate_request(service.parse_raw_request(payload))
        )
    )


def generation_input_of(payload: StagePayload) -> Drama3GenerationInput:
    generation_input = payload.data
    assert isinstance(generation_input, Drama3GenerationInput)
    return generation_input


def refused(preprocessor: Drama3SpeechPreprocessor, payload: StagePayload) -> str:
    """The message of the bad request the preprocessor raises instead of a payload."""
    with pytest.raises(ValueError) as rejection:
        preprocessor(payload)
    assert is_bad_request_error(rejection.value)
    return str(rejection.value)


@pytest.mark.parametrize(
    "text", ["hello", "<|speaker:0|> hello"], ids=["untagged", "tagged"]
)
def test_text_becomes_a_tts_prompt_spoken_by_speaker_zero(
    tokenizer: Tokenizer, text: str
) -> None:
    payload = lowered({"input": text})

    preprocessed = speech_preprocessor(tokenizer)(payload)

    assert preprocessed.request is payload.request
    assert torch.equal(
        generation_input_of(preprocessed).prompt_rows,
        expected_rows(
            tokenizer,
            DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END, VOICE_TURN],
        ),
    )


@pytest.mark.parametrize(
    ("sampling_fields", "sampling"),
    [
        pytest.param({}, (1.0, 0.9, 30, None), id="omitted"),
        pytest.param(dict.fromkeys(GENERATION_FIELDS), (1.0, 0.9, 30, None), id="null"),
        pytest.param({"temperature": 0.5}, (0.5, 0.9, 30, None), id="only-temperature"),
        pytest.param(
            {
                "temperature": 0.0,
                "top_p": 0.5,
                "top_k": 4,
                "max_new_tokens": 12,
                "repetition_penalty": 1.0,
            },
            (0.0, 0.5, 4, 12),
            id="all-set",
        ),
    ],
)
def test_sampling_takes_the_set_fields_and_drama3_defaults_for_the_rest(
    tokenizer: Tokenizer,
    sampling_fields: dict[str, object],
    sampling: tuple[float, float, int, int | None],
) -> None:
    generation_input = generation_input_of(
        speech_preprocessor(tokenizer)(lowered({"input": "hello", **sampling_fields}))
    )

    assert (
        generation_input.temperature,
        generation_input.top_p,
        generation_input.top_k,
        generation_input.max_new_tokens,
    ) == sampling


def test_a_generate_request_keeps_its_set_fields_and_max_tokens(
    tokenizer: Tokenizer,
) -> None:
    """/v1/generate passes the fields it sets, with max_tokens as max_new_tokens."""
    request = Client.build_omni_request(
        build_rollout_generate_request(
            RolloutGenerateRequest.model_validate(
                {
                    "prompt": "hello",
                    "sampling_params": {"max_tokens": 12, "temperature": 0.5},
                }
            )
        )
    )

    generation_input = generation_input_of(
        speech_preprocessor(tokenizer)(first_stage_payload(request))
    )

    assert (
        generation_input.temperature,
        generation_input.top_p,
        generation_input.top_k,
        generation_input.max_new_tokens,
    ) == (0.5, 0.9, 30, 12)


def test_references_become_a_reference_prompt(tokenizer: Tokenizer) -> None:
    """Given codes are used as they are, audio goes through the shared encoder, and a
    blank transcript takes only the speaker tag."""
    hook = FixedCodesEncodeHook(clip_codes(3, 7))
    given_codes = clip_codes(4, 1)
    reference_data = base64.b64encode(REFERENCE_WAV).decode("ascii")
    payload = lowered(
        {
            "input": "hello",
            "references": [
                {"text": "first take", "vq_codes": given_codes.tolist()},
                {"data": reference_data, "media_type": "audio/wav"},
            ],
        }
    )

    generation_input = generation_input_of(
        speech_preprocessor(tokenizer, encode_hook=hook)(payload)
    )

    assert torch.equal(
        generation_input.prompt_rows,
        expected_rows(
            tokenizer,
            [SYSTEM, REFERENCE_INSTRUCTION]
            + [reference_header(1, "<|speaker:0|> first take"), given_codes]
            + [reference_header(2, "<|speaker:0|> "), hook.audio_codes, END]
            + [USER, "<|speaker:0|> hello", END, VOICE_TURN],
        ),
    )
    assert hook.encoded == [FishReferenceInput("base64", reference_data, "audio/wav")]


def test_an_uploaded_voice_speaks_from_its_reference(
    tokenizer: Tokenizer, tmp_path: Path
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=REFERENCE_WAV,
        filename="anchor.wav",
        content_type="audio/wav",
        ref_text="anchor take",
    )
    hook = FixedCodesEncodeHook(clip_codes(3, 7))
    payload = lowered({"input": "hello", "voice": "anchor"}, voice_store=store)
    tts_params = payload.request.metadata["tts_params"]
    assert isinstance(tts_params, dict)

    generation_input = generation_input_of(
        speech_preprocessor(tokenizer, encode_hook=hook)(payload)
    )

    assert tts_params["task_type"] == "Base"
    assert torch.equal(
        generation_input.prompt_rows,
        expected_rows(
            tokenizer,
            [SYSTEM, REFERENCE_INSTRUCTION]
            + [reference_header(1, "<|speaker:0|> anchor take"), hook.audio_codes, END]
            + [USER, "<|speaker:0|> hello", END, VOICE_TURN],
        ),
    )
    assert [reference_input.source_kind for reference_input in hook.encoded] == ["path"]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"task_type": "Base"}, id="base-task-type"),
        pytest.param({"voice": "DEFAULT"}, id="default-voice-in-any-case"),
        pytest.param({"voice": ""}, id="empty-voice"),
    ],
)
def test_base_requests_with_the_default_voice_are_accepted(
    tokenizer: Tokenizer, change: dict[str, object]
) -> None:
    generation_input_of(
        speech_preprocessor(tokenizer)(lowered({"input": "hello", **change}))
    )


TWO_FRAME_CODES = clip_codes(2, 0).tolist()
REFUSED_REQUESTS = [
    pytest.param({"seed": 0}, "seed not supported", id="zero-seed"),
    pytest.param(
        {"instructions": "whisper"}, "instructions not supported", id="instructions"
    ),
    pytest.param(
        {"repetition_penalty": 1.2},
        "repetition_penalty must be 1.0",
        id="repetition-penalty",
    ),
    pytest.param(
        {"task_type": "CustomVoice"}, "task_type 'CustomVoice'", id="custom-voice-task"
    ),
    pytest.param({"voice": "narrator"}, "unknown voice 'narrator'", id="unknown-voice"),
    pytest.param(
        {"input": "<|speaker:1|> hello"}, "speaker tags [1]", id="speaker-one-in-input"
    ),
    pytest.param(
        {"references": [{"text": "<|speaker:2|> take", "vq_codes": TWO_FRAME_CODES}]},
        "speaker tags [2]",
        id="speaker-two-in-a-transcript",
    ),
    pytest.param(
        {"references": [{"text": "take"}]},
        "references.0 needs audio or vq_codes",
        id="reference-without-audio",
    ),
    pytest.param(
        {"references": [{"text": "take", "vq_codes": [1, 2]}]},
        "references.0.vq_codes",
        id="flat-vq-codes",
    ),
    pytest.param(
        {"references": [{"vq_codes": [[CODEBOOK_SIZE]] * NUM_CODEBOOKS}]},
        f"[0, {CODEBOOK_SIZE})",
        id="code-past-the-codebook",
    ),
    pytest.param(
        {"references": [{"vq_codes": [[2**63]] * NUM_CODEBOOKS}]},
        "references.0.vq_codes must be a [codebooks, frames] grid of int64 integers",
        id="code-past-int64",
    ),
    pytest.param(
        {"references": [{"vq_codes": [[-(2**63) - 1]] * NUM_CODEBOOKS}]},
        "references.0.vq_codes must be a [codebooks, frames] grid of int64 integers",
        id="code-below-int64",
    ),
    pytest.param(
        {
            "references": [
                {"text": f"<|speaker:{'9' * 5000}|> take", "vq_codes": TWO_FRAME_CODES}
            ]
        },
        "speaker tags must be written",
        id="speaker-tag-of-5000-digits-in-a-transcript",
    ),
    pytest.param(
        {"input": "say <|im_end|>"}, "<|im_end|>", id="structural-token-in-input"
    ),
]


@pytest.mark.parametrize(("change", "error"), REFUSED_REQUESTS)
def test_requests_the_model_cannot_honor_are_bad_requests(
    tokenizer: Tokenizer, change: dict[str, object], error: str
) -> None:
    message = refused(
        speech_preprocessor(tokenizer), lowered({"input": "hello", **change})
    )

    assert error in message


@pytest.mark.parametrize(
    ("location", "field_name", "value"),
    [
        pytest.param("tts_params", "instructions", "", id="empty-instructions"),
        pytest.param("tts_params", "language", "English", id="language"),
        pytest.param("tts_params", "x_vector_only_mode", False, id="false-x-vector"),
        pytest.param("tts_params", "token_count", 0, id="zero-token-count"),
        pytest.param("params", "seed", 0, id="zero-seed"),
        pytest.param("params", "stage_params", {}, id="empty-stage-params"),
        pytest.param("params", "stage_sampling", {}, id="empty-stage-sampling"),
    ],
)
def test_unsupported_fields_refuse_every_value_but_none(
    tokenizer: Tokenizer,
    location: Literal["params", "tts_params"],
    field_name: str,
    value: object,
) -> None:
    """Covers callers that skip the speech entry: None is unset, while {}, False and
    0 are values the caller gave."""
    preprocessor = speech_preprocessor(tokenizer)

    def with_field(field_value: object) -> StagePayload:
        payload = lowered({"input": "hello"})
        fields = (
            payload.request.params
            if location == "params"
            else payload.request.metadata["tts_params"]
        )
        assert isinstance(fields, dict)
        fields[field_name] = field_value
        return payload

    generation_input_of(preprocessor(with_field(None)))
    assert field_name in refused(preprocessor, with_field(value))


def test_prompt_cap_counts_reference_rows_and_refuses_one_row_more(
    tokenizer: Tokenizer,
) -> None:
    preprocessor = speech_preprocessor(tokenizer)

    def with_reference_frames(frame_count: int) -> StagePayload:
        vq_codes = [[frame % CODEBOOK_SIZE for frame in range(frame_count)]]
        return lowered(
            {
                "input": "hello",
                "references": [{"text": "take", "vq_codes": vq_codes * NUM_CODEBOOKS}],
            }
        )

    one_frame_rows = generation_input_of(preprocessor(with_reference_frames(1)))
    frames_at_cap = SERVING_PROMPT_CAP_ROWS - (one_frame_rows.prompt_rows.shape[0] - 1)
    at_cap = generation_input_of(preprocessor(with_reference_frames(frames_at_cap)))

    assert at_cap.prompt_rows.shape[0] == SERVING_PROMPT_CAP_ROWS
    assert "exceeds the maximum allowed length" in refused(
        preprocessor, with_reference_frames(frames_at_cap + 1)
    )


def test_prompt_cap_follows_the_configured_value(tokenizer: Tokenizer) -> None:
    prompt_row_count = generation_input_of(
        speech_preprocessor(tokenizer)(lowered({"input": "hello"}))
    ).prompt_rows.shape[0]

    generation_input_of(
        speech_preprocessor(tokenizer, max_prompt_rows=prompt_row_count)(
            lowered({"input": "hello"})
        )
    )
    assert f"maximum allowed length of {prompt_row_count - 1} rows" in refused(
        speech_preprocessor(tokenizer, max_prompt_rows=prompt_row_count - 1),
        lowered({"input": "hello"}),
    )


def test_a_tokenizer_off_the_config_semantic_band_is_refused(
    tokenizer: Tokenizer,
) -> None:
    with pytest.raises(ValueError, match="semantic_start_token_id"):
        Drama3SpeechPreprocessor(
            tokenizer=tokenizer,
            config=FishOmniConfig(**fish_config_dict()),
            max_prompt_rows=SERVING_PROMPT_CAP_ROWS,
            reference_encoder=ReferenceEncodeService(
                FixedCodesEncodeHook(clip_codes(3, 7))
            ),
        )


def engine_request_builder(tokenizer: Tokenizer) -> Drama3RequestBuilder:
    return Drama3RequestBuilder(
        config=toy_config(tokenizer),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=SERVING_PROMPT_CAP_ROWS,
        max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
    )


def test_a_preprocessed_request_is_admitted_with_the_engine_cap(
    tokenizer: Tokenizer,
) -> None:
    request_data = engine_request_builder(tokenizer)(
        speech_preprocessor(tokenizer)(lowered({"input": "hello"}))
    )

    assert request_data.req is not None
    assert request_data.max_new_tokens == MAX_NEW_TOKENS_CAP
    assert request_data.req.sampling_params.max_new_tokens == MAX_NEW_TOKENS_CAP
    assert (request_data.temperature, request_data.top_p, request_data.top_k) == (
        1.0,
        0.9,
        30,
    )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param(
            {"temperature": -0.5},
            "temperature must be finite and >= 0",
            id="negative-temperature",
        ),
        pytest.param({"top_p": 1.5}, "top_p must be in (0, 1]", id="top-p-above-one"),
        pytest.param({"top_k": 0}, "top_k must be in [1, 2147483647]", id="zero-top-k"),
        pytest.param(
            {"top_k": 2**31}, "top_k must be in [1, 2147483647]", id="top-k-past-int32"
        ),
        pytest.param(
            {"top_k": 2**63}, "top_k must be in [1, 2147483647]", id="top-k-past-int64"
        ),
        pytest.param(
            {"max_new_tokens": MAX_NEW_TOKENS_CAP + 1},
            f"max_new_tokens must be at most {MAX_NEW_TOKENS_CAP}",
            id="past-the-cap",
        ),
    ],
)
def test_out_of_range_sampling_is_refused_before_admission(
    tokenizer: Tokenizer, change: dict[str, object], error: str
) -> None:
    payload = speech_preprocessor(tokenizer)(lowered({"input": "hello", **change}))

    with pytest.raises(ValueError, match=re.escape(error)) as rejection:
        engine_request_builder(tokenizer)(payload)

    assert is_bad_request_error(rejection.value)


@pytest.mark.parametrize("top_k", [1, 2**31 - 1], ids=["one", "int32-limit"])
def test_top_k_within_int32_is_admitted(tokenizer: Tokenizer, top_k: int) -> None:
    request_data = engine_request_builder(tokenizer)(
        speech_preprocessor(tokenizer)(lowered({"input": "hello", "top_k": top_k}))
    )

    assert request_data.top_k == top_k


def test_preprocessing_builds_cpu_rows_under_another_default_device(
    tokenizer: Tokenizer,
) -> None:
    given_codes = clip_codes(4, 1)
    payload = lowered(
        {
            "input": "hello",
            "references": [{"text": "take", "vq_codes": given_codes.tolist()}],
        }
    )
    rows = expected_rows(
        tokenizer,
        [SYSTEM, REFERENCE_INSTRUCTION, reference_header(1, "<|speaker:0|> take")]
        + [given_codes, END, USER, "<|speaker:0|> hello", END, VOICE_TURN],
    )
    preprocessor = speech_preprocessor(tokenizer)

    with torch.device("meta"):
        prompt_rows = generation_input_of(preprocessor(payload)).prompt_rows

    assert (prompt_rows.device.type, prompt_rows.dtype) == ("cpu", torch.int64)
    assert torch.equal(prompt_rows, rows)


def test_undecodable_reference_audio_is_a_bad_request(tokenizer: Tokenizer) -> None:
    """The shared hook decodes the audio before any codec call, so its codec is a stub."""
    hook = FishReferenceEncodeHook(
        codec=SimpleNamespace(sample_rate=DEFAULT_SAMPLE_RATE),
        checkpoint_id="codec.pth",
    )
    reference_data = base64.b64encode(b"not an audio file").decode("ascii")

    message = refused(
        speech_preprocessor(tokenizer, encode_hook=hook),
        lowered(
            {
                "input": "hello",
                "references": [{"data": reference_data, "media_type": "audio/wav"}],
            }
        ),
    )

    assert message.startswith("Drama 3 request: references.0:")


def test_a_reference_longer_than_the_limit_is_a_bad_request(
    tokenizer: Tokenizer,
) -> None:
    hook = FishReferenceEncodeHook(
        codec=SimpleNamespace(sample_rate=DEFAULT_SAMPLE_RATE),
        checkpoint_id="codec.pth",
        max_reference_audio_seconds=1.0,
    )
    reference_data = base64.b64encode(REFERENCE_WAV).decode("ascii")

    message = refused(
        speech_preprocessor(tokenizer, encode_hook=hook),
        lowered(
            {
                "input": "hello",
                "references": [{"data": reference_data, "media_type": "audio/wav"}],
            }
        ),
    )

    assert message == (
        "Drama 3 request: references.0: reference audio lasts 1.2 s, longer than "
        "the 1 s limit; shorten it or raise max_reference_audio_seconds"
    )


def test_a_codec_failure_while_encoding_stays_an_internal_error(
    tokenizer: Tokenizer,
) -> None:
    class FailingCodecEncodeHook(FishReferenceEncodeHook):
        def encode_one(self, item: FishReferenceInput) -> torch.Tensor:
            raise RuntimeError("the codec ran out of memory")

    hook = FailingCodecEncodeHook(
        codec=SimpleNamespace(sample_rate=DEFAULT_SAMPLE_RATE),
        checkpoint_id="codec.pth",
    )
    reference_data = base64.b64encode(REFERENCE_WAV).decode("ascii")
    payload = lowered(
        {
            "input": "hello",
            "references": [{"data": reference_data, "media_type": "audio/wav"}],
        }
    )

    with pytest.raises(RuntimeError, match="ran out of memory") as failure:
        speech_preprocessor(tokenizer, encode_hook=hook)(payload)

    assert not is_bad_request_error(failure.value)


def direct_client_payload(sampling: SamplingParams) -> StagePayload:
    """A request a Python caller hands to Client itself."""
    return first_stage_payload(
        Client.build_omni_request(GenerateRequest(prompt="hello", sampling=sampling))
    )


@pytest.mark.parametrize(
    ("sampling", "expected_sampling"),
    [
        pytest.param(SamplingParams(), (1.0, 0.9, 30), id="unset"),
        pytest.param(
            SamplingParams(temperature=0.5), (0.5, 0.9, 30), id="only-temperature"
        ),
        pytest.param(
            SamplingParams(temperature=0.0, top_p=1.0, top_k=1),
            (0.0, 1.0, 1),
            id="all-set",
        ),
    ],
)
def test_a_direct_client_request_takes_the_fields_it_sets(
    tokenizer: Tokenizer,
    sampling: SamplingParams,
    expected_sampling: tuple[float, float, int],
) -> None:
    """A Client caller's set sampling fields reach the model with no metadata record."""
    generation_input = generation_input_of(
        speech_preprocessor(tokenizer)(direct_client_payload(sampling))
    )

    assert (
        generation_input.temperature,
        generation_input.top_p,
        generation_input.top_k,
    ) == expected_sampling


def test_a_direct_client_request_cannot_set_a_repetition_penalty(
    tokenizer: Tokenizer,
) -> None:
    message = refused(
        speech_preprocessor(tokenizer),
        direct_client_payload(SamplingParams(repetition_penalty=1.2)),
    )

    assert "repetition_penalty must be 1.0" in message
