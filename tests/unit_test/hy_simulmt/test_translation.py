# SPDX-License-Identifier: Apache-2.0
"""Speech translation protocol and streaming state regression tests."""

import io
import wave
from typing import Literal

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor

from sglang_omni.models.hy_simulmt.request_builders import (
    ASSISTANT_PREFIX,
    AUDIO_END,
    AUDIO_START,
    AUDIO_TOKEN,
    PROMPT_PREFIX,
    SIMULTANEOUS_INSTRUCTIONS,
    TRANSLATION_INSTRUCTIONS,
    TURN_END,
    USER_TOKEN,
    HySimulMTRequestBuilder,
    HySimulMTResultAdapter,
)
from sglang_omni.models.hy_simulmt.sglang_model import FrameStackAdapter
from sglang_omni.models.hy_simulmt.streaming import HySimulMTStreamingStrategy
from sglang_omni.proto.request import OmniRequest, StagePayload


def wav_bytes(sample_count: int, value: int = 100) -> bytes:
    return waveform_to_wav(np.full(sample_count, value, dtype=np.int16))


def waveform_to_wav(waveform_samples: NDArray[np.int16]) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as waveform:
        waveform.setnchannels(1)
        waveform.setsampwidth(2)
        waveform.setframerate(16000)
        waveform.writeframes(waveform_samples.tobytes())
    return buffer.getvalue()


@pytest.fixture
def builder() -> HySimulMTRequestBuilder:
    markers = [
        "<unk>",
        PROMPT_PREFIX,
        USER_TOKEN,
        ASSISTANT_PREFIX,
        AUDIO_START,
        AUDIO_END,
        AUDIO_TOKEN,
        TURN_END,
        TRANSLATION_INSTRUCTIONS["en"],
        SIMULTANEOUS_INSTRUCTIONS["en"],
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {token: index for index, token in enumerate(markers)}, unk_token="<unk>"
            )
        ),
        unk_token="<unk>",
        additional_special_tokens=markers[1:],
    )
    tokenizer.add_tokens(["first", "second", " has left. "])
    return HySimulMTRequestBuilder(
        tokenizer=tokenizer,
        feature_extractor=WhisperFeatureExtractor(feature_size=128),
        max_new_tokens=64,
        context_length=8192,
        audio_token_id=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
        stop_token_ids=[tokenizer.convert_tokens_to_ids(TURN_END)],
    )


def payload(audio: bytes, parameters: dict[str, object]) -> StagePayload:
    return StagePayload(
        request_id="translation",
        request=OmniRequest(inputs={"audio_bytes": audio}, params=parameters),
        data={},
    )


@pytest.mark.parametrize("sample_count", [15840, 16000, 16160, 25600, 480000])
def test_audio_span_matches_convolution_and_frame_stacking(
    builder: HySimulMTRequestBuilder, sample_count: int
) -> None:
    """Convolution blocks pad independently at the one-second boundary."""
    request = builder(payload(wav_bytes(sample_count), {}))
    audio = request.req.multimodal_inputs.mm_items[0]
    convolutions = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 3, 2, padding=1),
        torch.nn.Conv2d(1, 1, 3, 2, padding=1),
        torch.nn.Conv2d(1, 1, 3, 2, padding=1),
    )
    encoder_frames = sum(
        convolutions(chunk[None, None]).shape[-1]
        for chunk in audio.feature.split(100, dim=-1)
    )
    adapter = FrameStackAdapter(encoder_dim=4, llm_dim=4, fsk_size=2)
    expected_tokens = adapter(torch.zeros(encoder_frames, 4)).shape[0]
    start, end = audio.offsets[0]
    assert end - start + 1 == expected_tokens
    assert (
        request.input_ids[start : end + 1].tolist()
        == [audio.pad_value] * expected_tokens
    )
    assert request.input_ids.tolist().count(audio.pad_value) == expected_tokens


@pytest.mark.parametrize("frame_count", [3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_frame_stacking_preserves_the_odd_tail(frame_count: int, batched: bool) -> None:
    """Singleton batches must stack frames along time, padding only an odd tail."""
    adapter = FrameStackAdapter(encoder_dim=2, llm_dim=4, fsk_size=2)
    adapter.layers = torch.nn.Identity()
    frames = torch.arange(frame_count * 2).reshape(frame_count, 2).float()
    actual = adapter(frames.unsqueeze(0) if batched else frames)
    expected = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 0.0, 0.0]])
    if frame_count == 4:
        expected[1, 2:] = torch.tensor([6.0, 7.0])
    else:
        pass
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected)


def test_history_preserves_empty_turns_and_separate_audio_spans(
    builder: HySimulMTRequestBuilder,
) -> None:
    """WAIT turns still retain audio and EOT in the next model prompt."""
    request = builder(
        payload(
            wav_bytes(51200),
            {
                "hy_simulmt_history": [{"end_sample": 25600, "text": ""}],
                "language": "en",
            },
        )
    )
    items = request.req.multimodal_inputs.mm_items
    assert len(items) == 2
    assert items[0].offsets[0][1] < items[1].offsets[0][0]
    tokens = request.input_ids.tolist()
    assert tokens.count(builder.tokenizer.convert_tokens_to_ids(USER_TOKEN)) == 2
    assert tokens.count(builder.tokenizer.convert_tokens_to_ids(TURN_END)) == 1
    assert request.target_language == "zh"


@pytest.mark.parametrize(
    "is_final,history",
    [
        (False, [{"end_sample": 51200, "text": ""}]),
        (False, [{"end_sample": 25600, "text": ""}, {"end_sample": 16000, "text": ""}]),
        (True, [{"end_sample": 51201, "text": ""}]),
        (True, [{"end_sample": 25600, "text": ""}, {"end_sample": 16000, "text": ""}]),
    ],
)
def test_history_rejects_invalid_audio_boundaries(
    builder: HySimulMTRequestBuilder,
    is_final: bool,
    history: list[dict[str, int | str]],
) -> None:
    """Incremental and final prompts must reject rewound or out-of-range audio history."""
    with pytest.raises(ValueError, match="increasing boundaries"):
        builder(
            payload(
                wav_bytes(51200),
                {"hy_simulmt_history": history, "hy_simulmt_final": is_final},
            )
        )


@pytest.mark.parametrize(
    "parameters",
    [{"language": "fr"}, {"target_language": "de"}, {"hy_simulmt_final": 1}],
)
def test_invalid_translation_parameters_are_rejected(
    builder: HySimulMTRequestBuilder, parameters: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        builder(payload(wav_bytes(25600), parameters))


def test_overlong_audio_is_rejected_instead_of_truncated(
    builder: HySimulMTRequestBuilder,
) -> None:
    with pytest.raises(ValueError, match="up to 30"):
        builder(payload(wav_bytes(480001), {}))


def test_native_result_carries_tokens_and_requested_logprobs(
    builder: HySimulMTRequestBuilder,
) -> None:
    """Native generation must return one logprob and usage count per output token."""
    request = builder(payload(wav_bytes(25600), {"return_logprob": True}))
    request.output_ids = [0, builder.stop_token_ids[0]]
    request.output_token_logprobs = [[-0.5, token] for token in request.output_ids]
    request.finish_reason = "stop"
    result = HySimulMTResultAdapter(tokenizer=builder.tokenizer)(request)
    assert request.return_logprob
    assert result.data["token_ids"] == request.output_ids
    assert result.data["output_token_logprobs"] == request.output_token_logprobs
    assert result.data["usage"]["completion_tokens"] == len(request.output_ids)
    assert result.data["usage"]["prompt_tokens"] == len(request.input_ids)
    assert result.data["finish_reason"] == "stop"


def test_native_stage_parameters_preserve_translation_direction_and_history(
    builder: HySimulMTRequestBuilder,
) -> None:
    """Native requests must not silently lose their source language or audio turns."""
    parameters = {
        "stage_params": {
            "asr": {
                "language": "en",
                "hy_simulmt_history": [{"end_sample": 25600, "text": ""}],
            }
        }
    }
    request = builder(payload(wav_bytes(51200), parameters))
    assert request.target_language == "zh"
    assert len(request.req.multimodal_inputs.mm_items) == 2


@pytest.mark.parametrize(
    "history,sample_count,context_start_sample,assistant_prefix",
    [
        ([], 25600, 0, ""),
        ([{"end_sample": 25600, "text": ""}], 25600, 0, ""),
        ([{"end_sample": 25600, "text": "first"}], 25600, 0, "first"),
        (
            [
                {"end_sample": 12800, "text": ""},
                {"end_sample": 25600, "text": "second"},
                {"end_sample": 38400, "text": ""},
            ],
            51200,
            12800,
            "second",
        ),
        (
            [
                {"end_sample": 12800, "text": "first"},
                {"end_sample": 25600, "text": ""},
                {"end_sample": 38400, "text": "second"},
                {"end_sample": 51200, "text": ""},
            ],
            64000,
            25600,
            "second",
        ),
        ([{"end_sample": 25600, "text": "   "}], 51200, 0, ""),
    ],
)
def test_final_continues_the_last_submitted_text_over_its_audio(
    builder: HySimulMTRequestBuilder,
    history: list[dict[str, int | str]],
    sample_count: int,
    context_start_sample: int,
    assistant_prefix: str,
) -> None:
    """EOF continues the last submitted text over that turn's audio and speech after it."""
    waveform_samples = np.arange(sample_count, dtype=np.int16)
    request = builder(
        payload(
            waveform_to_wav(waveform_samples),
            {"hy_simulmt_history": history, "hy_simulmt_final": True},
        )
    )
    audio_items = request.req.multimodal_inputs.mm_items
    assert len(audio_items) == 1
    expected_waveform = (
        waveform_samples[context_start_sample:].astype(np.float32) / 32768
    )
    extracted = builder.feature_extractor(
        expected_waveform,
        sampling_rate=16000,
        padding="longest",
        pad_to_multiple_of=160,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mel_frames = int(extracted.attention_mask.sum())
    torch.testing.assert_close(
        audio_items[0].feature, extracted.input_features[0, :, :mel_frames]
    )
    tokens = request.input_ids.tolist()
    expected_suffix = builder.tokenizer.encode(
        AUDIO_END + ASSISTANT_PREFIX + assistant_prefix, add_special_tokens=False
    )
    assert tokens[-len(expected_suffix) :] == expected_suffix
    assert builder.tokenizer.convert_tokens_to_ids(TURN_END) not in tokens
    assert (
        builder.tokenizer.convert_tokens_to_ids(TRANSLATION_INSTRUCTIONS["en"])
        in tokens
    )
    assert (
        builder.tokenizer.convert_tokens_to_ids(SIMULTANEOUS_INSTRUCTIONS["en"])
        not in tokens
    )
    first_token_id = builder.tokenizer.convert_tokens_to_ids("first")
    if assistant_prefix == "first":
        assert tokens.count(first_token_id) == 1
    else:
        assert first_token_id not in tokens


@pytest.mark.parametrize(
    "parameters,finish_reason,expected_text",
    [
        (
            {
                "hy_simulmt_history": [{"end_sample": 25600, "text": "first"}],
                "hy_simulmt_final": True,
            },
            "stop",
            " has left.",
        ),
        (
            {
                "hy_simulmt_history": [{"end_sample": 25600, "text": "first"}],
                "hy_simulmt_final": True,
            },
            "length",
            " has left.",
        ),
        (
            {
                "hy_simulmt_history": [{"end_sample": 25600, "text": "first"}],
                "hy_simulmt_final": False,
            },
            "length",
            "has left. ",
        ),
        (
            {
                "hy_simulmt_history": [{"end_sample": 25600, "text": "first"}],
                "hy_simulmt_final": False,
            },
            "stop",
            "has left.",
        ),
        ({"hy_simulmt_final": True}, "stop", "has left."),
        ({}, "length", "has left."),
    ],
)
def test_result_keeps_only_the_whitespace_needed_to_resume_text(
    builder: HySimulMTRequestBuilder,
    parameters: dict[str, object],
    finish_reason: Literal["stop", "length"],
    expected_text: str,
) -> None:
    """A suffix keeps its leading space, and a truncated partial keeps its trailing space."""
    request = builder(payload(wav_bytes(51200), parameters))
    request.output_ids = builder.tokenizer.encode(
        " has left. ", add_special_tokens=False
    )
    request.finish_reason = finish_reason
    result = HySimulMTResultAdapter(tokenizer=builder.tokenizer)(request)
    assert result.data["text"] == expected_text


@pytest.mark.parametrize(
    "language,first_text,tail_text,has_wait,expected_text",
    [
        ("en", "The train", " has left.", True, "The train has left."),
        ("en", "Hello", ".", False, "Hello."),
        ("en", "trans", "lation", False, "translation"),
        ("zh", "你好", "，世界。", True, "你好，世界。"),
        ("en", "The train", "", True, "The train"),
    ],
)
def test_final_decodes_unchanged_audio_and_preserves_committed_text(
    language: str,
    first_text: str,
    tail_text: str,
    has_wait: bool,
    expected_text: str,
) -> None:
    """EOF must decode unchanged audio and retain submitted text when appending or receiving no suffix."""
    strategy = HySimulMTStreamingStrategy()
    source_language = "zh" if language == "en" else "en"
    state = strategy.create_state(model_name="translation", language=source_language)
    other_state = strategy.create_state(model_name="translation", language="en")
    request = strategy.build_decode_request(
        audio=wav_bytes(25600), state=state, is_final=False, request_id="first"
    )
    assert request.extra_params["hy_simulmt_history"] == []
    assert (
        strategy.update_hypothesis(
            generated_text=first_text, language=language, state=state
        )
        == first_text
    )
    assert other_state.history == []
    history = [{"end_sample": 25600, "text": first_text}]
    sample_count = 51200 if has_wait else 25600
    if has_wait:
        strategy.build_decode_request(
            audio=wav_bytes(sample_count),
            state=state,
            is_final=False,
            request_id="wait",
        )
        assert (
            strategy.update_hypothesis(
                generated_text="", language=language, state=state
            )
            == first_text
        )
        history.append({"end_sample": sample_count, "text": ""})
    else:
        pass
    request = strategy.build_decode_request(
        audio=wav_bytes(sample_count), state=state, is_final=True, request_id="final"
    )
    assert request.extra_params["language"] == source_language
    assert request.extra_params["hy_simulmt_history"] == history
    assert request.extra_params["hy_simulmt_final"] is True
    assert (
        strategy.update_hypothesis(
            generated_text=tail_text, language=language, state=state
        )
        == expected_text
    )


def test_truncated_partial_joins_english_with_one_space(
    builder: HySimulMTRequestBuilder,
) -> None:
    """A truncated partial must not add a second space when the next turn is joined."""
    request = builder(
        payload(
            wav_bytes(51200),
            {
                "hy_simulmt_history": [{"end_sample": 25600, "text": "Hello"}],
                "hy_simulmt_final": False,
            },
        )
    )
    request.output_ids = builder.tokenizer.encode(
        " has left. ", add_special_tokens=False
    )
    request.finish_reason = "length"
    suffix = HySimulMTResultAdapter(tokenizer=builder.tokenizer)(request).data["text"]
    strategy = HySimulMTStreamingStrategy()
    state = strategy.create_state(model_name="translation", language="zh")
    strategy.build_decode_request(
        audio=wav_bytes(25600), state=state, is_final=False, request_id="first"
    )
    strategy.update_hypothesis(generated_text="Hello", language="en", state=state)
    strategy.build_decode_request(
        audio=wav_bytes(51200), state=state, is_final=False, request_id="next"
    )
    assert (
        strategy.update_hypothesis(generated_text=suffix, language="en", state=state)
        == "Hello has left. "
    )
    strategy.build_decode_request(
        audio=wav_bytes(64000), state=state, is_final=False, request_id="tail"
    )
    assert (
        strategy.update_hypothesis(
            generated_text=" station", language="en", state=state
        )
        == "Hello has left. station"
    )


def test_blank_partial_does_not_prefix_the_final_suffix() -> None:
    """A whitespace-only partial is a wait, so the final suffix stands alone."""
    strategy = HySimulMTStreamingStrategy()
    state = strategy.create_state(model_name="translation", language="zh")
    strategy.build_decode_request(
        audio=wav_bytes(25600), state=state, is_final=False, request_id="blank"
    )
    assert (
        strategy.update_hypothesis(generated_text="   ", language="en", state=state)
        == ""
    )
    strategy.build_decode_request(
        audio=wav_bytes(51200), state=state, is_final=True, request_id="final"
    )
    assert (
        strategy.update_hypothesis(generated_text="tail", language="en", state=state)
        == "tail"
    )
