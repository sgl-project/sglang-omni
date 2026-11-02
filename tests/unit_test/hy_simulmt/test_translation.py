# SPDX-License-Identifier: Apache-2.0
"""Speech translation protocol and streaming state regression tests."""

import io
import wave

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor

from sglang_omni.models.hy_simulmt.request_builders import (
    ASSISTANT_PREFIX,
    AUDIO_END,
    AUDIO_START,
    AUDIO_TOKEN,
    PROMPT_PREFIX,
    TURN_END,
    USER_TOKEN,
    HySimulMTRequestBuilder,
    HySimulMTResultAdapter,
)
from sglang_omni.models.hy_simulmt.sglang_model import FrameStackAdapter
from sglang_omni.models.hy_simulmt.streaming import HySimulMTStreamingStrategy
from sglang_omni.proto.request import OmniRequest, StagePayload


def wav_bytes(sample_count: int, value: int = 100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as waveform:
        waveform.setnchannels(1)
        waveform.setsampwidth(2)
        waveform.setframerate(16000)
        waveform.writeframes(np.full(sample_count, value, dtype=np.int16).tobytes())
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
    "history",
    [
        [{"end_sample": 51200, "text": ""}],
        [{"end_sample": 25600, "text": ""}, {"end_sample": 16000, "text": ""}],
    ],
)
def test_history_cannot_rewind_or_consume_the_new_turn(
    builder: HySimulMTRequestBuilder, history: list[dict[str, int | str]]
) -> None:
    with pytest.raises(ValueError, match="increasing boundaries"):
        builder(payload(wav_bytes(51200), {"hy_simulmt_history": history}))


@pytest.mark.parametrize("parameters", [{"language": "fr"}, {"target_language": "de"}])
def test_unsupported_translation_direction_is_rejected(
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


def test_realtime_history_is_committed_after_success_and_final_flush_is_complete() -> (
    None
):
    strategy = HySimulMTStreamingStrategy()
    first = strategy.create_state(model_name="translation", language="en")
    second = strategy.create_state(model_name="translation", language="zh")
    request = strategy.build_decode_request(
        audio=wav_bytes(25600),
        state=first,
        is_final=False,
        request_id="first",
    )
    assert first.history == second.history == []
    assert request.extra_params["hy_simulmt_history"] == []
    assert (
        strategy.update_hypothesis(generated_text="你好", language="zh", state=first)
        == "你好"
    )
    next_request = strategy.build_decode_request(
        audio=wav_bytes(51200),
        state=first,
        is_final=False,
        request_id="next",
    )
    assert next_request.extra_params["language"] == "en"
    assert next_request.extra_params["hy_simulmt_history"] == [
        {"end_sample": 25600, "text": "你好"}
    ]
    assert second.history == []
    final_request = strategy.build_decode_request(
        audio=wav_bytes(51200),
        state=first,
        is_final=True,
        request_id="final",
    )
    assert "hy_simulmt_history" not in final_request.extra_params
    assert (
        strategy.update_hypothesis(
            generated_text="你好，世界。", language="zh", state=first
        )
        == "你好，世界。"
    )
