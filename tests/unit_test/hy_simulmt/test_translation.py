# SPDX-License-Identifier: Apache-2.0
"""Speech translation protocol and streaming state regression tests."""

import base64
import io
import json
import wave
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi import WebSocket
from starlette.websockets import WebSocketState
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor

from sglang_omni.client import Client, CompletionResult, GenerateRequest
from sglang_omni.models.hy_simulmt.config import HySimulMTPipelineConfig
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
from sglang_omni.models.hy_simulmt.stages import (
    HySimulMTEngineBuilder,
    create_hy_simulmt_executor,
)
from sglang_omni.models.hy_simulmt.streaming import HySimulMTStreamingStrategy
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)


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


@pytest.mark.parametrize(
    "sample_count", [1, 29, 160, 161, 15840, 16000, 16160, 25600, 480000]
)
def test_audio_span_matches_convolution_and_frame_stacking(
    builder: HySimulMTRequestBuilder, sample_count: int
) -> None:
    """Audio frames and token spans must stay valid for short and long inputs."""
    request = builder(payload(wav_bytes(sample_count), {}))
    audio = request.req.multimodal_inputs.mm_items[0]
    assert audio.feature.shape[-1] == (sample_count + 159) // 160
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


@pytest.mark.parametrize(
    ("sample_count", "history_end_sample", "expected_mel_frames"),
    [
        (51200, 25600, [160, 160]),
        (25629, 25600, [160, 1]),
        (25629, 29, [1, 160]),
    ],
)
def test_history_preserves_empty_turns_and_separate_audio_spans(
    builder: HySimulMTRequestBuilder,
    sample_count: int,
    history_end_sample: int,
    expected_mel_frames: list[int],
) -> None:
    """WAIT turns still retain audio and EOT in the next model prompt."""
    request = builder(
        payload(
            wav_bytes(sample_count),
            {
                "hy_simulmt_history": [{"end_sample": history_end_sample, "text": ""}],
                "language": "en",
            },
        )
    )
    items = request.req.multimodal_inputs.mm_items
    assert len(items) == 2
    assert [item.feature.shape[-1] for item in items] == expected_mel_frames
    assert items[0].offsets[0][1] < items[1].offsets[0][0]
    tokens = request.input_ids.tolist()
    assert tokens.count(builder.tokenizer.convert_tokens_to_ids(USER_TOKEN)) == 2
    assert tokens.count(builder.tokenizer.convert_tokens_to_ids(TURN_END)) == 1
    assert request.target_language == "zh"


@pytest.mark.parametrize("sample_count", [161, 319, 16000])
def test_normal_audio_matches_whisper_features(
    builder: HySimulMTRequestBuilder, sample_count: int
) -> None:
    """Padding short audio must not alter features for longer audio."""
    waveform = np.full(sample_count, 100 / 32768, dtype=np.float32)
    reference = builder.feature_extractor(
        waveform,
        sampling_rate=16000,
        padding="longest",
        pad_to_multiple_of=160,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    request = builder(payload(wav_bytes(sample_count), {}))
    audio = request.req.multimodal_inputs.mm_items[0]
    mel_frame_count = int(reference.attention_mask.sum())
    torch.testing.assert_close(
        audio.feature, reference.input_features[0, :, :mel_frame_count], rtol=0, atol=0
    )


@pytest.mark.asyncio
async def test_realtime_short_tail_preserves_committed_segments(
    builder: HySimulMTRequestBuilder,
) -> None:
    """A 300-second stream with a short tail must finish without losing prior text."""
    websocket = MagicMock(spec=WebSocket)
    websocket.application_state = WebSocketState.CONNECTED
    websocket.client_state = WebSocketState.CONNECTED

    async def complete(
        request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        builder(
            StagePayload(
                request_id=request_id,
                request=OmniRequest(inputs=request.prompt, params=request.extra_params),
                data={},
            )
        )
        return CompletionResult(request_id=request_id, text=request_id, language="en")

    client = MagicMock(spec=Client)
    client.completion.side_effect = complete
    session = RealtimeTranscriptionSession(
        websocket,
        client=client,
        model_name="translation",
        transcription_config=HySimulMTPipelineConfig.realtime_transcription,
        strategy=HySimulMTStreamingStrategy(),
        session_id="short-tail",
    )
    speech_pcm_bytes = np.full(480000, 1000, dtype=np.int16).tobytes()
    silence_pcm_bytes = np.zeros(480000, dtype=np.int16).tobytes()
    tail_pcm_bytes = np.full(29, 1000, dtype=np.int16).tobytes()
    try:
        for pcm_bytes in [speech_pcm_bytes, *([silence_pcm_bytes] * 9), tail_pcm_bytes]:
            await session.dispatch(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm_bytes).decode("ascii"),
                }
            )
        await session.dispatch({"type": "transcription.done"})
    finally:
        await session.teardown()

    events = [json.loads(call.args[0]) for call in websocket.send_text.await_args_list]
    assert not [event for event in events if event["type"] == "error"]
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert len(finals) == 11
    assert (finals[-1]["audio_start_ms"], finals[-1]["audio_end_ms"]) == (
        300000,
        300001,
    )
    completed = [
        event for event in events if event["type"] == "transcription.completed"
    ]
    assert len(completed) == 1
    assert completed[0]["text"] == "short-tail:0:1 short-tail:10:1"


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


@dataclass(kw_only=True)
class CapturedBuildCall:
    server_args_overrides: dict[str, int]
    infra_kwargs: InfrastructureOptions


@pytest.mark.parametrize(
    ("tp_rank", "tp_size", "nccl_port"),
    [(2, 4, 29501), (0, 1, None)],
)
def test_tensor_parallel_arguments_reach_server_args_and_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
    tp_rank: int,
    tp_size: int,
    nccl_port: int | None,
) -> None:
    captured: list[CapturedBuildCall] = []

    def fake_build(
        self: HySimulMTEngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        dtype: str,
        server_args_overrides: dict[str, int],
    ) -> str:
        captured.append(
            CapturedBuildCall(
                server_args_overrides=dict(server_args_overrides),
                infra_kwargs=self.infra_kwargs(),
            )
        )
        return "scheduler"

    monkeypatch.setattr(HySimulMTEngineBuilder, "build", fake_build)

    create_hy_simulmt_executor(
        "dummy",
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
        server_args_overrides={"max_running_requests": 8},
    )

    (call,) = captured
    assert call.server_args_overrides == {
        "max_running_requests": 8,
        "tp_size": tp_size,
    }
    assert call.infra_kwargs == {"tp_rank": tp_rank, "nccl_port": nccl_port}
