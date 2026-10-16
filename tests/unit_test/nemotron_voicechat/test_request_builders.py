# SPDX-License-Identifier: Apache-2.0
"""Frame counts and local code handoff between Nemotron stages."""

import logging
from contextlib import nullcontext
from unittest.mock import Mock, call

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.config import NemotronVoiceChatPipelineConfig
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.request_builders import (
    build_talker_request,
    build_thinker_request,
    merge_for_talker,
    talker_stream_output_builder,
)
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

PROMPT_FRAMES = 37


def make_payload(num_frames, params=None):
    return StagePayload(
        "r",
        request=OmniRequest(inputs={}, params=params or {}),
        data=NemotronVoiceChatState(num_frames=num_frames).to_dict(),
    )


@pytest.mark.parametrize("num_frames", [1, 2, PROMPT_FRAMES, 513])
def test_talker_decode_steps_match_thinker_tokens(num_frames):
    """The prefill emits no codes, so the talker needs one extra generation step."""
    payload = make_payload(num_frames)
    thinker = build_thinker_request(
        payload, vocab_size=8, prompt_token_ids=[1, 2], pad_token_id=3
    )
    talker = build_talker_request(payload, vocab_size=8, prompt_frames=PROMPT_FRAMES)

    assert thinker.max_new_tokens == num_frames
    assert thinker.req.sampling_params.max_new_tokens == num_frames
    assert talker.max_new_tokens == num_frames + 1
    assert talker.req.sampling_params.max_new_tokens == num_frames + 1
    assert len(talker.input_ids) == PROMPT_FRAMES


def test_thinker_prefill_carries_prompt_then_one_pad_position():
    data = build_thinker_request(
        make_payload(4), vocab_size=8, prompt_token_ids=[1, 2, 5], pad_token_id=3
    )
    assert data.input_ids.tolist() == [1, 2, 5, 3]
    assert data.req.origin_input_ids == [1, 2, 5, 3]
    assert data.pending_stream_tokens == []


def test_thinker_is_greedy_and_warns_on_ignored_temperature(caplog):
    with caplog.at_level(logging.WARNING):
        data = build_thinker_request(
            make_payload(4, {"temperature": 0.7}),
            vocab_size=8,
            prompt_token_ids=[1],
            pad_token_id=3,
        )
    # SamplingParams.normalize() expresses greedy as top_k=1.
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.ignore_eos
    assert any("temperature" in record.getMessage() for record in caplog.records)


def test_merge_for_talker_keeps_only_the_frame_count():
    perception = make_payload(9)
    state = NemotronVoiceChatState.from_dict(perception.data)
    state.text_ids = [1, 2, 3]
    perception.data = state.to_dict()

    merged = merge_for_talker({"perception": perception})
    merged_state = NemotronVoiceChatState.from_dict(merged.data)
    assert merged.request_id == "r"
    assert merged_state.num_frames == 9
    assert merged_state.text_ids == []
    assert merged_state.acoustic_frames is None


@pytest.mark.parametrize(
    "stage_name,field,value,expected",
    [
        ("talker", "gpu", 0, True),
        ("talker", "gpu", [0], True),
        ("code2wav", "gpu", [0], True),
        ("code2wav", "process", "codec", False),
        ("code2wav", "gpu", 1, False),
        ("code2wav", "gpu", None, False),
        ("talker", "tp_size", 2, False),
        ("code2wav", "tp_size", 2, False),
    ],
)
def test_local_code_handoff_depends_only_on_topology(
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    field: str,
    value: str | int | list[int] | None,
    expected: bool,
) -> None:
    for operation in ("is_available", "is_initialized", "init", "current_stream"):
        monkeypatch.setattr(
            torch.cuda, operation, Mock(side_effect=AssertionError("CUDA in config"))
        )
    config = NemotronVoiceChatPipelineConfig(model_path="unused")
    setattr(config.stage_named(stage_name), field, value)
    for stage_name in ("talker", "code2wav"):
        assert config.stage_factory_kwargs(stage_name) == {
            "can_use_local_code_handoff": expected
        }
    assert config.stage_factory_kwargs("thinker") == {}


class DeviceCodesTensor(torch.Tensor):
    """CPU storage with a CUDA device contract for producer dispatch tests."""

    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:2")

    def cpu(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)


def make_decode_runner(
    codes: list[torch.Tensor], can_use_local_code_handoff: bool = True
) -> NemotronVoiceChatTalkerModelRunner:
    runner = NemotronVoiceChatTalkerModelRunner.__new__(
        NemotronVoiceChatTalkerModelRunner
    )
    runner.can_use_local_code_handoff = can_use_local_code_handoff
    runner.generate_codes = Mock(side_effect=codes)
    return runner


def make_decode_requests(count: int = 2) -> list[SchedulerRequest]:
    return [
        SchedulerRequest(
            request_id=f"request-{index}",
            data=SGLangARRequestData(talker_model_inputs={"codes_rows": []}),
        )
        for index in range(count)
    ]


def test_batch_code_messages_share_event_recorded_after_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codes = [
        torch.full((1, 4), index, dtype=torch.long).as_subclass(DeviceCodesTensor)
        for index in range(2)
    ]
    runner = make_decode_runner(codes)
    requests = make_decode_requests()
    producer_stream = Mock(spec=torch.cuda.Stream)
    event = Mock(spec=torch.cuda.Event)
    event_factory = Mock(return_value=event)
    current_stream = Mock(return_value=producer_stream)
    monkeypatch.setattr(torch.cuda, "Event", event_factory)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    operations = Mock()
    operations.attach_mock(runner.generate_codes, "generate")
    operations.attach_mock(event.record, "record")

    runner.post_decode(None, None, None, requests)
    assert operations.mock_calls == [
        call.generate(0),
        call.generate(1),
        call.record(producer_stream),
    ]
    event_factory.assert_called_once_with()
    current_stream.assert_called_once_with(codes[0].device)
    for request, chunk in zip(requests, codes, strict=True):
        message = talker_stream_output_builder(request.request_id, request.data, None)[
            0
        ]
        assert message.data is chunk
        assert message.metadata == {
            "modality": "audio_codes",
            "codes_ready_event": event,
        }
        assert (
            talker_stream_output_builder(request.request_id, request.data, None) == []
        )


@pytest.mark.parametrize(
    "can_use_local_code_handoff,is_cuda", [(False, False), (False, True), (True, False)]
)
def test_cpu_code_fallback_has_no_cuda_event(
    monkeypatch: pytest.MonkeyPatch,
    can_use_local_code_handoff: bool,
    is_cuda: bool,
) -> None:
    codes = torch.tensor([[1, 2, 3, 4]])
    if is_cuda:
        codes = codes.as_subclass(DeviceCodesTensor)
    else:
        pass
    runner = make_decode_runner([codes], can_use_local_code_handoff)
    request = make_decode_requests(1)[0]
    for operation in ("Event", "current_stream"):
        monkeypatch.setattr(
            torch.cuda, operation, Mock(side_effect=AssertionError("CUDA in fallback"))
        )

    runner.post_decode(None, None, None, [request])
    message = talker_stream_output_builder(request.request_id, request.data, None)[0]
    assert message.data.device.type == "cpu"
    assert torch.equal(message.data, codes.cpu())
    assert message.metadata == {"modality": "audio_codes"}


@pytest.mark.parametrize(
    "can_use_local_code_handoff,token_is_cuda,expected_operations",
    [
        (True, True, ["sample", "stage", "generate"]),
        (True, False, ["sample", "generate"]),
        (False, True, ["generate", "sample"]),
    ],
)
def test_execute_stages_sampled_tokens_before_code_generation(
    can_use_local_code_handoff: bool,
    token_is_cuda: bool,
    expected_operations: list[str],
) -> None:
    runner = make_decode_runner(
        [torch.zeros(1, 4, dtype=torch.long)], can_use_local_code_handoff
    )
    requests = make_decode_requests(1)
    schedule_batch = Mock(is_prefill_only=False)
    forward_batch = Mock()
    result = Mock(next_token_ids=None, logits_output=Mock())
    runner.execution_context = Mock(return_value=nullcontext())
    runner.build_forward_batch = Mock(
        return_value=(forward_batch, schedule_batch, False)
    )
    runner.before_decode = Mock()
    runner.custom_decode_forward = Mock(return_value=result)
    runner.publish_next_tokens = Mock()
    runner.finalize = Mock()
    sampled_tokens = Mock(is_cuda=token_is_cuda)
    runner.sample_next_token_ids = Mock(return_value=sampled_tokens)
    runner.stage_token_ids = Mock()
    operations = Mock()
    operations.attach_mock(runner.sample_next_token_ids, "sample")
    operations.attach_mock(runner.stage_token_ids, "stage")
    operations.attach_mock(runner.generate_codes, "generate")

    assert (
        runner.execute(SchedulerOutput(requests, schedule_batch))
        is runner.finalize.return_value
    )
    assert [operation[0] for operation in operations.mock_calls] == expected_operations
    assert result.next_token_ids is sampled_tokens
    if can_use_local_code_handoff and token_is_cuda:
        runner.stage_token_ids.assert_called_once_with(result, sampled_tokens)
    else:
        runner.stage_token_ids.assert_not_called()
