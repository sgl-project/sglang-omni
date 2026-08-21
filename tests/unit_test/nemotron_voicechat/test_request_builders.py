# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.engine_builder import (
    VoiceChatTalkerEngineBuilder,
    VoiceChatThinkerEngineBuilder,
)
from sglang_omni.models.nemotron_voicechat.model_runner import VoiceChatModelRunner
from sglang_omni.models.nemotron_voicechat.request_builders import (
    TalkerAdapters,
    ThinkerAdapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import (
    DeferredAdmission,
    FollowupAdmission,
    RequestOutput,
)


class _Tokenizer:
    bos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text
        assert not add_special_tokens
        return [10, 11]

    def decode(self, ids: list[int], **_: object) -> str:
        return f"token-{ids[0]}"


def test_engine_builders_reserve_session_and_inflight_request_slots() -> None:
    common = {
        "context_length": 64,
        "max_sessions": 3,
        "total_gpu_memory_fraction": 0.5,
    }
    thinker = VoiceChatThinkerEngineBuilder(**common)
    talker = VoiceChatTalkerEngineBuilder(**common)

    for builder in (thinker, talker):
        assert builder.extra_scheduler_kwargs() == {
            "request_build_max_workers": 1,
            "enable_streaming_sessions": True,
        }
    assert thinker.generation_defaults(dtype="bfloat16")["max_running_requests"] == 6
    assert thinker.generation_defaults(dtype="bfloat16")["disable_cuda_graph"] is True
    talker_defaults = talker.generation_defaults(dtype="float32")
    assert talker_defaults["max_running_requests"] == 6
    assert talker_defaults["enable_tf32_matmul"] is True
    assert talker_defaults["disable_cuda_graph"] is True
    assert talker_defaults["chunked_prefill_size"] == -1

    with pytest.raises(ValueError, match="requires dtype='float32'"):
        talker.generation_defaults(dtype="bfloat16")


def _payload(frame_index: int, **data: object) -> StagePayload:
    return StagePayload(
        request_id=f"request-{frame_index}",
        request=OmniRequest(inputs={}),
        data={
            "event": "audio_frame",
            "session_id": "session-1",
            "frame_index": frame_index,
            **data,
        },
    )


def test_thinker_carries_function_token_between_session_turns() -> None:
    config = SimpleNamespace(
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        vocab_size=32,
    )
    adapters = ThinkerAdapters(
        config=config,
        tokenizer=_Tokenizer(),
        context_length=64,
    )
    first = adapters.request_builder(
        _payload(0, acoustic_embedding=torch.ones(1, 2), instructions="Be concise")
    )
    assert list(first.req.origin_input_ids) == [1, 10, 11, 2, 3]
    assert first.custom_inputs["is_initial_prefill"] is True
    second_deferred = adapters.request_builder(
        _payload(1, acoustic_embedding=torch.ones(1, 2))
    )
    assert isinstance(second_deferred, DeferredAdmission)
    assert not second_deferred.ready.done()
    first.output_ids = [7]
    first.extra_model_outputs = {"function_tokens": 9}

    output = adapters.result_adapter(first)
    assert output.data["text_token"] == 7
    assert output.data["function_token"] == 9
    assert output.data["text_delta"] == "token-7"

    assert second_deferred.ready.done()
    second = second_deferred.value
    assert list(second.req.origin_input_ids) == []
    assert second.custom_inputs["input_function_ids"] == [9]
    second.output_ids = [8]
    second.extra_model_outputs = {"function_tokens": 10}
    adapters.result_adapter(second)

    third = adapters.request_builder(_payload(2, acoustic_embedding=torch.ones(1, 2)))
    assert not isinstance(third, DeferredAdmission)
    assert third.custom_inputs["input_function_ids"] == [10]


def test_talker_chains_speaker_prefill_before_first_frame() -> None:
    adapters = TalkerAdapters(
        config=SimpleNamespace(vocab_size=2),
        speaker=torch.ones(3, 4),
        context_length=64,
    )
    prefill = adapters.request_builder(_payload(0, text_token=5))
    assert prefill.kind == "speaker_prefill"
    assert prefill.req.rid == prefill.stage_payload.request_id
    assert list(prefill.req.origin_input_ids) == [0, 0, 0]
    assert prefill.followup is not None
    assert prefill.followup.custom_inputs == {
        "text_token": 5,
        "previous_audio_codes": None,
    }
    second_deferred = adapters.request_builder(_payload(1, text_token=6))
    assert isinstance(second_deferred, DeferredAdmission)
    assert not second_deferred.ready.done()

    followup = adapters.result_adapter(prefill)
    assert isinstance(followup, FollowupAdmission)
    frame = followup.value
    frame.extra_model_outputs = {"audio_codes": [4, 3, 2]}
    output = adapters.result_adapter(frame)
    assert output.data["audio_codes"] == [4, 3, 2]

    assert second_deferred.ready.done()
    second = second_deferred.value
    assert second.custom_inputs == {
        "text_token": 6,
        "previous_audio_codes": [4, 3, 2],
    }
    second.extra_model_outputs = {"audio_codes": [7, 8, 9]}
    adapters.result_adapter(second)

    third = adapters.request_builder(_payload(2, text_token=7))
    assert not isinstance(third, DeferredAdmission)
    assert third.custom_inputs == {
        "text_token": 7,
        "previous_audio_codes": [7, 8, 9],
    }


def test_model_runner_preserves_customized_info_per_request() -> None:
    runner = object.__new__(VoiceChatModelRunner)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(customized_info={"audio_codes": [[1, 2], [3, 4]]})
    )
    scheduler_output = SimpleNamespace(
        requests=[
            SimpleNamespace(request_id="a"),
            SimpleNamespace(request_id="b"),
        ]
    )
    outputs = {
        "a": RequestOutput(request_id="a"),
        "b": RequestOutput(request_id="b"),
    }

    runner.post_process_outputs(result, scheduler_output, outputs)

    assert outputs["a"].extra == {"audio_codes": [1, 2]}
    assert outputs["b"].extra == {"audio_codes": [3, 4]}


def test_model_runner_installs_model_local_custom_inputs() -> None:
    runner = object.__new__(VoiceChatModelRunner)
    installed = []
    runner.model = SimpleNamespace(set_voicechat_custom_inputs=installed.append)
    requests = [
        SimpleNamespace(data=SimpleNamespace(custom_inputs={"frame": 1})),
        SimpleNamespace(data=SimpleNamespace(custom_inputs={"frame": 2})),
    ]

    runner.before_prefill(object(), object(), requests)

    assert installed == [[{"frame": 1}, {"frame": 2}]]
