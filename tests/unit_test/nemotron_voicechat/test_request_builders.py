# SPDX-License-Identifier: Apache-2.0
"""Frame-count contract between the thinker and talker requests."""

import logging
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.model_runner import (
    NemotronVoiceChatModelRunner,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.request_builders import (
    apply_thinker_result,
    build_talker_request,
    build_thinker_request,
    merge_for_talker,
)
from sglang_omni.models.nemotron_voicechat.thinker import NemotronVoiceChatForCausalLM
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.types import SchedulerRequest

PROMPT_FRAMES = 37


def make_payload(
    num_frames: int, params: dict[str, float] | None = None
) -> StagePayload:
    return StagePayload(
        "r",
        request=OmniRequest(inputs={}, params=params or {}),
        data=NemotronVoiceChatState(
            num_frames=num_frames,
            acoustic_frames=torch.zeros(num_frames + 1, 4),
        ).to_dict(),
    )


@pytest.mark.parametrize("num_frames", [1, 2, PROMPT_FRAMES, 513])
def test_talker_decode_steps_match_thinker_tokens(num_frames: int) -> None:
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


def test_thinker_prefill_carries_prompt_then_one_pad_position() -> None:
    data = build_thinker_request(
        make_payload(4), vocab_size=8, prompt_token_ids=[1, 2, 5], pad_token_id=3
    )
    assert data.input_ids.tolist() == [1, 2, 5, 3]
    assert data.req.origin_input_ids == [1, 2, 5, 3]
    assert data.pending_stream_tokens == []
    assert data.acoustic_frames is not None
    assert data.acoustic_frames.shape == (5, 4)
    assert data.stage_payload is not None
    assert "acoustic_frames_bytes" not in data.stage_payload.data


def test_thinker_runner_uses_current_request_prefill_span() -> None:
    """The Thinker prefill follows the scheduling span exposed by SGLang Req."""
    data = build_thinker_request(
        make_payload(4), vocab_size=8, prompt_token_ids=[1, 2], pad_token_id=3
    )
    data.req.prefix_len = 0
    data.req.extend_end = len(data.req.origin_input_ids)
    embedding = torch.nn.Embedding(8, 4)
    model = Mock(
        fusion_buffer=torch.empty(1, 4),
        fusion=Mock(user_weight=1.0, text_weight=1.0, function_weight=2.0),
    )
    model.llm.get_input_embeddings.return_value = embedding
    runner = object.__new__(NemotronVoiceChatModelRunner)
    runner.model = model

    rows = runner.build_prefill_rows([SchedulerRequest("r", data=data)])

    assert rows.shape == (3, 4)


def test_thinker_decode_always_consumes_staged_fusion_rows() -> None:
    """Every decode forward consumes the newly staged acoustic fusion rows."""
    model = object.__new__(NemotronVoiceChatForCausalLM)
    torch.nn.Module.__init__(model)
    model.fusion_buffer = torch.zeros(2, 4)
    model.has_staged_decode = True
    forwarded_rows: list[torch.Tensor] = []
    llm = Mock()

    def record_forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Mock,
        input_embeds_placeholder: None,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids, positions, forward_batch, input_embeds_placeholder
        forwarded_rows.append(input_embeds.clone())
        return torch.zeros(2, 4)

    llm.model.forward.side_effect = record_forward
    model.llm = llm
    model.sample_function_ids = Mock()
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    forward_batch = Mock()

    first_rows = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    second_rows = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
    model.fusion_buffer.copy_(first_rows)
    model.forward(input_ids, positions, forward_batch)
    model.fusion_buffer.copy_(second_rows)
    model.forward(input_ids, positions, forward_batch)

    torch.testing.assert_close(forwarded_rows[0], first_rows)
    torch.testing.assert_close(forwarded_rows[1], second_rows)
    llm.get_input_embeddings.assert_not_called()


def test_thinker_is_greedy_and_warns_on_ignored_temperature(
    caplog: pytest.LogCaptureFixture,
) -> None:
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


def test_thinker_result_releases_acoustic_frames() -> None:
    data = build_thinker_request(
        make_payload(4), vocab_size=8, prompt_token_ids=[1], pad_token_id=3
    )
    data.output_ids = [5, 6]

    payload = apply_thinker_result(data)

    assert data.acoustic_frames is None
    assert NemotronVoiceChatState.from_dict(payload.data).text_ids == [5, 6]


def test_merge_for_talker_keeps_only_the_frame_count() -> None:
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
