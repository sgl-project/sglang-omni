# SPDX-License-Identifier: Apache-2.0
"""Frame-count contract between the thinker and talker requests."""

import logging

import pytest

from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.request_builders import (
    build_talker_request,
    build_thinker_request,
    merge_for_talker,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

PROMPT_FRAMES = 37


def _payload(num_frames, params=None):
    return StagePayload(
        "r",
        request=OmniRequest(inputs={}, params=params or {}),
        data=NemotronVoiceChatState(num_frames=num_frames).to_dict(),
    )


@pytest.mark.parametrize("num_frames", [1, 2, PROMPT_FRAMES, 513])
def test_talker_decode_steps_match_thinker_tokens(num_frames):
    """The prefill emits no codes, so the talker needs one extra generation step."""
    payload = _payload(num_frames)
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
        _payload(4), vocab_size=8, prompt_token_ids=[1, 2, 5], pad_token_id=3
    )
    assert data.input_ids.tolist() == [1, 2, 5, 3]
    assert data.req.origin_input_ids == [1, 2, 5, 3]
    assert data.pending_stream_tokens == []


def test_thinker_is_greedy_and_warns_on_ignored_temperature(caplog):
    with caplog.at_level(logging.WARNING):
        data = build_thinker_request(
            _payload(4, {"temperature": 0.7}),
            vocab_size=8,
            prompt_token_ids=[1],
            pad_token_id=3,
        )
    # SamplingParams.normalize() expresses greedy as top_k=1.
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.ignore_eos
    assert any("temperature" in record.getMessage() for record in caplog.records)


def test_merge_for_talker_keeps_only_the_frame_count():
    perception = _payload(9)
    state = NemotronVoiceChatState.from_dict(perception.data)
    state.text_ids = [1, 2, 3]
    perception.data = state.to_dict()

    merged = merge_for_talker({"perception": perception})
    merged_state = NemotronVoiceChatState.from_dict(merged.data)
    assert merged.request_id == "r"
    assert merged_state.num_frames == 9
    assert merged_state.text_ids == []
    assert merged_state.acoustic_frames is None
