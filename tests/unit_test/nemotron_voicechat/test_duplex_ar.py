# SPDX-License-Identifier: Apache-2.0
"""AR adapters preserve fusion inputs across streaming continuations."""

from unittest.mock import Mock

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import PreTrainedTokenizerBase

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.nemotron_voicechat.duplex_ar import (
    DuplexThinkerRunner,
    ThinkerAdapter,
)
from sglang_omni.models.nemotron_voicechat.fusion import AddFusion
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.types import SchedulerRequest


@pytest.fixture
def thinker_adapter() -> ThinkerAdapter:
    embeddings = torch.nn.Embedding.from_pretrained(
        torch.arange(40).float().reshape(10, 4)
    )
    model = Mock(
        llm=Mock(
            get_input_embeddings=Mock(return_value=embeddings),
            config=Mock(vocab_size=10),
        ),
        fusion=AddFusion(
            {
                "duplex_user_channel_weight": 2,
                "duplex_text_channel_weight": 3,
                "duplex_function_channel_weight": 5,
            }
        ),
    )
    tokenizer = Mock(spec=PreTrainedTokenizerBase, all_special_ids=[0])
    tokenizer.convert_tokens_to_ids.return_value = 0
    tokenizer.decode.side_effect = lambda token_ids: "x" * len(token_ids)
    return ThinkerAdapter(
        Mock(spec=DuplexThinkerRunner, model=model),
        prompt_token_ids=[1, 2],
        pad_token_id=0,
        tokenizer=tokenizer,
        context_length=8,
    )


@pytest.mark.parametrize("cached_positions", [0, 2, 3])
def test_runner_forwards_only_uncached_fusion_inputs(
    thinker_adapter: ThinkerAdapter,
    cached_positions: int,
) -> None:
    session_identity = SessionIdentity("fusion")
    thinker_adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    first_request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("first", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
    )
    first_request.output_ids = [3]
    first_request.extra_model_outputs = {"function_ids": [4]}
    thinker_adapter.result(session_identity, first_request)
    next_request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("next", OmniRequest(None), {"acoustic": torch.full((1, 4), 7)}),
    )
    forward_batch = Mock(
        spec=ForwardBatch,
        extend_prefix_lens_cpu=[cached_positions],
        extend_seq_lens_cpu=[4 - cached_positions],
        replace_embeds=None,
        input_ids=torch.zeros(4 - cached_positions, dtype=torch.long),
    )
    DuplexThinkerRunner.before_prefill(
        thinker_adapter.runner,
        forward_batch,
        Mock(),
        [SchedulerRequest("next", data=next_request)],
    )
    embeddings = thinker_adapter.runner.model.llm.get_input_embeddings().weight
    expected_rows = torch.stack(
        [
            2 * embeddings[1] + 8 * embeddings[0],
            2 * embeddings[2] + 8 * embeddings[0],
            2 * torch.ones(4) + 8 * embeddings[0],
            2 * torch.full((4,), 7) + 3 * embeddings[3] + 5 * embeddings[4],
        ]
    )
    actual_inputs = get_omni_prefill_inputs(forward_batch)
    torch.testing.assert_close(
        actual_inputs.input_embeds, expected_rows[cached_positions:]
    )


def test_thinker_continuation_reuses_pending_token_position(
    thinker_adapter: ThinkerAdapter,
) -> None:
    session_identity = SessionIdentity("continuation")
    thinker_adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    first_request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("first", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
    )
    assert first_request.input_ids.tolist() == [1, 2, 0]
    first_request.output_ids = [3]
    first_request.extra_model_outputs = {"function_ids": [4]}
    first_output = thinker_adapter.result(session_identity, first_request)
    assert first_output.data["text"] == "x"
    assert first_output.data["text_token"] == 3
    assert first_output.data["function_token"] == 4
    next_request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("next", OmniRequest(None), {"acoustic": torch.full((1, 4), 7)}),
    )
    assert next_request.input_ids.numel() == 0
    assert next_request.max_new_tokens == 1
    thinker_adapter.close(session_identity)
    thinker_adapter.open(session_identity, OmniRequest(None))
    reopened_request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("reopened", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
    )
    assert reopened_request.input_ids.tolist() == [1, 2, 0]
