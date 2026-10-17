# SPDX-License-Identifier: Apache-2.0
"""AR adapters preserve fusion inputs across streaming continuations."""

from collections import deque
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import PreTrainedTokenizerBase

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.nemotron_voicechat.duplex_ar import (
    DuplexTalkerRunner,
    DuplexThinkerRunner,
    TalkerAdapter,
    ThinkerAdapter,
)
from sglang_omni.models.nemotron_voicechat.fusion import AddFusion
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
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


def test_context_limit_rejects_unit_without_corrupting_reopened_session(
    thinker_adapter: ThinkerAdapter,
) -> None:
    session_identity = SessionIdentity("context-limit")
    thinker_adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    for unit_index in range(5):
        request = thinker_adapter.build(
            session_identity,
            chunk,
            StagePayload(
                str(unit_index), OmniRequest(None), {"acoustic": torch.ones(1, 4)}
            ),
        )
        request.output_ids = [3]
        request.extra_model_outputs = {"function_ids": [4]}
        thinker_adapter.result(session_identity, request)
    for attempt in range(2):
        with pytest.raises(ValueError, match="context limit"):
            thinker_adapter.build(
                session_identity,
                chunk,
                StagePayload(
                    f"overflow-{attempt}",
                    OmniRequest(None),
                    {"acoustic": torch.ones(1, 4)},
                ),
            )
    thinker_adapter.close(session_identity)
    thinker_adapter.open(session_identity, OmniRequest(None))
    request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("reopened", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
    )
    assert request.input_ids.tolist() == [1, 2, 0]


def test_thinker_sessions_keep_text_history_independent(
    thinker_adapter: ThinkerAdapter,
) -> None:
    first_session = SessionIdentity("first")
    second_session = SessionIdentity("second")
    for session_identity in (first_session, second_session):
        thinker_adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    texts: list[str] = []
    for session_identity, token_ids in [
        (first_session, [3]),
        (second_session, [5]),
        (first_session, [3, 4]),
    ]:
        request = thinker_adapter.build(
            session_identity,
            chunk,
            StagePayload("unit", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
        )
        request.output_ids = token_ids[-1:]
        request.extra_model_outputs = {"function_ids": [4]}
        texts.append(thinker_adapter.result(session_identity, request).data["text"])
        thinker_adapter.tokenizer.decode.assert_called_with(token_ids)
    assert texts == ["x", "x", "x"]
    thinker_adapter.close(first_session)
    request = thinker_adapter.build(
        second_session,
        chunk,
        StagePayload("still-open", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
    )
    assert request.input_ids.numel() == 0
    request.output_ids = [6]
    request.extra_model_outputs = {"function_ids": [4]}
    assert thinker_adapter.result(second_session, request).data["text"] == "x"
    thinker_adapter.tokenizer.decode.assert_called_with([5, 6])


def test_thinker_withholds_incomplete_unicode_and_special_tokens(
    thinker_adapter: ThinkerAdapter,
) -> None:
    thinker_adapter.tokenizer.decode.side_effect = ["\ufffd", "你", "你好"]
    session_identity = SessionIdentity("unicode")
    thinker_adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    deltas: list[str] = []
    for token_id in (0, 1, 2, 3):
        request = thinker_adapter.build(
            session_identity,
            chunk,
            StagePayload("unit", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
        )
        request.output_ids = [token_id]
        request.extra_model_outputs = {"function_ids": [4]}
        deltas.append(thinker_adapter.result(session_identity, request).data["text"])
    assert deltas == ["", "", "你", "好"]


def test_empty_eos_skips_ar_generation(thinker_adapter: ThinkerAdapter) -> None:
    session_identity = SessionIdentity("drain")
    thinker_adapter.open(session_identity, OmniRequest(None))
    eos_payload = StagePayload(
        "eos", OmniRequest(None), {"eos": True, "acoustic": None}
    )
    assert thinker_adapter.finish_input(session_identity, eos_payload) is eos_payload
    audio_payload = StagePayload(
        "last-audio", OmniRequest(None), {"eos": True, "acoustic": torch.ones(1, 4)}
    )
    assert thinker_adapter.finish_input(session_identity, audio_payload) is None


def test_runner_rejects_misaligned_fusion_history(
    thinker_adapter: ThinkerAdapter,
) -> None:
    session_identity = SessionIdentity("misaligned")
    thinker_adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    request = thinker_adapter.build(
        session_identity,
        chunk,
        StagePayload("unit", OmniRequest(None), {"acoustic": torch.ones(1, 4)}),
    )
    forward_batch = Mock(
        spec=ForwardBatch, extend_prefix_lens_cpu=[2], extend_seq_lens_cpu=[2]
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        DuplexThinkerRunner.before_prefill(
            thinker_adapter.runner,
            forward_batch,
            Mock(),
            [SchedulerRequest("unit", data=request)],
        )


def test_talker_continuation_fuses_previous_codes_with_current_text() -> None:
    def fuse_codes_and_text(
        previous_codes: torch.Tensor, text_token_id: int
    ) -> torch.Tensor:
        return previous_codes.float() + text_token_id

    runner = Mock(
        spec=DuplexTalkerRunner,
        model=Mock(fusion_buffer=torch.zeros(1, 4), config=Mock(vocab_size=8)),
    )
    runner.pad_codes.return_value = torch.zeros(1, 4, dtype=torch.long)
    runner.warmup.return_value = torch.ones(2, 4)
    runner.step_row.side_effect = fuse_codes_and_text
    runner.generate_codes.return_value = torch.tensor([[5, 6, 7, 8]])
    adapter = TalkerAdapter(runner, context_length=8)
    session_identity = SessionIdentity("talker")
    adapter.open(session_identity, OmniRequest(None))
    chunk = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    first_request = adapter.build(
        session_identity,
        chunk,
        StagePayload("first", OmniRequest(None), {"text_token": 3}),
    )
    assert first_request.input_ids.tolist() == [0, 0, 0]
    DuplexTalkerRunner.post_prefill(
        runner, Mock(), Mock(), Mock(), [SchedulerRequest("first", data=first_request)]
    )
    first_output = adapter.result(session_identity, first_request)
    torch.testing.assert_close(first_output.data["codes"], torch.tensor([[5, 6, 7, 8]]))
    next_request = adapter.build(
        session_identity,
        chunk,
        StagePayload("next", OmniRequest(None), {"text_token": 4}),
    )
    assert next_request.input_ids.numel() == 0
    forward_batch = Mock(
        spec=ForwardBatch,
        extend_prefix_lens_cpu=[3],
        extend_seq_lens_cpu=[1],
        replace_embeds=None,
        input_ids=torch.zeros(1, dtype=torch.long),
    )
    DuplexTalkerRunner.before_prefill(
        runner, forward_batch, Mock(), [SchedulerRequest("next", data=next_request)]
    )
    torch.testing.assert_close(
        get_omni_prefill_inputs(forward_batch).input_embeds,
        torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
    )


def test_talker_decode_rows_reach_fusion_buffer_without_autograd_history() -> None:
    projection = torch.nn.Linear(4, 4)

    def project_codes_and_text(
        previous_codes: torch.Tensor, text_token_id: int
    ) -> torch.Tensor:
        return projection(previous_codes.float() + text_token_id)

    runner = Mock(
        spec=NemotronVoiceChatTalkerModelRunner,
        model=Mock(
            fusion_buffer=torch.zeros(1, 4),
            fusion_mask=torch.zeros(1, dtype=torch.bool),
        ),
    )
    runner.step_row.side_effect = project_codes_and_text
    request_data = Mock(
        pending_text_queue=deque([3]),
        talker_model_inputs={"prev_codes": torch.zeros(1, 4, dtype=torch.long)},
    )
    NemotronVoiceChatTalkerModelRunner.before_decode(
        runner,
        Mock(spec=ForwardBatch),
        Mock(reqs=[Mock()]),
        [SchedulerRequest("unit", data=request_data)],
    )
    assert runner.model.fusion_buffer.grad_fn is None
    with torch.no_grad():
        expected_row = projection(torch.full((1, 4), 3.0))
    torch.testing.assert_close(runner.model.fusion_buffer, expected_row)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires GPU")
@torch.inference_mode()
def test_sampler_replay_uses_new_hidden_states_and_randomness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample_codes(
        hidden: torch.Tensor,
        head: torch.nn.Module,
        *,
        num_iter: int,
        exponent: float,
        top_p: float,
        noise_scale: float,
        assignment_counts: tuple[int, ...],
    ) -> torch.Tensor:
        return hidden + torch.rand_like(hidden)

    def initialize_runner(
        runner: NemotronVoiceChatTalkerModelRunner,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
    ) -> None:
        runner.model = Mock(
            hidden_out=torch.zeros(1, 16, device="cuda"),
            talker=Mock(num_quantizers=8, generate_codes=sample_codes),
            mog_head=Mock(),
        )
        runner.exponent = 1.0
        runner.top_p = 0.9
        runner.noise_scale = 1.0

    monkeypatch.setattr(
        NemotronVoiceChatTalkerModelRunner, "__init__", initialize_runner
    )
    runner = DuplexTalkerRunner(
        Mock(spec=ModelWorker), Mock(spec=SGLangOutputProcessor)
    )
    first_codes = runner.generate_codes(0)
    saved_codes = first_codes.clone()
    second_codes = runner.generate_codes(0)
    assert not torch.equal(first_codes, second_codes)
    runner.model.hidden_out.fill_(10)
    changed_codes = runner.generate_codes(0)
    assert torch.all(changed_codes >= 10)
    assert torch.all(changed_codes < 11)
    torch.testing.assert_close(first_codes, saved_codes, rtol=0, atol=0)
