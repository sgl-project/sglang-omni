# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts_realtime import model_runner as model_runner_module
from sglang_omni.models.moss_tts_realtime.model_runner import MossTTSRealtimeModelRunner
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimePendingInput,
    MossTTSRealtimeRequestData,
    MossTTSRealtimeTurnLedger,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeTurnState,
)
from sglang_omni.models.moss_tts_realtime.state_pool import (
    MossTTSRealtimeDecodeStatePool,
)
from sglang_omni.scheduling.types import RequestOutput, SchedulerRequest
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_VOCAB_SIZE as MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE,
)
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG
from tests.unit_test.moss_tts_realtime.runtime_config import (
    REFERENCE_AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
)


class _FakeModel:
    def __init__(self) -> None:
        self.hidden_size = 4
        self.dtype = torch.float32
        self.config = MODEL_CONFIG
        self.local_transformer = SimpleNamespace(
            local_lm_heads=[
                torch.nn.Linear(
                    self.hidden_size,
                    MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE,
                    bias=False,
                )
            ]
        )
        self._decode_input_embedding = torch.nn.Embedding(4, self.hidden_size)
        self._decode_input_embedding.weight.requires_grad_(False)
        self._state_pool: MossTTSRealtimeDecodeStatePool | None = None
        self.seen_rows: list[torch.Tensor] = []
        self.fixed_frame: torch.Tensor | None = None

    def init_decode_state_pool(
        self,
        *,
        max_running_requests: int,
        max_history_frames: int = 1000,
    ) -> MossTTSRealtimeDecodeStatePool:
        self._state_pool = MossTTSRealtimeDecodeStatePool(
            self,
            max_running_requests=max_running_requests,
            max_history_frames=max_history_frames,
        )
        return self._state_pool

    @property
    def state_pool(self) -> MossTTSRealtimeDecodeStatePool:
        assert self._state_pool is not None
        return self._state_pool

    def _prepare_multi_modal_inputs(self, rows: torch.Tensor) -> torch.Tensor:
        self.seen_rows.append(rows.detach().clone())
        values = rows.to(torch.float32).sum(dim=-1, keepdim=True)
        return values.repeat(1, self.hidden_size)

    def decode_local_frame(self, hidden_states, *, sample_audio):
        if self.fixed_frame is not None:
            return self.fixed_frame.to(hidden_states.device).expand(
                hidden_states.shape[0], -1
            )
        codes = []
        for codebook in range(16):
            logits = torch.zeros(
                hidden_states.shape[0],
                MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE,
                device=hidden_states.device,
            )
            logits[:, codebook + 1] = 10
            codes.append(sample_audio(logits, codebook))
        return torch.stack(codes, dim=-1)

    def reset_request(self, rid: str, turn_state=None):
        return self.state_pool.release_row(rid, turn_state)


class _Outbox:
    def __init__(self) -> None:
        self.messages = []

    def put(self, message) -> None:
        self.messages.append(message)


def _turn(rid: str) -> MossTTSRealtimeTurnState:
    turn = MossTTSRealtimeTurnState(
        session_id=f"session-{rid}",
        turn_id=f"turn-{rid}",
        request_id=rid,
        pending_input=MossTTSRealtimePendingInput(
            max_tokens=32,
            max_bytes=1024,
            max_updates=32,
        ),
        ledger=MossTTSRealtimeTurnLedger(model_config=MODEL_CONFIG),
    )
    turn.phase = MossTTSRealtimeTurnPhase.RUNNING
    return turn


def _request(
    rid: str,
    *,
    prompt_rows: torch.Tensor | None = None,
    seed: int = 7,
    stream: bool = True,
) -> SchedulerRequest:
    turn = _turn(rid)
    state = MossTTSRealtimeState(
        session_id=f"session-{rid}",
        turn_id=f"turn-{rid}",
        turn_index=0,
        generation_kwargs={
            "temperature": 0.0,
            "top_p": 0.6,
            "top_k": 30,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "repetition_window": 50,
            "seed": seed,
        },
    )
    extend_input_len = 0 if prompt_rows is None else int(prompt_rows.shape[0])
    req = SimpleNamespace(
        rid=rid,
        extend_range=SimpleNamespace(length=extend_input_len),
        prefix_indices=[],
        output_ids=[],
        is_chunked=0,
    )
    data = MossTTSRealtimeRequestData(
        req=req,
        state=state,
        turn_state=turn,
        prompt_rows=prompt_rows,
        provisional_output_id=MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
        sampling_seed=seed,
        stream_metadata={"n_vq": 16} if stream else None,
    )
    return SchedulerRequest(request_id=rid, data=data)


def _runner() -> tuple[MossTTSRealtimeModelRunner, _FakeModel]:
    model = _FakeModel()
    model.init_decode_state_pool(max_running_requests=4, max_history_frames=1000)
    runner = object.__new__(MossTTSRealtimeModelRunner)
    runner.model = model
    runner._outbox = None
    return runner, model


def _result(batch_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        logits_output=SimpleNamespace(
            hidden_states=torch.zeros(batch_size, 4, dtype=torch.float32)
        ),
        next_token_ids=None,
    )


def test_prefill_projects_only_the_uncached_canonical_rows() -> None:
    runner, model = _runner()
    rows = torch.arange(4 * 17, dtype=torch.long).reshape(4, 17)
    rows[:, 1:] %= MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE
    request = _request("a", prompt_rows=rows)
    request.data.req.prefix_indices = [1]
    request.data.req.extend_range = SimpleNamespace(length=2)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    runner.custom_prefill_forward(
        forward_batch,
        SimpleNamespace(),
        [request],
    )

    assert torch.equal(model.seen_rows[-1], rows[1:3])
    assert forward_batch.input_embeds.shape == (2, model.hidden_size)
    assert request.data.input_embeds_are_projected


def test_prefill_dispatch_events_include_shape_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _runner()
    events: list[dict] = []
    monkeypatch.setattr(model_runner_module, "realtime_events_active", lambda: True)
    monkeypatch.setattr(
        model_runner_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    rows = torch.arange(3 * 17, dtype=torch.long).reshape(3, 17)
    request = _request("a", prompt_rows=rows)
    request.data.req.prefix_indices = [0]
    request.data.req.extend_range = SimpleNamespace(length=2)
    schedule_batch = SimpleNamespace(
        is_prefill_only=True,
        is_extend_in_batch=True,
    )

    runner.before_prefill(SimpleNamespace(), schedule_batch, [request])
    runner.post_prefill(
        SimpleNamespace(can_run_cuda_graph=False),
        SimpleNamespace(),
        schedule_batch,
        [request],
    )

    assert [event["event_name"] for event in events] == [
        "prefill_dispatch_start",
        "prefill_dispatch_end",
    ]
    assert events[0]["metadata"] == {
        "session_id": "session-a",
        "turn_id": "turn-a",
        "turn_index": 0,
        "seq_no": None,
        "stable_token_count": 0,
        "prompt_rows": 3,
        "prefill_cached_rows": 1,
        "prefill_dispatch_rows": 2,
        "batch_size": 1,
        "is_prefill_only": True,
        "is_extend_in_batch": True,
        "is_chunked": False,
    }
    assert events[1]["metadata"]["can_run_cuda_graph"] is False


def test_first_codec_frame_ready_event_is_emitted_after_host_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _runner()
    events: list[dict] = []
    monkeypatch.setattr(model_runner_module, "realtime_events_active", lambda: True)
    monkeypatch.setattr(
        model_runner_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    outbox = _Outbox()
    runner.set_stream_outbox(outbox)
    request = _request("a")
    result = _result()
    runner._collect_frame(
        result,
        SimpleNamespace(),
        SimpleNamespace(output_ids=None),
        [request],
    )

    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=[request]),
        {"a": RequestOutput(request_id="a", data=result.next_token_ids[0])},
    )

    assert len(events) == 1
    assert events[0]["event_name"] == "first_codec_frame_ready"
    assert events[0]["request_id"] == "a"
    assert events[0]["metadata"]["frame_index"] == 0
    assert events[0]["metadata"]["ar_batch_size"] == 1
    assert request.data.turn_state.provisional_frame is not None
    assert outbox.messages[0].metadata == {
        "n_vq": 16,
        "session_id": "session-a",
        "turn_id": "turn-a",
        "turn_index": 0,
    }


def test_collect_journals_nonterminal_frame_and_streams_after_host_apply() -> None:
    runner, model = _runner()
    request = _request("a")
    schedule_batch = SimpleNamespace(output_ids=None)
    result = _result()

    runner._collect_frame(result, SimpleNamespace(), schedule_batch, [request])

    expected_frame = torch.arange(1, 17, dtype=torch.long).view(1, 16)
    assert result.next_token_ids.tolist() == [
        MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID
    ]
    assert torch.equal(result.moss_realtime_journal.frames, expected_frame)
    row = model.state_pool.row_for("a")
    assert row is not None
    assert bool(model.state_pool.provisional[row])
    assert request.data.turn_state.provisional_frame is None

    outbox = _Outbox()
    runner.set_stream_outbox(outbox)
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=[request]),
        {"a": RequestOutput(request_id="a", data=result.next_token_ids[0])},
    )

    assert request.data.turn_state.provisional_frame is not None
    assert request.data.turn_state.provisional_frame.audio_codes == tuple(range(1, 17))
    assert len(outbox.messages) == 1
    assert torch.equal(outbox.messages[0].data, expected_frame[0])
    assert outbox.messages[0].metadata == {"n_vq": 16}


def test_decode_rejects_unresolved_provisional_then_accepts_materialized_row() -> None:
    runner, model = _runner()
    request = _request("a")
    schedule_batch = SimpleNamespace(output_ids=None)
    result = _result()
    runner._collect_frame(result, SimpleNamespace(), schedule_batch, [request])
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=[request]),
        {"a": RequestOutput(request_id="a", data=result.next_token_ids[0])},
    )
    cache_key = 991
    forward_batch = SimpleNamespace(input_ids=torch.tensor([cache_key]))

    with pytest.raises(RuntimeError, match="unresolved provisional frame"):
        runner.before_decode(
            forward_batch,
            SimpleNamespace(),
            [request],
        )

    request.data.turn_state.materialize_provisional(
        next_text_token=42,
        cache_key=cache_key,
    )
    request.data.req.output_ids = [cache_key]
    runner.before_decode(
        forward_batch,
        SimpleNamespace(),
        [request],
    )

    row = model.state_pool.row_for("a")
    assert row is not None
    assert not bool(model.state_pool.provisional[row])
    assert bool(model.state_pool.feedback_valid[row])
    assert forward_batch.input_embeds is None
    assert forward_batch.input_ids.tolist() == [0]
    torch.testing.assert_close(
        model._decode_input_embedding.weight[0],
        model.state_pool.feedback_embeds[row],
    )
    assert torch.equal(
        model.seen_rows[-1],
        torch.tensor([[42, *range(1, 17)]], dtype=torch.long),
    )


def test_decode_rejects_scalar_key_that_does_not_match_materialized_row() -> None:
    runner, _ = _runner()
    request = _request("a")
    result = _result()
    runner._collect_frame(
        result,
        SimpleNamespace(),
        SimpleNamespace(output_ids=None),
        [request],
    )
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=[request]),
        {"a": RequestOutput(request_id="a", data=result.next_token_ids[0])},
    )
    request.data.turn_state.materialize_provisional(
        next_text_token=42,
        cache_key=100,
    )

    with pytest.raises(RuntimeError, match="do not match materialized row hashes"):
        runner.before_decode(
            SimpleNamespace(input_ids=torch.tensor([101])),
            SimpleNamespace(),
            [request],
        )


def test_audio_eos_is_terminal_id_and_is_not_streamed_as_audio() -> None:
    runner, model = _runner()
    terminal = torch.ones(1, 16, dtype=torch.long)
    terminal[0, 0] = MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID
    model.fixed_frame = terminal
    request = _request("a")
    result = _result()
    schedule_batch = SimpleNamespace(output_ids=None)

    runner._collect_frame(result, SimpleNamespace(), schedule_batch, [request])
    outbox = _Outbox()
    runner.set_stream_outbox(outbox)
    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=[request]),
        {"a": RequestOutput(request_id="a", data=result.next_token_ids[0])},
    )

    assert result.next_token_ids.tolist() == [MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID]
    assert request.data.turn_state.audio_eos_seen
    assert request.data.turn_state.provisional_frame is None
    assert outbox.messages == []
    row = model.state_pool.row_for("a")
    assert row is not None
    assert not bool(model.state_pool.has_previous_frame[row])


def test_journal_request_order_mismatch_fails_closed() -> None:
    runner, _ = _runner()
    request = _request("a")
    result = _result()
    runner._collect_frame(
        result,
        SimpleNamespace(),
        SimpleNamespace(output_ids=None),
        [request],
    )
    other = _request("b")

    with pytest.raises(RuntimeError, match="journal/batch alignment broken"):
        runner.post_process_outputs(
            result,
            SimpleNamespace(requests=[other]),
            {"b": RequestOutput(request_id="b", data=0)},
        )


def test_release_request_clears_pool_and_host_slot() -> None:
    runner, model = _runner()
    request = _request("a")
    model.state_pool.acquire_row("a", request.data)

    released = runner.release_request(request)

    assert released is not None
    assert model.state_pool.row_for("a") is None
    assert request.data.turn_state.model_state_slot_id is None


def test_async_lookahead_is_never_eligible() -> None:
    runner, _ = _runner()
    assert not runner.lookahead_eligible(SimpleNamespace())
    with pytest.raises(RuntimeError, match="does not support async lookahead"):
        runner.before_decode(
            SimpleNamespace(),
            SimpleNamespace(),
            [],
            is_lookahead=True,
        )
