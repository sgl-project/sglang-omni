# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimePendingInput,
    MossTTSRealtimeRequestData,
    MossTTSRealtimeTurnLedger,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeTurnState,
)
from sglang_omni.models.moss_tts_realtime.state_pool import (
    MossTTSRealtimeDecodeJournal,
    MossTTSRealtimeDecodeStatePool,
)
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_VOCAB_SIZE as MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE,
)
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG


def _model(hidden_size: int = 8) -> SimpleNamespace:
    head = torch.nn.Linear(hidden_size, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE, bias=False)
    return SimpleNamespace(
        config=MODEL_CONFIG,
        local_transformer=SimpleNamespace(local_lm_heads=[head]),
    )


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
    seed: int = 7,
    temperature: float = 0.8,
    top_p: float = 0.6,
    top_k: int = 30,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    repetition_window: int = 50,
) -> SchedulerRequest:
    state = MossTTSRealtimeState(
        generation_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "repetition_window": repetition_window,
            "seed": seed,
        }
    )
    data = MossTTSRealtimeRequestData(
        state=state,
        turn_state=_turn(rid),
        sampling_seed=seed,
    )
    return SchedulerRequest(request_id=rid, data=data)


def _frame(token: int) -> torch.Tensor:
    return torch.full((1, 16), token, dtype=torch.long)


def _materialize_pool_frame(
    pool: MossTTSRealtimeDecodeStatePool,
    rid: str,
    frame: torch.Tensor,
) -> None:
    row = (123, *tuple(int(value) for value in frame[0].tolist()))
    pool.mark_materialized(rid, row)


def _hf_sample(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    logits = logits / temperature
    top_k = min(max(top_k, 1), int(logits.shape[-1]))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -1:] = False
    remove = torch.zeros_like(logits, dtype=torch.bool).scatter(
        1,
        sorted_indices,
        sorted_indices_to_remove,
    )
    probs = F.softmax(logits.masked_fill(remove, float("-inf")), dim=-1)
    return torch.multinomial(probs, 1, generator=generator).view(-1)


def test_pool_capacity_slot_ownership_and_full_reset() -> None:
    pool = MossTTSRealtimeDecodeStatePool(
        _model(),
        max_running_requests=2,
        max_history_frames=64,
    )
    request_a = _request("a")
    request_b = _request("b")
    row_a = pool.acquire_row("a", request_a.data)
    row_b = pool.acquire_row("b", request_b.data)

    assert {row_a, row_b} == {0, 1}
    assert pool.padding_row == 2
    assert request_a.data.turn_state.model_state_slot_id == row_a
    with pytest.raises(RuntimeError, match="pool exhausted"):
        pool.acquire_row("c", _request("c").data)

    frames = _frame(5)
    pool.commit_frames(
        rids=["a"],
        pool_rows=[row_a],
        sample_positions=[0],
        frames=frames,
    )
    released = pool.release_row("a", request_a.data.turn_state)

    assert released == row_a
    assert request_a.data.turn_state.model_state_slot_id is None
    assert pool.row_for("a") is None
    assert torch.all(
        pool.previous_audio_frames[row_a] == MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID
    )
    assert not bool(pool.has_previous_frame[row_a])
    assert not bool(pool.provisional[row_a])
    assert pool.history_for("b").numel() == 0


def test_pool_resource_snapshot_tracks_current_and_high_water_rows() -> None:
    pool = MossTTSRealtimeDecodeStatePool(
        _model(),
        max_running_requests=2,
        max_history_frames=64,
    )
    request_a = _request("a")
    request_b = _request("b")

    assert pool.resource_snapshot() == {
        "model_state_capacity": 2,
        "model_state_active_rows": 0,
        "model_state_free_rows": 2,
        "model_state_max_active_rows_observed": 0,
    }
    pool.acquire_row("a", request_a.data)
    pool.acquire_row("b", request_b.data)
    assert pool.resource_snapshot() == {
        "model_state_capacity": 2,
        "model_state_active_rows": 2,
        "model_state_free_rows": 0,
        "model_state_max_active_rows_observed": 2,
    }

    pool.release_row("a", request_a.data.turn_state)
    pool.release_row("b", request_b.data.turn_state)
    assert pool.resource_snapshot() == {
        "model_state_capacity": 2,
        "model_state_active_rows": 0,
        "model_state_free_rows": 2,
        "model_state_max_active_rows_observed": 2,
    }


def test_acquire_is_idempotent_but_sampling_params_are_immutable() -> None:
    pool = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=1)
    request = _request("a", seed=11)
    row = pool.acquire_row("a", request.data)

    assert pool.acquire_row("a", request.data) == row
    changed = _request("a", seed=12)
    changed.data.turn_state.assign_model_state_slot(row)
    with pytest.raises(RuntimeError, match="sampling parameters cannot change"):
        pool.acquire_row("a", changed.data)


def test_repetition_window_must_fit_fixed_capacity() -> None:
    pool = MossTTSRealtimeDecodeStatePool(
        _model(),
        max_running_requests=1,
        max_history_frames=4,
    )
    with pytest.raises(ValueError, match="repetition_window exceeds"):
        pool.acquire_row("a", _request("a", repetition_window=5).data)
    assert pool.row_for("a") is None


def test_sampling_matches_hf_filter_and_explicit_generator_stream() -> None:
    pool = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=1)
    request = _request("a", seed=1234, temperature=0.7, top_p=0.72, top_k=31)
    row = pool.acquire_row("a", request.data)
    logits = torch.linspace(-4, 4, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE).view(1, -1)
    reference_generator = torch.Generator(device="cpu").manual_seed(1234)

    for codebook in range(16):
        expected = _hf_sample(
            logits,
            generator=reference_generator,
            temperature=0.7,
            top_p=0.72,
            top_k=31,
        )
        actual = pool.sample_audio(logits, codebook, [row])
        assert torch.equal(actual, expected)


def test_greedy_sampling_does_not_advance_generator() -> None:
    pool = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=1)
    request = _request("a", seed=9, temperature=0.0)
    row = pool.acquire_row("a", request.data)
    before = pool.snapshot_generator_states([row])
    logits = torch.randn(1, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE)

    sampled = pool.sample_audio(logits, 0, [row])
    after = pool.snapshot_generator_states([row])

    assert sampled.item() == int(torch.argmax(logits, dim=-1).item())
    assert torch.equal(before[0], after[0])


def test_generator_state_can_be_rolled_back_after_failed_step() -> None:
    pool = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=1)
    request = _request("a", seed=77)
    row = pool.acquire_row("a", request.data)
    logits = torch.randn(1, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE)
    before = pool.snapshot_generator_states([row])

    first = pool.sample_audio(logits, 0, [row])
    pool.restore_generator_states([row], before)
    replay = pool.sample_audio(logits, 0, [row])

    assert torch.equal(first, replay)


def test_repetition_penalty_uses_only_recent_per_codebook_history() -> None:
    pool = MossTTSRealtimeDecodeStatePool(
        _model(),
        max_running_requests=1,
        max_history_frames=4,
    )
    request = _request(
        "a",
        temperature=0.0,
        repetition_penalty=2.0,
        repetition_window=1,
    )
    row = pool.acquire_row("a", request.data)
    first = _frame(0)
    second = _frame(2)
    pool.commit_frames(rids=["a"], pool_rows=[row], sample_positions=[0], frames=first)
    _materialize_pool_frame(pool, "a", first)
    pool.commit_frames(rids=["a"], pool_rows=[row], sample_positions=[1], frames=second)
    _materialize_pool_frame(pool, "a", second)
    logits = torch.zeros(1, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE)
    logits[0, 0] = 10
    logits[0, 2] = 9

    sampled = pool.sample_audio(logits, 0, [row])

    assert sampled.item() == 0


def test_duplicate_history_token_receives_one_penalty_application() -> None:
    pool = MossTTSRealtimeDecodeStatePool(
        _model(),
        max_running_requests=1,
        max_history_frames=4,
    )
    request = _request("a", repetition_penalty=2.0, repetition_window=4)
    row = pool.acquire_row("a", request.data)
    repeated = _frame(3)
    for position in range(2):
        pool.commit_frames(
            rids=["a"],
            pool_rows=[row],
            sample_positions=[position],
            frames=repeated,
        )
        _materialize_pool_frame(pool, "a", repeated)
    params = pool.sampling_params_for(row)
    logits = torch.zeros(1, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE)
    logits[0, 3] = 10

    penalized = pool._apply_repetition_penalty(
        logits,
        row_idx=row,
        codebook=0,
        params=params,
    )

    assert penalized[0, 3].item() == 5


def test_nonterminal_frame_requires_materialization_before_next_commit() -> None:
    pool = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=1)
    request = _request("a")
    row = pool.acquire_row("a", request.data)
    frame = _frame(4)
    pool.commit_frames(rids=["a"], pool_rows=[row], sample_positions=[0], frames=frame)

    with pytest.raises(RuntimeError, match="must materialize"):
        pool.commit_frames(
            rids=["a"], pool_rows=[row], sample_positions=[1], frames=frame
        )
    with pytest.raises(ValueError, match="do not match"):
        pool.mark_materialized("a", (1, *([5] * 16)))

    pool.mark_materialized("a", (1, *([4] * 16)))
    embedding = torch.randn(1, pool.hidden_size)
    pool.stage_feedback(torch.tensor([row]), embedding)
    assert torch.equal(pool.feedback_for(torch.tensor([row])), embedding)


def test_audio_eos_has_no_previous_frame_or_provisional_state() -> None:
    pool = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=1)
    request = _request("a")
    row = pool.acquire_row("a", request.data)
    terminal = _frame(1)
    terminal[0, 0] = MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID

    eos = pool.commit_frames(
        rids=["a"],
        pool_rows=[row],
        sample_positions=[0],
        frames=terminal,
    )

    assert eos.tolist() == [True]
    assert not bool(pool.has_previous_frame[row])
    assert not bool(pool.provisional[row])
    assert bool(pool.audio_eos_seen[row])
    with pytest.raises(RuntimeError, match="already sampled audio EOS"):
        pool.commit_frames(
            rids=["a"],
            pool_rows=[row],
            sample_positions=[1],
            frames=terminal,
        )


def test_batch_order_does_not_change_per_request_rng_stream() -> None:
    logits_a = torch.randn(1, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE)
    logits_b = torch.randn(1, MOSS_TTS_REALTIME_AUDIO_VOCAB_SIZE)

    pool_ab = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=2)
    row_a = pool_ab.acquire_row("a", _request("a", seed=101).data)
    row_b = pool_ab.acquire_row("b", _request("b", seed=202).data)
    sampled_ab = pool_ab.sample_audio(
        torch.cat([logits_a, logits_b]),
        0,
        [row_a, row_b],
    )

    pool_ba = MossTTSRealtimeDecodeStatePool(_model(), max_running_requests=2)
    row_b2 = pool_ba.acquire_row("b", _request("b", seed=202).data)
    row_a2 = pool_ba.acquire_row("a", _request("a", seed=101).data)
    sampled_ba = pool_ba.sample_audio(
        torch.cat([logits_b, logits_a]),
        0,
        [row_b2, row_a2],
    )

    assert sampled_ab[0].item() == sampled_ba[1].item()
    assert sampled_ab[1].item() == sampled_ba[0].item()


def test_journal_rejects_misaligned_metadata() -> None:
    with pytest.raises(ValueError, match="pool row count"):
        MossTTSRealtimeDecodeJournal(
            rids=["a"],
            pool_rows=[],
            sample_positions=[0],
            frames=torch.zeros(1, int(MODEL_CONFIG.rvq), dtype=torch.long),
            eos_mask=torch.zeros(1, dtype=torch.bool),
            generator_states_before=(torch.zeros(1, dtype=torch.uint8),),
            model_config=MODEL_CONFIG,
        )
