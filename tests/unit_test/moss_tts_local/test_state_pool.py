# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-TTS Local decode-state pool (PR-A c3).

CPU-only: the pool derives its sizing/placement from a fake model exposing a
``_decode_input_embedding.weight`` tensor, so no CUDA is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
    make_moss_tts_local_scheduler_adapters,
)
from sglang_omni.models.moss_tts_local.state_pool import (
    MossTTSLocalDecodeJournal,
    MossTTSLocalDecodeStatePool,
)
from sglang_omni.proto import OmniRequest, StagePayload

_HIDDEN = 8


def _model(max_running_requests: int = 4) -> SimpleNamespace:
    """Fake model exposing only what the pool reads."""
    weight = torch.zeros(max_running_requests, _HIDDEN, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    return SimpleNamespace(
        _decode_input_embedding=embedding,
        config=SimpleNamespace(n_vq=12, audio_vocab_size=1024),
    )


def _params(seed: int = 7, audio_repetition_penalty: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        text_temperature=0.5,
        text_top_p=0.9,
        text_top_k=40,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
        sampling_seed=seed,
        audio_repetition_penalty=audio_repetition_penalty,
    )


def test_pool_dims_derive_from_embedding_weight():
    """P = weight.shape[0] + 1; no literal row count, padding row reserved."""
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=4))
    assert pool.num_rows == 5
    assert pool.padding_row == 4
    assert pool.hidden_size == _HIDDEN
    assert pool.feedback_embeds.shape == (5, _HIDDEN)
    assert pool.feedback_embeds.dtype == torch.bfloat16
    for field in (pool.text_temp, pool.text_top_p, pool.audio_temp, pool.audio_top_p):
        assert field.shape == (5,)
        assert field.dtype == torch.float32
    for field in (pool.text_top_k, pool.audio_top_k, pool.seeds):
        assert field.shape == (5,)
        assert field.dtype == torch.int64
    assert pool.generation_steps.shape == (5,)
    assert pool.generation_steps.dtype == torch.int64
    assert pool.sampling_steps.shape == (5,)
    assert pool.sampling_steps.dtype == torch.int64
    assert pool.audio_repetition_penalty.shape == (5,)
    assert pool.audio_repetition_penalty.dtype == torch.float32
    assert pool.audio_token_presence.shape == (5, 12, 1024)
    assert pool.audio_token_presence.dtype == torch.bool


def test_acquire_is_idempotent_by_rid():
    pool = MossTTSLocalDecodeStatePool(_model())
    first = pool.acquire_row("a")
    again = pool.acquire_row("a")
    assert first == again
    # A second rid takes a different row.
    other = pool.acquire_row("b")
    assert other != first


def test_padding_row_never_acquired():
    """Real rows are 0..P-2; the padding row stays out of every assignment."""
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=4))
    acquired = {pool.acquire_row(f"r{i}") for i in range(4)}
    assert acquired == {0, 1, 2, 3}
    assert pool.padding_row not in acquired


def test_pool_exhaustion_raises():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    pool.acquire_row("a")
    pool.acquire_row("b")
    try:
        pool.acquire_row("c")
    except RuntimeError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on pool exhaustion")


def test_release_is_noop_for_unheld_rid():
    pool = MossTTSLocalDecodeStatePool(_model())
    # No row held: release must not raise or perturb the free list.
    free_before = list(pool._free_rows)
    pool.release_row("ghost")
    assert pool._free_rows == free_before


def test_release_frees_and_recycles_row():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    row_a = pool.acquire_row("a")
    pool.acquire_row("b")
    pool.release_row("a")
    assert pool.row_for("a") is None
    # The freed row is reusable.
    row_c = pool.acquire_row("c")
    assert row_c == row_a


def test_release_resets_row_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=123))
    pool.commit_generation_step("a", 3)
    pool.feedback_embeds[row].fill_(1.0)
    pool.release_row("a")
    assert torch.all(pool.feedback_embeds[row] == 0)
    assert pool.text_temp[row] == 0.0
    assert pool.audio_top_k[row] == 0
    assert pool.seeds[row] == 0
    assert pool.generation_steps[row] == 0
    assert pool.sampling_steps[row] == 0
    assert pool.audio_repetition_penalty[row] == 0.0
    assert int(torch.count_nonzero(pool.audio_token_presence[row])) == 0


def test_reset_row_zeroes_all_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=99))
    pool.feedback_embeds[row].fill_(2.0)
    pool.reset_row(row)
    assert torch.all(pool.feedback_embeds[row] == 0)
    for field in (
        pool.text_temp,
        pool.text_top_p,
        pool.audio_temp,
        pool.audio_top_p,
        pool.text_top_k,
        pool.audio_top_k,
        pool.seeds,
        pool.generation_steps,
        pool.sampling_steps,
        pool.audio_repetition_penalty,
    ):
        assert field[row] == 0
    assert int(torch.count_nonzero(pool.audio_token_presence[row])) == 0


def test_write_params_writes_request_static_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=555, audio_repetition_penalty=1.25))
    assert pool.text_temp[row].item() == torch.tensor(0.5, dtype=torch.float32).item()
    assert pool.text_top_p[row].item() == torch.tensor(0.9, dtype=torch.float32).item()
    assert pool.audio_temp[row].item() == torch.tensor(1.7, dtype=torch.float32).item()
    assert pool.audio_top_p[row].item() == torch.tensor(0.8, dtype=torch.float32).item()
    assert int(pool.text_top_k[row]) == 40
    assert int(pool.audio_top_k[row]) == 25
    assert int(pool.seeds[row]) == 555
    assert (
        pool.audio_repetition_penalty[row].item()
        == torch.tensor(1.25, dtype=torch.float32).item()
    )
    assert pool.rows_have_audio_repetition_penalty([row]) is True


def test_write_params_does_not_touch_other_rows():
    pool = MossTTSLocalDecodeStatePool(_model())
    row_a = pool.acquire_row("a")
    row_b = pool.acquire_row("b")
    pool.write_params(row_a, _params(seed=1))
    assert pool.seeds[row_b] == 0
    assert pool.text_temp[row_b] == 0.0


def test_ensure_params_writes_once_until_invalidated():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.ensure_params(row, "a", _params(seed=1))
    pool.ensure_params(row, "a", _params(seed=2))
    assert int(pool.seeds[row]) == 1

    pool.invalidate_params("a")
    pool.ensure_params(row, "a", _params(seed=2))
    assert int(pool.seeds[row]) == 2


def test_row_for_returns_none_when_unheld():
    pool = MossTTSLocalDecodeStatePool(_model())
    assert pool.row_for("nobody") is None
    row = pool.acquire_row("a")
    assert pool.row_for("a") == row


def test_commit_generation_step_updates_active_row():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")

    pool.commit_generation_step("a", 7)
    pool.commit_generation_step("ghost", 3)

    assert int(pool.generation_steps[row]) == 7
    assert int(pool.sampling_steps[row]) == 7


def test_commit_generation_steps_updates_active_rows():
    pool = MossTTSLocalDecodeStatePool(_model())
    row_a = pool.acquire_row("a")
    row_b = pool.acquire_row("b")

    pool.commit_generation_steps(
        torch.tensor([row_a, row_b], dtype=torch.long),
        torch.tensor([7, 9], dtype=torch.long),
    )

    assert int(pool.generation_steps[row_a]) == 7
    assert int(pool.generation_steps[row_b]) == 9
    assert int(pool.sampling_steps[row_a]) == 7
    assert int(pool.sampling_steps[row_b]) == 9


def test_reset_for_refill_clears_active_row():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.ensure_params(row, "a", _params(seed=1))
    pool.commit_generation_step("a", 4)
    pool.feedback_embeds[row] = 1.0
    pool.audio_token_presence[row, 0, 7] = True

    assert pool.reset_for_refill("a", generation_steps=4) is True
    assert int(pool.seeds[row]) == 0
    assert int(torch.count_nonzero(pool.feedback_embeds[row])) == 0
    assert int(torch.count_nonzero(pool.audio_token_presence[row])) == 0
    assert int(pool.generation_steps[row]) == 4
    assert int(pool.sampling_steps[row]) == 4
    # params were invalidated, so the next ensure_params re-writes them.
    pool.ensure_params(row, "a", _params(seed=2))
    assert int(pool.seeds[row]) == 2


def test_reset_for_refill_is_noop_for_unheld_rid():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.ensure_params(row, "a", _params(seed=1))

    assert pool.reset_for_refill("nobody") is False
    # the held row and its write-once flag are untouched.
    pool.ensure_params(row, "a", _params(seed=9))
    assert int(pool.seeds[row]) == 1


def test_prepare_active_rows_gathers_rows_and_params():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    reqs = [
        SimpleNamespace(request_id="a", data=_params(seed=11)),
        SimpleNamespace(request_id="b", data=_params(seed=22)),
    ]

    row_t, rows, has_audio_repetition_penalty = pool.prepare_active_rows(reqs)

    assert row_t.dtype == torch.long
    assert row_t.device == pool.device
    assert rows == [pool.row_for("a"), pool.row_for("b")]
    assert torch.equal(pool.seeds[row_t], torch.tensor([11, 22], dtype=torch.long))
    assert has_audio_repetition_penalty is False


def test_prepare_active_rows_reports_audio_repetition_penalty_from_pool():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    reqs = [
        SimpleNamespace(request_id="a", data=_params(seed=11)),
        SimpleNamespace(
            request_id="b", data=_params(seed=22, audio_repetition_penalty=1.2)
        ),
    ]

    _, rows, has_audio_repetition_penalty = pool.prepare_active_rows(reqs)

    assert has_audio_repetition_penalty is True
    assert pool.rows_have_audio_repetition_penalty(rows) is True


def test_audio_history_updates_pool_presence_mask():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    row_a = pool.acquire_row("a")
    row_b = pool.acquire_row("b")
    rows = torch.full((2, 13), 999, dtype=torch.long)
    rows[:, 1:] = torch.arange(24, dtype=torch.long).reshape(2, 12)
    rows[1, 2] = 2048

    pool.update_audio_history(torch.tensor([row_a, row_b]), rows)

    assert bool(pool.audio_token_presence[row_a, 0, 0])
    assert bool(pool.audio_token_presence[row_a, 11, 11])
    assert bool(pool.audio_token_presence[row_b, 0, 12])
    assert not bool(pool.audio_token_presence[row_b, 1, 2047 % 1024])


def test_rebuild_audio_history_clears_stale_presence():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.audio_token_presence[row, 0, 99] = True
    rows = []
    for token in (3, 5):
        row_t = torch.full((13,), 0, dtype=torch.long)
        row_t[1:] = token
        rows.append(row_t)

    assert pool.rebuild_audio_history("a", rows) is True

    assert not bool(pool.audio_token_presence[row, 0, 99])
    assert bool(pool.audio_token_presence[row, 0, 3])
    assert bool(pool.audio_token_presence[row, 0, 5])


def test_journal_holds_fields():
    rows = torch.arange(2 * 13, dtype=torch.long).reshape(2, 13)
    journal = MossTTSLocalDecodeJournal(rids=["a", "b"], pool_rows=[0, 1], rows=rows)
    assert journal.rids == ["a", "b"]
    assert journal.pool_rows == [0, 1]
    assert torch.equal(journal.rows, rows)


def test_feedback_gather_equals_old_popleft():
    model = _model(max_running_requests=4)
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    rows = [pool.acquire_row("a"), pool.acquire_row("b")]
    expected = torch.stack(
        [
            torch.arange(_HIDDEN, dtype=torch.bfloat16),
            torch.arange(_HIDDEN, dtype=torch.bfloat16) + 10,
        ],
        dim=0,
    )
    pool.feedback_embeds[torch.tensor(rows, dtype=torch.long)] = expected

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(input_ids=torch.full((2,), -1, dtype=torch.long))
    requests = [
        SimpleNamespace(request_id="a", data=_params(seed=1)),
        SimpleNamespace(request_id="b", data=_params(seed=2)),
    ]

    runner._write_decode_input_embedding(forward_batch, requests)

    assert torch.equal(model._decode_input_embedding.weight[:2], expected)
    assert torch.equal(forward_batch.input_ids, torch.tensor([0, 1]))


def test_fresh_row_zeros_feedback():
    model = _model(max_running_requests=2)
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(input_ids=torch.full((1,), -1, dtype=torch.long))
    requests = [SimpleNamespace(request_id="fresh", data=_params(seed=1))]

    runner._write_decode_input_embedding(forward_batch, requests)

    assert torch.equal(
        model._decode_input_embedding.weight[:1],
        torch.zeros((1, _HIDDEN), dtype=torch.bfloat16),
    )


def test_double_collect_overwrites_feedback():
    hidden_size = 4
    weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        _decode_input_embedding=embedding,
        _state_pool=None,
        config=SimpleNamespace(
            n_vq=12,
            audio_assistant_slot_token_id=1000,
            audio_end_token_id=1001,
        ),
        frame_graph_max_bs=0,
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    model.acquire_row = pool.acquire_row
    embeds = [
        torch.full((1, hidden_size), 1, dtype=torch.bfloat16),
        torch.full((1, hidden_size), 2, dtype=torch.bfloat16),
    ]

    def decode_frame(hidden_states, *, sample_text, sample_audio):
        del hidden_states, sample_text, sample_audio
        return (
            torch.zeros(1, dtype=torch.long),
            torch.full((1, 12), 7, dtype=torch.long),
        )

    def prepare_multi_modal_inputs(rows):
        del rows
        return embeds.pop(0)

    model.decode_frame = decode_frame
    model._prepare_multi_modal_inputs = prepare_multi_modal_inputs

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    data = SimpleNamespace(
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.0,
        audio_top_p=1.0,
        audio_top_k=50,
        sampling_seed=0,
        generation_steps=0,
        audio_repetition_penalty=1.0,
        output_rows=[],
    )
    request = SimpleNamespace(request_id="rid", data=data)

    for _ in range(2):
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=torch.zeros(1, hidden_size))
        )
        schedule_batch = SimpleNamespace()
        runner._collect_frame(result, None, schedule_batch, [request])

    row = pool.row_for("rid")
    assert row is not None
    assert torch.equal(
        pool.feedback_embeds[row],
        torch.full((hidden_size,), 2, dtype=torch.bfloat16),
    )


def test_collect_frame_reads_generation_steps_from_pool():
    hidden_size = 4
    weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        _decode_input_embedding=embedding,
        _state_pool=None,
        config=SimpleNamespace(
            n_vq=12,
            audio_assistant_slot_token_id=1000,
            audio_end_token_id=1001,
        ),
        frame_graph_max_bs=1,
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    captured = {}

    def decode_frame_graphed(hidden_states, **kwargs):
        del hidden_states
        captured["base_positions"] = kwargs["base_positions"].detach().clone()
        return (
            torch.zeros(1, dtype=torch.long),
            torch.full((1, 12), 7, dtype=torch.long),
            torch.ones((1, hidden_size), dtype=torch.bfloat16),
        )

    model.decode_frame_graphed = decode_frame_graphed

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.0,
        audio_top_p=1.0,
        audio_top_k=50,
        sampling_seed=0,
        generation_steps=99,
        audio_repetition_penalty=1.0,
        output_rows=[],
    )
    request = SimpleNamespace(request_id="rid", data=data)
    row = pool.acquire_row("rid")
    pool.ensure_params(row, "rid", data)
    pool.commit_generation_step("rid", 4)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, hidden_size))
    )

    runner._collect_frame(result, SimpleNamespace(), SimpleNamespace(), [request])

    assert torch.equal(captured["base_positions"], torch.tensor([4 * 13]))
    assert int(pool.sampling_steps[row]) == 5


def test_pool_sampling_position_leads_unresolved_lookahead_launches():
    hidden_size = 4
    weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        _decode_input_embedding=embedding,
        _state_pool=None,
        config=SimpleNamespace(
            n_vq=12,
            audio_assistant_slot_token_id=1000,
            audio_end_token_id=1001,
        ),
        frame_graph_max_bs=1,
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    captured = []

    def decode_frame_graphed(hidden_states, **kwargs):
        del hidden_states
        captured.append(kwargs["base_positions"].detach().clone())
        return (
            torch.zeros(1, dtype=torch.long),
            torch.full((1, 12), 7, dtype=torch.long),
            torch.ones((1, hidden_size), dtype=torch.bfloat16),
        )

    model.decode_frame_graphed = decode_frame_graphed

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.0,
        audio_top_p=1.0,
        audio_top_k=50,
        sampling_seed=0,
        generation_steps=0,
        audio_repetition_penalty=1.0,
        output_rows=[],
    )
    request = SimpleNamespace(request_id="rid", data=data)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, hidden_size))
    )

    runner._collect_frame(result, SimpleNamespace(), SimpleNamespace(), [request])
    runner._collect_frame(result, SimpleNamespace(), SimpleNamespace(), [request])

    row = pool.row_for("rid")
    assert row is not None
    assert torch.equal(captured[0], torch.tensor([0]))
    assert torch.equal(captured[1], torch.tensor([13]))
    assert int(pool.generation_steps[row]) == 0
    assert int(pool.sampling_steps[row]) == 2


def test_collect_frame_uses_eager_path_when_audio_repetition_penalty_active(
    monkeypatch,
):
    hidden_size = 4
    weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        _decode_input_embedding=embedding,
        _state_pool=None,
        config=SimpleNamespace(
            n_vq=12,
            audio_vocab_size=1024,
            audio_assistant_slot_token_id=1000,
            audio_end_token_id=1001,
        ),
        frame_graph_max_bs=1,
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    called = {"eager": False}
    sampled_audio_logits = []

    def sample_tokens(logits, *, temperature, top_p, top_k, seeds, positions):
        del temperature, top_p, top_k, seeds, positions
        sampled_audio_logits.append(logits.detach().clone())
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(
        MossTTSModelRunner,
        "_sample_tokens",
        staticmethod(sample_tokens),
    )

    def decode_frame_graphed(*args, **kwargs):
        del args, kwargs
        raise AssertionError("penalty-enabled frames must not use graph replay")

    def decode_frame(hidden_states, *, sample_text, sample_audio):
        del hidden_states, sample_text
        called["eager"] = True
        audio_logits = torch.zeros(1, 1024, dtype=torch.float32)
        audio_logits[0, 7] = 10.0
        audio_logits[0, 8] = 6.0
        channel0 = sample_audio(audio_logits, 0)
        codes = torch.full((1, 12), 99, dtype=torch.long)
        codes[:, 0] = channel0
        return (
            torch.zeros(1, dtype=torch.long),
            codes,
        )

    model.decode_frame_graphed = decode_frame_graphed
    model.decode_frame = decode_frame
    model._prepare_multi_modal_inputs = lambda rows: torch.ones(
        (rows.shape[0], hidden_size), dtype=torch.bfloat16
    )

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.0,
        audio_top_p=1.0,
        audio_top_k=50,
        sampling_seed=0,
        generation_steps=0,
        audio_repetition_penalty=2.0,
        output_rows=[],
    )
    request = SimpleNamespace(request_id="rid", data=data)
    row = pool.acquire_row("rid")
    pool.ensure_params(row, "rid", data)
    pool.audio_token_presence[row, 0, 7] = True
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, hidden_size))
    )

    runner._collect_frame(result, SimpleNamespace(), SimpleNamespace(), [request])

    assert called["eager"] is True
    assert len(sampled_audio_logits) == 1
    assert sampled_audio_logits[0][0, 7].item() == 5.0
    assert sampled_audio_logits[0][0, 8].item() == 6.0
    assert pool.row_for("rid") == row
    assert bool(pool.audio_token_presence[row, 0, 7])
    assert bool(pool.audio_token_presence[row, 0, 8])
    assert not bool(pool.audio_token_presence[row, 1, 8])


def test_cached_pool_rows_drive_collect_and_batched_step_commit():
    hidden_size = 4
    weight = torch.zeros(4, hidden_size, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        _decode_input_embedding=embedding,
        _state_pool=None,
        config=SimpleNamespace(
            n_vq=12,
            audio_vocab_size=1024,
            audio_assistant_slot_token_id=1000,
            audio_end_token_id=1001,
        ),
        frame_graph_max_bs=4,
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    runner.output_processor = SimpleNamespace(
        process=lambda batch_result, scheduler_output: {
            req.request_id: SimpleNamespace(data=1000, extra=None)
            for req in scheduler_output.requests
        }
    )

    def data(step, seed):
        return SimpleNamespace(
            req=SimpleNamespace(is_chunked=0),
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=50,
            sampling_seed=seed,
            generation_steps=step,
            audio_repetition_penalty=1.0,
            output_rows=[],
            extra_model_outputs={},
        )

    requests = [
        SimpleNamespace(request_id="a", data=data(step=4, seed=11)),
        SimpleNamespace(request_id="b", data=data(step=8, seed=22)),
    ]

    forward_batch = SimpleNamespace(input_ids=torch.full((2,), -1, dtype=torch.long))
    runner._write_decode_input_embedding(forward_batch, requests)

    row_t = forward_batch.moss_pool_row_t.clone()
    row_a, row_b = int(row_t[0]), int(row_t[1])
    assert [row_a, row_b] != [0, 1]
    assert forward_batch.moss_pool_rows == [row_a, row_b]
    assert torch.equal(forward_batch.input_ids, torch.tensor([0, 1]))

    def fail_prepare_active_rows(*args, **kwargs):
        del args, kwargs
        raise AssertionError("collect path must reuse cached moss_pool_row_t")

    pool.prepare_active_rows = fail_prepare_active_rows
    pool.commit_generation_steps(row_t, torch.tensor([4, 8], dtype=torch.long))
    captured = {}

    def decode_frame_graphed(hidden_states, **kwargs):
        del hidden_states
        captured["base_positions"] = kwargs["base_positions"].detach().clone()
        return (
            torch.zeros(2, dtype=torch.long),
            torch.stack(
                [
                    torch.full((12,), 7, dtype=torch.long),
                    torch.full((12,), 9, dtype=torch.long),
                ],
                dim=0,
            ),
            torch.tensor(
                [[3, 3, 3, 3], [5, 5, 5, 5]],
                dtype=torch.bfloat16,
            ),
        )

    model.decode_frame_graphed = decode_frame_graphed
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(2, hidden_size)),
        can_run_cuda_graph=False,
    )
    schedule_batch = SimpleNamespace(is_prefill_only=False, output_ids=None)
    scheduler_output = SimpleNamespace(requests=requests)

    runner._collect_frame(result, forward_batch, schedule_batch, requests)

    assert torch.equal(captured["base_positions"], torch.tensor([4 * 13, 8 * 13]))
    assert result.moss_journal.pool_rows == [row_a, row_b]
    assert torch.equal(pool.feedback_embeds[row_a], torch.full((4,), 3).bfloat16())
    assert torch.equal(pool.feedback_embeds[row_b], torch.full((4,), 5).bfloat16())
    assert int(pool.sampling_steps[row_a]) == 5
    assert int(pool.sampling_steps[row_b]) == 9

    runner._finalize(
        result,
        forward_batch,
        schedule_batch,
        SimpleNamespace(seq_lens=[1, 1], input_ids=torch.zeros(2, dtype=torch.long)),
        scheduler_output,
    )

    assert requests[0].data.generation_steps == 5
    assert requests[1].data.generation_steps == 9
    assert int(pool.generation_steps[row_a]) == 5
    assert int(pool.generation_steps[row_b]) == 9


def test_finalize_commits_generation_steps_to_pool():
    model = _model(max_running_requests=2)
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    model.config = SimpleNamespace(audio_end_token_id=1001)
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    runner.output_processor = SimpleNamespace(
        process=lambda batch_result, scheduler_output: {
            req.request_id: SimpleNamespace(data=1000, extra=None)
            for req in scheduler_output.requests
        }
    )
    data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        generation_steps=0,
        extra_model_outputs={},
        output_rows=[],
    )
    sched_req = SimpleNamespace(request_id="rid", data=data)
    row = pool.acquire_row("rid")

    runner._finalize(
        SimpleNamespace(
            next_token_ids=torch.tensor([0]),
            logits_output=None,
            can_run_cuda_graph=False,
            moss_journal=None,
        ),
        SimpleNamespace(),
        SimpleNamespace(is_prefill_only=False, output_ids=None),
        SimpleNamespace(seq_lens=[1], input_ids=torch.zeros(1, dtype=torch.long)),
        SimpleNamespace(requests=[sched_req]),
    )

    assert data.generation_steps == 1
    assert int(pool.generation_steps[row]) == 1


def test_resume_reprefill_overwrites_stranded_feedback():
    """Retraction resume wipes the stranded feedback row and forces a param
    re-write — the pool-row replacement for the old
    ``pending_feedback_queue.clear()``. Drives the retraction branch of
    ``_build_prefill_input_embeds`` (the only path that resets a live row).
    """
    model = _model(max_running_requests=4)
    model.hidden_size = _HIDDEN
    model.dtype = torch.bfloat16
    model._prepare_multi_modal_inputs = lambda rows: torch.zeros(
        (rows.shape[0], _HIDDEN), dtype=torch.bfloat16
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool

    row = pool.acquire_row("a")
    pool.ensure_params(row, "a", _params(seed=1))
    pool.commit_generation_step("a", 3)
    # Feedback stranded by the retraction (must be wiped by the resume).
    pool.feedback_embeds[row].fill_(5.0)
    pool.audio_token_presence[row, 0, 99] = True

    # prompt_rows (2 frames) + already-generated output_rows (3 frames); the
    # resume re-prefills the whole span, so extend_input_len = 2 + 3.
    width = 13
    prompt_rows = torch.zeros((2, width), dtype=torch.long)
    generated = []
    for token in (4, 5, 6):
        row_t = torch.zeros(width, dtype=torch.long)
        row_t[1:] = token
        generated.append(row_t)
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=5, prefix_indices=[], rid="a"),
        prompt_rows=prompt_rows,
        output_rows=generated,
        generation_steps=3,
    )
    sched_req = SimpleNamespace(request_id="a", data=data)

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    runner._build_prefill_input_embeds(forward_batch, [sched_req])

    assert torch.all(pool.feedback_embeds[row] == 0), "stranded feedback must be wiped"
    assert int(pool.generation_steps[row]) == 3, "resume must preserve sample position"
    assert not bool(pool.audio_token_presence[row, 0, 99])
    assert bool(pool.audio_token_presence[row, 0, 4])
    assert bool(pool.audio_token_presence[row, 0, 5])
    assert bool(pool.audio_token_presence[row, 0, 6])
    pool.ensure_params(row, "a", _params(seed=2))
    assert int(pool.seeds[row]) == 2, "params must be re-written on resume"


def test_collect_frame_skips_chunked_feedback_and_journal():
    hidden_size = 4
    weight = torch.zeros(3, hidden_size, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        _decode_input_embedding=embedding,
        _state_pool=None,
        config=SimpleNamespace(
            n_vq=12,
            audio_assistant_slot_token_id=1000,
            audio_end_token_id=1001,
        ),
        frame_graph_max_bs=0,
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    model.acquire_row = pool.acquire_row

    def decode_frame(hidden_states, *, sample_text, sample_audio):
        del hidden_states, sample_text, sample_audio
        return (
            torch.zeros(2, dtype=torch.long),
            torch.full((2, 12), 7, dtype=torch.long),
        )

    def prepare_multi_modal_inputs(rows):
        del rows
        return torch.tensor(
            [[1, 1, 1, 1], [2, 2, 2, 2]],
            dtype=torch.bfloat16,
        )

    model.decode_frame = decode_frame
    model._prepare_multi_modal_inputs = prepare_multi_modal_inputs

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model

    def data(is_chunked):
        return SimpleNamespace(
            req=SimpleNamespace(is_chunked=is_chunked),
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=50,
            sampling_seed=0,
            generation_steps=0,
            audio_repetition_penalty=1.0,
            output_rows=[],
        )

    requests = [
        SimpleNamespace(request_id="chunked", data=data(is_chunked=1)),
        SimpleNamespace(request_id="normal", data=data(is_chunked=0)),
    ]
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros(2, hidden_size))
    )
    schedule_batch = SimpleNamespace()

    runner._collect_frame(result, None, schedule_batch, requests)

    chunked_row = pool.row_for("chunked")
    normal_row = pool.row_for("normal")
    assert chunked_row is not None
    assert normal_row is not None
    assert torch.equal(
        pool.feedback_embeds[chunked_row],
        torch.zeros(hidden_size, dtype=torch.bfloat16),
    )
    assert torch.equal(
        pool.feedback_embeds[normal_row],
        torch.full((hidden_size,), 2, dtype=torch.bfloat16),
    )
    assert result.moss_journal.rids == ["normal"]
    assert result.moss_journal.pool_rows == [normal_row]
    assert result.moss_journal.rows.shape == (1, 13)
    assert int(torch.count_nonzero(pool.audio_token_presence[chunked_row])) == 0
    assert int(torch.count_nonzero(pool.audio_token_presence[normal_row])) == 0


def test_sampling_position_floor_is_sync_noop():
    """C5 soul: on the sync path generation_steps is incremented after every
    collect (base _finalize, the sole increment), so the floor
    max(sampling_steps, generation_steps) is a no-op — the RNG position is
    exactly generation_steps every step, bit-identical to pre-C5.
    """
    data = SimpleNamespace(generation_steps=0, sampling_steps=None)
    for step in range(5):
        pos = MossTTSLocalModelRunner._advance_sampling_position(data)
        # floor no-op: position == generation_steps == the true step index
        assert pos == data.generation_steps == step
        assert data.sampling_steps == step + 1
        data.generation_steps += 1  # base _finalize, after each sync collect


def test_sampling_position_floor_leads_under_lookahead():
    """C5 soul (async): generation_steps lags (it only moves at resolve), so the
    floor lifts the position to the launch-advanced sampling_steps — launch(N+1)
    samples at N+1, not the stale generation_steps (N). This is what makes async
    ON bit-identical to sync.
    """
    data = SimpleNamespace(generation_steps=0, sampling_steps=None)
    # launch(0): position 0; generation_steps NOT yet bumped (resolve lags).
    assert MossTTSLocalModelRunner._advance_sampling_position(data) == 0
    # launch(1) before resolve(0): generation_steps still 0, but the floor uses
    # the launch-advanced sampling_steps (1), so the position is 1, not stale 0.
    assert MossTTSLocalModelRunner._advance_sampling_position(data) == 1
    assert data.sampling_steps == 2


def test_resume_resets_sampling_steps_to_generation_steps():
    """C5 soul: the retraction-resume branch resets sampling_steps to
    generation_steps so a lookahead-advanced counter does not skip the resumed
    frame's RNG position. No-op on the sync path (already equal).
    """
    model = _model(max_running_requests=4)
    model.hidden_size = _HIDDEN
    model.dtype = torch.bfloat16
    model._prepare_multi_modal_inputs = lambda rows: torch.zeros(
        (rows.shape[0], _HIDDEN), dtype=torch.bfloat16
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    row = pool.acquire_row("a")
    pool.sampling_steps[row] = 99

    width = 13
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=5, prefix_indices=[], rid="a"),
        prompt_rows=torch.zeros((2, width), dtype=torch.long),
        output_rows=[torch.zeros(width, dtype=torch.long) for _ in range(3)],
        generation_steps=3,
        sampling_steps=99,  # lookahead-advanced, must be reset to generation_steps
    )
    sched_req = SimpleNamespace(request_id="a", data=data)

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    runner._build_prefill_input_embeds(forward_batch, [sched_req])

    assert (
        data.sampling_steps == 3
    ), "resume must reset sampling_steps to generation_steps"
    assert int(pool.sampling_steps[row]) == 3


def test_resume_with_empty_output_rows_still_resets_sampling_steps():
    """A request retracted before it emitted any frame has output_rows == [] but
    still holds a pool row and may carry a launch-advanced sampling_steps. The
    refill reset must fire off the held row, not off output_rows, so the resumed
    frame samples at generation_steps, not the stale launch position.
    """
    model = _model(max_running_requests=4)
    model.hidden_size = _HIDDEN
    model.dtype = torch.bfloat16
    model._prepare_multi_modal_inputs = lambda rows: torch.zeros(
        (rows.shape[0], _HIDDEN), dtype=torch.bfloat16
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    row = pool.acquire_row("a")  # launched once, so it holds a row
    pool.sampling_steps[row] = 1

    width = 13
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=2, prefix_indices=[], rid="a"),
        prompt_rows=torch.zeros((2, width), dtype=torch.long),
        output_rows=[],  # retracted before emitting any frame
        generation_steps=0,
        sampling_steps=1,  # one launch advanced it; resolve never ran
    )
    sched_req = SimpleNamespace(request_id="a", data=data)

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    runner._build_prefill_input_embeds(forward_batch, [sched_req])

    assert data.sampling_steps == 0, "empty-output_rows resume must still reset"
    assert int(pool.sampling_steps[row]) == 0
    # The next collect then samples the resumed frame at position 0, not stale 1.
    assert MossTTSLocalModelRunner._advance_sampling_position(data) == 0
    assert data.sampling_steps == 1


def test_journal_rid_assertion_fires():
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(audio_end_token_id=1001))
    journal = MossTTSLocalDecodeJournal(
        rids=["other"],
        pool_rows=[0],
        rows=torch.zeros((1, 13), dtype=torch.long),
    )
    result = SimpleNamespace(moss_journal=journal)
    sched_req = SimpleNamespace(
        request_id="rid",
        data=SimpleNamespace(req=None, output_rows=[]),
    )
    scheduler_output = SimpleNamespace(requests=[sched_req])
    outputs = {"rid": SimpleNamespace(data=1000)}

    try:
        runner.post_process_outputs(result, scheduler_output, outputs)
    except RuntimeError as exc:
        assert "journal/batch alignment broken" in str(exc)
    else:
        raise AssertionError("expected journal rid mismatch to raise")


def test_journal_length_mismatch_raises_runtime_error():
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(audio_end_token_id=1001))
    journal = MossTTSLocalDecodeJournal(
        rids=["rid"],
        pool_rows=[],
        rows=torch.zeros((1, 13), dtype=torch.long),
    )
    result = SimpleNamespace(moss_journal=journal)
    sched_req = SimpleNamespace(
        request_id="rid",
        data=SimpleNamespace(req=None, output_rows=[]),
    )
    scheduler_output = SimpleNamespace(requests=[sched_req])
    outputs = {"rid": SimpleNamespace(data=1000)}

    try:
        runner.post_process_outputs(result, scheduler_output, outputs)
    except RuntimeError as exc:
        assert "journal length mismatch" in str(exc)
    else:
        raise AssertionError("expected journal length mismatch to raise")


def test_stop_row_not_appended_via_journal():
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(audio_end_token_id=1001))
    row = torch.arange(13, dtype=torch.long)
    result = SimpleNamespace(
        moss_journal=MossTTSLocalDecodeJournal(
            rids=["rid"],
            pool_rows=[0],
            rows=row.reshape(1, 13),
        )
    )
    data = SimpleNamespace(req=None, output_rows=[])
    sched_req = SimpleNamespace(request_id="rid", data=data)
    scheduler_output = SimpleNamespace(requests=[sched_req])
    outputs = {"rid": SimpleNamespace(data=1001)}

    runner.post_process_outputs(result, scheduler_output, outputs)

    assert data.output_rows == []


def test_journal_rows_appended_to_output_rows():
    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(audio_end_token_id=1001))
    row = torch.arange(13, dtype=torch.long)
    result = SimpleNamespace(
        moss_journal=MossTTSLocalDecodeJournal(
            rids=["rid"],
            pool_rows=[0],
            rows=row.reshape(1, 13),
        )
    )
    data = SimpleNamespace(req=None, output_rows=[])
    sched_req = SimpleNamespace(request_id="rid", data=data)
    scheduler_output = SimpleNamespace(requests=[sched_req])
    outputs = {"rid": SimpleNamespace(data=1000)}

    runner.post_process_outputs(result, scheduler_output, outputs)

    assert len(data.output_rows) == 1
    assert torch.equal(data.output_rows[0], row)


def test_param_gather_matches_old_cache():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    data = _params(seed=12345)
    row = pool.acquire_row("rid")
    pool.write_params(row, data)
    row_t = torch.tensor([row], dtype=torch.long, device=pool.device)

    params = {
        "text_temp": pool.text_temp[row_t],
        "text_top_p": pool.text_top_p[row_t],
        "text_top_k": pool.text_top_k[row_t],
        "audio_temp": pool.audio_temp[row_t],
        "audio_top_p": pool.audio_top_p[row_t],
        "audio_top_k": pool.audio_top_k[row_t],
        "seeds": pool.seeds[row_t],
    }

    assert torch.equal(
        params["text_temp"],
        torch.tensor([float(data.text_temperature)], dtype=torch.float32),
    )
    assert torch.equal(
        params["text_top_p"],
        torch.tensor([float(data.text_top_p)], dtype=torch.float32),
    )
    assert torch.equal(
        params["text_top_k"],
        torch.tensor([int(data.text_top_k)], dtype=torch.long),
    )
    assert torch.equal(
        params["audio_temp"],
        torch.tensor([float(data.audio_temperature)], dtype=torch.float32),
    )
    assert torch.equal(
        params["audio_top_p"],
        torch.tensor([float(data.audio_top_p)], dtype=torch.float32),
    )
    assert torch.equal(
        params["audio_top_k"],
        torch.tensor([int(data.audio_top_k)], dtype=torch.long),
    )
    assert torch.equal(
        params["seeds"],
        torch.tensor([int(data.sampling_seed)], dtype=torch.long),
    )


def test_result_adapter_releases_row_when_apply_raises():
    reset_calls = []
    model = SimpleNamespace(reset_request=lambda rid: reset_calls.append(rid))
    _, result_adapter = make_moss_tts_local_scheduler_adapters(model=model)
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={}, params={}, metadata={}),
        data={},
    )
    data = MossTTSLocalSGLangRequestData(
        input_ids=torch.zeros(1, dtype=torch.long),
        max_new_tokens=1,
        temperature=0.0,
        output_ids=[],
        prompt_rows=torch.zeros((1, 13), dtype=torch.long),
        output_rows=[
            torch.zeros(13, dtype=torch.long),
            torch.zeros(12, dtype=torch.long),
        ],
        stage_payload=payload,
    )

    try:
        result_adapter(data)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected malformed output_rows to raise")

    assert reset_calls == ["rid"]
