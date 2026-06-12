# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-TTS Local decode-state pool (PR-A c3).

CPU-only: the pool derives its sizing/placement from a fake model exposing a
``_decode_input_embedding.weight`` tensor, so no CUDA is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

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
    return SimpleNamespace(_decode_input_embedding=embedding)


def _params(seed: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        text_temperature=0.5,
        text_top_p=0.9,
        text_top_k=40,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
        sampling_seed=seed,
    )


def test_pool_dims_derive_from_embedding_weight():
    """P = weight.shape[0] + 1; no literal row count, padding row reserved."""
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=4))
    assert pool.num_rows == 5
    assert pool.padding_row == 4
    assert pool.hidden_size == _HIDDEN
    assert pool.feedback_embeds.shape == (5, _HIDDEN)
    assert pool.feedback_embeds.dtype == torch.bfloat16
    for name in ("text_temp", "text_top_p", "audio_temp", "audio_top_p"):
        assert getattr(pool, name).shape == (5,)
        assert getattr(pool, name).dtype == torch.float32
    for name in ("text_top_k", "audio_top_k", "seeds"):
        assert getattr(pool, name).shape == (5,)
        assert getattr(pool, name).dtype == torch.int64


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
    pool.feedback_embeds[row].fill_(1.0)
    pool.release_row("a")
    assert torch.all(pool.feedback_embeds[row] == 0)
    assert pool.text_temp[row] == 0.0
    assert pool.audio_top_k[row] == 0
    assert pool.seeds[row] == 0


def test_reset_row_zeroes_all_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=99))
    pool.feedback_embeds[row].fill_(2.0)
    pool.reset_row(row)
    assert torch.all(pool.feedback_embeds[row] == 0)
    for name in (
        "text_temp",
        "text_top_p",
        "audio_temp",
        "audio_top_p",
        "text_top_k",
        "audio_top_k",
        "seeds",
    ):
        assert getattr(pool, name)[row] == 0


def test_write_params_writes_request_static_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=555))
    assert pool.text_temp[row].item() == torch.tensor(0.5, dtype=torch.float32).item()
    assert pool.text_top_p[row].item() == torch.tensor(0.9, dtype=torch.float32).item()
    assert pool.audio_temp[row].item() == torch.tensor(1.7, dtype=torch.float32).item()
    assert pool.audio_top_p[row].item() == torch.tensor(0.8, dtype=torch.float32).item()
    assert int(pool.text_top_k[row]) == 40
    assert int(pool.audio_top_k[row]) == 25
    assert int(pool.seeds[row]) == 555


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


def test_reset_for_refill_clears_active_row():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.ensure_params(row, "a", _params(seed=1))
    pool.feedback_embeds[row] = 1.0

    assert pool.reset_for_refill("a") is True
    assert int(pool.seeds[row]) == 0
    assert int(torch.count_nonzero(pool.feedback_embeds[row])) == 0
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
        SimpleNamespace(request_id="a", data=SimpleNamespace()),
        SimpleNamespace(request_id="b", data=SimpleNamespace()),
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
    requests = [SimpleNamespace(request_id="fresh", data=SimpleNamespace())]

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
    # Feedback stranded by the retraction (must be wiped by the resume).
    pool.feedback_embeds[row].fill_(5.0)

    # prompt_rows (2 frames) + already-generated output_rows (3 frames); the
    # resume re-prefills the whole span, so extend_input_len = 2 + 3.
    width = 13
    prompt_rows = torch.zeros((2, width), dtype=torch.long)
    generated = [torch.zeros(width, dtype=torch.long) for _ in range(3)]
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=5, prefix_indices=[], rid="a"),
        prompt_rows=prompt_rows,
        output_rows=generated,
    )
    sched_req = SimpleNamespace(request_id="a", data=data)

    runner = object.__new__(MossTTSLocalModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    runner._build_prefill_input_embeds(forward_batch, [sched_req])

    assert torch.all(pool.feedback_embeds[row] == 0), "stranded feedback must be wiped"
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
