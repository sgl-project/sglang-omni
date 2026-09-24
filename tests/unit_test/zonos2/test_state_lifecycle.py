# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import threading
import time
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang_omni.models.zonos2 import request_builders
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder
from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner
from sglang_omni.models.zonos2.payload_types import (
    FRAME_WIDTH,
    N_CODEBOOKS,
    Zonos2State,
)
from sglang_omni.models.zonos2.request_builders import (
    Zonos2SGLangRequestData,
    make_zonos2_scheduler_adapters,
)
from sglang_omni.models.zonos2.sglang_model import Zonos2SGLangModel
from sglang_omni.models.zonos2.state_pool import Zonos2DecodeStatePool
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class _ModelHarness:
    reset_request = Zonos2SGLangModel.reset_request

    def __init__(self, pool: Zonos2DecodeStatePool) -> None:
        self.decode_state_pool = pool


class _FakeCopyStream:
    def wait_event(self, _event) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _model_and_pool() -> tuple[_ModelHarness, Zonos2DecodeStatePool]:
    pool_owner = SimpleNamespace(
        decode_input_embedding=SimpleNamespace(
            weight=torch.zeros((2, 3), dtype=torch.float32)
        ),
        n_codebooks=N_CODEBOOKS,
    )
    pool = Zonos2DecodeStatePool(pool_owner)
    return _ModelHarness(pool), pool


def _poison_row(pool: Zonos2DecodeStatePool, row: int, value: int = 7) -> None:
    pool.feedback_embeds[row].fill_(value)
    pool.eos_frame_set[row] = True
    pool.eos_frame_val[row] = value
    pool.eos_countdown[row] = value
    pool.generation_step[row] = value
    pool.rep_hist[row].fill_(value)
    pool.rep_len[row] = value


def _assert_row_reset(pool: Zonos2DecodeStatePool, row: int) -> None:
    assert torch.count_nonzero(pool.feedback_embeds[row]) == 0
    assert not bool(pool.eos_frame_set[row])
    assert int(pool.eos_frame_val[row]) == 0
    assert int(pool.eos_countdown[row]) == 0
    assert int(pool.generation_step[row]) == 0
    assert torch.all(pool.rep_hist[row] == -1)
    assert int(pool.rep_len[row]) == 0


def _terminal_data(request_id: str = "req-zonos2") -> Zonos2SGLangRequestData:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=Zonos2State().to_dict(),
    )
    return Zonos2SGLangRequestData(
        prompt_rows=torch.zeros((1, FRAME_WIDTH), dtype=torch.long),
        output_codes=[torch.zeros(N_CODEBOOKS, dtype=torch.long)],
        engine_start_s=time.perf_counter(),
        stage_payload=payload,
    )


def test_length_terminal_releases_pool_row_through_scheduler_result_path(
    monkeypatch,
) -> None:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    request_id = "req-length"
    model, pool = _model_and_pool()
    _, result_adapter = make_zonos2_scheduler_adapters(model=model)
    data = _terminal_data(request_id)
    row = pool.acquire_row(request_id)
    _poison_row(pool, row)

    sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
    sampling_params.normalize(tokenizer=None)
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=sampling_params,
        vocab_size=2,
    )
    req._omni_data = data
    req._omni_terminal_claimed = False
    req.output_ids.append(1)
    req.update_finish_state()
    assert req.finished_reason.to_json()["type"] == "length"

    scheduler = object.__new__(OmniScheduler)
    scheduler.request_admission_lock = threading.RLock()
    scheduler.outbox = Queue()
    scheduler.aborted_request_ids = set()
    scheduler.completed_request_ids = {}
    scheduler.pending_stream_ingress = {}
    scheduler.request_finished_callback = None
    scheduler.first_emit_done = {request_id}
    scheduler.prefill_start_done = {request_id}
    scheduler.prefill_end_done = set()
    scheduler.result_adapter = result_adapter
    scheduler.model_runner = None
    scheduler.stream_output_builder = None
    monkeypatch.setattr(
        omni_scheduler_module,
        "get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )

    scheduler.stream_output([req])

    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert data.finish_reason == "length"
    assert result.data.data["completion_tokens"] == 1
    assert request_id not in pool.rid_to_row
    assert len(pool.free_rows) == pool.padding_row
    _assert_row_reset(pool, row)


def test_result_adapter_releases_state_when_serialization_fails(monkeypatch) -> None:
    request_id = "req-adapter-error"
    model, pool = _model_and_pool()
    _, result_adapter = make_zonos2_scheduler_adapters(model=model)
    row = pool.acquire_row(request_id)
    _poison_row(pool, row)

    def fail_result(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(request_builders, "apply_sglang_zonos2_result", fail_result)

    with pytest.raises(RuntimeError, match="serialization failed"):
        result_adapter(_terminal_data(request_id))

    assert request_id not in pool.rid_to_row
    assert len(pool.free_rows) == pool.padding_row
    _assert_row_reset(pool, row)


def test_engine_builder_abort_callback_is_safe_before_and_after_allocation() -> None:
    request_id = "req-abort"
    model, pool = _model_and_pool()
    builder = Zonos2EngineBuilder()
    builder.model = model
    abort_callback = builder.make_abort_callback()
    builder.model = None

    free_rows = list(pool.free_rows)
    abort_callback(request_id)
    assert pool.free_rows == free_rows

    row = pool.acquire_row(request_id)
    _poison_row(pool, row)
    abort_callback(request_id)
    abort_callback(request_id)

    assert request_id not in pool.rid_to_row
    assert len(pool.free_rows) == pool.padding_row
    assert len(set(pool.free_rows)) == pool.padding_row
    _assert_row_reset(pool, row)


def test_release_resets_reused_row_without_touching_mixed_batch_survivor() -> None:
    model, pool = _model_and_pool()
    done = SimpleNamespace(request_id="done")
    live = SimpleNamespace(request_id="live")
    done_row, live_row = (
        int(row) for row in pool.prepare_active_rows([done, live]).tolist()
    )
    _poison_row(pool, done_row, value=3)
    _poison_row(pool, live_row, value=11)

    model.reset_request(done.request_id)
    free_rows_after_release = len(pool.free_rows)
    model.reset_request(done.request_id)

    assert done.request_id not in pool.rid_to_row
    assert pool.row_for(live.request_id) == live_row
    assert pool.active_ids is None
    assert pool.active_rows is None
    _assert_row_reset(pool, done_row)
    assert torch.all(pool.feedback_embeds[live_row] == 11)
    assert bool(pool.eos_frame_set[live_row])
    assert int(pool.eos_frame_val[live_row]) == 11
    assert int(pool.eos_countdown[live_row]) == 11
    assert int(pool.generation_step[live_row]) == 11
    assert torch.all(pool.rep_hist[live_row] == 11)
    assert int(pool.rep_len[live_row]) == 11
    assert len(pool.free_rows) == free_rows_after_release
    assert len(set(pool.free_rows)) == len(pool.free_rows)

    active_rows = pool.prepare_active_rows([live, done])

    assert active_rows.tolist() == [live_row, done_row]
    _assert_row_reset(pool, done_row)
    owned_rows = set(pool.rid_to_row.values())
    free_rows = set(pool.free_rows)
    assert len(owned_rows) == len(pool.rid_to_row)
    assert len(free_rows) == len(pool.free_rows)
    assert owned_rows.isdisjoint(free_rows)
    assert len(owned_rows) + len(free_rows) == pool.padding_row


@pytest.mark.parametrize("state", ["active", "retracted", "finished"])
def test_resolve_collects_only_active_metadata_without_releasing_state(
    state: str,
) -> None:
    request_id = "req-resolve"
    model, pool = _model_and_pool()
    row = pool.acquire_row(request_id)

    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model = model
    runner.decode_requests = {}
    runner.copy_stream = _FakeCopyStream()
    data = SimpleNamespace(
        output_codes=[],
        eos_frame=None,
        req=SimpleNamespace(
            is_retracted=state == "retracted", finished=lambda: state == "finished"
        ),
    )
    request = SimpleNamespace(request_id=request_id, data=data)
    codes = list(range(N_CODEBOOKS))
    packed = torch.tensor([codes + [1, 5]], dtype=torch.int64)
    next_ids = torch.tensor([123], dtype=torch.int64)
    result = SimpleNamespace(next_token_ids=None)
    launch_buf = ([request], packed, N_CODEBOOKS, next_ids, object())

    with mock.patch("torch.cuda.stream", lambda _stream: contextlib.nullcontext()):
        runner.collect_resolve(launch_buf, result)

    if state == "active":
        assert data.output_codes[0].tolist() == codes
        assert data.eos_frame == 5
    else:
        assert data.output_codes == []
        assert data.eos_frame is None
    assert torch.equal(result.next_token_ids, next_ids)
    assert pool.row_for(request_id) == row
    assert row not in pool.free_rows


@pytest.mark.parametrize("start", [0, 1, 3])
@pytest.mark.parametrize("has_eos", [False, True])
def test_reprefill_replays_full_frames_and_rebuilds_decode_state(
    start: int, has_eos: bool
) -> None:
    prompt = torch.arange(2 * FRAME_WIDTH).reshape(2, FRAME_WIDTH)
    codes = torch.arange(3 * N_CODEBOOKS).reshape(3, N_CODEBOOKS)
    if has_eos:
        codes[1, 2] = 99
    else:
        pass
    model = SimpleNamespace(
        decode_input_embedding=SimpleNamespace(weight=torch.zeros(2, FRAME_WIDTH)),
        n_codebooks=N_CODEBOOKS,
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=SimpleNamespace(text_vocab=100, eoa_id=99),
        embed_frames=lambda rows: rows.float(),
    )
    pool = Zonos2DecodeStatePool(model)
    model.decode_state_pool = pool
    row = pool.acquire_row("replay")
    _poison_row(pool, row)
    survivor = pool.acquire_row("survivor")
    _poison_row(pool, survivor, 11)
    data = SimpleNamespace(
        req=SimpleNamespace(
            extend_range=SimpleNamespace(start=start, end=5, length=5 - start),
            output_ids=[1, 2, 3],
        ),
        prompt_rows=prompt,
        output_codes=list(codes.unbind()),
        speaker_emb=None,
    )
    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model = model
    runner.decode_requests = {}

    actual = runner.build_prefill_embeds(
        None, [SimpleNamespace(request_id="replay", data=data)]
    )

    expected = torch.cat([prompt, torch.cat([codes, torch.full((3, 1), 100)], dim=1)])
    assert torch.equal(actual, expected[start:].float())
    assert int(pool.generation_step[row]) == 3
    assert int(pool.rep_len[row]) == 3
    assert torch.equal(pool.rep_hist[row, -3:], codes)
    assert torch.all(pool.rep_hist[row, :-3] == -1)
    assert bool(pool.eos_frame_set[row]) == has_eos
    assert int(pool.eos_frame_val[row]) == 0
    assert int(pool.eos_countdown[row]) == (8 if has_eos else 0)
    assert torch.count_nonzero(pool.feedback_embeds[row]) == 0
    assert int(pool.generation_step[survivor]) == 11
    assert torch.all(pool.rep_hist[survivor] == 11)


@pytest.mark.parametrize("committed_tokens", [0, 1])
def test_reprefill_missing_frames_fails_before_resetting_decode_state(
    committed_tokens: int,
) -> None:
    model, pool = _model_and_pool()
    model.device = torch.device("cpu")
    row = pool.acquire_row("replay")
    _poison_row(pool, row)
    data = SimpleNamespace(
        req=SimpleNamespace(
            extend_range=SimpleNamespace(start=0, end=3, length=3),
            output_ids=[1] * committed_tokens,
        ),
        prompt_rows=torch.zeros(2, FRAME_WIDTH, dtype=torch.long),
        output_codes=[],
    )
    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model = model
    runner.decode_requests = {}
    with pytest.raises(
        AssertionError, match="missing generated frames|committed tokens"
    ):
        runner.build_prefill_embeds(
            None, [SimpleNamespace(request_id="replay", data=data)]
        )
    assert int(pool.generation_step[row]) == 7
    assert int(pool.eos_countdown[row]) == 7
    assert torch.all(pool.rep_hist[row] == 7)


def test_full_pool_reclaims_waiting_owners_before_prefill_and_reentry() -> None:
    model = SimpleNamespace(
        decode_input_embedding=SimpleNamespace(weight=torch.zeros(1, FRAME_WIDTH)),
        n_codebooks=N_CODEBOOKS,
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=SimpleNamespace(text_vocab=100, eoa_id=99),
        embed_frames=lambda rows: rows.float(),
    )
    pool = Zonos2DecodeStatePool(model)
    model.decode_state_pool = pool
    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model = model
    runner.decode_requests = {}
    for request_id in ["waiting", "finished", "survivor-a", "survivor-b"]:
        row = pool.acquire_row(request_id)
        _poison_row(pool, row, 11)
        runner.decode_requests[request_id] = SimpleNamespace(
            is_retracted=request_id == "waiting",
            finished=lambda request_id=request_id: request_id == "finished",
        )
    assert not pool.free_rows
    survivor_row = pool.row_for("survivor-a")
    data = SimpleNamespace(
        req=SimpleNamespace(
            extend_range=SimpleNamespace(start=0, end=1, length=1),
            output_ids=[],
            is_retracted=False,
            finished=lambda: False,
        ),
        prompt_rows=torch.zeros(1, FRAME_WIDTH, dtype=torch.long),
        output_codes=[],
        speaker_emb=None,
        _stream_emit_idx=0,
    )
    request = SimpleNamespace(request_id="fresh", data=data)

    runner.build_prefill_embeds(None, [request])

    assert set(pool.rid_to_row) == {"fresh", "survivor-a", "survivor-b"}
    assert set(runner.decode_requests) == set(pool.rid_to_row)
    assert int(pool.generation_step[survivor_row]) == 11
    assert torch.all(pool.feedback_embeds[survivor_row] == 11)
    codes = torch.arange(2 * N_CODEBOOKS).reshape(2, N_CODEBOOKS)
    data = SimpleNamespace(
        req=SimpleNamespace(
            extend_range=SimpleNamespace(start=1, end=3, length=2),
            output_ids=[1, 2],
            is_retracted=False,
            finished=lambda: False,
        ),
        prompt_rows=data.prompt_rows,
        output_codes=list(codes.unbind()),
        speaker_emb=None,
        _stream_emit_idx=2,
    )
    request = SimpleNamespace(request_id="waiting", data=data)

    embeddings = runner.build_prefill_embeds(None, [request])

    expected = torch.cat([codes, torch.full((2, 1), 100)], dim=1).float()
    torch.testing.assert_close(embeddings, expected)
    assert int(pool.generation_step[pool.row_for("waiting")]) == 2
    assert data._stream_emit_idx == 2
    assert len(data.output_codes) == 2
    assert int(pool.generation_step[survivor_row]) == 11
    runner.on_request_finished("waiting", data)
    runner.on_request_finished("waiting", data)
    assert "waiting" not in runner.decode_requests


def test_zonos2_radix_namespace_is_unique_per_request_lifecycle() -> None:
    from sglang.srt.mem_cache.radix_cache import RadixKey

    payload = StagePayload(
        request_id="reused",
        request=OmniRequest(inputs=""),
        data=Zonos2State(
            input_ids=torch.zeros(2, FRAME_WIDTH, dtype=torch.long)
        ).to_dict(),
    )
    model = SimpleNamespace(config=SimpleNamespace(n_codebooks=N_CODEBOOKS))
    first = request_builders.build_sglang_zonos2_request(payload, model=model)
    second = request_builders.build_sglang_zonos2_request(payload, model=model)
    assert first.req.origin_input_ids == second.req.origin_input_ids
    assert first.req.extra_key != second.req.extra_key
    assert (
        RadixKey(first.req.origin_input_ids, first.req.extra_key).child_key()
        != RadixKey(second.req.origin_input_ids, second.req.extra_key).child_key()
    )
    key = first.req.extra_key
    first.req.reset_for_retract()
    assert first.req.extra_key == key
