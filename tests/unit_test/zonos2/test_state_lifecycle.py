# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.zonos2 import request_builders
from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams
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


class ModelHarness:
    reset_request = Zonos2SGLangModel.reset_request

    def __init__(self, pool: Zonos2DecodeStatePool) -> None:
        self.decode_state_pool = pool


class FakeCopyStream:
    def wait_event(self, event) -> None:
        pass

    def synchronize(self) -> None:
        pass


def model_and_pool() -> tuple[ModelHarness, Zonos2DecodeStatePool]:
    pool_owner = SimpleNamespace(
        decode_input_embedding=SimpleNamespace(
            weight=torch.zeros((2, 3), dtype=torch.float32)
        ),
        n_codebooks=N_CODEBOOKS,
    )
    pool = Zonos2DecodeStatePool(pool_owner)
    return ModelHarness(pool), pool


def poison_row(pool: Zonos2DecodeStatePool, row: int, value: int = 7) -> None:
    pool.feedback_embeds[row].fill_(value)
    pool.eos_frame_set[row] = True
    pool.eos_frame_val[row] = value
    pool.eos_countdown[row] = value
    pool.generation_step[row] = value
    pool.rep_hist[row].fill_(value)
    pool.rep_len[row] = value


def assert_row_reset(pool: Zonos2DecodeStatePool, row: int) -> None:
    assert torch.count_nonzero(pool.feedback_embeds[row]) == 0
    assert not bool(pool.eos_frame_set[row])
    assert int(pool.eos_frame_val[row]) == 0
    assert int(pool.eos_countdown[row]) == 0
    assert int(pool.generation_step[row]) == 0
    assert torch.all(pool.rep_hist[row] == -1)
    assert int(pool.rep_len[row]) == 0


def terminal_data(request_id: str = "req-zonos2") -> Zonos2SGLangRequestData:
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
    model, pool = model_and_pool()
    _, result_adapter = make_zonos2_scheduler_adapters(model=model)
    data = terminal_data(request_id)
    row = pool.acquire_row(request_id)
    poison_row(pool, row)

    sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
    sampling_params.normalize(tokenizer=None)
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=sampling_params,
        vocab_size=2,
    )
    req.omni_data = data
    req._omni_terminal_claimed = False  # noqa: leading-underscore  # production name
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
    assert_row_reset(pool, row)


def test_result_adapter_releases_state_when_serialization_fails(monkeypatch) -> None:
    request_id = "req-adapter-error"
    model, pool = model_and_pool()
    _, result_adapter = make_zonos2_scheduler_adapters(model=model)
    row = pool.acquire_row(request_id)
    poison_row(pool, row)

    def fail_result(*args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(request_builders, "apply_sglang_zonos2_result", fail_result)

    with pytest.raises(RuntimeError, match="serialization failed"):
        result_adapter(terminal_data(request_id))

    assert request_id not in pool.rid_to_row
    assert len(pool.free_rows) == pool.padding_row
    assert_row_reset(pool, row)


def test_engine_builder_abort_callback_is_safe_before_and_after_allocation() -> None:
    request_id = "req-abort"
    model, pool = model_and_pool()
    builder = Zonos2EngineBuilder()
    builder.model = model
    abort_callback = builder.make_abort_callback()
    builder.model = None

    free_rows = list(pool.free_rows)
    abort_callback(request_id)
    assert pool.free_rows == free_rows

    row = pool.acquire_row(request_id)
    poison_row(pool, row)
    abort_callback(request_id)
    abort_callback(request_id)

    assert request_id not in pool.rid_to_row
    assert len(pool.free_rows) == pool.padding_row
    assert len(set(pool.free_rows)) == pool.padding_row
    assert_row_reset(pool, row)


class FakeEvent:
    pass


class FakeStream:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.waited_events: list[FakeEvent] = []
        self.synchronized = False

    def wait_stream(self, other: FakeStream) -> None:
        pass

    def wait_event(self, event: FakeEvent) -> None:
        self.waited_events.append(event)

    def synchronize(self) -> None:
        self.synchronized = True


class FakeDeviceModule:

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.entered_streams: list[FakeStream] = []

    def Stream(self, device: torch.device) -> FakeStream:
        stream = FakeStream(device)
        self.streams.append(stream)
        return stream

    def current_stream(self, device: torch.device) -> FakeStream:
        stream = FakeStream(device)
        self.streams.append(stream)
        return stream

    def synchronize(self, device: torch.device) -> None:
        pass

    @contextlib.contextmanager
    def stream(self, stream: FakeStream) -> Iterator[None]:
        self.entered_streams.append(stream)
        yield

    @contextlib.contextmanager
    def device(self, device: torch.device) -> Iterator[None]:
        yield


class FakeGraph:
    pass


class FakeGraphBackend:
    def __init__(self, failing_capture_index: int | None = None) -> None:
        self.failing_capture_index = failing_capture_index
        self.graphs: list[FakeGraph] = []

    @contextlib.contextmanager
    def capture(self) -> Iterator[FakeGraph]:
        if len(self.graphs) == self.failing_capture_index:
            raise RuntimeError("backend ran out of capture memory")
        else:
            pass
        graph = FakeGraph()
        self.graphs.append(graph)
        yield graph


class CaptureHarness:
    capture_tail_graphs = Zonos2SGLangModel.capture_tail_graphs

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.dtype = torch.float32
        self.n_codebooks = 2
        self.audio_vocab = 8
        self.config = SimpleNamespace(dim=4)
        self.tail_buckets: list[int] = []
        self.tail_graphs: dict[int, FakeGraph] = {}

    def tail_compute(self, batch_size: int) -> None:
        pass


def patch_device_module(monkeypatch: pytest.MonkeyPatch) -> FakeDeviceModule:
    device_module = FakeDeviceModule()
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)
    return device_module


def test_tail_graph_capture_binds_every_stream_to_the_model_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_module = patch_device_module(monkeypatch)
    graph_backend = FakeGraphBackend()

    device = torch.device("cpu")
    harness = CaptureHarness(device)
    harness.capture_tail_graphs([1, 2], TTSSamplingParams(), graph_backend)

    assert {stream.device for stream in device_module.streams} == {device}
    assert harness.tail_buckets == [1, 2]
    assert harness.tail_graphs == dict(zip([1, 2], graph_backend.graphs))


def test_tail_graph_capture_stays_disarmed_when_a_bucket_fails_to_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_device_module(monkeypatch)
    graph_backend = FakeGraphBackend(failing_capture_index=1)

    harness = CaptureHarness(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="capture memory"):
        harness.capture_tail_graphs([1, 2], TTSSamplingParams(), graph_backend)

    assert len(graph_backend.graphs) == 1
    assert harness.tail_buckets == []
    assert harness.tail_graphs == {}


def test_release_resets_reused_row_without_touching_mixed_batch_survivor() -> None:
    model, pool = model_and_pool()
    done = SimpleNamespace(request_id="done")
    live = SimpleNamespace(request_id="live")
    done_row, live_row = (
        int(row) for row in pool.prepare_active_rows([done, live]).tolist()
    )
    poison_row(pool, done_row, value=3)
    poison_row(pool, live_row, value=11)

    model.reset_request(done.request_id)
    free_rows_after_release = len(pool.free_rows)
    model.reset_request(done.request_id)

    assert done.request_id not in pool.rid_to_row
    assert pool.row_for(live.request_id) == live_row
    assert pool.active_ids is None
    assert pool.active_rows is None
    assert_row_reset(pool, done_row)
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
    assert_row_reset(pool, done_row)
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
    model, pool = model_and_pool()
    row = pool.acquire_row(request_id)

    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model = model
    runner.decode_requests = {}
    runner.copy_stream = FakeCopyStream()
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


@pytest.mark.parametrize("start,has_eos", [(0, True), (1, False), (3, True)])
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
    poison_row(pool, row)
    survivor = pool.acquire_row("survivor")
    poison_row(pool, survivor, 11)
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


def test_reprefill_without_generated_frames_fails_loudly() -> None:
    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model, _ = model_and_pool()
    runner.decode_requests = {}
    data = SimpleNamespace(
        req=SimpleNamespace(
            extend_range=SimpleNamespace(start=0, end=5, length=5),
            output_ids=[1, 2, 3],
        ),
        prompt_rows=torch.zeros(2, FRAME_WIDTH, dtype=torch.long),
        output_codes=[],
        speaker_emb=None,
    )

    with pytest.raises(AssertionError, match="frame history does not match"):
        runner.build_prefill_embeds(
            None, [SimpleNamespace(request_id="replay", data=data)]
        )


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
        poison_row(pool, row, 11)
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
    assert (
        data._stream_emit_idx == 2
    )  # noqa: leading-underscore  # Existing request or scheduler interface.
    assert len(data.output_codes) == 2
    assert int(pool.generation_step[survivor_row]) == 11
    runner.on_request_finished("waiting", data)
    runner.on_request_finished("waiting", data)
    assert "waiting" not in runner.decode_requests


def test_zonos2_radix_namespace_is_shared_across_prompt_texts() -> None:
    from sglang.srt.mem_cache.radix_cache import RadixKey

    def build(request_id: str, rows: torch.Tensor):
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs=""),
            data=Zonos2State(input_ids=rows).to_dict(),
        )
        model = SimpleNamespace(config=SimpleNamespace(n_codebooks=N_CODEBOOKS))
        return request_builders.build_sglang_zonos2_request(payload, model=model)

    first = build("first", torch.zeros(2, FRAME_WIDTH, dtype=torch.long))
    second = build("second", torch.zeros(2, FRAME_WIDTH, dtype=torch.long))
    other_text = build("other-text", torch.ones(3, FRAME_WIDTH, dtype=torch.long))
    assert first.req.use_private_radix_on_retract
    assert (
        first.req._omni_prompt_only_radix
    )  # noqa: leading-underscore  # Existing request or scheduler interface.
    # note (0xtoward): row keys already hash every prompt row, so the namespace must not
    # depend on the text or different prompts lose their shared speaker prefix.
    assert first.req.extra_key == second.req.extra_key == other_text.req.extra_key
    assert (
        RadixKey(first.req.origin_input_ids, first.req.extra_key).child_key()
        == RadixKey(second.req.origin_input_ids, second.req.extra_key).child_key()
    )


def test_resolve_takes_its_stream_from_the_tensors_own_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = torch.device("privateuseone", 3)
    device_module = FakeDeviceModule()
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: device_module if device == accelerator else None,
    )
    codes = list(range(N_CODEBOOKS))

    class OffDeviceTensor:
        device = accelerator

        def __init__(self) -> None:
            self.copies: list[tuple[str, bool]] = []

        def to(self, target: str, non_blocking: bool = False) -> torch.Tensor:
            self.copies.append((target, non_blocking))
            return torch.tensor([codes + [1, 5]], dtype=torch.int64)

    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model, _ = model_and_pool()
    runner.copy_stream = None
    data = SimpleNamespace(
        output_codes=[],
        eos_frame=None,
        req=SimpleNamespace(is_retracted=False, finished=lambda: False),
    )
    request = SimpleNamespace(request_id="req-stream", data=data)
    packed = OffDeviceTensor()
    event = FakeEvent()

    runner.collect_resolve(
        ([request], packed, N_CODEBOOKS, torch.tensor([1]), event), None
    )

    assert runner.copy_stream.device == accelerator
    assert device_module.entered_streams == [runner.copy_stream]
    assert runner.copy_stream.waited_events == [event]
    assert runner.copy_stream.synchronized is True
    assert packed.copies == [("cpu", True)]
    assert data.output_codes[0].tolist() == codes
    assert data.eos_frame == 5
