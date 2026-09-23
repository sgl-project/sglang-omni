# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from queue import Queue
from types import SimpleNamespace

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
        self._decode_state_pool = pool


def _model_and_pool() -> tuple[_ModelHarness, Zonos2DecodeStatePool]:
    pool_owner = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
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
    scheduler._request_admission_lock = threading.RLock()
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._request_finished_callback = None
    scheduler._first_emit_done = {request_id}
    scheduler._prefill_start_done = {request_id}
    scheduler._prefill_end_done = set()
    scheduler._result_adapter = result_adapter
    scheduler._model_runner = None
    scheduler._stream_output_builder = None
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
    assert request_id not in pool._rid_to_row
    assert len(pool._free_rows) == pool.padding_row
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

    assert request_id not in pool._rid_to_row
    assert len(pool._free_rows) == pool.padding_row
    _assert_row_reset(pool, row)


def test_engine_builder_abort_callback_is_safe_before_and_after_allocation() -> None:
    request_id = "req-abort"
    model, pool = _model_and_pool()
    builder = Zonos2EngineBuilder()
    builder.model = model
    abort_callback = builder.make_abort_callback()
    builder.model = None

    free_rows = list(pool._free_rows)
    abort_callback(request_id)
    assert pool._free_rows == free_rows

    row = pool.acquire_row(request_id)
    _poison_row(pool, row)
    abort_callback(request_id)
    abort_callback(request_id)

    assert request_id not in pool._rid_to_row
    assert len(pool._free_rows) == pool.padding_row
    assert len(set(pool._free_rows)) == pool.padding_row
    _assert_row_reset(pool, row)


class _FakeStream:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def wait_stream(self, other: _FakeStream) -> None:
        pass


class _FakeDeviceModule:
    """Records the device every stream, guard and sync was asked for."""

    def __init__(self) -> None:
        self.stream_devices: list[torch.device] = []
        self.current_stream_devices: list[torch.device] = []
        self.synchronize_devices: list[torch.device] = []
        self.guarded_devices: list[torch.device] = []
        self.entered_streams: list[_FakeStream] = []

    def Stream(self, device: torch.device) -> _FakeStream:
        self.stream_devices.append(device)
        return _FakeStream(device)

    def current_stream(self, device: torch.device) -> _FakeStream:
        self.current_stream_devices.append(device)
        return _FakeStream(device)

    def synchronize(self, device: torch.device) -> None:
        self.synchronize_devices.append(device)

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream) -> Iterator[None]:
        self.entered_streams.append(stream)
        yield

    @contextlib.contextmanager
    def device(self, device: torch.device) -> Iterator[None]:
        self.guarded_devices.append(device)
        yield


class _CaptureHarness:
    capture_tail_graphs = Zonos2SGLangModel.capture_tail_graphs

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.dtype = torch.float32
        self.n_codebooks = 2
        self.audio_vocab = 8
        self.config = SimpleNamespace(dim=4)
        self.compute_calls: list[int] = []
        self._tail_buckets: list[int] = []
        self._tail_graphs: dict[int, object] = {}

    def tail_compute(self, bs: int) -> None:
        self.compute_calls.append(bs)


def _patch_tail_capture_backend(
    monkeypatch: pytest.MonkeyPatch,
    capture: Callable[[], contextlib.AbstractContextManager[object]],
) -> _FakeDeviceModule:
    """Point capture_tail_graphs at a fake device module and graph backend."""
    from sglang_omni.models.zonos2 import sglang_model as sglang_model_module

    device_module = _FakeDeviceModule()
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)
    monkeypatch.setattr(
        sglang_model_module,
        "current_platform",
        SimpleNamespace(
            get_device_graph_backend=lambda device: SimpleNamespace(capture=capture),
            device_type="fake",
        ),
    )
    return device_module


def test_tail_graph_capture_binds_every_stream_to_the_model_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An indexed device is not the process's implicit one: xpu:1 is not xpu:0."""
    from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams

    captures: list[object] = []

    @contextlib.contextmanager
    def capture() -> Iterator[object]:
        graph = object()
        captures.append(graph)
        yield graph

    device_module = _patch_tail_capture_backend(monkeypatch, capture)

    dev = torch.device("cpu")
    harness = _CaptureHarness(dev)
    harness.capture_tail_graphs([1, 2], TTSSamplingParams())

    assert device_module.stream_devices == [dev]
    assert device_module.current_stream_devices == [dev, dev]
    assert device_module.synchronize_devices == [dev, dev]
    assert [stream.device for stream in device_module.entered_streams] == [dev]
    assert device_module.guarded_devices == [dev], "the capture runs under a guard"
    assert len(captures) == 2
    assert harness._tail_buckets == [1, 2]
    assert sorted(harness._tail_graphs) == [1, 2]


def test_tail_graph_capture_stays_disarmed_when_a_bucket_fails_to_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reads the bucket list alone to decide it may replay, so a
    half-recorded set must leave the per-frame tail eager."""
    from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams

    captures: list[object] = []

    @contextlib.contextmanager
    def capture() -> Iterator[object]:
        if captures:
            raise RuntimeError("backend ran out of capture memory")
        graph = object()
        captures.append(graph)
        yield graph

    _patch_tail_capture_backend(monkeypatch, capture)

    harness = _CaptureHarness(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="capture memory"):
        harness.capture_tail_graphs([1, 2], TTSSamplingParams())

    assert len(captures) == 1, "the first bucket did record"
    assert harness._tail_buckets == []
    assert harness._tail_graphs == {}


def test_a_failed_recapture_disarms_what_the_last_capture_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recapture reallocates the buffers the armed graphs recorded against, so a
    bucket that fails must leave none of the previous capture replayable."""
    from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams

    captures: list[object] = []

    @contextlib.contextmanager
    def capture() -> Iterator[object]:
        if len(captures) >= 2:
            raise RuntimeError("backend ran out of capture memory")
        graph = object()
        captures.append(graph)
        yield graph

    _patch_tail_capture_backend(monkeypatch, capture)

    harness = _CaptureHarness(torch.device("cpu"))
    harness.capture_tail_graphs([1, 2], TTSSamplingParams())
    armed_graphs = harness._tail_graphs
    armed_buffers = harness._cg

    with pytest.raises(RuntimeError, match="capture memory"):
        harness.capture_tail_graphs([1, 2], TTSSamplingParams())

    assert sorted(armed_graphs) == [1, 2], "the first capture really did arm both"
    assert harness._cg is not armed_buffers, "and the second replaced their buffers"
    assert harness._tail_buckets == []
    assert harness._tail_graphs == {}


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
    free_rows_after_release = len(pool._free_rows)
    model.reset_request(done.request_id)

    assert done.request_id not in pool._rid_to_row
    assert pool.row_for(live.request_id) == live_row
    assert pool._active_ids is None
    assert pool._active_rows is None
    _assert_row_reset(pool, done_row)
    assert torch.all(pool.feedback_embeds[live_row] == 11)
    assert bool(pool.eos_frame_set[live_row])
    assert int(pool.eos_frame_val[live_row]) == 11
    assert int(pool.eos_countdown[live_row]) == 11
    assert int(pool.generation_step[live_row]) == 11
    assert torch.all(pool.rep_hist[live_row] == 11)
    assert int(pool.rep_len[live_row]) == 11
    assert len(pool._free_rows) == free_rows_after_release
    assert len(set(pool._free_rows)) == len(pool._free_rows)

    active_rows = pool.prepare_active_rows([live, done])

    assert active_rows.tolist() == [live_row, done_row]
    _assert_row_reset(pool, done_row)
    owned_rows = set(pool._rid_to_row.values())
    free_rows = set(pool._free_rows)
    assert len(owned_rows) == len(pool._rid_to_row)
    assert len(free_rows) == len(pool._free_rows)
    assert owned_rows.isdisjoint(free_rows)
    assert len(owned_rows) + len(free_rows) == pool.padding_row


def test_resolve_collects_compact_metadata_without_releasing_state() -> None:
    request_id = "req-resolve"
    model, pool = _model_and_pool()
    row = pool.acquire_row(request_id)

    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model = model
    runner._copy_stream = None
    data = SimpleNamespace(output_codes=[], eos_frame=None)
    request = SimpleNamespace(request_id=request_id, data=data)
    codes = list(range(N_CODEBOOKS))
    packed = torch.tensor([codes + [1, 5]], dtype=torch.int64)
    next_ids = torch.tensor([123], dtype=torch.int64)
    result = SimpleNamespace(next_token_ids=None)
    launch_buf = ([request], packed, N_CODEBOOKS, next_ids, None)

    runner.collect_resolve(launch_buf, result)

    assert runner._copy_stream is None
    assert data.output_codes[0].tolist() == codes
    assert data.eos_frame == 5
    assert torch.equal(result.next_token_ids, next_ids)
    assert pool.row_for(request_id) == row
    assert row not in pool._free_rows


def test_resolve_takes_its_stream_from_the_tensors_own_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The D2H overlap must follow the device the codes live on, not torch.cuda."""
    seen: dict[str, object] = {}

    class _Stream:
        def wait_event(self, event: object) -> None:
            seen["waited"] = event

        def synchronize(self) -> None:
            seen["synchronized"] = True

    class _DeviceModule:
        @staticmethod
        def Stream(*, device: torch.device) -> _Stream:
            seen["stream_device"] = device
            return _Stream()

        @staticmethod
        @contextlib.contextmanager
        def stream(stream: _Stream) -> Iterator[None]:
            seen["entered"] = stream
            yield

    fake_device = torch.device("privateuseone", 3)
    host = torch.tensor([list(range(N_CODEBOOKS)) + [1, 5]], dtype=torch.int64)

    class _OffDeviceTensor:
        """Reports a device this host lacks; the D2H copy yields the real rows."""

        device = fake_device

        def to(self, target: str, non_blocking: bool = False) -> torch.Tensor:
            seen["copy"] = (target, non_blocking)
            return host

    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: _DeviceModule if device == fake_device else None,
    )

    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner.model, _pool = _model_and_pool()
    runner._copy_stream = None
    data = SimpleNamespace(output_codes=[], eos_frame=None)
    request = SimpleNamespace(request_id="req-stream", data=data)
    event = object()

    runner.collect_resolve(
        ([request], _OffDeviceTensor(), N_CODEBOOKS, torch.tensor([1]), event), None
    )

    assert seen["stream_device"] == fake_device
    assert seen["waited"] is event
    assert isinstance(seen["entered"], _Stream)
    assert seen["synchronized"] is True
    assert seen["copy"] == ("cpu", True)
    assert data.output_codes[0].tolist() == list(range(N_CODEBOOKS))
    assert data.eos_frame == 5
