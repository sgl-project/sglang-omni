# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.utils.env import env_flag


def _install_fake_forward_batch_module(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.model_executor",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    class CaptureHiddenMode:
        LAST = "last"

    class ForwardBatch:
        @staticmethod
        def init_new(model_worker_batch, model_runner):
            del model_runner
            return SimpleNamespace(
                input_ids=torch.tensor([1]),
                marker=model_worker_batch.marker,
            )

    forward_batch_info = types.ModuleType(
        "sglang.srt.model_executor.forward_batch_info"
    )
    forward_batch_info.CaptureHiddenMode = CaptureHiddenMode
    forward_batch_info.ForwardBatch = ForwardBatch
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.forward_batch_info",
        forward_batch_info,
    )


class _ForwardMode:
    def __init__(self, *, is_prefill: bool) -> None:
        self._is_prefill = is_prefill

    def is_extend(self) -> bool:
        return self._is_prefill


def _scheduler_output(*, is_prefill: bool):
    model_worker_batch = SimpleNamespace(marker="worker-batch")
    schedule_batch = SimpleNamespace(
        forward_mode=_ForwardMode(is_prefill=is_prefill),
        is_prefill_only=False,
        output_ids=None,
        get_model_worker_batch=lambda: model_worker_batch,
    )
    req_state = SimpleNamespace(finished=lambda: False, is_retracted=False)
    request_data = SimpleNamespace(
        generation_steps=0,
        extra_model_outputs={},
        req=req_state,
    )
    request = SimpleNamespace(request_id="req-1", data=request_data)
    return SimpleNamespace(batch_data=schedule_batch, requests=[request])


def _runner(calls: list[str], *, custom_result):
    class RecordingRunner(ModelRunner):
        def before_prefill(self, forward_batch, schedule_batch, requests):
            del forward_batch, schedule_batch, requests
            calls.append("before_prefill")

        def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
            del forward_batch, schedule_batch, requests
            calls.append("custom_prefill")
            return custom_result

        def before_decode(
            self,
            forward_batch,
            schedule_batch,
            requests,
            *,
            is_lookahead: bool = False,
        ):
            del forward_batch, schedule_batch, requests, is_lookahead
            calls.append("before_decode")

        def custom_decode_forward(self, forward_batch, schedule_batch, requests):
            del forward_batch, schedule_batch, requests
            calls.append("custom_decode")
            return custom_result

        def post_prefill(self, result, forward_batch, schedule_batch, requests):
            del result, forward_batch, schedule_batch, requests
            calls.append("post_prefill")

        def post_decode(self, result, forward_batch, schedule_batch, requests):
            del result, forward_batch, schedule_batch, requests
            calls.append("post_decode")

        def post_decode_launch(self, result, forward_batch, requests):
            del result, forward_batch, requests
            calls.append("post_decode_launch")
            return "launch-buf"

        def post_decode_resolve(
            self,
            launch_buf,
            result,
            forward_batch,
            schedule_batch,
            requests,
        ):
            del launch_buf, result, forward_batch, schedule_batch, requests
            calls.append("post_decode_resolve")

    runner = object.__new__(RecordingRunner)
    runner.device = torch.device("cpu")
    runner.output_processor = SimpleNamespace(
        _capture_hidden=False,
        process=lambda result, scheduler_output: {
            "req-1": SimpleNamespace(extra={}),
        },
    )

    def standard_forward(forward_batch):
        del forward_batch
        calls.append("standard_forward")
        return SimpleNamespace(
            logits_output=None,
            next_token_ids=torch.tensor([5]),
            can_run_cuda_graph=False,
        )

    runner.tp_worker = SimpleNamespace(
        model_runner=object(),
        forward_batch_generation=standard_forward,
    )
    runner._async_query_hit = 0
    runner._async_query_miss = 0
    runner._phase_on = False
    runner._phase_sync = False
    runner._last_phase = None
    runner._build_profile_on = False
    runner._last_build_phase = None
    runner._mrope_patch_enabled = False
    return runner


@pytest.mark.parametrize(
    "value",
    ["0", "false", "False", "no", "off", ""],
)
def test_env_flag_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SGLANG_OMNI_PHASE_PROFILE", value)

    assert env_flag("SGLANG_OMNI_PHASE_PROFILE") is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_env_flag_true_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SGLANG_OMNI_PHASE_PROFILE", value)

    assert env_flag("SGLANG_OMNI_PHASE_PROFILE") is True


@pytest.mark.parametrize(
    ("is_prefill", "expected"),
    [
        (True, ["before_prefill", "custom_prefill", "post_prefill"]),
        (False, ["before_decode", "custom_decode", "post_decode"]),
    ],
)
def test_execute_uses_explicit_custom_forward_hook(
    monkeypatch: pytest.MonkeyPatch,
    is_prefill: bool,
    expected: list[str],
) -> None:
    _install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []
    custom_result = SimpleNamespace(
        logits_output=None,
        next_token_ids=torch.tensor([7]),
        can_run_cuda_graph=True,
    )

    output = _runner(calls, custom_result=custom_result).execute(
        _scheduler_output(is_prefill=is_prefill)
    )

    assert calls == expected
    assert output.can_run_cuda_graph is True


def test_execute_falls_back_to_standard_forward_after_before_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []

    output = _runner(calls, custom_result=None).execute(
        _scheduler_output(is_prefill=True)
    )

    assert calls == [
        "before_prefill",
        "custom_prefill",
        "standard_forward",
        "post_prefill",
    ]
    assert output.can_run_cuda_graph is False
    assert not hasattr(ModelRunner, "prepare_prefill")


def test_async_resolve_records_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_forward_batch_module(monkeypatch)

    class FakeEvent:
        def record(self) -> None:
            return None

        def query(self) -> bool:
            return True

        def synchronize(self) -> None:
            raise AssertionError("query path should not synchronize")

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    calls: list[str] = []
    custom_result = SimpleNamespace(
        logits_output=None,
        next_token_ids=torch.tensor([7]),
        can_run_cuda_graph=True,
    )
    runner = _runner(calls, custom_result=custom_result)
    runner._phase_on = True
    runner._build_profile_on = True

    pending = runner.execute_launch(_scheduler_output(is_prefill=False))

    assert pending is not None
    assert pending.phase is not None
    assert runner._last_phase is None

    output = runner.execute_resolve(pending)

    assert output is not None
    assert output.req_ids == ["req-1"]
    assert runner._async_query_hit == 1
    assert runner._last_phase is not None
    assert runner._last_phase["is_prefill"] is False
    assert {
        "build",
        "forward",
        "post",
        "finalize",
        "build_detail",
    } <= set(runner._last_phase)
    assert calls == [
        "before_decode",
        "custom_decode",
        "post_decode_launch",
        "post_decode_resolve",
    ]


class _DecodeMode:
    def is_decode(self) -> bool:
        return True


def test_mrope_text_decode_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_FAST_MROPE_DECODE", "1")
    monkeypatch.setenv("SGLANG_OMNI_BUILD_PROFILE", "1")

    class ForwardBatch:
        def __init__(self) -> None:
            self.forward_mode = _DecodeMode()
            self.positions = torch.tensor([4, 7], dtype=torch.int32)
            self.spec_info = None
            self.mrope_positions = None

        def _compute_mrope_positions(self, model_runner, batch):
            del model_runner, batch
            self.mrope_positions = torch.full((3, 2), -1, dtype=torch.int64)

    runner = object.__new__(ModelRunner)
    runner._patch_forward_batch_mrope(ForwardBatch)

    forward_batch = ForwardBatch()
    forward_batch._compute_mrope_positions(
        object(),
        SimpleNamespace(multimodal_inputs=[None, None]),
    )

    expected = torch.tensor([[4, 7], [4, 7], [4, 7]], dtype=torch.int64)
    assert torch.equal(forward_batch.mrope_positions, expected)
    assert forward_batch._sglang_omni_mrope_fast is True
    assert forward_batch._sglang_omni_mrope_seconds >= 0.0


def test_mrope_text_decode_fast_path_accepts_zero_delta_mm_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_FAST_MROPE_DECODE", "1")
    monkeypatch.setenv("SGLANG_OMNI_BUILD_PROFILE", "1")

    class ForwardBatch:
        def __init__(self) -> None:
            self.forward_mode = _DecodeMode()
            self.positions = torch.tensor([8], dtype=torch.int64)
            self.spec_info = None
            self.mrope_positions = None

        def _compute_mrope_positions(self, model_runner, batch):
            del model_runner, batch
            self.mrope_positions = torch.full((3, 1), -1, dtype=torch.int64)

    mm_input = SimpleNamespace(
        mrope_position_delta=torch.zeros((1, 1), dtype=torch.int64),
        contains_mm_input=lambda: False,
    )

    runner = object.__new__(ModelRunner)
    runner._patch_forward_batch_mrope(ForwardBatch)

    forward_batch = ForwardBatch()
    forward_batch._compute_mrope_positions(
        object(),
        SimpleNamespace(multimodal_inputs=[mm_input]),
    )

    assert torch.equal(
        forward_batch.mrope_positions,
        torch.tensor([[8], [8], [8]], dtype=torch.int64),
    )
    assert forward_batch._sglang_omni_mrope_fast is True


def test_mrope_fast_path_skips_nonzero_delta_mm_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_FAST_MROPE_DECODE", "1")
    monkeypatch.setenv("SGLANG_OMNI_BUILD_PROFILE", "1")

    class ForwardBatch:
        def __init__(self) -> None:
            self.forward_mode = _DecodeMode()
            self.positions = torch.tensor([8], dtype=torch.int64)
            self.spec_info = None
            self.mrope_positions = None

        def _compute_mrope_positions(self, model_runner, batch):
            del model_runner, batch
            self.mrope_positions = torch.full((3, 1), -1, dtype=torch.int64)

    mm_input = SimpleNamespace(
        mrope_position_delta=torch.ones((1, 1), dtype=torch.int64),
        contains_mm_input=lambda: False,
    )

    runner = object.__new__(ModelRunner)
    runner._patch_forward_batch_mrope(ForwardBatch)

    forward_batch = ForwardBatch()
    forward_batch._compute_mrope_positions(
        object(),
        SimpleNamespace(multimodal_inputs=[mm_input]),
    )

    assert torch.equal(
        forward_batch.mrope_positions,
        torch.full((3, 1), -1, dtype=torch.int64),
    )
    assert forward_batch._sglang_omni_mrope_fast is False


def test_mrope_fast_path_skips_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_FAST_MROPE_DECODE", "1")

    class ForwardBatch:
        def __init__(self) -> None:
            self.forward_mode = _DecodeMode()
            self.positions = torch.tensor([4], dtype=torch.int32)
            self.spec_info = None
            self.mrope_positions = None

        def _compute_mrope_positions(self, model_runner, batch):
            del model_runner, batch
            self.mrope_positions = torch.full((3, 1), -1, dtype=torch.int64)

    runner = object.__new__(ModelRunner)
    runner._patch_forward_batch_mrope(ForwardBatch)

    forward_batch = ForwardBatch()
    forward_batch._compute_mrope_positions(
        object(),
        SimpleNamespace(multimodal_inputs=[object()]),
    )

    assert torch.equal(
        forward_batch.mrope_positions,
        torch.full((3, 1), -1, dtype=torch.int64),
    )


def test_mrope_patch_warns_when_hook_missing(caplog: pytest.LogCaptureFixture) -> None:
    class ForwardBatch:
        pass

    runner = object.__new__(ModelRunner)

    with caplog.at_level(logging.WARNING, logger="sglang_omni.model_runner.base"):
        runner._patch_forward_batch_mrope(ForwardBatch)

    assert "has no _compute_mrope_positions" in caplog.text
    assert ForwardBatch._sglang_omni_mrope_patch_missing_logged is True


def test_mrope_patch_keeps_original_method_on_reentry() -> None:
    class ForwardBatch:
        def _compute_mrope_positions(self, model_runner, batch):
            del model_runner, batch
            return "original"

    original = ForwardBatch._compute_mrope_positions
    runner = object.__new__(ModelRunner)

    runner._patch_forward_batch_mrope(ForwardBatch)
    patched = ForwardBatch._compute_mrope_positions
    runner._patch_forward_batch_mrope(ForwardBatch)

    assert ForwardBatch._sglang_omni_orig_compute_mrope_positions is original
    assert ForwardBatch._compute_mrope_positions is patched


def test_finalize_default_batch_generation_hook_calls_single_hook() -> None:
    calls: list[tuple[str, int]] = []

    class RecordingRunner(ModelRunner):
        def on_generation_step_advanced(self, sched_req, generation_steps):
            calls.append((sched_req.request_id, generation_steps))

    runner = object.__new__(RecordingRunner)
    runner.output_processor = SimpleNamespace(
        process=lambda result, scheduler_output: {
            req.request_id: SimpleNamespace(extra={})
            for req in scheduler_output.requests
        },
    )
    requests = [
        SimpleNamespace(
            request_id="req-1",
            data=SimpleNamespace(generation_steps=0, extra_model_outputs={}),
        ),
        SimpleNamespace(
            request_id="req-2",
            data=SimpleNamespace(generation_steps=4, extra_model_outputs={}),
        ),
    ]

    runner._finalize(
        SimpleNamespace(
            next_token_ids=torch.tensor([1, 2]),
            logits_output=None,
            can_run_cuda_graph=False,
        ),
        SimpleNamespace(),
        SimpleNamespace(is_prefill_only=False, output_ids=None),
        SimpleNamespace(seq_lens=[1, 1], input_ids=torch.zeros(2, dtype=torch.long)),
        SimpleNamespace(requests=requests),
    )

    assert calls == [("req-1", 1), ("req-2", 5)]
