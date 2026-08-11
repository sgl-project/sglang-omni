# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from tests.unit_test.fakes import FakeExecutionBridge


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
        def init_new(
            model_worker_batch,
            model_runner,
            *,
            capture_hidden_mode=None,
            return_hidden_states_before_norm,
        ):
            # Mirrors the sglang 0.5.16 signature: both overrides are
            # keyword-only and return_hidden_states_before_norm is required.
            del model_runner, return_hidden_states_before_norm
            return SimpleNamespace(
                input_ids=torch.tensor([1]),
                marker=model_worker_batch.marker,
                capture_hidden_mode=capture_hidden_mode,
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
    schedule_batch = SimpleNamespace(
        forward_mode=_ForwardMode(is_prefill=is_prefill),
        is_prefill_only=False,
        output_ids=None,
        marker="worker-batch",
        prefill_input_ids_cpu=None,
        mix_running_indices=None,
    )
    request_data = SimpleNamespace(generation_steps=0, extra_model_outputs={})
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

    runner = object.__new__(RecordingRunner)
    runner.device = torch.device("cpu")
    runner._execution_bridge = FakeExecutionBridge()
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
    return runner


def test_resolve_deferred_prefill_inputs_materializes_staged_ids():
    from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs

    staged = torch.tensor([11, 12], dtype=torch.long)
    batch = SimpleNamespace(
        input_ids=None,
        prefill_input_ids_cpu=staged,
        mix_running_indices=None,
    )

    resolve_deferred_prefill_inputs(batch, torch.device("cpu"))

    assert batch.prefill_input_ids_cpu is None
    assert torch.equal(batch.input_ids, staged)


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


def _prefill_forward_batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[None],
        input_ids=torch.tensor([1]),
        batch_size=1,
    )


def test_prepare_and_forward_clears_sidecar_before_cleanup_on_forward_error() -> None:
    runner = object.__new__(ModelRunner)
    forward_batch = _prefill_forward_batch()
    payload = OmniPrefillInputs(input_embeds=torch.zeros(1, 4))
    cleanup_observations: list[object] = []

    runner.before_prefill = lambda *_args: attach_omni_prefill_inputs(
        forward_batch, payload
    )

    def fail_forward(*_args):
        raise ValueError("forward failed")

    runner.custom_prefill_forward = fail_forward
    runner.cleanup_prefill = lambda *_args: cleanup_observations.append(
        get_omni_prefill_inputs(forward_batch)
    )

    with pytest.raises(ValueError, match="forward failed"):
        runner._prepare_and_forward(
            forward_batch,
            SimpleNamespace(is_prefill_only=True),
            [],
            True,
        )

    assert cleanup_observations == [None]
    assert get_omni_prefill_inputs(forward_batch) is None


def test_execute_isolates_scheduler_sampling_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_forward_batch_module(monkeypatch)
    isolate_sampling_values = []

    @contextmanager
    def forward_context(_batch, *, isolate_sampling=False):
        isolate_sampling_values.append(isolate_sampling)
        yield

    runner = _runner(
        [],
        custom_result=SimpleNamespace(
            logits_output=None,
            next_token_ids=torch.tensor([7]),
            can_run_cuda_graph=True,
        ),
    )
    runner.bind_execution_bridge(
        SimpleNamespace(
            forward_context=forward_context,
            publish_next_tokens=lambda *_args: None,
        )
    )

    runner.execute(_scheduler_output(is_prefill=False))

    assert isolate_sampling_values == [True]


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
        SimpleNamespace(is_prefill_only=False),
        SimpleNamespace(requests=requests),
    )

    assert calls == [("req-1", 1), ("req-2", 5)]
