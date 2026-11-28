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


def install_fake_forward_batch_module(monkeypatch: pytest.MonkeyPatch) -> None:
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
            # Mirrors the upstream signature: both overrides are
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


class ForwardMode:
    def __init__(self, *, is_prefill: bool) -> None:
        self.is_prefill = is_prefill

    def is_extend(self) -> bool:
        return self.is_prefill


def make_scheduler_output(*, is_prefill: bool):
    schedule_batch = SimpleNamespace(
        forward_mode=ForwardMode(is_prefill=is_prefill),
        is_prefill_only=False,
        output_ids=None,
        marker="worker-batch",
        sampling_info=SimpleNamespace(penalizer_orchestrator=None),
        prefill_input_ids_cpu=None,
        mix_running_indices=None,
    )
    request_data = SimpleNamespace(generation_steps=0, extra_model_outputs={})
    request = SimpleNamespace(request_id="req-1", data=request_data)
    return SimpleNamespace(batch_data=schedule_batch, requests=[request])


def make_runner(calls: list[str], *, custom_result):
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
    runner.execution_bridge = FakeExecutionBridge()
    runner.output_processor = SimpleNamespace(
        capture_hidden=False,
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
    install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []
    custom_result = SimpleNamespace(
        logits_output=None,
        next_token_ids=torch.tensor([7]),
        can_run_cuda_graph=True,
    )

    output = make_runner(calls, custom_result=custom_result).execute(
        make_scheduler_output(is_prefill=is_prefill)
    )

    assert calls == expected
    assert output.can_run_cuda_graph is True


def test_execute_pins_the_runners_own_device_not_the_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing this through current_platform broke cpu-resident runners: an
    accelerator's set_device rejects a cpu device, so a CUDA host raised
    'Expected a cuda device, but got: cpu' while an XPU host silently accepted it.
    """
    import sglang_omni.platforms as platforms

    def reject(device):
        raise AssertionError(f"platform set_device called with {device!r}")

    monkeypatch.setattr(platforms.current_platform, "set_device", reject, raising=False)
    install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []

    make_runner(calls, custom_result=None).execute(
        make_scheduler_output(is_prefill=True)
    )

    assert calls[0] == "before_prefill"


def test_execute_never_reaches_for_a_device_module_on_a_cpu_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cpu runner has no per-device context to bind, so it must not touch a device
    module at all. torch.cpu.set_device is only incidentally a tolerant no-op, so
    calling it would leave cpu-resident runners at the mercy of that detail.
    """

    install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []
    runner = make_runner(calls, custom_result=None)

    def reject(device):
        raise AssertionError(f"get_device_module called with {device!r}")

    monkeypatch.setattr(torch, "get_device_module", reject)
    runner.execute(make_scheduler_output(is_prefill=True))

    assert calls[0] == "before_prefill"


def test_execute_still_binds_the_index_of_an_accelerator_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping cpu must not skip accelerators: the runner still binds its own card,
    by index, since torch.xpu.set_device rejects a device object.
    """
    bound: list[object] = []
    install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []

    runner = make_runner(calls, custom_result=None)
    runner.device = torch.device("xpu", 1)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: SimpleNamespace(set_device=bound.append),
    )
    runner.execute(make_scheduler_output(is_prefill=True))

    assert bound == [1]


def test_execute_falls_back_to_standard_forward_after_before_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_forward_batch_module(monkeypatch)
    calls: list[str] = []

    output = make_runner(calls, custom_result=None).execute(
        make_scheduler_output(is_prefill=True)
    )

    assert calls == [
        "before_prefill",
        "custom_prefill",
        "standard_forward",
        "post_prefill",
    ]
    assert output.can_run_cuda_graph is False
    assert not hasattr(ModelRunner, "prepare_prefill")


def prefill_forward_batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=[None],
        input_ids=torch.tensor([1]),
        batch_size=1,
    )


def test_prepare_and_forward_clears_sidecar_before_cleanup_on_forward_error() -> None:
    runner = object.__new__(ModelRunner)
    forward_batch = prefill_forward_batch()
    payload = OmniPrefillInputs(input_embeds=torch.zeros(1, 4))
    cleanup_observations: list[object] = []

    runner.before_prefill = lambda *args: attach_omni_prefill_inputs(
        forward_batch, payload
    )

    def fail_forward(*args):
        raise ValueError("forward failed")

    runner.custom_prefill_forward = fail_forward
    runner.cleanup_prefill = lambda *args: cleanup_observations.append(
        get_omni_prefill_inputs(forward_batch)
    )

    with pytest.raises(ValueError, match="forward failed"):
        runner.prepare_and_forward(
            forward_batch,
            SimpleNamespace(is_prefill_only=True),
            [],
            True,
        )

    assert cleanup_observations == [None]
    assert get_omni_prefill_inputs(forward_batch) is None


@pytest.mark.parametrize("retain_prefill_input_embeds", [True, False])
def test_prepare_and_forward_keeps_the_composed_embeds_only_for_a_draft_that_reads_them(
    retain_prefill_input_embeds: bool,
) -> None:
    """The sidecar is cleared at cleanup, so an EAGLE draft gets its own reference."""
    runner = object.__new__(ModelRunner)
    runner.retain_prefill_input_embeds = retain_prefill_input_embeds
    forward_batch = prefill_forward_batch()
    payload = OmniPrefillInputs(input_embeds=torch.zeros(1, 4))
    runner.before_prefill = lambda *args: attach_omni_prefill_inputs(
        forward_batch, payload
    )
    runner.custom_prefill_forward = lambda *args: SimpleNamespace(
        logits_output=None, next_token_ids=torch.tensor([5])
    )
    runner.cleanup_prefill = lambda *args: None

    runner.prepare_and_forward(
        forward_batch,
        SimpleNamespace(is_prefill_only=True),
        [],
        True,
    )

    assert get_omni_prefill_inputs(forward_batch) is None
    assert runner.retained_prefill_input_embeds is (
        payload.input_embeds if retain_prefill_input_embeds else None
    )


@pytest.mark.parametrize("runner_retains_itself", [False, True])
def test_a_failed_prefill_keeps_no_composed_embeds(runner_retains_itself: bool) -> None:
    """The target adapter releases the retained embeddings only from a returned
    result, so a failed prefill keeps none, from the sidecar or set by the runner."""
    runner = object.__new__(ModelRunner)
    runner.retain_prefill_input_embeds = True
    forward_batch = prefill_forward_batch()
    payload = OmniPrefillInputs(input_embeds=torch.zeros(1, 4))
    runner.before_prefill = lambda *args: attach_omni_prefill_inputs(
        forward_batch, payload
    )

    def fail_forward(*args):
        if runner_retains_itself:
            runner.retained_prefill_input_embeds = payload.input_embeds
        else:
            pass
        raise torch.OutOfMemoryError("CUDA out of memory")

    runner.custom_prefill_forward = fail_forward
    runner.cleanup_prefill = lambda *args: None

    with pytest.raises(torch.OutOfMemoryError):
        runner.prepare_and_forward(
            forward_batch,
            SimpleNamespace(is_prefill_only=True),
            [],
            True,
        )

    assert runner.retained_prefill_input_embeds is None
    assert get_omni_prefill_inputs(forward_batch) is None


def test_execute_isolates_scheduler_sampling_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_forward_batch_module(monkeypatch)
    isolate_sampling_values = []

    @contextmanager
    def forward_context(batch, *, isolate_sampling=False):
        isolate_sampling_values.append(isolate_sampling)
        yield

    runner = make_runner(
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
            publish_next_tokens=lambda *args: None,
        )
    )

    runner.execute(make_scheduler_output(is_prefill=False))

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

    runner.finalize(
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


def test_launch_ids_go_straight_to_the_pinned_buffer_where_the_copy_is_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "supports_async_device_to_host_copy", lambda: True
    )
    runner = make_runner([], custom_result=None)
    ids = torch.tensor([7, 8])
    host_buf = torch.zeros(4, dtype=ids.dtype)

    launch_ids = runner.snapshot_launch_ids(ids, host_buf)
    ids.fill_(0)  # the next launch reuses the sampler's output buffer

    assert launch_ids is host_buf
    assert runner.host_launch_ids(launch_ids)[:2].tolist() == [7, 8]


def test_launch_ids_stay_on_device_where_the_host_copy_would_wait_for_the_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "supports_async_device_to_host_copy", lambda: False
    )
    runner = make_runner([], custom_result=None)
    ids = torch.tensor([7, 8])
    host_buf = torch.zeros(4, dtype=ids.dtype)

    launch_ids = runner.snapshot_launch_ids(ids, host_buf)
    ids.fill_(0)  # the next launch reuses the sampler's output buffer

    assert launch_ids is not host_buf
    assert host_buf.tolist() == [0, 0, 0, 0]
    assert runner.host_launch_ids(launch_ids).tolist() == [7, 8]


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="the device snapshot path needs an accelerator whose pinned copy blocks the host",
)
def test_resolve_hands_back_the_launched_step_after_waiting_for_its_copy(
    monkeypatch,
) -> None:
    """One async-decode step on real hardware. Two things must hold: the snapshot survives the
    sampler buffer being reused for the next step, and because the host copy runs on a stream of
    its own that nothing else orders, resolve records a completion event on that stream and waits
    on it before handing the buffer back."""
    from sglang_omni.platforms import current_platform

    assert current_platform.supports_async_device_to_host_copy() is False
    device = torch.device("xpu", torch.xpu.current_device())
    device_module = torch.get_device_module(device)
    waits: list[str] = []

    class SpyEvent:
        def __init__(self):
            self.event = device_module.Event()

        def record(self, stream=None):
            waits.append("record")
            self.event.record(stream)

        def synchronize(self):
            waits.append("synchronize")
            self.event.synchronize()

    spied_module = SimpleNamespace(
        Event=SpyEvent, Stream=device_module.Stream, stream=device_module.stream
    )
    runner = make_runner([], custom_result=None)
    runner.device = device
    runner.host_staging_buffers = []
    runner.staging_slot = 0
    runner.host_copy_stream = None
    runner.host_copy_done = None

    rows = 2_000_000
    sampled = torch.arange(1, rows + 1, device=device)
    launch_ids = runner.snapshot_launch_ids(
        sampled, runner.next_host_staging((rows,), sampled.dtype)
    )
    launched = device_module.Event()
    launched.record()
    sampled.fill_(-1)  # the next launch samples into the same buffer
    filler = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    for _ in range(10):  # the next step's forward, queued ahead of resolve
        filler.matmul(filler)
    launched.synchronize()  # what execute_resolve waits on before it reads the snapshot

    monkeypatch.setattr(torch, "get_device_module", lambda requested: spied_module)
    host_ids = runner.host_launch_ids(launch_ids)

    assert waits == ["record", "synchronize"]
    assert torch.equal(host_ids, torch.arange(1, rows + 1))
