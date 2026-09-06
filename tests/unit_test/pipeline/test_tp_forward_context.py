# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

from sglang_omni.scheduling.omni_scheduler import _FAILED_BATCH_RESULT, OmniScheduler


@pytest.fixture
def scheduler(monkeypatch):
    comm = object.__new__(PyNcclCommunicator)
    comm.available = True
    comm.disabled = True
    scheduler = object.__new__(OmniScheduler)
    scheduler.device = "cuda"
    scheduler.tp_group = SimpleNamespace(world_size=2, pynccl_comm=comm)
    scheduler._native_speculative = False
    scheduler.forward_ct = 0
    scheduler._sched_idled = False
    scheduler.future_map = None
    scheduler.device_module = SimpleNamespace(Event=object)
    scheduler._forward_isolation = lambda *_args, **_kwargs: nullcontext()
    scheduler._emit_prefill_start_for_batch = lambda _batch: None
    scheduler._emit_prefill_end_for_batch = lambda _batch: None
    scheduler._emit_stream_output = lambda *_args: None
    scheduler._build_sched_output = lambda batch: batch
    scheduler._make_batch_result = lambda output: output
    scheduler.update_cache_from_scheduler = lambda *_args: None
    scheduler._model_runner = SimpleNamespace()
    scheduler.model_worker = SimpleNamespace()
    monkeypatch.setattr(
        "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
        lambda *_args: None,
    )
    return scheduler


def _launch(scheduler, mode, forward):
    batch = SimpleNamespace(return_logprob=False, return_hidden_states=False)
    scheduler._model_runner.execute = forward
    scheduler._model_runner.execute_launch = forward
    scheduler.model_worker.forward_batch_generation = forward
    scheduler._native_speculative = mode == "speculative"
    if mode == "async":
        return scheduler._run_batch_launch(batch)[1]
    return scheduler.run_batch(batch)


@pytest.mark.parametrize("mode", ["sync", "speculative", "async"])
@pytest.mark.parametrize("previously_disabled", [False, True])
def test_forward_reuses_communicator_and_restores_state(
    scheduler, mode, previously_disabled
):
    comm = scheduler.tp_group.pynccl_comm
    comm.disabled = previously_disabled
    output = SimpleNamespace(next_draft_input=None, new_seq_lens=None)

    def forward(_batch):
        assert not comm.disabled
        with scheduler._tp_forward_context():
            assert not comm.disabled
        assert not comm.disabled
        return output

    def copy_to_cpu(**_kwargs):
        assert not comm.disabled

    output.copy_to_cpu = copy_to_cpu
    assert _launch(scheduler, mode, forward) is output
    assert comm.disabled is previously_disabled
    assert scheduler.forward_ct == 1


@pytest.mark.parametrize("mode", ["sync", "speculative", "async"])
@pytest.mark.parametrize("previously_disabled", [False, True])
def test_forward_failure_restores_state_before_error_handling(
    scheduler, mode, previously_disabled
):
    comm = scheduler.tp_group.pynccl_comm
    comm.disabled = previously_disabled
    failure = RuntimeError("forward failed")
    handled = []
    scheduler._handle_batch_failure = lambda _batch, exc: handled.append(
        (exc, comm.disabled)
    )

    def forward(_batch):
        assert not comm.disabled
        with scheduler._tp_forward_context():
            raise failure

    if mode == "async":
        with pytest.raises(RuntimeError, match="forward failed"):
            _launch(scheduler, mode, forward)
        assert handled == []
    else:
        assert _launch(scheduler, mode, forward) is _FAILED_BATCH_RESULT
        assert handled == [(failure, previously_disabled)]
    assert comm.disabled is previously_disabled


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("fallback", ["cpu", "tp1", "unavailable", "missing"])
def test_forward_preserves_unsupported_communicator_path(scheduler, mode, fallback):
    comm = scheduler.tp_group.pynccl_comm
    if fallback == "cpu":
        scheduler.device = "cpu"
    elif fallback == "tp1":
        scheduler.tp_group.world_size = 1
    elif fallback == "unavailable":
        comm.available = False
    else:
        scheduler.tp_group.pynccl_comm = None
    output = object()

    def forward(_batch):
        assert comm.disabled
        return output

    assert _launch(scheduler, mode, forward) is output
    assert comm.disabled
