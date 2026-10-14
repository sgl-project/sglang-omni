# SPDX-License-Identifier: Apache-2.0
"""Unconfirmed publication drains must escape recoverable batch cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

from sglang_omni.scheduling import omni_scheduler as scheduler_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import PrefillTokenDrainError
from tests.unit_test.model_runner.test_speculative_target_worker import (
    make_speculative_target_probe,
)


@pytest.mark.parametrize("drain_fails", [False, True])
def test_failed_publication_drains_before_outer_batch_cleanup(
    monkeypatch: pytest.MonkeyPatch, drain_fails: bool
) -> None:
    adapter, forward_batch, calls = make_speculative_target_probe()
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[SimpleNamespace(rid="request", omni_data=SimpleNamespace())],
    )
    monkeypatch.setattr(ForwardBatch, "init_new", lambda *args, **kwargs: forward_batch)
    publication_error = ValueError("output queue failed after partial publication")
    drain_error = RuntimeError("GPU completion could not be verified")

    def publish(*args) -> None:
        calls.append("publication_failed")
        raise publication_error

    def drain() -> None:
        calls.append("drain")
        if drain_fails:
            raise drain_error

    adapter.prefill_token_publisher = publish
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=drain)
    )
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.running = True
    scheduler._run_batch = (
        Mock(  # noqa: leading-underscore  # Real outer handler, fake hardware launch.
            side_effect=lambda *args: adapter.forward_batch_generation(
                batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )
        )
    )
    scheduler.handle_batch_failure = Mock(
        side_effect=lambda *args: calls.append("cleanup")
    )

    if drain_fails:
        with pytest.raises(PrefillTokenDrainError) as captured:
            scheduler.run_batch(batch)
        assert captured.value.__cause__ is drain_error
        assert drain_error.__context__ is publication_error
        assert scheduler.running is False
        scheduler.handle_batch_failure.assert_not_called()
        assert calls[-2:] == ["publication_failed", "drain"]
    else:
        scheduler.run_batch(batch)
        scheduler.handle_batch_failure.assert_called_once_with(batch, publication_error)
        assert scheduler.running is True
        assert calls[-3:] == ["publication_failed", "drain", "cleanup"]


def test_fatal_drain_exits_real_scheduler_loop_without_another_batch() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.running = True
    scheduler.process_admin_requests = Mock()
    scheduler.recv_requests = Mock(return_value=[])
    scheduler.take_deferred_request_payloads = Mock(return_value=[])
    scheduler.process_input_requests = Mock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # Actual loop gate.
    batch = object()
    scheduler.get_next_batch_to_run = Mock(return_value=batch)
    scheduler._run_batch = (
        Mock(  # noqa: leading-underscore  # Inject only the launch failure.
            side_effect=PrefillTokenDrainError("unconfirmed GPU completion")
        )
    )
    scheduler.handle_batch_failure = Mock()
    scheduler.process_batch_result = Mock()

    with pytest.raises(PrefillTokenDrainError):
        scheduler.event_loop_normal()

    assert scheduler.running is False
    scheduler.get_next_batch_to_run.assert_called_once()
    scheduler.handle_batch_failure.assert_not_called()
    scheduler.process_batch_result.assert_not_called()


def test_pending_output_mode_retires_worker_on_unconfirmed_callback_drain(monkeypatch):
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.wait_for_prefill_publication_on_stop = True
    scheduler.running = True
    scheduler._run_batch = Mock(
        side_effect=PrefillTokenDrainError("device lost")
    )  # noqa: leading-underscore
    scheduler.handle_batch_failure = Mock()
    monkeypatch.setattr(scheduler_module.os, "write", Mock())
    monkeypatch.setattr(scheduler_module.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit, match="1"):
        scheduler.run_batch(object())
    scheduler.handle_batch_failure.assert_not_called()
    assert not scheduler.running
