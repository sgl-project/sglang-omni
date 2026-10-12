# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.token_metrics import (
    batch_metrics_messages,
    token_metrics_message,
)


@pytest.mark.parametrize(
    "enabled,leader,aborted",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_disabled_follower_and_abort_do_not_emit(
    enabled: bool, leader: bool, aborted: bool
) -> None:
    req = request(enabled)
    assert token_metrics_message(req, is_entry_rank=leader, aborted=aborted) is None


def request(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        rid="r",
        origin_input_ids=[1, 2, 3],
        output_ids=[4, 5],
        omni_data=SimpleNamespace(
            stage_payload=SimpleNamespace(
                request=SimpleNamespace(metadata={"omni_metrics_enabled": enabled}),
            )
        ),
    )


def test_telemetry_is_cumulative_and_does_not_mutate_output() -> None:
    req = request()
    message = token_metrics_message(req, is_entry_rank=True, aborted=False)
    assert message.type == "metrics"
    assert message.data == {"prompt_tokens": 3, "completion_tokens": 2}
    assert req.output_ids == [4, 5]
    req.output_ids.clear()
    assert token_metrics_message(req, is_entry_rank=True, aborted=False) is None


def test_batch_reports_queue_once_and_each_forward_size() -> None:
    req = request()
    req.time_stats = SimpleNamespace(wait_queue_entry_time=1)
    first = batch_metrics_messages([req], is_entry_rank=True, now=3)
    assert [message.data for message in first] == [
        {"event": "queue", "duration": 2},
        {"event": "batch", "size": 1},
    ]
    assert [
        message.data
        for message in batch_metrics_messages([req], is_entry_rank=True, now=4)
    ] == [{"event": "batch", "size": 1}]
    assert batch_metrics_messages([req], is_entry_rank=False, now=4) == []


def test_disabled_batch_has_no_events() -> None:
    assert batch_metrics_messages([request(False)], is_entry_rank=True, now=4) == []


def test_deferred_request_features_do_not_expand_token_telemetry() -> None:
    req = request()
    req.spec_verify_ct = 4
    req.grammar = object()
    message = token_metrics_message(req, is_entry_rank=True, aborted=False)
    assert message.data == {"prompt_tokens": 3, "completion_tokens": 2}
