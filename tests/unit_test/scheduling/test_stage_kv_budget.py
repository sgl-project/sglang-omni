# SPDX-License-Identifier: Apache-2.0
"""The ambient KV byte budget scope and its stage-worker wiring.

The budget must reach the engine bootstrap without touching model factory
signatures, and a declared budget that no engine consumes must fail startup
instead of being silently dropped.
"""

from __future__ import annotations

import logging

import pytest

from sglang_omni.pipeline.stage_workers import StageLaunchConfig, _construct_scheduler
from sglang_omni.scheduling.stage_kv_budget import (
    consume_stage_kv_cache_bytes,
    peek_stage_kv_cache_bytes,
    stage_kv_cache_budget,
)
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path

_LOG = logging.getLogger(__name__)


def test_consume_outside_scope_returns_none() -> None:
    assert consume_stage_kv_cache_bytes() is None
    assert peek_stage_kv_cache_bytes() is None


def test_scope_delivers_budget_once_consumed() -> None:
    with stage_kv_cache_budget("thinker", 2 * 1024**3):
        assert peek_stage_kv_cache_bytes() == 2 * 1024**3
        assert consume_stage_kv_cache_bytes() == 2 * 1024**3
    assert consume_stage_kv_cache_bytes() is None


def test_second_consume_in_one_scope_raises() -> None:
    """Two engines each taking the full stage budget would silently commit
    twice the declared bytes."""
    with stage_kv_cache_budget("thinker", 2 * 1024**3):
        assert consume_stage_kv_cache_bytes() == 2 * 1024**3
        with pytest.raises(RuntimeError, match="second SGLang engine"):
            consume_stage_kv_cache_bytes()
    assert peek_stage_kv_cache_bytes() is None


def test_unconsumed_scope_raises_on_exit() -> None:
    with pytest.raises(RuntimeError, match="'vocoder'.*did not build"):
        with stage_kv_cache_budget("vocoder", 1024**3):
            pass
    assert peek_stage_kv_cache_bytes() is None


def test_peek_does_not_count_as_consumption() -> None:
    with pytest.raises(RuntimeError, match="did not build"):
        with stage_kv_cache_budget("thinker", 1024**3):
            peek_stage_kv_cache_bytes()


def test_factory_exception_is_not_masked_by_consumption_check() -> None:
    with pytest.raises(ValueError, match="factory boom"):
        with stage_kv_cache_budget("thinker", 1024**3):
            raise ValueError("factory boom")
    assert peek_stage_kv_cache_bytes() is None


def test_nested_scopes_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="cannot nest"):
        with stage_kv_cache_budget("thinker", 1024**3):
            with stage_kv_cache_budget("talker_ar", 1024**3):
                pass


def _spec(factory_name: str, defaults: dict) -> StageLaunchConfig:
    return StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path(factory_name),
        factory_arg_defaults=defaults,
    )


def test_construct_scheduler_scopes_budget_around_factory() -> None:
    spec = _spec(
        "make_scheduler_consuming_kv_budget",
        {"kv_cache_bytes": 3 * 1024**3},
    )

    scheduler = _construct_scheduler(spec, None, _LOG)

    assert scheduler.consumed_kv_cache_bytes == 3 * 1024**3
    assert "kv_cache_bytes" not in scheduler.factory_kwargs


def test_construct_scheduler_fails_when_budget_is_not_consumed() -> None:
    spec = _spec("make_scheduler", {"kv_cache_bytes": 3 * 1024**3})

    with pytest.raises(RuntimeError, match="'thinker'.*did not build"):
        _construct_scheduler(spec, None, _LOG)


def test_construct_scheduler_without_budget_opens_no_scope() -> None:
    spec = _spec("make_scheduler_consuming_kv_budget", {})

    scheduler = _construct_scheduler(spec, None, _LOG)

    assert scheduler.consumed_kv_cache_bytes is None
