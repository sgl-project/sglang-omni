# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config.schema import PDExecution
from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import StageLaunchConfig
from sglang_omni.scheduling.omni_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
    OmniScheduler,
)
from sglang_omni.scheduling.pd_runtime import PDTransportBinding
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path
from tests.unit_test.pipeline.helpers import make_stage


class _Log:
    def info(self, *_args) -> None:
        pass


def _spec(role: str, *, factory: str | None = None, **factory_kwargs):
    partner = "thinker_decode" if role == "prefill" else "thinker_prefill"
    implementation = (
        "sglang_omni.scheduling.omni_scheduler.OmniPrefillScheduler"
        if role == "prefill"
        else "sglang_omni.scheduling.omni_scheduler.OmniDecodeScheduler"
    )
    return StageLaunchConfig(
        stage_name=f"thinker_{role}",
        factory=factory or fake_factory_path("pd_capable_factory"),
        factory_kwargs=factory_kwargs,
        pd_execution=PDExecution(
            role=role,
            partner=partner,
            scheduler_class=implementation,
        ),
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [("prefill", OmniPrefillScheduler), ("decode", OmniDecodeScheduler)],
)
def test_factory_constructs_compiler_selected_concrete_scheduler(role, expected):
    observed = []
    scheduler = stage_workers._construct_scheduler(
        _spec(role, post_setup=lambda value: observed.append(type(value))),
        None,
        _Log(),
    )

    assert type(scheduler) is expected
    assert observed == [expected]
    assert scheduler.scheduler_kwargs == {
        "stage_name": f"thinker_{role}",
        "partner": "thinker_decode" if role == "prefill" else "thinker_prefill",
    }


def test_non_pd_factory_constructs_plain_scheduler():
    scheduler = stage_workers._construct_scheduler(
        StageLaunchConfig(
            stage_name="thinker",
            factory=fake_factory_path("pd_capable_factory"),
        ),
        None,
        _Log(),
    )

    assert type(scheduler) is OmniScheduler
    assert scheduler.scheduler_kwargs == {}


def test_factory_kwargs_survive_concrete_scheduler_selection():
    callback = object()
    scheduler = stage_workers._construct_scheduler(
        _spec("prefill", callback=callback, typed="value"), None, _Log()
    )

    assert scheduler.factory_kwargs == {"callback": callback, "typed": "value"}


def test_mismatched_factory_result_fails_during_construction():
    with pytest.raises(TypeError, match="OmniPrefillScheduler"):
        stage_workers._construct_scheduler(
            _spec(
                "prefill",
                factory=fake_factory_path("mismatched_pd_scheduler_factory"),
            ),
            None,
            _Log(),
        )


def test_mismatched_compiled_role_and_scheduler_class_fail_before_factory():
    spec = _spec("prefill")
    spec.pd_execution = PDExecution(
        role="prefill",
        partner="thinker_decode",
        scheduler_class=("sglang_omni.scheduling.omni_scheduler.OmniDecodeScheduler"),
    )

    with pytest.raises(TypeError, match="role 'prefill' requires .*Prefill"):
        stage_workers._construct_scheduler(spec, None, _Log())


def test_scheduler_role_cannot_be_changed_after_construction():
    scheduler = stage_workers._construct_scheduler(_spec("decode"), None, _Log())

    assert type(scheduler) is OmniDecodeScheduler
    assert not hasattr(scheduler, "bind_pd_runtime")
    assert not hasattr(scheduler, "_pd_role")
    assert not hasattr(scheduler, "pd_role")


def test_pd_types_do_not_share_role_specific_state():
    prefill = object.__new__(OmniPrefillScheduler)
    decode = object.__new__(OmniDecodeScheduler)
    generic = object.__new__(OmniScheduler)
    prefill._transport_binding = SimpleNamespace(pool=object(), receiver=None)

    assert not hasattr(prefill, "_ready_queue")
    assert not hasattr(prefill, "_deferred_admission")
    assert not hasattr(prefill, "_drain_admissions")
    assert not hasattr(decode, "_source_pool_id")
    assert not hasattr(decode, "_queue_prefill_handoffs")
    assert not hasattr(generic, "_pd_role")
    assert not hasattr(generic, "transport_binding")


def test_stage_only_registers_typed_transport_binding():
    pool = SimpleNamespace(pool_id="thinker_decode:kv")
    receiver = object()
    stage = make_stage(
        name="thinker_decode",
        pd_transport_binding=PDTransportBinding(pool=pool, receiver=receiver),
    )

    assert stage._comm._kv_pools == {pool.pool_id: pool}
    assert stage._comm._kv_receivers == {pool.pool_id: receiver}
    assert not hasattr(stage.scheduler, "bind_pd_runtime")


def test_scheduler_has_final_type_before_stage_construction(monkeypatch):
    observed = {}

    class RecordingStage:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(stage_workers, "Stage", RecordingStage)
    monkeypatch.setattr(
        stage_workers,
        "StageControlPlane",
        lambda **_kwargs: object(),
    )

    stage_workers._construct_stage(_spec("prefill"), _Log())

    assert type(observed["scheduler"]) is OmniPrefillScheduler
    assert observed["pd_transport_binding"].pool.pool_id == "thinker_prefill:kv"
    assert "pd_execution" not in observed
