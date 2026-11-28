# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PD replica binding policies, non-PD lockstep preservation, and Coordinator binding release."""

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.config.schema import PipelineConfig, ProcessConfig, StageConfig
from sglang_omni.config.topology import (
    LogicalProcess,
    LogicalProcessPlan,
    compile_logical_processes,
    resolve_tracked_binding_processes,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.replicas import (
    LeastInFlightBindingPolicy,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
    expand_replica_stages,
    release_replica_bindings,
)
from sglang_omni.proto.messages import AbortMessage, CompleteMessage, SubmitMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingCoordinatorControlPlane,
    fake_factory_path,
)

FACTORY_PATH = fake_factory_path("dummy_factory")


def make_stage(
    name: str,
    *,
    process: str,
    next_stage: str | list[str] | None = None,
    kv_transfer_to: str | None = None,
    stream_to: list[str] | None = None,
    terminal: bool = False,
    gpu: int | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        factory_path=FACTORY_PATH,
        process=process,
        next=next_stage,
        kv_transfer_to=kv_transfer_to,
        stream_to=stream_to or [],
        terminal=terminal,
        gpu=gpu,
    )


class FailingControlPlane(RecordingCoordinatorControlPlane):
    """In-memory coordinator control plane supporting submit and abort fault injection."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_submit_error: Exception | None = None
        self.fail_abort_error: Exception | None = None

    async def submit_to_stage(
        self, stage: str, endpoint: str, message: SubmitMessage
    ) -> None:
        if self.fail_submit_error is not None:
            raise self.fail_submit_error
        await super().submit_to_stage(stage, endpoint, message)

    async def broadcast_abort(self, message: AbortMessage) -> None:
        if self.fail_abort_error is not None:
            raise self.fail_abort_error
        await super().broadcast_abort(message)


def build_pd_coordinator(
    *,
    terminal_stages: list[str] | None = None,
    register_all_replicas: bool = True,
) -> tuple[Coordinator, LeastInFlightBindingPolicy, FailingControlPlane]:
    has_postprocess = terminal_stages is not None and "postprocess" in terminal_stages
    stage_configs = [
        make_stage("prefill", process="prefill", kv_transfer_to="decode", gpu=0),
        make_stage(
            "decode",
            process="decode",
            next_stage="postprocess" if has_postprocess else None,
            terminal=not has_postprocess,
            gpu=1,
        ),
    ]
    if has_postprocess:
        stage_configs.append(
            make_stage("postprocess", process="decode", terminal=True, gpu=1)
        )
    config = PipelineConfig(
        model_path="dummy",
        stages=stage_configs,
        processes={
            "prefill": ProcessConfig(num_replicas=2, replica_devices=[0, 1]),
            "decode": ProcessConfig(num_replicas=2, replica_devices=[2, 3]),
        },
    )
    plan, copied_stages = compile_logical_processes(config)
    expanded_stages, topology = expand_replica_stages(copied_stages, plan)
    policy = LeastInFlightBindingPolicy(
        tracked_processes=resolve_tracked_binding_processes(config, plan)
    )
    coordinator = Coordinator(
        completion_endpoint="inproc://complete",
        abort_endpoint="inproc://abort",
        entry_stage="prefill",
        terminal_stages=terminal_stages or ["decode"],
        replica_topology=topology,
        logical_process_plan=plan,
        binding_policy=policy,
    )
    control_plane = FailingControlPlane()
    coordinator.control_plane, coordinator.running = control_plane, True
    for stage_config in expanded_stages:
        if register_all_replicas or not stage_config.name.endswith("@r1"):
            coordinator.register_stage(
                stage_config.name, f"inproc://{stage_config.name}"
            )
    return coordinator, policy, control_plane


@pytest.mark.parametrize(
    ("first_stage", "second_stage"),
    [("talker_ar", "code2wav"), ("tts_engine", "vocoder")],
)
def test_non_pd_paired_replicas_keep_lockstep_while_pd_tracks_only_kv_targets(
    first_stage: str, second_stage: str
) -> None:
    paired_config = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage(
                first_stage,
                process="upstream",
                next_stage=second_stage,
                stream_to=[second_stage],
            ),
            make_stage(second_stage, process="downstream", terminal=True),
        ],
        processes={
            "upstream": ProcessConfig(num_replicas=2),
            "downstream": ProcessConfig(num_replicas=2),
        },
    )
    paired_plan, _ = compile_logical_processes(paired_config)
    assert resolve_tracked_binding_processes(paired_config, paired_plan) == frozenset()
    round_robin = RoundRobinBindingPolicy()
    paired_picks = [
        assign_replica_bindings(paired_plan, round_robin, f"req-{index}")
        for index in range(4)
    ]
    assert [(item[first_stage], item[second_stage]) for item in paired_picks] == [
        (0, 0),
        (1, 1),
        (0, 0),
        (1, 1),
    ]

    pd_three_stage = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage("prefill", process="prefill", kv_transfer_to="decode", gpu=0),
            make_stage("decode", process="decode", next_stage="postprocess", gpu=1),
            make_stage("postprocess", process="postprocess", terminal=True),
        ],
    )
    pd_three_plan, _ = compile_logical_processes(pd_three_stage)
    assert resolve_tracked_binding_processes(
        pd_three_stage, pd_three_plan
    ) == frozenset({"decode"})


def test_least_in_flight_policy_projection_tie_break_2p2d_balance_and_rollback() -> (
    None
):
    tail_config = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage("prefill", process="prefill", kv_transfer_to="decode", gpu=0),
            make_stage("decode", process="tail", next_stage="postprocess", gpu=1),
            make_stage("postprocess", process="tail", terminal=True, gpu=1),
        ],
        processes={
            "prefill": ProcessConfig(num_replicas=2, replica_devices=[0, 1]),
            "tail": ProcessConfig(num_replicas=2, replica_devices=[2, 3]),
        },
    )
    tail_plan, _ = compile_logical_processes(tail_config)
    tail_policy = LeastInFlightBindingPolicy(
        tracked_processes=resolve_tracked_binding_processes(tail_config, tail_plan)
    )
    first = assign_replica_bindings(tail_plan, tail_policy, "req-1")
    second = assign_replica_bindings(tail_plan, tail_policy, "req-2")
    assert first == {"prefill": 0, "decode": 0, "postprocess": 0} and second == {
        "prefill": 1,
        "decode": 1,
        "postprocess": 1,
    }
    release_replica_bindings(tail_plan, tail_policy, second)
    third = assign_replica_bindings(tail_plan, tail_policy, "req-3")
    assert third == {"prefill": 0, "decode": 1, "postprocess": 1}
    release_replica_bindings(tail_plan, tail_policy, third)
    fourth = assign_replica_bindings(tail_plan, tail_policy, "req-4")
    assert fourth == {"prefill": 1, "decode": 1, "postprocess": 1}
    for bindings in (first, fourth, fourth):
        release_replica_bindings(tail_plan, tail_policy, bindings)
    assert tail_policy.in_flight_counts == {}

    three_replica_policy = LeastInFlightBindingPolicy(
        tracked_processes=frozenset({"decode"})
    )
    sequential_picks: list[int] = []
    for request_index in range(6):
        chosen_replica = three_replica_policy.bind("decode", 3, f"seq-{request_index}")
        sequential_picks.append(chosen_replica)
        three_replica_policy.release({"decode": chosen_replica})
    assert sequential_picks == [0, 1, 2, 0, 1, 2]

    class FailSecondProcessPolicy(LeastInFlightBindingPolicy):
        def bind(self, process_name: str, num_replicas: int, request_id: str) -> int:
            if process_name == "second":
                raise RuntimeError("boom on second process")
            return super().bind(process_name, num_replicas, request_id)

    rollback_plan = LogicalProcessPlan(
        processes=(
            LogicalProcess(
                name="first",
                stage_names=("decode_a",),
                tp_size=1,
                num_replicas=2,
                replica_devices=None,
            ),
            LogicalProcess(
                name="second",
                stage_names=("decode_b",),
                tp_size=1,
                num_replicas=2,
                replica_devices=None,
            ),
        ),
        stage_to_process={"decode_a": "first", "decode_b": "second"},
    )
    rollback_policy = FailSecondProcessPolicy(
        tracked_processes=frozenset({"first", "second"})
    )
    with pytest.raises(RuntimeError, match="boom on second process"):
        assign_replica_bindings(rollback_plan, rollback_policy, "req-rollback")
    assert rollback_policy.in_flight_counts == {}


def test_coordinator_pop_request_releases_owned_replica_bindings_once_across_all_paths() -> (
    None
):
    async def run() -> None:
        coordinator, policy, control_plane = build_pd_coordinator()
        await coordinator.submit_request("req-ok", "hello")
        assert policy.in_flight_counts == {("decode", 0): 1}
        await coordinator.handle_completion(
            CompleteMessage("req-ok", "decode@r0", True, result={"ok": True})
        )
        await coordinator.handle_completion(
            CompleteMessage("req-ok", "decode@r0", True, result={"late": True})
        )
        assert policy.in_flight_counts == {}

        await coordinator.submit_request("req-abort", "hello")
        assert (
            await coordinator.abort("req-abort") is True
            and await coordinator.abort("req-abort") is False
            and policy.in_flight_counts == {}
        )

        await coordinator.submit_request("req-fail", "hello")
        await coordinator.handle_completion(
            CompleteMessage("req-fail", "decode@r0", False, error="oom")
        )
        assert policy.in_flight_counts == {}

        await coordinator.submit_request("req-fail-retry", "hello")
        control_plane.fail_abort_error = RuntimeError("abort busy")
        await coordinator.handle_completion(
            CompleteMessage("req-fail-retry", "decode@r1", False, error="oom")
        )
        assert policy.in_flight_counts == {("decode", 1): 1}
        control_plane.fail_abort_error = None
        assert (
            await coordinator.abort("req-fail-retry") is True
            and policy.in_flight_counts == {}
        )

        control_plane.fail_submit_error = RuntimeError("zmq send failed")
        with pytest.raises(RuntimeError, match="zmq send failed"):
            await coordinator.submit_request("req-send-err", "hello")
        assert policy.in_flight_counts == {}
        control_plane.fail_submit_error = None

        bad_coordinator, bad_policy, _ = build_pd_coordinator()
        bad_coordinator.terminal_stages_resolver = lambda request: ["not_a_terminal"]
        with pytest.raises(ValueError, match="outside the static terminal stages"):
            await bad_coordinator.submit_request("req-bad-term", "hello")
        assert bad_policy.in_flight_counts == {}

        partial_registration_coordinator, partial_registration_policy, _ = (
            build_pd_coordinator(register_all_replicas=False)
        )
        await partial_registration_coordinator.submit_request("req-r0", "hello")
        with pytest.raises(ValueError, match="not registered"):
            await partial_registration_coordinator.submit_request("req-r1", "hello")
        assert partial_registration_policy.in_flight_counts == {("decode", 0): 1}
        await partial_registration_coordinator.fail_pending_requests("shutdown")
        assert partial_registration_policy.in_flight_counts == {}

        multi_coordinator, multi_policy, _ = build_pd_coordinator(
            terminal_stages=["decode", "postprocess"]
        )
        await multi_coordinator.submit_request("req-multi", "hello")
        await multi_coordinator.handle_completion(
            CompleteMessage("req-multi", "decode@r0", True, result={"part": 1})
        )
        assert multi_policy.in_flight_counts == {("decode", 0): 1}
        await multi_coordinator.handle_completion(
            CompleteMessage("req-multi", "postprocess@r0", True, result={"part": 2})
        )
        assert multi_policy.in_flight_counts == {}

    asyncio.run(run())
