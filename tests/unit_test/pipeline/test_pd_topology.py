# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PD config schema, path visibility, topology validation, comm wiring, and Stage KV guards."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni import platforms
from sglang_omni.admission import AdminDroppedRequestError, QueueFullError
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.kv_transfer import KVPageTransfer
from sglang_omni.config.path import ConfigPath, ConfigPathError, PathVisibility
from sglang_omni.config.runtime import resolve_stage_factory_arg_defaults
from sglang_omni.config.schema import (
    PLACEMENT_OWNED_FACTORY_KWARGS,
    CommConfig,
    EngineArgs,
    EngineStageConfig,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.topology import (
    ProcessGroupPlacement,
    ProcessTopologyPlan,
    compile_logical_processes,
    pipeline_edges,
    resolve_coordinator_max_in_flight,
)
from sglang_omni.pipeline.mp_runner import (
    resolve_comm_config,
    resolve_same_process_targets,
)
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.runtime_config import allocate_endpoints
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    fake_factory_path,
)

FACTORY_PATH = fake_factory_path("dummy_factory")


def make_stage(
    name: str,
    *,
    process: str,
    next_stage: str | list[str] | None = None,
    kv_transfer_to: str | None = None,
    terminal: bool = False,
    route_fn: str | None = None,
    gpu: int | list[int] | None = None,
    tp_size: int = 1,
    comm: CommConfig | None = None,
    engine_limits: tuple[int, int] | None = None,
    is_engine: bool = False,
) -> StageConfig:
    if is_engine or engine_limits is not None:
        engine = (
            EngineArgs(
                max_running_requests=engine_limits[0],
                max_queued_requests=engine_limits[1],
            )
            if engine_limits is not None
            else EngineArgs()
        )
        return EngineStageConfig(
            name=name,
            factory_path=FACTORY_PATH,
            process=process,
            next=next_stage,
            kv_transfer_to=kv_transfer_to,
            terminal=terminal,
            gpu=gpu,
            tp_size=tp_size,
            comm=comm,
            engine=engine,
        )
    return StageConfig(
        name=name,
        factory_path=FACTORY_PATH,
        process=process,
        next=next_stage,
        kv_transfer_to=kv_transfer_to,
        terminal=terminal,
        route_fn=route_fn,
        gpu=gpu,
        tp_size=tp_size,
        comm=comm,
    )


@pytest.mark.parametrize(
    ("next_stage", "kv_transfer_to", "terminal", "route_fn", "expected_error"),
    [
        (None, None, False, None, "exactly one"),
        ("decode", "decode", False, None, "exactly one"),
        (None, "decode", True, None, "exactly one"),
        ("decode", "decode", True, None, "exactly one"),
        ("decode", None, True, None, "exactly one"),
        (None, "decode", False, fake_factory_path("identity_route"), "route_fn"),
        (None, "prefill", False, None, "cannot reference itself"),
        (None, "missing", False, None, "unknown stage"),
    ],
)
def test_stage_kv_transfer_schema_guards(
    next_stage: str | None,
    kv_transfer_to: str | None,
    terminal: bool,
    route_fn: str | None,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        PipelineConfig(
            model_path="dummy",
            stages=[
                make_stage(
                    "prefill",
                    process="prefill",
                    next_stage=next_stage,
                    kv_transfer_to=kv_transfer_to,
                    terminal=terminal,
                    route_fn=route_fn,
                ),
                make_stage("decode", process="decode", terminal=True),
            ],
        )


def test_kv_transfer_path_visibility_session_route_and_timeout_ownership() -> None:
    kv_path = ConfigPath.parse("stages.prefill.kv_transfer_to")
    assert kv_path.visibility is PathVisibility.INTERNAL
    with pytest.raises(ConfigPathError, match="internal"):
        kv_path.require_writable()

    timeout_path = ConfigPath.parse("stages.decode.comm.prealloc_timeout_seconds")
    assert timeout_path.visibility is PathVisibility.PUBLIC
    assert timeout_path.coerce("20") == 20.0 and isinstance(
        timeout_path.coerce("20"), float
    )
    with pytest.raises(ConfigPathError, match="boolean"):
        timeout_path.coerce("true")

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage("prefill", process="prefill", kv_transfer_to="decode", gpu=0),
            make_stage("decode", process="decode", terminal=True, gpu=1),
        ],
    )
    assert config.terminal_stages == ["decode"]
    with pytest.raises(ValueError, match="Session routes are not supported"):
        config.session_route()

    dumped = config.model_dump()
    timeout_path.write(dumped, timeout_path.coerce("42"))
    rebuilt = PipelineConfig.model_validate(dumped)
    assert (
        rebuilt.stage_named("decode").comm is not None
        and rebuilt.stage_named("decode").comm.prealloc_timeout_seconds == 42.0
    )

    bad_owner = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage(
                "prefill",
                process="prefill",
                kv_transfer_to="decode",
                gpu=0,
                comm=CommConfig(prealloc_timeout_seconds=30.0),
            ),
            make_stage("decode", process="decode", terminal=True, gpu=1),
        ],
    )
    with pytest.raises(ValueError, match="only valid on a kv_transfer_to target stage"):
        compile_logical_processes(bad_owner)


@pytest.mark.parametrize(
    ("stages", "expected_error"),
    [
        (
            [
                make_stage("prefill", process="shared", kv_transfer_to="decode", gpu=0),
                make_stage("decode", process="shared", terminal=True, gpu=0),
            ],
            "distinct processes",
        ),
        (
            [
                make_stage(
                    "prefill",
                    process="prefill",
                    kv_transfer_to="decode",
                    tp_size=2,
                    gpu=[0, 1],
                ),
                make_stage("decode", process="decode", terminal=True, tp_size=1, gpu=2),
            ],
            "matching tp_size",
        ),
        (
            [
                make_stage("a", process="pa", kv_transfer_to="b", gpu=0),
                make_stage("b", process="pb", kv_transfer_to="c", gpu=1),
                make_stage("c", process="pc", terminal=True, gpu=2),
            ],
            "cannot chain",
        ),
        (
            [
                make_stage("a", process="pa", kv_transfer_to="b", gpu=0),
                make_stage("b", process="pb", kv_transfer_to="a", gpu=1),
            ],
            "cannot chain",
        ),
        (
            [
                make_stage(
                    "prefill", process="prefill", kv_transfer_to="decode", gpu=None
                ),
                make_stage("decode", process="decode", terminal=True, gpu=None),
            ],
            "GPU placement on both stages",
        ),
        (
            [
                make_stage(
                    "prefill", process="prefill", kv_transfer_to="decode", gpu=0
                ),
                make_stage("decode", process="decode", terminal=True, gpu=None),
            ],
            "GPU placement on both stages",
        ),
        (
            [
                make_stage(
                    "prefill", process="prefill", kv_transfer_to="decode", gpu=None
                ),
                make_stage("decode", process="decode", terminal=True, gpu=0),
            ],
            "GPU placement on both stages",
        ),
    ],
)
def test_validate_kv_transfer_edges_rejects_invalid_topology(
    stages: list[StageConfig], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        compile_logical_processes(PipelineConfig(model_path="dummy", stages=stages))


def test_valid_pd_topology_comm_timeout_mirroring_and_same_process_defense() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage(
                "prefill",
                process="prefill",
                kv_transfer_to="decode",
                tp_size=2,
                gpu=[0, 1],
            ),
            make_stage(
                "decode",
                process="decode",
                terminal=True,
                tp_size=2,
                gpu=[2, 3],
                comm=CommConfig(prealloc_timeout_seconds=60.0),
            ),
        ],
        processes={
            "prefill": ProcessConfig(num_replicas=2, replica_devices=[0, 1, 2, 3]),
            "decode": ProcessConfig(num_replicas=2, replica_devices=[4, 5, 6, 7]),
        },
    )
    assert ("prefill", "decode") in pipeline_edges(config.stages)
    plan, copied_stages = compile_logical_processes(config)
    expanded_stages, topology = expand_replica_stages(copied_stages, plan)
    expanded_by_name = {stage.name: stage for stage in expanded_stages}
    assert (
        topology.instances("decode") == ("decode@r0", "decode@r1")
        and expanded_by_name["prefill@r0"].kv_transfer_to == "decode"
    )
    assert (
        resolve_comm_config(expanded_by_name["prefill@r0"], config=config, gpu_id=0)[
            "prealloc_timeout_seconds"
        ]
        == 60.0
    )

    default_pd = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage(
                "prefill",
                process="prefill",
                kv_transfer_to="decode",
                gpu=0,
                comm=CommConfig(credits=4),
            ),
            make_stage("decode", process="decode", terminal=True, gpu=1),
        ],
    )
    compile_logical_processes(PipelineConfig.model_validate(default_pd.model_dump()))
    assert (
        resolve_comm_config(
            default_pd.stage_named("prefill"), config=default_pd, gpu_id=0
        )["prealloc_timeout_seconds"]
        is None
    )

    same_prefill = make_stage(
        "prefill", process="shared", kv_transfer_to="decode", gpu=0
    )
    same_decode = make_stage("decode", process="shared", terminal=True, gpu=0)
    bypassed_plan = ProcessTopologyPlan(
        groups=(
            ProcessGroupPlacement(
                name="shared", stage_names=("prefill", "decode"), gpu_id=0
            ),
        ),
        stage_to_process={"prefill": "shared", "decode": "shared"},
        tp_stage_to_processes={},
    )
    assert resolve_same_process_targets(
        same_prefill, {"prefill": same_prefill, "decode": same_decode}, bypassed_plan
    ) == {"decode"}


@pytest.mark.parametrize(
    ("stages", "processes", "entry_cap", "expected_cap"),
    [
        (
            [
                make_stage("thinker", process="t", next_stage="talker", is_engine=True),
                make_stage("talker", process="k", terminal=True, engine_limits=(8, 8)),
            ],
            {},
            None,
            None,
        ),
        (
            [
                make_stage("e0", process="p0", next_stage="e1", engine_limits=(10, 10)),
                make_stage("e1", process="p1", terminal=True, engine_limits=(3, 1)),
            ],
            {},
            None,
            20,
        ),
        (
            [
                make_stage(
                    "p", process="p", kv_transfer_to="d", gpu=0, engine_limits=(4, 4)
                ),
                make_stage("d", process="d", terminal=True, gpu=1, is_engine=True),
            ],
            {},
            None,
            None,
        ),
        (
            [
                make_stage("p", process="p", kv_transfer_to="d", gpu=0, is_engine=True),
                make_stage(
                    "d", process="d", terminal=True, gpu=1, engine_limits=(8, 8)
                ),
            ],
            {},
            None,
            None,
        ),
        (
            [
                make_stage(
                    "p", process="p", kv_transfer_to="d", gpu=0, engine_limits=(4, 2)
                ),
                make_stage(
                    "d", process="d", terminal=True, gpu=1, engine_limits=(8, 4)
                ),
            ],
            {
                "p": ProcessConfig(num_replicas=2, replica_devices=[0, 1]),
                "d": ProcessConfig(num_replicas=3, replica_devices=[2, 3, 4]),
            },
            None,
            48,
        ),
        (
            [
                make_stage("e", process="e", next_stage=["pa", "pb"]),
                make_stage(
                    "pa",
                    process="pa",
                    kv_transfer_to="da",
                    gpu=0,
                    engine_limits=(10, 0),
                ),
                make_stage(
                    "da", process="da", terminal=True, gpu=1, engine_limits=(20, 0)
                ),
                make_stage(
                    "pb", process="pb", kv_transfer_to="db", gpu=2, engine_limits=(3, 1)
                ),
                make_stage(
                    "db", process="db", terminal=True, gpu=3, engine_limits=(5, 1)
                ),
            ],
            {},
            None,
            10,
        ),
        (
            [
                make_stage("pre", process="e", next_stage="p"),
                make_stage(
                    "p", process="p", kv_transfer_to="d", gpu=0, engine_limits=(4, 4)
                ),
                make_stage(
                    "d", process="d", terminal=True, gpu=1, engine_limits=(8, 8)
                ),
            ],
            {"e": ProcessConfig(num_replicas=2)},
            5,
            10,
        ),
    ],
)
def test_resolve_coordinator_max_in_flight_matrix(
    stages: list[StageConfig],
    processes: dict[str, ProcessConfig],
    entry_cap: int | None,
    expected_cap: int | None,
) -> None:
    class EntryLimitedConfig(PipelineConfig):
        cap_value: ClassVar[int | None] = entry_cap

        def entry_stage_admission_limit(self) -> int | None:
            return type(self).cap_value

    config = EntryLimitedConfig(model_path="dummy", stages=stages, processes=processes)
    plan, _ = compile_logical_processes(config)
    assert (
        resolve_coordinator_max_in_flight(config, logical_process_plan=plan)
        == expected_cap
    )


@pytest.mark.parametrize(
    ("scenario", "expected_exception", "match_pattern"),
    [
        (
            "same_process",
            NotImplementedError,
            "requires cuda_ipc transport, got local_object",
        ),
        ("shm_fallback", NotImplementedError, "requires cuda_ipc transport, got shm"),
        ("missing_rank_endpoints", ValueError, "missing rank_endpoints"),
        ("tp_mismatch", NotImplementedError, "has tp_size=1, expected 2"),
        ("valid_cuda_ipc", None, ""),
    ],
)
def test_stage_start_validates_kv_transfer_to_before_opening_resources(
    scenario: str,
    expected_exception: type[Exception] | None,
    match_pattern: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
        raising=False,
    )
    control_plane = RecordingStageControlPlane()
    tp_size = 2 if scenario == "tp_mismatch" else 1
    rank_endpoints: dict[str, tuple[str, ...]] = {
        "prefill": (
            ("inproc://prefill-r0", "inproc://prefill-r1")
            if tp_size == 2
            else ("inproc://prefill-r0",)
        ),
        "decode@r0": ("inproc://decode0-r0",),
        "decode@r1": (
            ("inproc://decode1-r0", "inproc://decode1-r1")
            if tp_size == 2
            else ("inproc://decode1-r0",)
        ),
    }
    if scenario == "missing_rank_endpoints":
        rank_endpoints.pop("decode@r1")
    stage = Stage(
        name="prefill",
        role="single",
        get_next=lambda request_id, output: None,
        gpu_id=None if scenario == "shm_fallback" else 0,
        endpoints={},
        control_plane=control_plane,
        rank_endpoints=rank_endpoints,
        tp_size=tp_size,
        relay=FakeRelay(),
        scheduler=None if expected_exception is None else FakeScheduler(),
        gpu_stage_names={"prefill", "decode@r0", "decode@r1"},
        same_process_targets={"decode@r1"} if scenario == "same_process" else set(),
        replica_topology={"decode": ["decode@r0", "decode@r1"]},
        kv_transfer_to="decode",
    )

    async def run() -> None:
        if expected_exception is None:
            try:
                await stage.start()
                assert stage.running and control_plane.started
            finally:
                await stage.stop()
        else:
            with pytest.raises(expected_exception, match=match_pattern):
                await stage.start()
            assert not stage.running and not control_plane.started
            assert (
                stage.scheduler_thread is None and stage.comm.rank_recv_socket is None
            )

    asyncio.run(run())


def test_stage_kv_transfer_route_result_and_oversize_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ResultOwningFakeScheduler(FakeScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.released: list[tuple[StagePayload, bool]] = []

        def claim_result(self, result: StagePayload, *, terminal: bool) -> bool:
            return True

        def release_result(self, result: StagePayload, *, delivered: bool) -> None:
            self.released.append((result, delivered))

    async def run() -> None:
        control_plane, scheduler = (
            RecordingStageControlPlane(),
            ResultOwningFakeScheduler(),
        )
        stage = Stage(
            name="prefill",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
            kv_transfer_to="decode",
        )
        stage.active_requests.add("request-1")
        payload = StagePayload(
            request_id="request-1",
            request=OmniRequest(inputs="hi"),
            data={"unexpected": True},
        )
        await stage.route_result("request-1", payload)
        assert control_plane.completions[
            -1
        ].success is False and "cannot route payload results" in (
            control_plane.completions[-1].error or ""
        )
        assert "request-1" not in stage.active_requests and scheduler.released == [
            (payload, False)
        ]

        send_mock = AsyncMock()
        stage.comm.send_kv_pages = send_mock
        ready_event, lease, bad_lease = Mock(), Mock(), Mock()
        await stage.send_kv_transfer(
            KVPageTransfer(
                request_id="r-ok",
                transfer_id="t0",
                source_pool_id="prefill:kv",
                source_page_indices=(0,),
                lease=lease,
                ready_event=ready_event,
            )
        )
        assert (
            send_mock.call_args.kwargs["to_stage"] == "decode"
            and send_mock.call_args.kwargs["target_pool_id"] == "kv"
        )
        assert (
            send_mock.call_args.kwargs["ready_event"] is ready_event
            and send_mock.call_args.kwargs["lease"] is lease
        )

        await stage.send_kv_transfer(
            KVPageTransfer(
                request_id="r-bad",
                transfer_id="t-bad",
                source_pool_id="prefill:kv",
                source_page_indices=(0,),
                to_stage="other",
                lease=bad_lease,
            )
        )
        bad_lease.release.assert_called_once_with()
        assert "disagrees with stage kv_transfer_to" in (
            control_plane.completions[-1].error or ""
        )

        oversize_message = (
            "Request request-oversize exceeds the maximum number of tokens: 9999 > 4096"
        )
        for index, rejected_error in enumerate(
            (
                QueueFullError(f"{QueueFullError.MESSAGE} full"),
                AdminDroppedRequestError(),
                RuntimeError(oversize_message),
            )
        ):
            stage.comm.send_kv_pages = AsyncMock(side_effect=rejected_error)
            caplog.clear()
            with caplog.at_level(
                logging.WARNING, logger="sglang_omni.pipeline.stage.runtime"
            ):
                await stage.send_kv_transfer(
                    KVPageTransfer(
                        request_id=f"request-rejected-{index}",
                        transfer_id=f"t-{index}",
                        source_pool_id="prefill:kv",
                        source_page_indices=(0,),
                    )
                )
            assert not any(record.levelno >= logging.ERROR for record in caplog.records)
            assert any(
                "KV transfer rejected" in record.message for record in caplog.records
            )

    asyncio.run(run())


def test_kv_participating_stage_endpoints_skip_non_pd_stages_and_omit_stage_instance_name() -> (
    None
):
    assert "stage_instance_name" not in PLACEMENT_OWNED_FACTORY_KWARGS
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            make_stage("preprocess", process="preprocess", next_stage="prefill"),
            make_stage("prefill", process="prefill", kv_transfer_to="decode", gpu=0),
            make_stage("decode", process="decode", next_stage="postprocess", gpu=1),
            make_stage("postprocess", process="postprocess", terminal=True),
        ],
        processes={
            "decode": ProcessConfig(num_replicas=2, replica_devices=[1, 2]),
        },
    )
    logical_plan, stages_cfg = compile_logical_processes(config)
    expanded_stages, _ = expand_replica_stages(stages_cfg, logical_plan)
    decode_replica = next(
        stage for stage in expanded_stages if stage.name == "decode@r1"
    )
    defaults = resolve_stage_factory_arg_defaults(
        decode_replica, config, gpu_id=decode_replica.gpu
    )
    assert "stage_instance_name" not in defaults
    assert "stage_name" not in defaults

    endpoints = allocate_endpoints(
        stages=expanded_stages,
        ipc_base_dir=Path("/tmp/pd_ep"),
    )
    assert "comm_prefill_rank0" in endpoints
    assert "comm_decode@r0_rank0" in endpoints
    assert "comm_decode@r1_rank0" in endpoints
    assert "comm_preprocess_rank0" not in endpoints
    assert "comm_postprocess_rank0" not in endpoints
