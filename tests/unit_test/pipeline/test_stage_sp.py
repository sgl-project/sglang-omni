# SPDX-License-Identifier: Apache-2.0
"""Static SP uses stage fanout, without changing the TP/KV contract."""

import asyncio
import queue
from types import SimpleNamespace
from typing import ClassVar

import pytest

from sglang_omni.config.schema import (
    FactoryArgs,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.pipeline.mp_runner import build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.tp_control import (
    ParallelAbortMessage,
    RequestDispatchTracker,
    TPLeaderFanout,
)
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.types import ParallelSchedulerCapabilities
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeMpContext,
    FakeScheduler,
    RecordingStageControlPlane,
)


class SPStageConfig(StageConfig):
    supports_sequence_parallel: ClassVar[bool] = True


def sp_config(**kwargs):
    return SPStageConfig(
        name="decode", factory_path="pkg.create", terminal=True, **kwargs
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sp_size": 0},
        {"sp_size": 2, "gpu": 0},
        {"sp_size": 2, "gpu": [0]},
        {"sp_size": 2, "gpu": [0, 0]},
        {"sp_size": 2},
        {"sp_size": 2, "tp_size": 2, "gpu": [0, 1]},
    ],
)
def test_sp_shape_rejected(kwargs):
    with pytest.raises(ValueError):
        sp_config(**kwargs)


def test_sp_requires_model_opt_in():
    with pytest.raises(ValueError, match="does not support"):
        StageConfig(name="other", factory_path="pkg.create", sp_size=2, gpu=[0, 1])


def test_sp_replica_placement_and_rank_specs(monkeypatch):
    # This is a four-device placement test, independent of host GPU visibility.
    monkeypatch.setattr(
        "sglang_omni.pipeline.runtime_config.visible_device_count", lambda: 4
    )
    config = PipelineConfig(
        model_path="model",
        stages=[sp_config(sp_size=2, gpu=[0, 1], factory=FactoryArgs(custom=7))],
        processes={
            "decode": ProcessConfig(num_replicas=2, replica_devices=[0, 1, 2, 3])
        },
    )
    runtime = prepare_pipeline_runtime(config)
    try:
        assert [s.gpu for s in runtime.stages_cfg] == [[0, 1], [2, 3]]
        groups = build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=runtime.stages_cfg,
            endpoints=runtime.endpoints,
            placement_plan=runtime.placement_plan,
            process_plan=runtime.process_plan,
            replica_topology=runtime.replica_topology,
        )
        specs = [p.stage_specs[0] for g in groups for p in g.process_specs]
        assert [p.process_name for g in groups for p in g.process_specs] == [
            "decode@r0_sp0",
            "decode@r0_sp1",
            "decode@r1_sp0",
            "decode@r1_sp1",
        ]
        assert [s.sp_rank for s in specs] == [0, 1, 0, 1]
        assert all(s.tp_size == 1 and s.tp_rank == 0 and s.sp_size == 2 for s in specs)
        assert all(s.typed_kwargs["custom"] == 7 for s in specs)
        assert all(s.factory_kwargs["stage_role"] == s.role for s in specs)
        assert all(
            "tp_size" not in s.factory_kwargs and "tp_rank" not in s.factory_kwargs
            for s in specs
        )
        assert all(len(s.rank_endpoints[s.stage_name]) == 1 for s in specs)
        assert specs[0].nccl_port == specs[1].nccl_port != specs[2].nccl_port
        assert specs[0].follower_work_queues[0] is specs[1].internal_work_queue
        assert specs[0].recv_endpoint and specs[1].recv_endpoint == ""
    finally:
        runtime.runtime_dir.close()


@pytest.mark.parametrize(
    "key", ["sp_size", "sp_rank", "tp_size", "tp_rank", "stage_role", "nccl_port"]
)
def test_typed_factory_cannot_override_rank_wiring(key):
    with pytest.raises(ValueError, match="owned"):
        PipelineConfig(
            model_path="m",
            stages=[sp_config(sp_size=2, gpu=[0, 1], factory=FactoryArgs(**{key: 4}))],
        )


def make_stage(*, follower=False, scheduler=None, rank_endpoints=None):
    work, abort = queue.Queue(), queue.Queue()
    fanout = TPLeaderFanout(
        "decode", follower_work_queues=[work], follower_abort_queues=[abort]
    )
    stage = Stage(
        name="decode",
        role="follower" if follower else "leader",
        get_next=lambda *_: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        scheduler=scheduler or SimpleScheduler(lambda payload: payload),
        sp_size=2,
        sp_rank=int(follower),
        tp_fanout=None if follower else fanout,
        rank_endpoints=rank_endpoints,
    )
    return stage, work, abort


def test_sp_follower_has_no_external_kv_endpoint():
    stage, _, _ = make_stage(
        follower=True, rank_endpoints={"decode": ("inproc://sp-follower-kv",)}
    )
    assert stage._comm.rank_endpoints == {}
    assert stage._comm.tp_size == 1


@pytest.mark.parametrize("follower", [False, True])
@pytest.mark.parametrize("terminal", ["result", "error"])
def test_abort_drains_committed_work_and_acknowledges_terminal(follower, terminal):
    cleaned = []
    computed = []
    scheduler = SimpleScheduler(
        lambda p: computed.append(p.request_id), abort_callback=cleaned.append
    )
    stage, work, _ = make_stage(follower=follower, scheduler=scheduler)
    payload = SimpleNamespace(request_id="r")

    async def run():
        if follower:
            # The abort queue can overtake the work queue.
            stage.on_abort("r", dispatch_id=1)
            await stage.execute(payload, dispatch_id=1)
        else:
            await stage.execute(payload)
            assert work.get_nowait().dispatch_id == 1
            stage.on_abort("r")
        stage.on_abort("r", dispatch_id=1)
        assert cleaned == [] and scheduler._aborted == set()
        msg = scheduler.inbox.get_nowait()
        loop = asyncio.new_event_loop()
        try:
            scheduler.run_single(msg, loop)
        finally:
            loop.close()
        scheduler.outbox.get_nowait()
        scheduler.outbox.put(
            OutgoingMessage(request_id="r", type=terminal, data="error")
        )
        await stage.drain_outbox()
        assert computed == ["r"] and cleaned == ["r"]
        assert stage._dispatches.current("r") is None
        stage.on_abort("r", dispatch_id=1)
        assert cleaned == ["r"] and not scheduler._draining_aborts

    asyncio.run(run())


def test_abort_before_dispatch_does_not_fanout():
    stage, work, _ = make_stage()
    stage.on_abort("r")
    asyncio.run(stage.execute(SimpleNamespace(request_id="r")))
    assert work.empty() and stage.scheduler.inbox.empty()


@pytest.mark.parametrize("parallel_kind", ["tp", "sp"])
def test_follower_preserves_pending_work_until_last_terminal(parallel_kind):
    scheduler = SimpleScheduler(lambda p: p, allow_multiple_inflight_per_request=True)
    stage = Stage(
        name="decode",
        role="follower",
        get_next=lambda *_: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        scheduler=scheduler,
        **{f"{parallel_kind}_size": 2, f"{parallel_kind}_rank": 1},
    )

    async def run():
        for dispatch in (1, 2):
            await stage.execute(
                SimpleNamespace(request_id="r"),
                dispatch_id=dispatch if parallel_kind == "sp" else None,
            )
        assert stage._inflight_work_pending["r"] == 2
        for pending in (1, 0):
            scheduler.outbox.put(OutgoingMessage(request_id="r", type="result"))
            await stage.drain_outbox()
            assert stage._inflight_work_pending.get("r", 0) == pending
            assert ("r" in stage._active_requests) == bool(pending)
        assert stage._dispatches.current("r") is None
        assert not stage.control_plane.completions

    asyncio.run(run())


def test_sp_failure_is_idempotent_and_drains_pending_work(monkeypatch):
    computed, cleaned, aborted = [], [], []
    scheduler = SimpleScheduler(
        lambda p: computed.append(p.request_id),
        allow_multiple_inflight_per_request=True,
        abort_callback=aborted.append,
    )
    stage, _, _ = make_stage(scheduler=scheduler)
    monkeypatch.setattr(stage._comm, "cleanup", cleaned.append)

    async def run():
        for _ in range(2):
            await stage.execute(SimpleNamespace(request_id="r"))
        await stage.send_failure("r", "relay failed")
        await stage.send_failure("r", "duplicate failure")
        stage.on_abort("r")
        stage.on_abort("r")
        assert cleaned == ["r"]
        assert len(stage.control_plane.completions) == 1
        assert not scheduler._aborted
        assert not aborted
        loop = asyncio.new_event_loop()
        try:
            for _ in range(2):
                scheduler.run_single(scheduler.inbox.get_nowait(), loop)
        finally:
            loop.close()
        await stage.drain_outbox()
        assert computed == ["r", "r"]
        assert not stage._inflight_work_pending
        assert stage._dispatches.current("r") is None
        assert not scheduler._draining_aborts
        assert aborted == ["r"]

    asyncio.run(run())


def test_late_abort_does_not_poison_next_dispatch():
    stage, _, _ = make_stage(follower=True)

    async def run():
        payload = SimpleNamespace(request_id="r")
        await stage.execute(payload, dispatch_id=1)
        stage.acknowledge_terminal("r")
        await stage.execute(payload, dispatch_id=2)
        stage.on_abort("r", dispatch_id=1)
        assert "r" not in stage._aborted
        assert stage._dispatches.current("r") == 2
        await stage.execute(payload, dispatch_id=2)
        assert stage.scheduler.inbox.qsize() == 2

    asyncio.run(run())


def test_completed_dispatch_history_is_compressed_and_not_evicted():
    tracker = RequestDispatchTracker()
    tracker.register_work("slow", 1)
    for dispatch in range(2, 20002):
        tracker.register_work("fast", dispatch)
        assert tracker.finish_terminal("fast") == dispatch
    assert tracker._completed == [(2, 20001)]
    assert not tracker.record_abort("fast", 2)
    tracker.finish_terminal("slow")
    assert tracker._watermark == 20001 and not tracker._completed


def test_sp_rejects_legacy_or_unordered_schedulers():
    with pytest.raises(ValueError, match="typed"):
        make_stage(scheduler=FakeScheduler())
    for kwargs in ({"max_concurrency": 2}, {"max_batch_size": 2}):
        with pytest.raises(ValueError, match="serial"):
            make_stage(scheduler=SimpleScheduler(lambda p: p, **kwargs))
    with pytest.raises(ValueError, match="drain"):
        ParallelSchedulerCapabilities(fanout_work=True)


def test_legacy_tp_scheduler_remains_accepted():
    Stage(
        name="legacy",
        role="follower",
        get_next=lambda *_: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        scheduler=FakeScheduler(),
        tp_size=2,
        tp_rank=1,
    )


def test_abort_listener_correlates_only_committed_work():
    async def run():
        stage, work, aborts = make_stage()

        async def abort_once():
            stage._running = False
            return SimpleNamespace(request_id="r")

        stage.control_plane.recv_abort = abort_once
        stage._running = True
        await stage.abort_listener()
        assert aborts.empty()

        stage, work, aborts = make_stage()
        await stage.execute(SimpleNamespace(request_id="r"))
        dispatch_id = work.get_nowait().dispatch_id
        stage.control_plane.recv_abort = abort_once
        stage._running = True
        await stage.abort_listener()
        assert aborts.get_nowait() == ParallelAbortMessage("r", dispatch_id)

    asyncio.run(run())


def test_sp_platform_environment_and_mps_identity(monkeypatch):
    import os

    from sglang_omni.mps.decision import collect_mps_facts
    from sglang_omni.pipeline import stage_workers
    from sglang_omni.pipeline.stage_workers import (
        StageLaunchConfig,
        StageWorkerProcessSpec,
    )
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    from sglang_omni.platforms.musa import MUSAOmniPlatform
    from sglang_omni.platforms.rocm import ROCMOmniPlatform
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    spec = StageLaunchConfig(
        stage_name="decode",
        role="follower",
        sp_rank=1,
        sp_size=2,
        gpu_id=1,
        placement_gpu_id=1,
        nccl_port=23456,
    )
    worker = StageWorkerProcessSpec(process_name="decode_sp1", stage_specs=[spec])
    cuda = CUDAOmniPlatform()
    monkeypatch.setattr(stage_workers, "current_platform", cuda)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    with stage_workers.patched_spawn_env(worker):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        assert os.environ["RANK"] == "1" and os.environ["WORLD_SIZE"] == "2"
        assert os.environ["LOCAL_RANK"] == "0" and os.environ["MASTER_PORT"] == "23456"
    assert os.environ["WORLD_SIZE"] == "8"
    assert spec.tp_rank == 0 and spec.tp_size == 1
    (fact,) = collect_mps_facts([worker])
    assert fact.contains_sp and not fact.contains_tp
    for platform in (MUSAOmniPlatform(), ROCMOmniPlatform(), XPUOmniPlatform()):
        with pytest.raises(ValueError, match="does not support SP"):
            platform.get_sp_stage_process_env(spec)


def test_sp_uses_existing_typed_path_compiler():
    from sglang_omni.config.path import ConfigPath

    class Config(PipelineConfig):
        stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
            "decode": SPStageConfig
        }

    path = ConfigPath.parse("stages.decode.sp_size", Config)
    assert path.coerce("2") == 2
    with pytest.raises(ValueError):
        sp_config(sp_size=path.coerce("0"), gpu=[0, 1])
    cfg = Config(model_path="m", stages=[sp_config(sp_size=2, gpu=[0, 1])])
    assert isinstance(Config.model_validate(cfg.model_dump()).stages[0], SPStageConfig)


def test_fanned_abort_preserves_collective_participation():
    import threading

    barrier = threading.Barrier(2, timeout=3)
    computed = []
    cleaned = []

    def compute(payload):
        barrier.wait()
        computed.append(payload.request_id)
        return payload

    leader, work, _ = make_stage(
        scheduler=SimpleScheduler(compute, abort_callback=cleaned.append)
    )
    follower, _, _ = make_stage(
        follower=True, scheduler=SimpleScheduler(compute, abort_callback=cleaned.append)
    )

    async def run():
        await leader.execute(SimpleNamespace(request_id="r"))
        msg = work.get_nowait()
        leader.on_abort("r")
        follower.on_abort("r", dispatch_id=msg.dispatch_id)
        await follower.execute(msg.data, dispatch_id=msg.dispatch_id)
        threads = [
            threading.Thread(target=s.scheduler.start, daemon=True)
            for s in (leader, follower)
        ]
        for thread in threads:
            thread.start()
        try:
            for stage in (leader, follower):
                terminal = stage.scheduler.outbox.get(timeout=5)
                assert terminal.type == "result"
                stage.acknowledge_terminal(terminal.request_id)
        finally:
            for stage in (leader, follower):
                stage.scheduler.stop()
            for thread in threads:
                thread.join(timeout=2)
                assert not thread.is_alive()
        assert computed == ["r", "r"] and cleaned == ["r", "r"]

    asyncio.run(run())


def test_fanout_preserves_scheduler_enqueue_hook():
    scheduler = SimpleScheduler(lambda p: p)
    enqueued = []
    scheduler.enqueue = enqueued.append
    stage, work, _ = make_stage(scheduler=scheduler)
    asyncio.run(stage.execute(SimpleNamespace(request_id="r")))
    assert work.get_nowait().request_id == "r"
    assert enqueued[0].request_id == "r" and scheduler.inbox.empty()


def test_shutdown_drains_committed_sp_work_before_cleanup(monkeypatch):
    import threading

    leader_in_b = threading.Event()
    follower_in_a = threading.Event()
    release_follower = threading.Event()
    collective = threading.Barrier(2, timeout=5)
    computed = [[], []]
    closed = []
    stop_called = [threading.Event(), threading.Event()]

    def compute(rank, payload):
        if rank == 0 and payload.request_id == "b":
            leader_in_b.set()
        collective.wait()
        if rank == 1 and payload.request_id == "a":
            follower_in_a.set()
            assert release_follower.wait(5)
        computed[rank].append(payload.request_id)
        return payload

    stages = []
    for rank in range(2):
        scheduler = SimpleScheduler(
            lambda p, rank=rank: compute(rank, p),
            shutdown_callback=lambda rank=rank: closed.append(
                (rank, tuple(computed[rank]))
            ),
        )
        stage, work, _ = make_stage(follower=bool(rank), scheduler=scheduler)
        stages.append(stage)
        if rank == 0:
            leader_work = work
        original_stop = scheduler.stop

        def stop(rank=rank, original_stop=original_stop):
            original_stop()
            stop_called[rank].set()

        monkeypatch.setattr(scheduler, "stop", stop)

    async def run():
        for request_id in ("a", "b"):
            await stages[0].execute(SimpleNamespace(request_id=request_id))
            msg = leader_work.get_nowait()
            await stages[1].execute(msg.data, dispatch_id=msg.dispatch_id)
        for stage in stages:
            await stage.start()
        try:
            assert await asyncio.to_thread(leader_in_b.wait, 5)
            assert await asyncio.to_thread(follower_in_a.wait, 5)
            stops = [asyncio.create_task(stage.stop()) for stage in stages]
            for called in stop_called:
                assert await asyncio.to_thread(called.wait, 5)
            assert not closed
            release_follower.set()
            await asyncio.wait_for(asyncio.gather(*stops), timeout=5)
            assert computed == [["a", "b"], ["a", "b"]]
            assert sorted(closed) == [(0, ("a", "b")), (1, ("a", "b"))]
            for stage in stages:
                assert stage._scheduler_thread is None
                assert stage.scheduler.inbox.empty()
                assert [stage.scheduler.outbox.get_nowait().type for _ in range(2)] == [
                    "result",
                    "result",
                ]
        finally:
            release_follower.set()
            collective.abort()
            await asyncio.gather(*(stage.stop() for stage in stages))

    asyncio.run(run())


def test_existing_tp_specs_keep_factory_and_kv_rank_wiring(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.pipeline.runtime_config.visible_device_count", lambda: 2
    )
    config = PipelineConfig(
        model_path="m",
        stages=[
            StageConfig(
                name="thinker",
                factory_path="pkg.create",
                terminal=True,
                tp_size=2,
                gpu=[0, 1],
                total_reserve_bytes=1024,
            )
        ],
    )
    runtime = prepare_pipeline_runtime(config)
    try:
        groups = build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=runtime.stages_cfg,
            endpoints=runtime.endpoints,
            placement_plan=runtime.placement_plan,
            process_plan=runtime.process_plan,
        )
        specs = [p.stage_specs[0] for g in groups for p in g.process_specs]
        assert [s.tp_rank for s in specs] == [0, 1]
        assert all(s.tp_size == 2 and s.sp_size == 1 and s.sp_rank == 0 for s in specs)
        assert [s.factory_kwargs["tp_rank"] for s in specs] == [0, 1]
        assert all(
            "sp_size" not in s.factory_kwargs and "stage_role" not in s.factory_kwargs
            for s in specs
        )
        assert all(len(s.rank_endpoints["thinker"]) == 2 for s in specs)
        assert all(s.total_reserve_bytes == 1024 for s in specs)
        assert specs[0].follower_work_queues[0] is specs[1].internal_work_queue
    finally:
        runtime.runtime_dir.close()


def test_existing_shutdown_callback_is_called_once():
    closed = []
    scheduler = SimpleScheduler(
        lambda p: p, shutdown_callback=lambda: closed.append(True)
    )
    scheduler.stop()
    scheduler.stop()
    assert closed == [True]
