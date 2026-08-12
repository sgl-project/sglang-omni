# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import multiprocessing
import queue
import sys
import threading
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.config import (
    ParallelismConfig,
    PipelineConfig,
    SequenceParallelPolicy,
    StageConfig,
    build_process_topology_plan,
    build_stage_placement_plan,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


@pytest.fixture
def stage_workers_module(monkeypatch):
    """Import stage_workers without requiring the optional SGLang runtime."""

    fake_platforms = ModuleType("sglang_omni.platforms")

    class Platform:
        def get_stage_process_env(self, spec, env=None):
            return {}

        def set_device(self, gpu_id):
            pass

    fake_platforms.current_platform = Platform()
    fake_platforms.get_platform_spec = lambda platform: "test.Platform"
    monkeypatch.setitem(sys.modules, "sglang_omni.platforms", fake_platforms)
    sys.modules.pop("sglang_omni.pipeline.stage_workers", None)
    module = import_module("sglang_omni.pipeline.stage_workers")
    yield module
    sys.modules.pop("sglang_omni.pipeline.stage_workers", None)


class _SequenceParallelPipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(
        cls, *, stage_name: str
    ) -> SequenceParallelPolicy | None:
        return SequenceParallelPolicy()


def _sp_stage() -> StageConfig:
    return StageConfig(
        name="decode",
        factory=_FACTORY,
        gpu=[0, 1],
        parallelism=ParallelismConfig(
            sp=2,
            ulysses_degree=2,
            ring_degree=1,
        ),
        terminal=True,
    )


def test_sp_stage_launch_assigns_rank_roles_and_factory_args(
    stage_workers_module,
) -> None:
    del stage_workers_module
    from sglang_omni.pipeline import mp_runner

    stage = _sp_stage()
    config = _SequenceParallelPipelineConfig(model_path="dummy", stages=[stage])
    placement = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, placement)
    groups = mp_runner._build_stage_groups(
        config,
        ctx=multiprocessing.get_context("spawn"),
        stages_cfg=[stage],
        name_map={"decode": "decode"},
        endpoints={
            "stage_decode": "ipc:///tmp/decode.sock",
            "completion": "ipc:///tmp/completion.sock",
            "abort": "ipc:///tmp/abort.sock",
        },
        placement_plan=placement,
        process_plan=topology,
    )
    try:
        assert len(groups) == 1
        assert [process.process_name for process in groups[0].process_specs] == [
            "decode_sp0",
            "decode_sp1",
        ]
        specs = groups[0].specs

        assert [spec.role for spec in specs] == ["leader", "follower"]
        assert [spec.sp_rank for spec in specs] == [0, 1]
        assert [spec.gpu_id for spec in specs] == [0, 1]
        assert [spec.parallel_kind for spec in specs] == ["sp", "sp"]
        assert [spec.parallel_rank for spec in specs] == [0, 1]
        assert [spec.parallel_size for spec in specs] == [2, 2]
        assert [spec.factory_args["stage_role"] for spec in specs] == [
            "leader",
            "follower",
        ]
        assert all(spec.factory_args["sp_size"] == 2 for spec in specs)
        assert all(spec.factory_args["ulysses_degree"] == 2 for spec in specs)
        assert all(spec.factory_args["ring_degree"] == 1 for spec in specs)
    finally:
        groups[0].close_control_channels()


def test_sp_process_env_sets_rank_and_rendezvous_metadata(stage_workers_module) -> None:
    stage_workers = stage_workers_module

    spec = stage_workers.StageLaunchConfig(
        stage_name="decode",
        role="follower",
        sp_rank=1,
        sp_size=2,
        gpu_id=1,
        nccl_port=29600,
    )

    env = stage_workers.get_sp_stage_process_env(spec, {"CUDA_VISIBLE_DEVICES": "3,4"})

    assert env == {
        "CUDA_VISIBLE_DEVICES": "4",
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        "WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29600",
    }


def test_sp_process_env_requires_gpu_and_nccl_port(stage_workers_module) -> None:
    stage_workers = stage_workers_module

    with pytest.raises(ValueError, match="SP stage .* requires a GPU id"):
        stage_workers.get_sp_stage_process_env(
            stage_workers.StageLaunchConfig(
                stage_name="decode",
                sp_size=2,
                nccl_port=29600,
            ),
            {},
        )

    with pytest.raises(ValueError, match="SP stage .* requires a distributed port"):
        stage_workers.get_sp_stage_process_env(
            stage_workers.StageLaunchConfig(
                stage_name="decode",
                sp_size=2,
                gpu_id=0,
            ),
            {},
        )


def test_sp_worker_process_env_rejects_colocated_stage(stage_workers_module) -> None:
    stage_workers = stage_workers_module

    sp_spec = stage_workers.StageLaunchConfig(
        stage_name="decode",
        role="leader",
        sp_size=2,
        gpu_id=0,
        nccl_port=29600,
    )
    process_spec = stage_workers.StageWorkerProcessSpec(
        process_name="decode_sp0",
        stage_specs=[sp_spec],
    )

    env = stage_workers._get_worker_process_env(process_spec)

    assert env["WORLD_SIZE"] == "2"
    assert env["RANK"] == "0"
    with pytest.raises(AssertionError, match="SP stages must own their OS process"):
        stage_workers._get_worker_process_env(
            stage_workers.StageWorkerProcessSpec(
                process_name="invalid",
                stage_specs=[
                    sp_spec,
                    stage_workers.StageLaunchConfig(stage_name="postprocess"),
                ],
            )
        )


def test_sp_follower_process_name_includes_parallel_kind_and_rank(
    stage_workers_module,
) -> None:
    stage_workers = stage_workers_module

    spec = stage_workers.StageLaunchConfig(
        stage_name="decode",
        role="follower",
        sp_rank=1,
        sp_size=2,
    )
    process_spec = stage_workers.StageWorkerProcessSpec(
        process_name="decode_sp1",
        stage_specs=[spec],
    )

    assert stage_workers._process_name(process_spec) == "stage-decode-sp1-follower"


def test_parallel_stage_context_validates_sp_rank_metadata() -> None:
    from sglang_omni.pipeline import parallel_control

    context = parallel_control.ParallelStageContext(
        kind="sp",
        rank=1,
        size=2,
        role="follower",
    )

    assert context.kind == "sp"
    assert context.rank == 1
    assert context.size == 2
    with pytest.raises(ValueError, match="leader requires fanout"):
        parallel_control.ParallelStageContext(
            kind="sp",
            rank=0,
            size=2,
            role="leader",
        )


def test_tp_control_names_remain_compatible_aliases() -> None:
    from sglang_omni.pipeline.parallel_control import ParallelWorkMessage
    from sglang_omni.pipeline.tp_control import TPWorkMessage

    payload = object()
    message = TPWorkMessage(request_id="request", data=payload)

    assert isinstance(message, ParallelWorkMessage)
    assert message.request_id == "request"
    assert message.data is payload


def test_sp_follower_construction_passes_generic_parallel_context(
    monkeypatch, stage_workers_module
) -> None:
    from sglang_omni.pipeline import parallel_control

    stage_workers = stage_workers_module

    captured: dict[str, object] = {}

    class CapturingStage:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class Log:
        def info(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(stage_workers, "Stage", CapturingStage)
    monkeypatch.setattr(
        stage_workers,
        "_construct_scheduler",
        lambda spec, gpu_id, log: object(),
    )
    spec = stage_workers.StageLaunchConfig(
        stage_name="decode",
        role="follower",
        sp_rank=1,
        sp_size=2,
        internal_work_queue=queue.Queue(),
        internal_abort_queue=queue.Queue(),
    )

    stage_workers._construct_stage(spec, Log())

    context = captured["parallel_context"]
    assert isinstance(context, parallel_control.ParallelStageContext)
    assert context.kind == "sp"
    assert context.rank == 1
    assert context.size == 2
    assert context.role == "follower"
    assert isinstance(
        captured["control_plane"],
        parallel_control.ParallelFollowerControlPlane,
    )
    assert "tp_fanout" not in captured


def test_sp_admin_result_reports_generic_and_sp_rank_metadata() -> None:
    from sglang_omni.pipeline.parallel_control import ParallelStageContext
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.proto import (
        AdminMessage,
        AdminOperation,
        AdminResult,
        AdminResultMessage,
    )
    from sglang_omni.scheduling.types import ParallelSchedulerCapabilities
    from tests.unit_test.fixtures.pipeline_fakes import (
        FakeRelay,
        RecordingStageControlPlane,
    )

    class Scheduler:
        parallel_capabilities = ParallelSchedulerCapabilities()

        def admin(self, action, payload):
            return {"success": True, "data": {"action": action, **payload}}

        def abort(self, request_id):
            pass

    class Fanout:
        async def collect_admin_results(self, op_id, *, timeout_s):
            return [
                AdminResultMessage(
                    AdminResult(
                        op_id=op_id,
                        stage="decode",
                        action="model_info",
                        success=True,
                        rank=1,
                        role="follower",
                    )
                )
            ]

    async def run() -> None:
        control_plane = RecordingStageControlPlane()
        stage = Stage(
            name="decode",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=Scheduler(),
            parallel_context=ParallelStageContext(
                kind="sp",
                rank=0,
                size=2,
                role="leader",
                fanout=Fanout(),
            ),
        )

        await stage._on_admin(
            AdminMessage(AdminOperation(op_id="op-1", action="model_info"))
        )

        result = control_plane.completions[0].result
        assert result.rank == 0
        assert result.data["parallel_kind"] == "sp"
        assert result.data["parallel_size"] == 2
        assert result.data["sp_size"] == 2
        assert "tp_size" not in result.data

    asyncio.run(run())


def test_sp_stage_uses_generic_scheduler_work_fanout_capability() -> None:
    from sglang_omni.pipeline.parallel_control import ParallelStageContext
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.scheduling.types import ParallelSchedulerCapabilities
    from tests.unit_test.fixtures.pipeline_fakes import (
        FakeRelay,
        RecordingStageControlPlane,
    )

    class Scheduler:
        parallel_capabilities = ParallelSchedulerCapabilities(
            fanout_work=True,
            drain_aborted_work=True,
        )

        def __init__(self) -> None:
            self.inbox = queue.Queue()

        def mark_request_aborted_for_drain(
            self, request_id: str, dispatch_id: int
        ) -> None:
            pass

        def acknowledge_request_terminal(
            self, request_id: str, dispatch_id: int
        ) -> None:
            pass

    class Fanout:
        def __init__(self) -> None:
            self.payloads: list[object] = []

        def fanout_work(self, payload: object) -> int:
            self.payloads.append(payload)
            return 1

    class Payload:
        request_id = "request"

    async def run() -> None:
        scheduler = Scheduler()
        fanout = Fanout()
        stage = Stage(
            name="decode",
            role="leader",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=RecordingStageControlPlane(),
            relay=FakeRelay(),
            scheduler=scheduler,
            parallel_context=ParallelStageContext(
                kind="sp",
                rank=0,
                size=2,
                role="leader",
                fanout=fanout,
            ),
        )
        payload = Payload()

        await stage._execute(payload)

        assert fanout.payloads == [payload]
        assert scheduler.inbox.get_nowait().data is payload

    asyncio.run(run())


def test_parallel_fanout_correlates_abort_with_work_dispatch() -> None:
    from sglang_omni.pipeline.parallel_control import (
        ParallelAbortMessage,
        ParallelLeaderFanout,
        ParallelWorkMessage,
    )
    from sglang_omni.proto import AbortMessage

    work_queue: queue.Queue[object] = queue.Queue()
    abort_queue: queue.Queue[object] = queue.Queue()
    fanout = ParallelLeaderFanout(
        "decode",
        follower_work_queues=[work_queue],
        follower_abort_queues=[abort_queue],
    )
    payload = type("Payload", (), {"request_id": "request"})()

    dispatch_id = fanout.fanout_work(payload)
    abort_dispatch_id = asyncio.run(
        fanout.fanout_abort(AbortMessage(request_id="request"))
    )

    work = work_queue.get_nowait()
    abort = abort_queue.get_nowait()
    assert isinstance(work, ParallelWorkMessage)
    assert isinstance(abort, ParallelAbortMessage)
    assert dispatch_id == abort_dispatch_id == work.dispatch_id == abort.dispatch_id


def test_parallel_follower_drains_abort_that_arrives_before_work() -> None:
    from sglang_omni.pipeline.parallel_control import (
        ParallelAbortMessage,
        ParallelFollowerControlPlane,
        ParallelLeaderFanout,
        ParallelStageContext,
    )
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.proto import AbortMessage, ShutdownMessage
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from tests.unit_test.fixtures.pipeline_fakes import FakeRelay

    request_id = "request-abort-before-work"
    allow_work = threading.Event()

    class GatedWorkQueue:
        def __init__(self) -> None:
            self._queue: queue.Queue[object] = queue.Queue()

        def put_nowait(self, item: object) -> None:
            self._queue.put_nowait(item)

        def get(self, timeout: float) -> object:
            if not allow_work.wait(timeout):
                raise queue.Empty
            return self._queue.get(timeout=timeout)

    work_queue = GatedWorkQueue()
    abort_queue: queue.Queue[object] = queue.Queue()
    fanout = ParallelLeaderFanout(
        "decode",
        follower_work_queues=[work_queue],
        follower_abort_queues=[abort_queue],
    )
    payload = SimpleNamespace(request_id=request_id)
    dispatch_id = fanout.fanout_work(payload)
    asyncio.run(fanout.fanout_abort(AbortMessage(request_id)))

    compute_calls: list[str] = []
    cleanup_calls: list[str] = []

    def compute(item: object) -> object:
        compute_calls.append(item.request_id)
        return item

    scheduler = SimpleScheduler(compute, abort_callback=cleanup_calls.append)
    control_plane = ParallelFollowerControlPlane(
        stage_name="decode",
        work_queue=work_queue,
        abort_queue=abort_queue,
    )
    stage = Stage(
        name="decode",
        role="follower",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=control_plane,
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=ParallelStageContext(
            kind="sp",
            rank=1,
            size=2,
            role="follower",
        ),
    )

    async def run() -> None:
        run_task = asyncio.create_task(stage.run())
        for _ in range(100):
            if abort_queue.empty():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("abort was not consumed before work")

        assert compute_calls == []
        assert cleanup_calls == []
        allow_work.set()
        for _ in range(100):
            if cleanup_calls == [request_id]:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("aborted stage-fanned work did not drain to terminal")

        abort_queue.put_nowait(ParallelAbortMessage(request_id, dispatch_id))
        for _ in range(100):
            if abort_queue.empty():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("replayed abort was not consumed")

        try:
            assert compute_calls == [request_id]
            assert cleanup_calls == [request_id]
            assert not scheduler._abort_drains.is_pending(request_id, dispatch_id)
        finally:
            work_queue.put_nowait(ShutdownMessage())
            await asyncio.wait_for(run_task, timeout=1.0)

    asyncio.run(run())


def test_parallel_leader_defers_abort_cleanup_until_terminal() -> None:
    from sglang_omni.pipeline.parallel_control import (
        ParallelLeaderFanout,
        ParallelStageContext,
    )
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from tests.unit_test.fixtures.pipeline_fakes import (
        FakeRelay,
        RecordingStageControlPlane,
    )

    request_id = "request-terminal-drain"
    cleanup_calls: list[str] = []
    scheduler = SimpleScheduler(
        lambda payload: payload,
        abort_callback=cleanup_calls.append,
    )
    fanout = ParallelLeaderFanout(
        "decode",
        follower_work_queues=[],
        follower_abort_queues=[],
    )
    stage = Stage(
        name="decode",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=ParallelStageContext(
            kind="sp",
            rank=0,
            size=2,
            role="leader",
            fanout=fanout,
        ),
    )
    stage._active_requests.add(request_id)

    asyncio.run(stage._execute(SimpleNamespace(request_id=request_id)))
    scheduler._emit_result(request_id, object(), scheduler.outbox)
    stage._on_abort(request_id)

    assert cleanup_calls == []
    asyncio.run(stage._drain_outbox_external())
    assert cleanup_calls == [request_id]


def test_simple_scheduler_reuses_request_id_after_consuming_stale_abort() -> None:
    from sglang_omni.scheduling.messages import IncomingMessage
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    request_id = "reused-request"
    scheduler = SimpleScheduler(lambda payload: payload)
    scheduler.abort(request_id)
    scheduler.inbox.put(
        IncomingMessage(request_id=request_id, type="new_request", data="stale")
    )
    scheduler.inbox.put(
        IncomingMessage(request_id=request_id, type="new_request", data="fresh")
    )
    scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
    scheduler_thread.start()

    try:
        result = scheduler.outbox.get(timeout=1.0)
        assert result.request_id == request_id
        assert result.type == "result"
        assert result.data == "fresh"
    finally:
        scheduler.stop()
        scheduler_thread.join(timeout=1.0)


def test_tp_parallel_context_uses_scheduler_owned_abort_synchronization() -> None:
    from sglang_omni.pipeline.parallel_control import (
        ParallelLeaderFanout,
        ParallelStageContext,
    )
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.scheduling.types import ParallelSchedulerCapabilities
    from tests.unit_test.fixtures.pipeline_fakes import (
        FakeRelay,
        RecordingStageControlPlane,
    )

    calls: list[str] = []
    scheduler = SimpleNamespace(
        parallel_capabilities=ParallelSchedulerCapabilities(
            synchronize_abort=True,
        ),
        propagate_abort=calls.append,
        abort=lambda request_id: calls.append(f"local:{request_id}"),
    )
    stage = Stage(
        name="thinker",
        role="leader",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        parallel_context=ParallelStageContext(
            kind="tp",
            rank=0,
            size=2,
            role="leader",
            fanout=ParallelLeaderFanout(
                "thinker",
                follower_work_queues=[],
                follower_abort_queues=[],
            ),
        ),
    )

    stage._on_abort("request")

    assert calls == ["request"]


def test_dllm_scheduler_parallel_contract_is_dispatch_aware() -> None:
    import ast
    from pathlib import Path

    module_path = (
        Path(__file__).parents[3] / "sglang_omni" / "scheduling" / "dllm_scheduler.py"
    )
    module = ast.parse(module_path.read_text())
    scheduler = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DllmScheduler"
    )
    method_names = {
        node.name for node in scheduler.body if isinstance(node, ast.FunctionDef)
    }

    assignments = [
        target.attr
        for method in scheduler.body
        if isinstance(method, ast.FunctionDef) and method.name == "__init__"
        for node in ast.walk(method)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
    ]

    assert "parallel_capabilities" in assignments
    assert "mark_request_aborted_for_drain" in method_names
    assert "acknowledge_request_terminal" in method_names


def test_stage_process_normal_exit_destroys_distributed_group_once(
    monkeypatch, stage_workers_module
) -> None:
    stage_workers = stage_workers_module

    calls: list[str] = []
    monkeypatch.setattr(
        stage_workers,
        "_prepare_cuda_environment",
        lambda spec, log: None,
    )
    monkeypatch.setattr(stage_workers, "apply_gpu_compat_env_defaults", lambda: None)
    monkeypatch.setattr(
        stage_workers,
        "_run_process",
        lambda spec, ready_event, log: calls.append("run"),
    )
    monkeypatch.setattr(
        stage_workers,
        "_destroy_torch_distributed_process_group",
        lambda log: calls.append("destroy"),
    )
    monkeypatch.setattr(
        stage_workers,
        "_reclaim_process_cuda_memory",
        lambda gpu_ids, log, *, reason: calls.append("reclaim"),
    )
    spec = stage_workers.StageWorkerProcessSpec(
        process_name="decode_sp0",
        stage_specs=[stage_workers.StageLaunchConfig(stage_name="decode")],
    )

    stage_workers.stage_process_main(spec, SimpleNamespace(set=lambda: None))

    assert calls == ["run", "destroy"]


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("boom")])
def test_stage_process_failure_destroys_distributed_group_once(
    monkeypatch, stage_workers_module, failure: BaseException
) -> None:
    stage_workers = stage_workers_module

    calls: list[str] = []
    monkeypatch.setattr(
        stage_workers,
        "_prepare_cuda_environment",
        lambda spec, log: None,
    )
    monkeypatch.setattr(stage_workers, "apply_gpu_compat_env_defaults", lambda: None)

    def fail(spec, ready_event, log) -> None:
        raise failure

    monkeypatch.setattr(stage_workers, "_run_process", fail)
    monkeypatch.setattr(
        stage_workers,
        "_destroy_torch_distributed_process_group",
        lambda log: calls.append("destroy"),
    )
    monkeypatch.setattr(
        stage_workers,
        "_reclaim_process_cuda_memory",
        lambda gpu_ids, log, *, reason: calls.append("reclaim"),
    )
    spec = stage_workers.StageWorkerProcessSpec(
        process_name="decode_sp0",
        stage_specs=[stage_workers.StageLaunchConfig(stage_name="decode")],
    )

    expected = (
        KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else SystemExit
    )
    with pytest.raises(expected):
        stage_workers.stage_process_main(spec, SimpleNamespace(set=lambda: None))

    assert calls.count("destroy") == 1
    assert calls.count("reclaim") == 1
