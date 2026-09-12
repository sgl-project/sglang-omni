# SPDX-License-Identifier: Apache-2.0
"""Multiprocess stage fixture with synthetic session hooks."""
from __future__ import annotations

import asyncio
import multiprocessing
import time
from contextlib import asynccontextmanager

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.session import ResourceUsage, TimedChunk
from sglang_omni.scheduling.session import SessionHooks, SessionScheduler


class Hooks(SessionHooks):
    def __init__(self, name, events):
        self.name, self.events = name, events

    def open(self, ref, request):
        self.events.put(("open", self.name, ref.session_id))
        time.sleep(request.params.get("open_delay", 0))
        if request.params.get("fail_open") == self.name:
            raise RuntimeError("open failed")
        return {"id": ref.session_id, "n": 0, "params": request.params}

    def append(self, state, chunk, payload, context):
        import torch

        self.events.put(("append", self.name, state["id"], chunk.seq))
        state["n"] += 1
        if self.name == "source":
            payload.data = {"tensor": torch.tensor([state["n"]])}
        elif self.name.split("@r")[0] == "middle":
            assert payload.data["tensor"].item() == state["n"]
        else:
            assert payload.data["tensor"].item() == state["n"]
            if state["n"] % state["params"].get("emit_every", 1) and not chunk.eos:
                payload.data = {"count": state["n"]}
                return payload
            cadence = state["params"].get("cadence", 1)
            for i in range(cadence):
                if state["params"].get("ignore_cancel"):
                    time.sleep(state["params"].get("delay", 0))
                    self.events.put(("finished", self.name, state["id"]))
                elif context.cancelled.wait(state["params"].get("delay", 0)):
                    self.events.put(("cancelled", self.name, state["id"]))
                    break
                context.emit(
                    TimedChunk(
                        "text",
                        chunk.t_start_ms,
                        0,
                        i,
                        [state["n"], i],
                        eos=chunk.eos and i == cadence - 1,
                    )
                )
            payload.data = {"count": state["n"]}
        return payload

    def abort(self, state, ref):
        self.events.put(("abort", self.name, state["id"]))
        if state["params"].get("cannot_abort"):
            raise RuntimeError("state cannot be retained")

    def close(self, state):
        self.events.put(("close", self.name, state["id"]))
        if state["params"].get("fail_close_once") == self.name:
            state["params"]["fail_close_once"] = None
            raise RuntimeError("close rejected")

    def usage(self, state):
        return ResourceUsage(bytes=state["n"])


def make_session_scheduler(name, events):
    return SessionScheduler(Hooks(name, events))


def worker(spec, ready):
    import logging

    from sglang_omni.pipeline.stage_workers import _construct_stage

    async def run():
        stage = _construct_stage(spec, logging.getLogger(__name__))
        await stage.start()
        ready.set()
        await stage.run()
        assert not stage.scheduler._sessions

    asyncio.run(run())


@asynccontextmanager
async def pipeline(tmp_path, *, stage_count=2, replicated=False, list_next=False):
    from sglang_omni.config.schema import PipelineConfig, ProcessConfig, StageConfig
    from sglang_omni.config.topology import compile_logical_processes
    from sglang_omni.pipeline.replicas import expand_replica_stages
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig

    ctx = multiprocessing.get_context("spawn")
    names = ["source", "middle", "sink"] if stage_count == 3 else ["source", "sink"]
    stages = []
    for index, name in enumerate(names):
        target = names[index + 1] if index + 1 < len(names) else None
        stages.append(
            StageConfig(
                name=name,
                process=name,
                terminal=target is None,
                next=[target] if list_next and target else target,
                factory_path=f"{__name__}.make_session_scheduler",
            )
        )
    config = PipelineConfig(
        model_path="mock",
        entry_stage="source",
        stages=stages,
        processes={"sink": ProcessConfig(num_replicas=2)} if replicated else {},
    )
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    endpoints = {stage.name: f"ipc://{tmp_path}/{stage.name}" for stage in expanded}
    completion, abort = f"ipc://{tmp_path}/done", f"ipc://{tmp_path}/abort"
    coordinator = Coordinator(
        completion,
        abort,
        "source",
        ["sink"],
        max_sessions=3,
        logical_process_plan=plan,
        replica_topology=topology,
    )
    events = ctx.Queue()
    processes = []
    await coordinator.start()
    loop = asyncio.create_task(coordinator.run_completion_loop())
    try:
        for stage in expanded:
            ready = ctx.Event()
            spec = StageLaunchConfig(
                stage_name=stage.name,
                factory=stage.factory_path,
                factory_kwargs={"name": stage.name, "events": events},
                next_stages=stage.next,
                is_terminal=stage.terminal,
                recv_endpoint=endpoints[stage.name],
                coordinator_endpoint=completion,
                abort_endpoint=abort,
                stage_endpoints=endpoints,
                replica_topology=topology.to_dict(),
            )
            process = ctx.Process(target=worker, args=(spec, ready))
            process.start()
            processes.append(process)
            assert await asyncio.to_thread(ready.wait, 30)
            coordinator.register_stage(stage.name, endpoints[stage.name])
        # PUB/SUB subscription is asynchronous; work begins after both workers bind.
        await asyncio.sleep(0.1)
        yield coordinator, events, processes
    finally:
        await coordinator.shutdown_stages()
        await coordinator.stop()
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        for process in processes:
            await asyncio.to_thread(process.join, 10)
            if process.is_alive():
                process.kill()
                process.join()
            assert process.exitcode == getattr(process, "expected_exitcode", 0)
        events.close()


def chunk(seq, eos=False):
    return TimedChunk("audio", seq * 20, 20, seq, b"pcm", eos=eos)


def block_async_call(monkeypatch, obj, name):
    entered, release, completed = (asyncio.Event() for _ in range(3))
    original = getattr(obj, name)

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        result = await original(*args, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(obj, name, blocked)
    return entered, release, completed


def compute_registered(scheduler, payload):
    """Run one session command on an unstarted scheduler through its inbox registration."""
    from sglang_omni.scheduling.messages import IncomingMessage

    scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    message = scheduler.inbox.get_nowait()
    return scheduler._compute(message.data)
