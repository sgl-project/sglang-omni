# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import signal
from collections.abc import Coroutine
from pathlib import Path
from traceback import format_exception
from types import FrameType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import sglang_omni.pipeline.mp_runner as mp_runner
import sglang_omni.pipeline.runtime_config as runtime_config
from sglang_omni.config.schema import (
    CustomVoiceConfig,
    EndpointsConfig,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.serve import launcher
from sglang_omni.serve.openai_api import ServerStatus
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage
from sglang_omni.serve.server_warmup import (
    TranscriptionWarmupRequest,
    warmup_tone_wav_bytes,
)
from sglang_omni.serve.speech_to_text import SpeechToTextForm, parse_speech_to_text_form
from sglang_omni.serve.transcriptions import TRANSCRIPTIONS_ENDPOINT
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext, FakeRelay


def noop_factory():
    return None


def failing_factory():
    raise RuntimeError("factory boom")


class FakeControlPlane:
    def __init__(self, recv_endpoint: str):
        self.recv_endpoint = recv_endpoint


class FakeStage:
    name = "preprocessing"

    def __init__(self, recv_endpoint: str):
        self.control_plane = FakeControlPlane(recv_endpoint)

    async def run(self) -> None:
        await asyncio.Event().wait()


class StubCoordinator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


def make_config(base_path: Path) -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory_path=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(base_path)),
    )


@pytest.fixture(autouse=True)
def fake_stage_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.comm.router.create_relay",
        lambda relay_type, **kwargs: FakeRelay(device=kwargs.get("device", "cpu")),
    )


def test_ipc_runtime_dir_creation_and_close_contracts(tmp_path: Path) -> None:
    """Preserves IPC runtime directory creation, uniqueness, and idempotent cleanup."""
    ipc_config = make_config(tmp_path)

    runtime_a = runtime_config.create_ipc_runtime_dir(ipc_config)
    runtime_b = runtime_config.create_ipc_runtime_dir(ipc_config)
    assert runtime_a is not None
    assert runtime_b is not None
    assert runtime_a.path != runtime_b.path

    runtime_path = runtime_a.path
    runtime_a.close()
    runtime_a.close()
    runtime_b.close()
    assert not runtime_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_owns_or_preserves_ipc_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves owned IPC cleanup and caller-owned IPC directory preservation."""
    config = make_config(tmp_path)

    def fail_allocate_endpoints(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_config, "allocate_endpoints", fail_allocate_endpoints)

    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config)
    assert list(tmp_path.iterdir()) == []

    caller_owned = runtime_config.create_ipc_runtime_dir(config)
    assert caller_owned is not None
    caller_path = caller_owned.path
    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config, ipc_runtime_dir=caller_owned)
    assert caller_path.exists()
    caller_owned.close()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_returns_managed_ipc_runtime_dir(
    tmp_path: Path,
) -> None:
    """Preserves managed IPC runtime directory ownership in runtime prep."""
    prep = runtime_config.prepare_pipeline_runtime(make_config(tmp_path))
    runtime_dir = prep.runtime_dir
    assert runtime_dir is not None
    try:
        assert runtime_dir.path.exists()
        assert str(runtime_dir.path) in prep.endpoints["stage_preprocessing"]
    finally:
        runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


def test_ipc_stage_groups_use_unique_endpoints_for_same_model_name(
    tmp_path: Path,
) -> None:
    """Preserves unique IPC endpoints across same-model pipeline instances."""
    config = make_config(tmp_path)
    prep_a = runtime_config.prepare_pipeline_runtime(config)
    prep_b = runtime_config.prepare_pipeline_runtime(config)
    assert prep_a.runtime_dir is not None
    assert prep_b.runtime_dir is not None

    try:
        groups_a = mp_runner.build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_a.stages_cfg,
            endpoints=prep_a.endpoints,
            placement_plan=prep_a.placement_plan,
            process_plan=prep_a.process_plan,
        )
        groups_b = mp_runner.build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_b.stages_cfg,
            endpoints=prep_b.endpoints,
            placement_plan=prep_b.placement_plan,
            process_plan=prep_b.process_plan,
        )

        assert prep_a.endpoints["completion"] != prep_b.endpoints["completion"]
        assert groups_a[0].leader_endpoint != groups_b[0].leader_endpoint
    finally:
        prep_a.runtime_dir.close()
        prep_b.runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_runtime_dir_on_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when runner startup fails."""

    class FailingCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self) -> None:
            raise RuntimeError("boom")

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(mp_runner, "Coordinator", FailingCoordinator)
    runner = mp_runner.MultiProcessPipelineRunner(make_config(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        await runner.start()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_spawned_groups_when_later_spawn_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves spawned process cleanup if a later stage group fails to spawn."""

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_count = 0
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.killed = True
            self.alive = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_count += 1

    class FakeGroup:
        def __init__(self, stage_name: str, *, fail_spawn: bool = False) -> None:
            self.stage_name = stage_name
            self.fail_spawn = fail_spawn
            self.process = FakeProcess() if not fail_spawn else None
            self.channels_closed = False
            self.process_specs = [
                StageWorkerProcessSpec(stage_name, [StageLaunchConfig(stage_name)])
            ]

        @property
        def processes(self) -> list[FakeProcess]:
            return [self.process] if self.process is not None else []

        def spawn(self, ctx) -> None:
            del ctx
            if self.fail_spawn:
                raise RuntimeError(f"spawn failed for {self.stage_name}")

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def close_control_channels(self) -> None:
            self.channels_closed = True

    first_group = FakeGroup("preprocessing")
    second_group = FakeGroup("thinker", fail_spawn=True)
    monkeypatch.setattr(mp_runner, "Coordinator", StubCoordinator)
    monkeypatch.setattr(
        mp_runner,
        "build_stage_groups",
        lambda *a, **k: [first_group, second_group],
    )

    runner = mp_runner.MultiProcessPipelineRunner(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="spawn failed"):
        await runner.start()

    assert first_group.process.terminated
    assert first_group.process.join_count >= 1
    assert first_group.channels_closed
    assert second_group.channels_closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_startup_failure_includes_child_factory_traceback(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        name="x",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory_path=f"{__name__}.failing_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = mp_runner.MultiProcessPipelineRunner(config)

    # A cold child can spend close to 10s importing torch before the factory
    # even runs; the dead-process fail-fast branch needs the child to have
    # exited, so give slow hosts room instead of racing the teardown.
    with pytest.raises(RuntimeError, match="factory boom"):
        await runner.start(timeout=30.0)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_stop_cleans_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when the runner stops."""

    class FakeCoordinator:
        def __init__(
            self,
            completion_endpoint: str,
            abort_endpoint: str,
            entry_stage: str,
            terminal_stages: list[str] | None = None,
            terminal_stages_resolver=None,
            replica_topology=None,
            logical_process_plan=None,
            max_in_flight=None,
        ) -> None:
            del (
                abort_endpoint,
                entry_stage,
                terminal_stages,
                terminal_stages_resolver,
                replica_topology,
                logical_process_plan,
                max_in_flight,
            )
            self.control_plane = SimpleNamespace(
                completion_endpoint=completion_endpoint
            )

        async def start(self) -> None:
            return None

        async def run_completion_loop(self) -> None:
            await asyncio.Event().wait()

        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

        async def shutdown_stages(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeGroup:
        stage_name = "preprocessing"
        leader_endpoint = "ipc://stage.sock"
        tp_size = 1
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.shutdown_called = False
            self.process_specs = [
                StageWorkerProcessSpec(
                    self.stage_name, [StageLaunchConfig(self.stage_name)]
                )
            ]

        def spawn(self, ctx) -> None:
            del ctx
            assert self.process_specs[0].cpu_threads == 8

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def any_dead(self) -> bool:
            return False

        def dead_summary(self) -> str:
            return "(none)"

        async def shutdown(self, before_signal=None) -> None:
            del before_signal
            self.shutdown_called = True

    group = FakeGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(mp_runner, "build_stage_groups", lambda *a, **k: [group])
    capacity = Mock(return_value=8)
    monkeypatch.setattr(mp_runner, "effective_cpu_count", capacity)

    runner = mp_runner.MultiProcessPipelineRunner(make_config(tmp_path))
    await runner.start()
    capacity.assert_called_once_with()
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1

    await runner.stop()

    assert group.shutdown_called
    assert list(tmp_path.iterdir()) == []


async def run_launcher_with_fake_runner(
    *,
    config: PipelineConfig,
    serve_mock: AsyncMock | None,
    monkeypatch: pytest.MonkeyPatch,
    replica_counts: tuple[int, ...] = (1,),
    host: str = "0.0.0.0",
) -> tuple[object, FastAPI, SimpleNamespace]:
    app = FastAPI()
    profiler_calls = SimpleNamespace(starts=[], stops=[])

    from sglang_omni.serve import launcher

    runner_ref = None

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config
            nonlocal runner_ref
            self.coordinator = StubCoordinator()
            self.stage_control_endpoints = {
                "preprocessing": "ipc://stage_preprocessing.sock"
            }
            self.started = False
            self.stopped = False
            # launcher._run_server reads .prep.placement_plan / .process_plan
            # after start() to log the resolved topology. Provide empty stubs
            # that satisfy _placement_log_summary's attribute access.
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}),
                process_plan=SimpleNamespace(
                    groups=(),
                    tp_stage_to_processes={},
                ),
                logical_process_plan=SimpleNamespace(
                    processes=tuple(
                        SimpleNamespace(num_replicas=count) for count in replica_counts
                    )
                ),
            )
            runner_ref = self

        async def start(self, timeout: float) -> None:
            del timeout
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def wait_failed(self) -> None:
            await asyncio.Future()

    class FakeProfilerControl:
        def __init__(self, stage_control_endpoints: dict[str, str]) -> None:
            del stage_control_endpoints

        async def broadcast_start(self, **kwargs) -> None:
            profiler_calls.starts.append(kwargs)

        async def broadcast_stop(self, **kwargs) -> None:
            profiler_calls.stops.append(kwargs)

    monkeypatch.setattr(launcher, "find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "ProfilerControlClient", FakeProfilerControl)

    def fake_create_app(*args, **kwargs):
        del args
        app.state.create_app_kwargs = kwargs
        return app

    monkeypatch.setattr(launcher, "create_app", fake_create_app)
    if serve_mock is not None:
        monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve_mock)

    await launcher.run_server(config, host=host, port=8000)
    assert runner_ref is not None
    return runner_ref, app, profiler_calls


@pytest.mark.asyncio
async def test_launcher_passes_one_resolved_custom_voice_config(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    custom_voice_config = CustomVoiceConfig(
        speakers=("speaker",), task_type="CustomVoice"
    )
    resolve = Mock(return_value=custom_voice_config)
    monkeypatch.setattr(PipelineConfig, "resolve_custom_voice_config", resolve)
    _, app, _ = await run_launcher_with_fake_runner(
        config=config,
        serve_mock=AsyncMock(return_value=None),
        monkeypatch=monkeypatch,
    )
    resolve.assert_called_once_with()
    kwargs = app.state.create_app_kwargs
    assert kwargs["custom_voice_config"] is custom_voice_config
    assert kwargs["requires_uploaded_voice_for_named_voice"] is False
    assert kwargs["supports_uploaded_voice_references"] is False


@pytest.mark.asyncio
async def test_launcher_passes_moss_tts_speech_input_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig

    config = MossTTSPipelineConfig(
        model_path="OpenMOSS-Team/MOSS-TTS-v1.5",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    _, app, _ = await run_launcher_with_fake_runner(
        config=config,
        serve_mock=AsyncMock(return_value=None),
        monkeypatch=monkeypatch,
    )

    assert app.state.create_app_kwargs["max_speech_input_chars"] is None


def route_warmup_to(
    monkeypatch: pytest.MonkeyPatch, transport: httpx.AsyncBaseTransport
) -> None:
    real_client = httpx.AsyncClient

    def routed_client(
        *, base_url: str, timeout: float, trust_env: bool
    ) -> httpx.AsyncClient:
        return real_client(
            base_url=base_url,
            timeout=timeout,
            trust_env=trust_env,
            transport=transport,
        )

    monkeypatch.setattr(launcher.httpx, "AsyncClient", routed_client)


def answer_warmup_with(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    posted: list[ChatCompletionRequest],
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        posted.append(ChatCompletionRequest.model_validate_json(request.content))
        return httpx.Response(status_code, text="warmup answer")

    route_warmup_to(monkeypatch, httpx.MockTransport(respond))


def build_test_warmup_request(pipeline_config: PipelineConfig) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hi")], max_tokens=8
    )


@pytest.mark.asyncio
async def test_server_reports_up_after_its_warmup_requests_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: list[ChatCompletionRequest] = []
    answer_warmup_with(monkeypatch, 200, posted)
    server = SimpleNamespace(started=True, should_exit=False)
    app = FastAPI()
    app.state.server_status = ServerStatus.STARTING
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hi")], max_tokens=8
    )

    await launcher.warm_up_server(
        server,
        app,
        base_url="http://127.0.0.1:8000",
        requests=[request, request],
        timeout_s=5,
    )

    assert app.state.server_status is ServerStatus.UP
    assert posted == [request, request]
    assert server.should_exit is False


@pytest.mark.asyncio
async def test_warmup_sends_its_requests_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_flight_counts: list[int] = []
    in_flight: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        in_flight.append(request)
        in_flight_counts.append(len(in_flight))
        await asyncio.sleep(0.01)
        in_flight.remove(request)
        return httpx.Response(200)

    route_warmup_to(monkeypatch, httpx.MockTransport(respond))
    server = SimpleNamespace(started=True, should_exit=False)
    app = FastAPI()
    app.state.server_status = ServerStatus.STARTING
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])

    await launcher.warm_up_server(
        server,
        app,
        base_url="http://127.0.0.1:8000",
        requests=[request] * 3,
        timeout_s=5,
    )

    assert in_flight_counts == [1, 1, 1]
    assert app.state.server_status is ServerStatus.UP


@pytest.mark.asyncio
async def test_warmup_drops_its_request_when_the_server_starts_shutting_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled: list[bool] = []

    async def hold(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200)

    route_warmup_to(monkeypatch, httpx.MockTransport(hold))
    server = SimpleNamespace(started=True, should_exit=False)
    app = FastAPI()
    app.state.server_status = ServerStatus.STARTING
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    warmup = asyncio.create_task(
        launcher.warm_up_server(
            server,
            app,
            base_url="http://127.0.0.1:8000",
            requests=[request],
            timeout_s=30,
        )
    )
    await started.wait()

    server.should_exit = True
    await asyncio.wait_for(warmup, timeout=5)

    assert cancelled == [True]
    assert app.state.server_status is ServerStatus.STARTING


@pytest.mark.asyncio
async def test_transcription_warmup_uploads_a_form_the_transcription_route_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forms: list[SpeechToTextForm] = []
    uploaded_wavs: list[bytes] = []
    route_app = FastAPI()

    @route_app.post(TRANSCRIPTIONS_ENDPOINT)
    async def transcribe(
        form: SpeechToTextForm = Depends(parse_speech_to_text_form),
    ) -> dict[str, str]:
        forms.append(form)
        uploaded_wavs.append(await form.file.read())
        return {"text": ""}

    route_warmup_to(monkeypatch, httpx.ASGITransport(app=route_app))
    server = SimpleNamespace(started=True, should_exit=False)
    app = FastAPI()
    app.state.server_status = ServerStatus.STARTING
    request = TranscriptionWarmupRequest(
        wav_bytes=warmup_tone_wav_bytes(), max_new_tokens=8
    )

    await launcher.warm_up_server(
        server,
        app,
        base_url="http://127.0.0.1:8000",
        requests=[request],
        timeout_s=5,
    )

    assert app.state.server_status is ServerStatus.UP
    assert uploaded_wavs == [request.wav_bytes]
    assert (forms[0].max_new_tokens, forms[0].response_format, forms[0].stream) == (
        8,
        "json",
        False,
    )


@pytest.mark.asyncio
async def test_launcher_sends_one_warmup_request_per_replica_of_the_most_replicated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PipelineConfig,
        "server_warmup_request_factory",
        f"{__name__}.build_test_warmup_request",
    )
    warmup_requests: list[list[ChatCompletionRequest]] = []

    def record_warmup(
        server: uvicorn.Server,
        app: FastAPI,
        *,
        base_url: str,
        requests: list[ChatCompletionRequest],
        timeout_s: float,
    ) -> Coroutine[None, None, None]:
        warmup_requests.append(requests)
        return asyncio.sleep(0)

    monkeypatch.setattr(launcher, "warm_up_server", record_warmup)
    await run_launcher_with_fake_runner(
        config=make_config(tmp_path),
        serve_mock=AsyncMock(return_value=None),
        monkeypatch=monkeypatch,
        replica_counts=(1, 3, 2),
    )

    assert warmup_requests == [[build_test_warmup_request(make_config(tmp_path))] * 3]


@pytest.mark.parametrize(
    ("host", "base_url"),
    [
        ("0.0.0.0", "http://127.0.0.1:8000"),
        ("", "http://127.0.0.1:8000"),
        ("::", "http://[::1]:8000"),
        ("::1", "http://[::1]:8000"),
        ("10.1.2.3", "http://10.1.2.3:8000"),
    ],
)
@pytest.mark.asyncio
async def test_launcher_warms_up_through_the_bind_host_or_loopback_for_a_wildcard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    base_url: str,
) -> None:
    warmup_base_urls: list[str] = []

    def record_warmup(
        server: uvicorn.Server,
        app: FastAPI,
        *,
        base_url: str,
        requests: list[ChatCompletionRequest],
        timeout_s: float,
    ) -> Coroutine[None, None, None]:
        warmup_base_urls.append(base_url)
        return asyncio.sleep(0)

    monkeypatch.setattr(launcher, "warm_up_server", record_warmup)
    await run_launcher_with_fake_runner(
        config=make_config(tmp_path),
        serve_mock=AsyncMock(return_value=None),
        monkeypatch=monkeypatch,
        host=host,
    )

    assert warmup_base_urls == [base_url]


@pytest.mark.asyncio
async def test_failed_warmup_stops_the_server_before_it_reports_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_warmup_with(monkeypatch, 500, [])
    server = SimpleNamespace(started=True, should_exit=False)
    app = FastAPI()
    app.state.server_status = ServerStatus.STARTING
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(RuntimeError, match="status 500: warmup answer"):
        await launcher.warm_up_server(
            server,
            app,
            base_url="http://127.0.0.1:8000",
            requests=[request],
            timeout_s=5,
        )

    assert app.state.server_status is ServerStatus.STARTING
    assert server.should_exit is True


@pytest.mark.asyncio
async def test_launcher_uses_runner_and_mounts_profiler_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    server_serve = AsyncMock(return_value=None)

    runner, app, profiler_calls = await run_launcher_with_fake_runner(
        config=config,
        serve_mock=server_serve,
        monkeypatch=monkeypatch,
    )

    assert runner.started
    assert runner.stopped
    server_serve.assert_awaited_once()
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/start_profile",
                json={
                    "enable_torch": False,
                    "event_dir": str(tmp_path / "events"),
                },
            )
            stop_resp = client.post("/stop_profile", json={})
        assert start_resp.status_code == 200
        assert stop_resp.status_code == 200
        assert profiler_calls.starts
        assert profiler_calls.starts[0]["enable_torch"] is False
        assert profiler_calls.starts[0]["event_dir"] == str(tmp_path / "events")
        assert profiler_calls.stops == [{"run_id": None}]
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_request_only_mode_does_not_require_trace_template(
    tmp_path: Path,
) -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        def __init__(self) -> None:
            self.starts: list[dict] = []

        async def broadcast_start(self, **kwargs) -> None:
            self.starts.append(kwargs)

    app = FastAPI()
    ctl = FakeProfilerControl()
    launcher.mount_profiler_routes(app, ctl, profiler_dir=None)
    event_dir = str(tmp_path / "events")

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/start_profile",
                json={"enable_torch": False, "event_dir": event_dir},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["enable_torch"] is False
        assert body["trace_path_template"] == ""
        assert body["event_dir"] == event_dir
        assert ctl.starts
        assert ctl.starts[0]["enable_torch"] is False
        assert ctl.starts[0]["trace_path_template"] == ""
        assert ctl.starts[0]["event_dir"] == event_dir
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


class RecordingProfilerControl:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def broadcast_start(self, **kwargs: object) -> None:
        self.calls.append("start")

    async def broadcast_stop(self, **kwargs: object) -> None:
        self.calls.append("stop")


@pytest.mark.parametrize(
    "path",
    [
        "/start_profile",
        "/stop_profile",
        "/start_request_profile",
        "/stop_request_profile",
    ],
)
def test_profiler_routes_take_the_admin_key(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang_omni.serve import launcher

    monkeypatch.setenv("SGLANG_OMNI_ADMIN_KEY", "admin-key")
    app = FastAPI()
    control = RecordingProfilerControl()
    launcher.mount_profiler_routes(app, control, profiler_dir=str(tmp_path))
    body = {"enable_torch": False, "event_dir": str(tmp_path / "events")}

    with TestClient(app) as client:
        assert client.post(path, json=body).status_code == 401
    assert control.calls == []


def test_start_profile_torch_mode_still_requires_trace_template() -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        async def broadcast_start(self, **kwargs) -> None:
            raise AssertionError("start_profile should fail before broadcasting")

    app = FastAPI()
    launcher.mount_profiler_routes(app, FakeProfilerControl(), profiler_dir=None)

    with TestClient(app) as client:
        resp = client.post("/start_profile", json={"enable_torch": True})
    assert resp.status_code == 400
    assert "trace_path_template is required" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_launcher_stops_runner_when_server_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    server_serve = AsyncMock(side_effect=RuntimeError("server failed"))

    with pytest.raises(RuntimeError, match="server failed"):
        await run_launcher_with_fake_runner(
            config=config,
            serve_mock=server_serve,
            monkeypatch=monkeypatch,
        )

    server_serve.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_uvicorn_server_consumes_handled_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.serve import launcher

    config = make_config(tmp_path)
    replayed_signals: list[int] = []
    server_ref: launcher.uvicorn.Server | None = None
    original_handler = signal.getsignal(signal.SIGTERM)

    def recording_handler(sig: int, frame: FrameType | None) -> None:
        del frame
        replayed_signals.append(sig)

    async def serve_until_sigterm(
        server: launcher.uvicorn.Server,
        sockets=None,
    ) -> None:
        del sockets
        nonlocal server_ref
        server_ref = server
        signal.raise_signal(signal.SIGTERM)
        assert server.should_exit

    monkeypatch.setattr(launcher.uvicorn.Server, "_serve", serve_until_sigterm)
    signal.signal(signal.SIGTERM, recording_handler)
    try:
        runner, _, _ = await run_launcher_with_fake_runner(
            config=config,
            serve_mock=None,
            monkeypatch=monkeypatch,
        )
        assert signal.getsignal(signal.SIGTERM) is recording_handler
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert isinstance(server_ref, launcher.PipelineUvicornServer)
    assert runner.started
    assert runner.stopped
    assert replayed_signals == []
    assert (
        server_ref._captured_signals == []
    )  # noqa: leading-underscore  # upstream name


@pytest.mark.asyncio
async def test_launcher_preserves_runner_start_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)

    from sglang_omni.serve import launcher

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("start failed")

        async def stop(self) -> None:
            raise AssertionError("launcher should not stop a runner that failed start")

    monkeypatch.setattr(launcher, "find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(RuntimeError, match="start failed"):
        await launcher.run_server(config, port=8000)


@pytest.mark.parametrize(
    "window", ["start", "handover", "serve", "stop", "failed_stop"]
)
def test_launcher_cleans_up_before_passing_on_sigterm(
    window: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.serve import launcher

    events: list[str] = []
    original_handler = signal.getsignal(signal.SIGTERM)

    def previous_handler(sig: int, frame: FrameType | None) -> None:
        events.append("previous handler")

    async def receive_sigterm_then_clean_up() -> None:
        signal.raise_signal(signal.SIGTERM)
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as cancelled:
            events.append("cancelled")
            signal.raise_signal(signal.SIGTERM)
            await asyncio.sleep(0)
            if window == "failed_stop":
                raise cancelled from RuntimeError("stop failed")
            else:
                pass
            events.append("cleaned up")
            raise

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            self.coordinator = StubCoordinator()
            self.stage_control_endpoints = {}
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}),
                process_plan=SimpleNamespace(groups=(), tp_stage_to_processes={}),
            )

        async def start(self, timeout: float) -> None:
            if window == "start":
                await receive_sigterm_then_clean_up()
            else:
                pass

        async def stop(self) -> None:
            events.append("stop")
            if window in ("stop", "failed_stop"):
                await receive_sigterm_then_clean_up()
            else:
                pass

        async def wait_failed(self) -> None:
            await asyncio.Future()

    def create_app(*args, **kwargs) -> FastAPI:
        if window == "handover":
            signal.raise_signal(signal.SIGTERM)
        else:
            pass
        return FastAPI()

    async def serve(server: launcher.uvicorn.Server, sockets=None) -> None:
        if window == "serve":
            signal.raise_signal(signal.SIGTERM)
            await asyncio.sleep(0.01)
            assert server.should_exit
        elif window == "handover":
            for _ in range(100):
                await asyncio.sleep(0)
        else:
            pass

    monkeypatch.setattr(launcher, "apply_gpu_compat_env_defaults", Mock())
    monkeypatch.setattr(launcher, "find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "ProfilerControlClient", Mock())
    monkeypatch.setattr(launcher, "create_app", create_app)
    monkeypatch.setattr(launcher.uvicorn.Server, "_serve", serve)
    signal.signal(signal.SIGTERM, previous_handler)
    try:
        if window == "serve":
            launcher.launch_server(make_config(tmp_path), port=8000)
        else:
            with pytest.raises(asyncio.CancelledError) as raised:
                launcher.launch_server(make_config(tmp_path), port=8000)
        assert signal.getsignal(signal.SIGTERM) is previous_handler
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    expected_events = {
        "start": ["cancelled", "cleaned up", "previous handler"],
        "handover": ["stop", "previous handler"],
        "serve": ["stop"],
        "stop": ["stop", "cancelled", "cleaned up", "previous handler"],
        "failed_stop": ["stop", "cancelled"],
    }
    assert events == expected_events[window]
    if window == "failed_stop":
        error_text = "".join(format_exception(raised.value))
        assert "RuntimeError: stop failed" in error_text
    else:
        pass
