# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.native_generation.stage import (
    NativeGenerationScheduler,
    build_sampling_params,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve.openai_errors import generation_error_status_code
from tests.unit_test.fixtures.pipeline_fakes import block_on_inbox


def payload(inputs="A quiet lake", **params):
    return StagePayload("request-1", OmniRequest(inputs, params), None)


class Generator:
    local_scheduler_process = None

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.shutdown_count = 0

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def shutdown(self):
        self.shutdown_count += 1


def test_preserves_native_parameters_and_owns_output_names(tmp_path):
    original = {
        "prompt": "A quiet lake",
        "guidance_scale": 0.0,
        "num_frames": 33,
        "seed": 0,
    }
    request = payload(original, diffusion={"guidance_scale": 5.0})
    first = build_sampling_params(request, str(tmp_path))
    second = build_sampling_params(request, str(tmp_path))
    assert first["guidance_scale"] == 5.0
    assert first["seed"] == 0
    assert first["num_frames"] == 33
    assert first["save_output"] is True
    assert first["return_file_paths_only"] is True
    assert first["output_path"] == str(tmp_path)
    assert first["output_file_name"] != second["output_file_name"]
    assert "/" not in first["output_file_name"]
    assert original["guidance_scale"] == 0.0


@pytest.mark.parametrize(
    "invalid_payload",
    [
        payload(""),
        payload("  "),
        payload([1, 2]),
        payload({"seed": 1}),
        payload("A lake", diffusion=[]),
        payload("A lake", stage_params=[]),
        payload("A lake", stage_sampling={"generation": 1}),
        payload("A lake", stream=True),
        payload({"prompt": "A lake", "prompt_path": "prompts.txt"}),
        payload("A lake", diffusion={"output_path": "/unowned"}),
        payload("A lake", stage_params={"generation": {"num_outputs_per_prompt": 11}}),
    ],
)
def test_invalid_request_is_rejected_before_native_dispatch(invalid_payload, tmp_path):
    generator = Generator(None)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    with pytest.raises(ValueError) as refused:
        scheduler.generate(invalid_payload)
    assert generation_error_status_code(refused.value) == 400
    assert not generator.calls


def test_saved_result_survives_scheduler_and_client_boundary(tmp_path):
    from pathlib import Path

    result = SimpleNamespace(
        output_file_path=None,
        size=(480, 832, 33),
        prompt="A lake",
        generation_time=2.5,
        peak_memory_mb=16000.0,
        metrics={"denoise": 1.8},
    )

    class SavedGenerator(Generator):
        def generate(self, **kwargs):
            params = kwargs["sampling_params_kwargs"]
            target = Path(params["output_path"]) / f"{params['output_file_name']}.mp4"
            target.write_bytes(b"video")
            result.output_file_path = str(target)
            return super().generate(**kwargs)

    generator = SavedGenerator(result)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload("A lake")))
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        message = scheduler.outbox.get(timeout=3)
        assert message.type == "result"
        chunk = Client.default_result_builder("request-1", message.data.data)
        # The saved file leaves the request workspace for the output directory.
        assert list(tmp_path.iterdir()) == [Path(chunk.media[0]["path"])]
        assert Path(chunk.media[0]["path"]).read_bytes() == b"video"
        assert chunk.media[0]["metrics"] == {"denoise": 1.8}
        assert len(generator.calls) == 1
    finally:
        scheduler.stop()
        scheduler.stop()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert generator.shutdown_count == 1


@pytest.mark.parametrize("result", [None, [], SimpleNamespace(output_file_path=None)])
def test_native_failure_cannot_become_empty_success(result, tmp_path):
    scheduler = NativeGenerationScheduler(Generator(result), str(tmp_path))
    with pytest.raises(RuntimeError):
        scheduler.generate(payload())


@pytest.mark.parametrize(
    "in_flight, wait",
    [(False, None), (True, None), (False, block_on_inbox)],
    ids=["idle", "in-flight", "idle-blocking-loop"],
)
def test_dead_native_worker_fails_the_owning_stage(
    tmp_path, monkeypatch, in_flight, wait
):
    import multiprocessing
    import time

    if wait is not None:
        # A loop that waits for work, so only the stage can wake it.
        wait(monkeypatch)
    else:
        pass
    worker = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(60,), daemon=True
    )
    worker.start()
    entered = threading.Event()

    class BlockedGenerator(Generator):
        def generate(self, **kwargs):
            # Native generation never returns once its scheduler is gone.
            entered.set()
            threading.Event().wait()

    generator = BlockedGenerator(None)
    generator.local_scheduler_process = [worker]
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    crashes = queue.Queue()

    def run_loop():
        # The stage stops once its scheduler loop raises.
        try:
            scheduler.start()
        except RuntimeError as exc:
            crashes.put(exc)

    loop = threading.Thread(target=run_loop, daemon=True)
    loop.start()
    try:
        if in_flight:
            scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload()))
            assert entered.wait(5)
        else:
            pass
        worker.kill()
        worker.join(5)
        if in_flight:
            failed = scheduler.outbox.get(timeout=5)
            assert failed.type == "error" and f"{worker.pid} exited" in str(failed.data)
        else:
            pass
        assert f"{worker.pid} exited" in str(crashes.get(timeout=5))
    finally:
        scheduler.stop()
        loop.join(5)
    assert list(tmp_path.iterdir()) == []


def test_queued_abort_does_not_dispatch_native_work(tmp_path):
    generator = Generator(None)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    scheduler.abort("request-1")
    scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload()))
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        scheduler.stop()
        thread.join(timeout=3)
    assert generator.calls == []


def test_cosmos_config_declares_native_child_ownership():
    config = Cosmos3PipelineConfig(model_path="checkpoint")
    assert len(config.stages) == 1
    assert config.stages[0].allow_child_processes
    assert config.stages[0].tp_size == 1
    assert config.model_dump()["stages"][0]["allow_child_processes"]


def test_generate_route_parameters_reach_the_native_request(tmp_path):
    request = payload(
        "A quiet lake",
        stage_params={
            "generation": {
                "width": 1280,
                "height": 720,
                "seed": 0,
                "num_inference_steps": 35,
            }
        },
    )
    params = build_sampling_params(request, str(tmp_path))
    assert (params["width"], params["height"]) == (1280, 720)
    assert params["seed"] == 0
    assert params["num_inference_steps"] == 35


def test_active_cancel_settles_native_work_and_removes_only_owned_files(tmp_path):
    from pathlib import Path

    from sglang_omni.native_generation import cancellation

    started = threading.Event()
    settled = threading.Event()
    keep = tmp_path / "preexisting.txt"
    keep.write_text("keep")

    class CancellableGenerator(Generator):
        def generate(self, *, sampling_params_kwargs, **kwargs):
            directory = Path(sampling_params_kwargs["output_path"])
            (directory / "partial.mp4").write_bytes(b"partial")
            started.set()
            # The native worker stops once its stage marks the request.
            assert (directory / cancellation.ARMED_MARKER).exists()
            marker = directory / cancellation.CANCELLED_MARKER
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists()
            settled.set()
            return None

    generator = CancellableGenerator(None)
    scheduler = NativeGenerationScheduler(
        generator, str(tmp_path), cancellation_mode="plugin"
    )
    scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload()))
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        assert started.wait(3)
        scheduler.abort("request-1")
        assert settled.wait(3)
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        scheduler.stop()
        thread.join(3)
    assert not thread.is_alive()
    assert list(tmp_path.iterdir()) == [keep]
    assert keep.read_text() == "keep"
    assert not scheduler.native_requests


def test_abort_between_native_completion_and_omni_emission_releases_media(tmp_path):
    from pathlib import Path

    class SavedGenerator(Generator):
        def generate(self, *, sampling_params_kwargs):
            target = Path(sampling_params_kwargs["output_path"]) / "image.png"
            target.write_bytes(b"completed")
            return SimpleNamespace(
                output_file_path=str(target),
                size=(1, 1, 1),
                prompt="image",
                generation_time=1.0,
                peak_memory_mb=1.0,
                metrics={},
            )

    scheduler = NativeGenerationScheduler(SavedGenerator(None), str(tmp_path))
    result = scheduler.generate(payload())
    assert list(tmp_path.iterdir())
    scheduler.abort("request-1")
    scheduler.emit_result("request-1", result, scheduler.outbox)
    assert scheduler.outbox.empty()
    assert not list(tmp_path.iterdir())
    assert not scheduler.native_requests


@pytest.mark.parametrize("inside", [True, False])
def test_media_paths_follow_the_server_media_policy(tmp_path, monkeypatch, inside):
    from sglang_omni.native_generation import stage
    from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector

    allowed = tmp_path / "media"
    allowed.mkdir()
    image = (allowed if inside else tmp_path) / "input.png"
    image.write_bytes(b"image")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    monkeypatch.setattr(stage, "get_global_resource_connector", lambda: connector)
    output = tmp_path / "outputs"
    output.mkdir()
    generator = Generator(None)
    scheduler = NativeGenerationScheduler(generator, str(output))
    request = payload({"prompt": "A lake", "image_path": str(image)})
    with pytest.raises((RuntimeError, ValueError)) as failure:
        scheduler.generate(request)
    if inside:
        assert "without producing an output" in str(failure.value)
        assert generator.calls[0]["sampling_params_kwargs"]["image_path"] == str(image)
    else:
        assert "image_path refused" in str(failure.value)
        assert generator.calls == []
    assert list(output.iterdir()) == []


def test_background_warmup_sends_one_step_while_the_stage_serves(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime import scheduler_client, server_warmup

    sent = []
    release = threading.Event()

    def build(server_args, *, warmup_input_path=None, step_limit=None):
        assert step_limit == 1 and server_args.input_save_path.startswith(str(tmp_path))
        return ["warmup"]

    def forward(req):
        sent.append(req)
        release.wait(5)
        return SimpleNamespace(error=None)

    monkeypatch.setattr(server_warmup, "build_client_warmup_reqs", build)
    monkeypatch.setattr(
        server_warmup, "should_include_warmup_image", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(scheduler_client.sync_scheduler_client, "forward", forward)
    generator = Generator(None)
    generator.server_args = SimpleNamespace(input_save_path=None)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    scheduler.warm_in_background = True
    scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload()))
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        # The stage serves while the warmup step is still in flight.
        assert scheduler.outbox.get(timeout=3).type == "error"
        assert sent == ["warmup"] and len(generator.calls) == 1
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=3)
    deadline = time.monotonic() + 3
    while list(tmp_path.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    # The warmup input goes once the step ends.
    assert list(tmp_path.iterdir()) == []


def test_stop_stops_waiting_on_native_shutdown_once_workers_exited(tmp_path):
    import multiprocessing

    parent, child = multiprocessing.get_context("fork").Pipe()
    worker = multiprocessing.get_context("fork").Process(target=child.recv)
    worker.start()
    release = threading.Event()

    class StuckGenerator(Generator):
        local_scheduler_process = [worker]

        def shutdown(self):
            super().shutdown()
            # Native waits for the scheduler's answer, which a worker that
            # already left never sends.
            release.wait(30)

    generator = StuckGenerator(None)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    threading.Timer(0.2, parent.send, args=("exit",)).start()
    started = time.monotonic()
    try:
        scheduler.stop()
        assert time.monotonic() - started < 10
        assert generator.shutdown_count == 1
    finally:
        release.set()
        worker.join(timeout=5)
