# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.models.cosmos3.stages import (
    NativeGenerationScheduler,
    build_sampling_params,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


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
        "output_path": "/unowned",
        "output_file_name": "../outside",
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
    assert original["output_file_name"] == "../outside"


@pytest.mark.parametrize(
    "invalid_payload",
    [
        payload(""),
        payload("  "),
        payload([1, 2]),
        payload({"seed": 1}),
        payload("A lake", diffusion=[]),
        payload("A lake", stream=True),
        payload({"prompt": "A lake", "prompt_file_path": "prompts.txt"}),
    ],
)
def test_invalid_request_is_rejected_before_native_dispatch(invalid_payload, tmp_path):
    generator = Generator(None)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    with pytest.raises(ValueError):
        scheduler._generate(invalid_payload)
    assert not generator.calls


def test_saved_result_survives_scheduler_and_client_boundary(tmp_path):
    result = SimpleNamespace(
        output_file_path=str(tmp_path / "output.mp4"),
        size=(480, 832, 33),
        prompt="A lake",
        generation_time=2.5,
        peak_memory_mb=16000.0,
        metrics={"denoise": 1.8},
    )
    generator = Generator(result)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    scheduler.inbox.put(IncomingMessage("request-1", "new_request", payload("A lake")))
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        message = scheduler.outbox.get(timeout=3)
        assert message.type == "result"
        chunk = Client._default_result_builder("request-1", message.data.data)
        assert chunk.media[0]["path"] == result.output_file_path
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
        scheduler._generate(payload())


def test_dead_native_worker_fails_the_owning_stage(tmp_path):
    generator = Generator(None)
    generator.local_scheduler_process = [
        SimpleNamespace(pid=1234, exitcode=2, is_alive=lambda: False)
    ]
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    with pytest.raises(RuntimeError, match="1234 exited with code 2"):
        scheduler._next_message()


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

    started = threading.Event()
    settled = threading.Event()
    keep = tmp_path / "preexisting.txt"
    keep.write_text("keep")

    class CancellableGenerator(Generator):
        supports_cancellation = True

        def generate(self, *, sampling_params_kwargs, cancellation_event):
            target = Path(sampling_params_kwargs["output_path"]) / "partial.mp4"
            target.write_bytes(b"partial")
            started.set()
            assert cancellation_event.wait(3)
            settled.set()
            raise RuntimeError("native cancellation settled")

    generator = CancellableGenerator(None)
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
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
    assert not scheduler._native_requests


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
    result = scheduler._generate(payload())
    assert list(tmp_path.iterdir())
    scheduler.abort("request-1")
    scheduler._emit_result("request-1", result, scheduler.outbox)
    assert scheduler.outbox.empty()
    assert not list(tmp_path.iterdir())
    assert not scheduler._native_requests
