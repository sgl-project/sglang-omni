# SPDX-License-Identifier: Apache-2.0
"""Action dispatch and result ownership; no weights or native GPU execution."""

import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.cosmos3.stages import NativeGenerationScheduler
from sglang_omni.proto import OmniRequest, StagePayload


def payload(mode="policy", **inputs):
    return StagePayload("action-1", OmniRequest({"action_mode": mode, **inputs}), None)


@pytest.fixture
def action_runtime(monkeypatch, native):
    protocol = ModuleType("sglang.multimodal_gen.runtime.entrypoints.action.protocol")
    response = {
        "object": "action.generation",
        "data": [{"action": {"values": [[0.5]]}}],
    }
    protocol.action_generation_response = Mock(return_value=response)
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    generator = SimpleNamespace(
        local_scheduler_process=None,
        supports_cancellation=True,
        server_args=object(),
        generate_action=Mock(return_value={"actions": [[0.5]]}),
        generate=Mock(side_effect=AssertionError("Expected action dispatch")),
        shutdown=Mock(),
    )
    return SimpleNamespace(
        generator=generator,
        response=response,
        formatted=protocol.action_generation_response,
    )


@pytest.mark.parametrize(
    "inputs", [{"action_mode": "unknown"}, {"num_outputs_per_prompt": 2}]
)
def test_invalid_action_dispatch_is_rejected_before_native_call(
    action_runtime, tmp_path, inputs
):
    scheduler = NativeGenerationScheduler(action_runtime.generator, str(tmp_path))
    request = payload()
    request.request.inputs.update(inputs)
    with pytest.raises(ValueError):
        scheduler._generate(request)
    assert not action_runtime.generator.generate_action.called
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics"])
def test_action_json_crosses_client_boundary_and_survives_delivery(
    action_runtime, tmp_path, mode
):
    scheduler = NativeGenerationScheduler(action_runtime.generator, str(tmp_path))
    result = scheduler._generate(
        payload(" " + mode.upper() + " ", num_frames=17, guidance_scale=1.0, seed=0)
    )
    chunk = Client._default_result_builder(result.request_id, result.data)
    assert chunk.media[0]["modality"] == "action"
    path = Path(chunk.media[0]["path"])
    assert path.parent.parent == tmp_path
    assert json.loads(path.read_text()) == action_runtime.response
    kwargs = action_runtime.generator.generate_action.call_args.kwargs
    assert "cancellation_event" not in kwargs
    params = kwargs["sampling_params_kwargs"]
    assert params["action_mode"] == mode and params["prompt"] == ""
    assert params["guidance_scale"] == 1.0 and params["seed"] == 0
    assert not params["save_output"] and not params["return_file_paths_only"]
    assert (
        action_runtime.formatted.call_args.args[1]
        is action_runtime.generator.server_args
    )
    assert scheduler.claim_result(result, terminal=True)
    scheduler.release_result(result, delivered=True)
    scheduler.stop()
    assert path.exists()


def test_forward_dynamics_keeps_native_video_path(action_runtime, tmp_path):
    calls = []
    result = SimpleNamespace(
        output_file_path=str(tmp_path / "video.mp4"),
        size=(480, 832, 17),
        prompt="",
        generation_time=1,
        peak_memory_mb=1,
        metrics={},
    )
    action_runtime.generator.generate = lambda **kwargs: calls.append(kwargs) or result
    scheduler = NativeGenerationScheduler(action_runtime.generator, str(tmp_path))
    actions = [[0.0] * 9] * 16
    output = scheduler._generate(
        payload("forward_dynamics", action=actions, image_path="input.png")
    )
    assert not action_runtime.generator.generate_action.called
    assert calls[0]["sampling_params_kwargs"]["action"] == actions
    assert output.data["media"][0]["path"] == result.output_file_path
    scheduler.release_result(output, delivered=False)


@pytest.mark.parametrize("failure", ["native", "serialization", "delivery"])
def test_failed_action_removes_only_its_owned_output(action_runtime, tmp_path, failure):
    keep = tmp_path / "input.png"
    keep.write_bytes(b"user input")
    scheduler = NativeGenerationScheduler(action_runtime.generator, str(tmp_path))
    if failure == "native":

        def fail(**kwargs):
            raise RuntimeError("native action failed")

        action_runtime.generator.generate_action = fail
    elif failure == "serialization":
        action_runtime.response["invalid"] = float("nan")
    if failure == "delivery":
        result = scheduler._generate(payload())
        assert scheduler.claim_result(result, terminal=True)
        scheduler.release_result(result, delivered=False)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            scheduler._generate(payload())
    assert list(tmp_path.iterdir()) == [keep]
    assert not scheduler._native_requests


def test_active_action_cancel_waits_for_native_settlement(action_runtime, tmp_path):
    started, finish = threading.Event(), threading.Event()
    errors = []

    def blocking_action(**kwargs):
        started.set()
        assert finish.wait(5)
        return {"actions": [[0.5]]}

    action_runtime.generator.generate_action = blocking_action
    scheduler = NativeGenerationScheduler(action_runtime.generator, str(tmp_path))

    def run():
        try:
            scheduler._generate(payload())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(3)
        directory = scheduler._native_requests["action-1"].directory
        scheduler.abort("action-1")
        assert directory.exists(), "Cleanup ran before native settled"
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert not directory.exists()
    assert not scheduler._native_requests
