# SPDX-License-Identifier: Apache-2.0
"""Check GPU-test configuration and artifact validation without loading models."""

import json
import wave

import numpy as np
import pytest
from PIL import Image

from sglang_omni.config import build_stage_placement_plan, compile_logical_processes
from sglang_omni.models.cosmos3.reasoner import build_chat_fields
from sglang_omni.models.cosmos3.stages import build_sampling_params
from sglang_omni.proto import OmniRequest, StagePayload
from tests.integration.cosmos3.test_super_gpu import (
    _check_action,
    _check_audio,
    _check_media,
    _config,
    _generation_case,
)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize("kind", ["generation", "reasoner"])
def test_gpu_campaign_uses_exact_allocation(count, kind, tmp_path, monkeypatch):
    monkeypatch.delenv("COSMOS3_SUPER_NATIVE_OVERRIDES", raising=False)
    devices = list(range(count))
    config = _config(kind, "local-super", devices, tmp_path)
    plan = build_stage_placement_plan(config)
    processes, _ = compile_logical_processes(config)
    assert set(plan.gpus) == set(devices)
    assert len(processes.processes) == 1
    assert processes.processes[0].tp_size == 1
    native = config.stages[0].factory.server_args_overrides
    if kind == "generation":
        assert "sp_degree" not in native
        assert "ulysses_degree" not in native
        assert "enable_cfg_parallel" not in native
        assert native["use_fsdp_inference"] == (count > 1)
        if count == 1:
            assert native["component_residency"]["transformer"] == "layerwise-offload"
    else:
        assert native["tp_size"] == count


def test_artifact_check_rejects_wrong_size_and_constant_image(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (32, 32), "white").save(path)
    with pytest.raises(AssertionError):
        _check_media({"path": str(path)}, 1, 64, 64)
    with pytest.raises(AssertionError, match="constant"):
        _check_media({"path": str(path)}, 1, 32, 32)


def test_artifact_check_decodes_and_counts_real_video_frames(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "synthetic.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=24)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for color in ("red", "green", "blue"):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    item = {"path": str(path)}
    assert _check_media(item, 3, 32, 32)["decoded_frames"] == 3
    with pytest.raises(AssertionError):
        _check_media(item, 4, 32, 32)


@pytest.mark.parametrize(
    "mode",
    [
        "t2i",
        "t2v",
        "i2v",
        "v2v",
        "sound",
        "policy",
        "inverse_dynamics",
        "forward_dynamics",
    ],
)
def test_modality_fixture_reaches_native_sampling_without_losing_options(
    tmp_path, mode
):
    inputs = _generation_case(mode, tmp_path)
    params = build_sampling_params(
        StagePayload("test", OmniRequest(inputs), None), str(tmp_path)
    )
    assert all(params[key] == value for key, value in inputs.items())
    if mode in ("v2v", "inverse_dynamics"):
        assert (
            _check_media({"path": inputs["video_path"]}, 17, 832, 480)["decoded_frames"]
            == 17
        )
    if mode == "sound":
        assert params["sound_duration"] == params["num_frames"] / params["fps"]


def test_video_reasoner_keeps_native_content_and_processing_options():
    content = [
        {"type": "text", "text": "Describe the motion."},
        {
            "type": "video_url",
            "video_url": {"url": "/local/video.mp4", "fps": 2, "max_frames": 8},
        },
    ]
    request = StagePayload(
        "video", OmniRequest({"messages": [{"role": "user", "content": content}]}), None
    )
    fields = build_chat_fields(request, "super", set())
    assert fields["messages"][0]["content"] == content
    assert fields["model"] == "super" and fields["rid"] == "video"


def test_audio_check_decodes_signal_and_rejects_silence_and_wrong_duration(tmp_path):
    rate = 16000
    path = tmp_path / "tone.wav"
    signal = (10000 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype("<i2")

    def write(values):
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            output.writeframes(values.tobytes())

    write(signal)
    assert _check_audio(path, 1.0)["audio_samples"] == rate
    with pytest.raises(AssertionError):
        _check_audio(path, 2.0)
    write(np.zeros_like(signal))
    with pytest.raises(AssertionError, match="silent"):
        _check_audio(path, 1.0)


@pytest.mark.parametrize("failure", [None, "shape", "nonfinite", "mode", "missing"])
def test_action_artifact_validation(tmp_path, failure):
    action = {
        "action_mode": "inverse_dynamics",
        "shape": [16, 9],
        "values": [[0.0] * 9 for _ in range(16)],
    }
    if failure == "shape":
        action["values"].pop()
    elif failure == "nonfinite":
        action["values"][0][0] = float("nan")
    elif failure == "mode":
        action["action_mode"] = "policy"
    response = {
        "object": "action.generation",
        "data": [] if failure == "missing" else [{"action": action}],
    }
    path = tmp_path / "action.json"
    path.write_text(json.dumps(response))
    item = {"path": str(path), "modality": "action"}
    if failure:
        with pytest.raises(AssertionError):
            _check_action(item, "inverse_dynamics", 16, 9)
    else:
        assert _check_action(item, "inverse_dynamics", 16, 9) == {
            "action_horizon": 16,
            "action_dim": 9,
        }
