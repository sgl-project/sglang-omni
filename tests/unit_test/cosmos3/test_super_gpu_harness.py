# SPDX-License-Identifier: Apache-2.0
"""Check GPU-test configuration and artifact validation without loading models."""

import pytest
from PIL import Image

from sglang_omni.config import build_stage_placement_plan, compile_logical_processes
from tests.integration.cosmos3.test_super_gpu import _check_media, _config


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
