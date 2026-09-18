# SPDX-License-Identifier: Apache-2.0
"""Transport contracts without model weights or CUDA."""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from sglang_omni.config.manager import ConfigManager
from tests.integration.cosmos3.super_harness import SuperClient, write_config
from tests.integration.cosmos3.test_super_gpu import _config


@pytest.mark.parametrize("kind", ["reasoner", "generation"])
@pytest.mark.parametrize("resident", [False, True])
def test_http_config_preserves_placement_and_residency(
    tmp_path, monkeypatch, kind, resident
):
    monkeypatch.setenv(
        "COSMOS3_SUPER_RESIDENCY", "resident" if resident else "layerwise-offload"
    )
    config = _config(kind, "local-snapshot", [0, 1, 2, 3], tmp_path)
    restored = ConfigManager.from_file(str(write_config(config, tmp_path))).config
    assert restored.model_path == "local-snapshot"
    assert restored.stages[0].runtime_gpu_ids == [0, 1, 2, 3]
    assert (
        restored.stages[0].factory.server_args_overrides
        == config.stages[0].factory.server_args_overrides
    )
    if resident and kind == "generation":
        assert (
            "layerwise_resident_layers"
            not in restored.stages[0].factory.server_args_overrides
        )


@pytest.mark.asyncio
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
async def test_http_upload_poll_download_and_parameters(tmp_path, mode):
    inputs = {
        "prompt": '{"caption":"test"}',
        "width": 1280,
        "height": 720,
        "num_frames": 1 if mode == "t2i" else 121,
        "seed": 17,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "use_duration_template": False,
    }
    if mode in ("i2v", "policy", "forward_dynamics", "v2v", "inverse_dynamics"):
        key = "video_path" if mode in ("v2v", "inverse_dynamics") else "image_path"
        path = tmp_path / ("input.mp4" if key == "video_path" else "input.png")
        path.write_bytes(b"uploaded-input")
        inputs[key] = str(path)
    if mode in ("policy", "inverse_dynamics", "forward_dynamics"):
        inputs.update(action_mode=mode, domain_name="av", raw_action_dim=9)
    if mode in ("policy", "inverse_dynamics"):
        inputs.pop("prompt")
    if mode == "forward_dynamics":
        inputs["action"] = [[0.0] * 9] * 120
    if mode == "sound":
        inputs["sound_duration"] = 121 / 24
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "POST":
            if mode == "t2i":
                assert request.url.path == "/v1/images/generations"
                assert json.loads(request.content)["flow_shift"] == 10.0
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"b64_json": base64.b64encode(b"image-output").decode()}
                        ]
                    },
                )
            body = request.content.decode()
            assert '"seed": 17' in body and '"use_duration_template": false' in body
            if "image_path" in inputs or "video_path" in inputs:
                assert "uploaded-input" in body
                assert (
                    str(tmp_path) not in body
                )  # media was uploaded, not sent as a local path
            if mode in ("policy", "inverse_dynamics"):
                assert request.url.path == "/v1/actions/generations"
                return httpx.Response(
                    200, json={"object": "action.generation", "data": []}
                )
            assert request.url.path == "/v1/videos"
            return httpx.Response(200, json={"id": "v1", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"video-output")
        return httpx.Response(200, json={"id": "v1", "status": "completed"})

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handle)
    ) as http:
        client = SuperClient("http", tmp_path, http=http)
        chunk = await client.generate(inputs)
    from pathlib import Path

    output = Path(chunk.media[0]["path"])
    assert output.is_file() and output.stat().st_size
    if mode not in ("t2i", "policy", "inverse_dynamics"):
        assert [r.method for r in requests] == ["POST", "GET", "GET"]


@pytest.mark.asyncio
async def test_http_failure_does_not_publish_artifact(tmp_path):
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"id": "bad", "status": "failed", "error": "OOM"}
            )
        ),
    ) as http:
        with pytest.raises(AssertionError, match="OOM"):
            await SuperClient("http", tmp_path, http=http).generate(
                {"prompt": "test", "num_frames": 17}
            )
    assert not list(tmp_path.glob("*.mp4"))


@pytest.mark.asyncio
@pytest.mark.parametrize("video", [False, True])
async def test_benchmark_completion_uses_same_payload_and_scorer(video):
    from benchmarks.tasks.video_understanding import make_video_send_fn
    from benchmarks.tasks.visual_understand import make_mmmu_send_fn

    seen = []

    async def complete(payload):
        seen.append(payload)
        return {
            "choices": [{"message": {"content": "Answer: B"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 3},
        }

    sample = SimpleNamespace(
        sample_id="example",
        prompt="question",
        video_path="video.mp4",
        image_data_uris=["data:image/png;base64,AA=="],
    )
    factory = make_video_send_fn if video else make_mmmu_send_fn
    result = await factory("super", "unused", completion=complete, max_tokens=99)(
        None, sample
    )
    assert result.is_success and result.text == "Answer: B"
    assert result.prompt_tokens == 42 and result.completion_tokens == 3
    assert seen[0]["max_tokens"] == 99
    assert seen[0]["videos" if video else "images"] == (
        ["video.mp4"] if video else sample.image_data_uris
    )


@pytest.mark.asyncio
async def test_sdk_completion_preserves_media_and_sampling(tmp_path, monkeypatch):
    from sglang_omni.client import GenerateChunk, UsageInfo

    captured = []

    class Client:
        def __init__(self, coordinator):
            pass

        async def generate(self, request):
            captured.append(request)
            yield GenerateChunk(
                request_id="test",
                text="Answer: C",
                finish_reason="stop",
                usage=UsageInfo(prompt_tokens=10, completion_tokens=2),
            )

    monkeypatch.setattr("sglang_omni.client.Client", Client)
    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "images": ["image"],
        "videos": ["video"],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    result = await SuperClient(
        "sdk", tmp_path, runner=SimpleNamespace(coordinator=None)
    ).completion(payload)
    assert captured[0].prompt == {
        k: payload[k] for k in ("messages", "images", "videos")
    }
    assert captured[0].sampling.max_new_tokens == 256
    assert captured[0].sampling.temperature == 0.0
    assert result["usage"]["completion_tokens"] == 2


def test_memory_sampler_records_peak_and_failure(tmp_path, monkeypatch):
    import threading

    from tests.integration.cosmos3.super_harness import measurements

    sampled = threading.Event()

    def query(*args, **kwargs):
        sampled.set()
        return SimpleNamespace(stdout="0, NVIDIA H100, 1234, 81920\n")

    monkeypatch.setattr("subprocess.run", query)
    with measurements(tmp_path, [0]):
        assert sampled.wait(2)
    report = json.loads((tmp_path / "memory.json").read_text())
    assert report["sampled_peak_mib"] == {"0": 1234.0}

    def fail(*args, **kwargs):
        raise OSError("no sampler")

    monkeypatch.setattr("subprocess.run", fail)
    with (
        pytest.raises(AssertionError, match="Memory sampling failed"),
        measurements(tmp_path, [0]),
    ):
        pass
    assert (
        json.loads((tmp_path / "memory.json").read_text())["samples"][0]["error"]
        == "no sampler"
    )


@pytest.mark.asyncio
async def test_http_polling_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS3_SUPER_REQUEST_TIMEOUT", "0.01")
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"id": "stuck", "status": "queued"}
            )
        ),
    ) as http:
        with pytest.raises(TimeoutError):
            await SuperClient("http", tmp_path, http=http).generate(
                {"prompt": "test", "num_frames": 17}
            )
    assert not list(tmp_path.glob("*.mp4"))
