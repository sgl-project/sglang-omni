# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-node Super GPU smoke tests; see docs/cookbook/cosmos3_super.md.

Imports of serving code and CUDA are delayed so ordinary CPU collection is safe.
Raw artifacts stay in pytest's temporary directory. Use pytest JUnit XML for
case results and revision metadata; these smoke tests do not benchmark quality.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("COSMOS3_SUPER_RUN_GPU") != "1",
    reason="Set COSMOS3_SUPER_RUN_GPU=1 on the GPU node to opt in",
)


@pytest.fixture(scope="module")
def deployment(record_testsuite_property):
    import torch

    model = os.environ["COSMOS3_SUPER_MODEL_PATH"]
    assert Path(model).is_dir(), "Use a local Super snapshot"
    config = json.loads((Path(model) / "config.json").read_text())
    assert config["architectures"] == ["Cosmos3ForConditionalGeneration"]
    assert (
        config["text_config"]["hidden_size"],
        config["text_config"]["num_hidden_layers"],
    ) == (5120, 64)
    devices = [
        int(v) for v in os.environ.get("COSMOS3_SUPER_GPU_IDS", "0,1,2,3").split(",")
    ]
    assert len(devices) in (1, 2, 4, 8) and len(set(devices)) == len(devices)
    assert (
        torch.cuda.is_available()
        and 0 <= min(devices) <= max(devices) < torch.cuda.device_count()
    )
    for name in ("CHECKPOINT_REVISION", "NATIVE_REVISION"):
        record_testsuite_property(name, os.environ[f"COSMOS3_SUPER_{name}"])
    root = Path(__file__).resolve().parents[3]
    record_testsuite_property(
        "omni_revision",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    )
    record_testsuite_property("gpu_ids", str(devices))
    record_testsuite_property(
        "gpu_names", str([torch.cuda.get_device_name(d) for d in devices])
    )
    extra = os.environ.get("COSMOS3_SUPER_NATIVE_OVERRIDES")
    if extra:
        record_testsuite_property("native_overrides", Path(extra).read_text())
    return model, devices


def _config(kind, model, devices, output):
    from sglang_omni.config.manager import ConfigManager

    root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(root / "examples/configs" / f"cosmos3_super_{kind}.yaml")
    ).config
    config.model_path = model
    stage = config.stages[0]
    stage.gpu, stage.runtime_gpu_ids = devices[0], devices
    overrides = stage.factory.server_args_overrides
    if kind == "generation":
        stage.factory.output_dir = str(output)
        overrides.update(
            use_fsdp_inference=len(devices) > 1, hsdp_shard_dim=len(devices)
        )
        if len(devices) == 1:
            overrides.update(
                component_residency={"transformer": "layerwise-offload"},
                layerwise_resident_layers={"transformer": 1},
            )
    else:
        overrides.update(
            tp_size=len(devices), mem_fraction_static=0.9 if len(devices) == 1 else 0.6
        )
    extra = os.environ.get("COSMOS3_SUPER_NATIVE_OVERRIDES")
    if extra:
        overrides.update(json.loads(Path(extra).read_text()).get(kind, {}))
    return config


@asynccontextmanager
async def _running(config):
    import psutil

    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    runner = MultiProcessPipelineRunner(config)
    owned = []
    try:
        await runner.start(
            timeout=float(os.environ.get("COSMOS3_SUPER_STARTUP_TIMEOUT", "1800"))
        )
        for group in runner._groups:
            for process in group.processes:
                owner = psutil.Process(process.pid)
                owned.extend([owner, *owner.children(recursive=True)])
        yield runner
    finally:
        await runner.stop()
        _, alive = psutil.wait_procs(owned, timeout=30)
        assert not alive, "Owned stage/native processes remained after shutdown"


async def _generate(runner, request, request_id=None):
    from sglang_omni.client import Client

    async def collect():
        chunks = []
        async for chunk in Client(runner.coordinator).generate(
            request, request_id=request_id
        ):
            chunks.append(chunk)
        return chunks

    chunks = await asyncio.wait_for(
        collect(),
        timeout=float(os.environ.get("COSMOS3_SUPER_REQUEST_TIMEOUT", "900")),
    )
    assert len(chunks) == 1 and chunks[0].finish_reason in ("stop", "length")
    return chunks[0]


def _check_media(item, frames, width, height):
    import numpy as np
    from PIL import Image

    path = Path(item["path"])
    assert path.is_file() and path.stat().st_size > 0
    if frames == 1:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert image.size == (width, height)
            assert pixels.std() > 0, "Image is constant"
        decoded = 1
    else:
        import av

        with av.open(str(path)) as container:
            decoded = 0
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (width, height)
                decoded += 1
            assert decoded == frames


def _write_video(path, images, fps):
    import av

    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = images[0].size
        stream.pix_fmt = "yuv420p"
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _generation_case(mode, directory):
    from PIL import Image, ImageDraw

    frames = 1 if mode == "t2i" else 49 if mode == "sound" else 17
    action = mode in ("policy", "inverse_dynamics", "forward_dynamics")
    inputs = {
        "prompt": "A red box moves slowly right on a white table, stationary camera.",
        "width": 832,
        "height": 480,
        "num_frames": frames,
        # Keep the same workload across GPU counts; retain native VAE frame validation.
        "adjust_frames": False,
        "fps": 5 if action else 24,
        "seed": 0,
        "num_inference_steps": 4,
        "guidance_scale": 1.0 if action else 5.0,
    }
    images = []
    for index in range(17):
        image = Image.new("RGB", (832, 480), "white")
        ImageDraw.Draw(image).rectangle(
            (200 + 10 * index, 140, 400 + 10 * index, 340), fill="red"
        )
        images.append(image)
    if mode in ("i2v", "policy", "forward_dynamics"):
        path = directory / "synthetic.png"
        images[0].save(path)
        inputs["image_path"] = str(path)
    if mode in ("v2v", "inverse_dynamics"):
        path = directory / "synthetic.mp4"
        _write_video(path, images, inputs["fps"])
        inputs["video_path"] = str(path)
        if mode == "v2v":
            inputs.update(condition_frame_indexes=[0, 1], condition_video_keep="first")
    if mode == "sound":
        inputs.update(
            prompt="A waterfall flows continuously with a loud rushing water sound.",
            sound_duration=frames / inputs["fps"],
        )
    if action:
        inputs.update(
            action_mode=mode,
            domain_name="av",
            raw_action_dim=9,
            use_system_prompt=False,
            use_duration_template=False,
        )
        if mode == "forward_dynamics":
            # Synthetic actions exercise the input contract, not trajectory quality.
            inputs["action"] = [[0.0] * 9 for _ in range(frames - 1)]
        else:
            inputs.pop("prompt")  # Exercise prompt-free action output.
    return inputs


def _check_audio(path, expected_seconds):
    import av
    import numpy as np

    seconds, samples, energy = 0.0, 0, 0.0
    with av.open(str(path)) as container:
        assert (
            len(container.streams.audio) == 1
        ), "Expected generated sound in the video"
        for frame in container.decode(audio=0):
            values = frame.to_ndarray().astype(np.float64)
            assert values.size and np.isfinite(values).all()
            samples += frame.samples
            seconds += frame.samples / frame.sample_rate
            energy += float(np.square(values).sum())
    assert samples > 0 and energy > 0, "Generated sound is empty or silent"
    # Allow AAC padding and tokenizer time quantization.
    assert abs(seconds - expected_seconds) <= 0.15


def _check_action(item, mode, horizon, dimension):
    import numpy as np

    assert item["modality"] == "action"
    response = json.loads(Path(item["path"]).read_text())
    assert response["object"] == "action.generation"
    assert len(response["data"]) == 1
    action = response["data"][0]["action"]
    assert action["action_mode"] == mode
    assert action["shape"] == [horizon, dimension]
    values = np.asarray(action["values"], dtype=float)
    assert values.shape == (horizon, dimension) and np.isfinite(values).all()


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
@pytest.mark.asyncio
async def test_super_generation(tmp_path, mode, deployment):
    from sglang_omni.client import GenerateRequest

    model, devices = deployment
    config = _config("generation", model, devices, tmp_path / "media")
    inputs = _generation_case(mode, tmp_path)
    # T2I also checks serving after the first owner completely shuts down.
    for _ in range(2 if mode == "t2i" else 1):
        async with _running(config) as runner:
            chunk = await _generate(
                runner, GenerateRequest(prompt=inputs, stream=False)
            )
            assert chunk.media and len(chunk.media) == 1
            if mode in ("policy", "inverse_dynamics"):
                _check_action(chunk.media[0], mode, inputs["num_frames"] - 1, 9)
            else:
                _check_media(chunk.media[0], inputs["num_frames"], 832, 480)
                if mode == "sound":
                    _check_audio(chunk.media[0]["path"], inputs["sound_duration"])


@pytest.mark.parametrize("mode", ["text", "image", "video"])
@pytest.mark.asyncio
async def test_super_reasoner(tmp_path, mode, deployment):
    import base64

    from PIL import Image

    from sglang_omni.client import GenerateRequest, SamplingParams

    model, devices = deployment
    config = _config("reasoner", model, devices, tmp_path)
    prompt = "What is two plus two? Give a short answer."
    if mode != "text":
        if mode == "image":
            reference = tmp_path / "red.png"
            Image.new("RGB", (224, 224), "red").save(reference)
            url = (
                "data:image/png;base64,"
                + base64.b64encode(reference.read_bytes()).decode()
            )
            question = "Name the dominant color."
        else:
            reference = tmp_path / "colors.mp4"
            _write_video(
                reference,
                [
                    Image.new("RGB", (224, 224), color)
                    for color in ("red", "blue")
                    for _ in range(8)
                ],
                4,
            )
            url = str(reference)
            question = "Describe how the color changes from the beginning to the end."
        key = f"{mode}_url"
        prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": key, key: {"url": url}},
                    ],
                }
            ]
        }
    async with _running(config) as runner:
        chunk = await _generate(
            runner,
            GenerateRequest(
                prompt=prompt,
                sampling=SamplingParams(temperature=0.0, max_new_tokens=128),
                stream=False,
            ),
        )
        assert chunk.text and chunk.text.strip()
        response = chunk.text.strip().lower()
        if mode == "text":
            assert response.endswith("4"), response
        elif mode == "image":
            assert "red" in response, response
        else:
            assert "red" in response and "blue" in response, response
            assert response.index("red") < response.index("blue"), response
        (tmp_path / "response.txt").write_text(chunk.text)


@pytest.mark.asyncio
async def test_super_reasoner_lifecycle(tmp_path, deployment):
    from sglang_omni.client import Client, GenerateRequest, SamplingParams

    model, devices = deployment
    config = _config("reasoner", model, devices, tmp_path)
    long_request = GenerateRequest(
        prompt="Count upward from one, writing every integer on a separate line.",
        sampling=SamplingParams(temperature=0.0, max_new_tokens=2048),
        stream=False,
    )

    # A native request can be cancelled without poisoning the live deployment.
    async with _running(config) as runner:
        client = Client(runner.coordinator)
        request_id = "cosmos3-super-cancel"
        pending = asyncio.create_task(
            _generate(runner, long_request, request_id=request_id), name=request_id
        )
        for _ in range(100):
            if await client.get_status(request_id) is not None:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("Cancellation request was not admitted")
        aborted = await client.abort(request_id)
        assert aborted.success
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=30)
        followup = await _generate(
            runner,
            GenerateRequest(
                prompt="What is two plus two? Give a short answer.",
                sampling=SamplingParams(temperature=0.0, max_new_tokens=128),
                stream=False,
            ),
        )
        assert followup.text and followup.text.strip().endswith("4")

    # The runner notices an unexpected stage death and releases the process tree.
    async with _running(config) as runner:
        worker = runner._groups[0].processes[0]
        worker.terminate()
        with pytest.raises(RuntimeError, match="Dead stage process"):
            await asyncio.wait_for(runner.wait_failed(), timeout=30)

    # A fresh native owner can start after the failed worker has been cleaned up.
    async with _running(config) as runner:
        restarted = await _generate(
            runner,
            GenerateRequest(
                prompt="What is two plus two? Give a short answer.",
                sampling=SamplingParams(temperature=0.0, max_new_tokens=128),
                stream=False,
            ),
        )
        assert restarted.text and restarted.text.strip().endswith("4")
