# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-node Super GPU smoke tests; see docs/cookbook/cosmos3_super.md.

Imports of serving code and CUDA are delayed so ordinary CPU collection is safe.
Raw artifacts stay in pytest's temporary directory; the JSON report contains
numeric measurements and revision/topology metadata, not request/response text.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("COSMOS3_SUPER_RUN_GPU") != "1",
    reason="Set COSMOS3_SUPER_RUN_GPU=1 on the GPU node to opt in",
)


def _settings():
    import torch

    model = os.environ.get("COSMOS3_SUPER_MODEL_PATH")
    assert (
        model and Path(model).is_dir()
    ), "COSMOS3_SUPER_MODEL_PATH must be a local snapshot"
    devices = [
        int(value)
        for value in os.environ.get("COSMOS3_SUPER_GPU_IDS", "0,1,2,3").split(",")
    ]
    assert len(devices) in (
        1,
        2,
        4,
        8,
    ), "Test one of the 1/2/4/8-GPU single-node allocations"
    assert len(set(devices)) == len(devices) and min(devices) >= 0
    assert torch.cuda.is_available() and max(devices) < torch.cuda.device_count()
    # Verify Super geometry before loading; the caller pins the base checkpoint.
    config = json.loads((Path(model) / "config.json").read_text())
    text = config["text_config"]
    assert config["architectures"] == ["Cosmos3ForConditionalGeneration"]
    assert (text["hidden_size"], text["num_hidden_layers"]) == (5120, 64)
    revision = os.environ.get("COSMOS3_SUPER_CHECKPOINT_REVISION")
    assert (
        revision
    ), "Record the local snapshot's commit in COSMOS3_SUPER_CHECKPOINT_REVISION"
    native_revision = os.environ.get("COSMOS3_SUPER_NATIVE_REVISION")
    assert (
        native_revision
    ), "Record patched native source revision in COSMOS3_SUPER_NATIVE_REVISION"
    return model, devices, revision, native_revision


def _config(kind, model, devices, output):
    from sglang_omni.models.cosmos3.config import (
        Cosmos3PipelineConfig,
        Cosmos3ReasonerPipelineConfig,
    )

    cls = (
        Cosmos3PipelineConfig if kind == "generation" else Cosmos3ReasonerPipelineConfig
    )
    config = cls(model_path=model)
    stage = config.stages[0]
    stage.gpu = devices[0]
    stage.runtime_gpu_ids = devices
    if kind == "generation":
        stage.factory.output_dir = str(output)
        overrides = {
            "tp_size": 1,
            "use_fsdp_inference": len(devices) > 1,
            "hsdp_shard_dim": len(devices),
            "hsdp_replicate_dim": 1,
            "warmup_mode": "off",
        }
        if len(devices) == 1:
            # Candidate low-memory path; host RAM and output activations still matter.
            overrides["component_residency"] = {"transformer": "layerwise-offload"}
            overrides["layerwise_resident_layers"] = {"transformer": 1}
    else:
        overrides = {
            "tp_size": len(devices),
            "context_length": 8192,
            "mem_fraction_static": 0.9 if len(devices) == 1 else 0.6,
        }
    extra = os.environ.get("COSMOS3_SUPER_NATIVE_OVERRIDES")
    if extra:
        overrides.update(json.loads(Path(extra).read_text()).get(kind, {}))
    stage.factory.server_args_overrides = overrides
    return config


def _revision():
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _report(kind, devices, checkpoint_revision, native_revision, config):
    import torch

    return {
        "kind": kind,
        "status": "running",
        "omni_revision": _revision(),
        "native_revision": native_revision,
        "checkpoint_revision": checkpoint_revision,
        "gpu_ids": devices,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(device),
                "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
            for device in devices
        ],
        "requested_native_options": config.stages[0].factory.server_args_overrides,
        "runs": [],
        "quality_qualified": False,
    }


def _save_report(report):
    target = Path(os.environ.get("COSMOS3_SUPER_REPORT_DIR", "results/cosmos3-super"))
    target.mkdir(parents=True, exist_ok=True)
    # Different topology/kind runs must not overwrite each other's evidence.
    path = (
        target / f"{report['kind']}-{len(report['gpu_ids'])}gpu-{time.time_ns()}.json"
    )
    path.write_text(json.dumps(report, indent=2) + "\n")


@asynccontextmanager
async def _running(config, record):
    import psutil

    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    runner = MultiProcessPipelineRunner(config)
    owned = []
    started = time.perf_counter()
    try:
        await runner.start(
            timeout=float(os.environ.get("COSMOS3_SUPER_STARTUP_TIMEOUT", "1800"))
        )
        record["startup_seconds"] = time.perf_counter() - started
        for group in runner._groups:
            for process in group.processes:
                owner = psutil.Process(process.pid)
                owned.extend([owner, *owner.children(recursive=True)])
        yield runner
    finally:
        started = time.perf_counter()
        await runner.stop()
        record["shutdown_seconds"] = time.perf_counter() - started
        _, alive = psutil.wait_procs(owned, timeout=30)
        record["remaining_owned_processes"] = len(alive)
        assert not alive, "Owned stage/native processes remained after shutdown"


async def _generate(runner, request):
    from sglang_omni.client import Client

    chunks = []
    async with asyncio.timeout(
        float(os.environ.get("COSMOS3_SUPER_REQUEST_TIMEOUT", "900"))
    ):
        async for chunk in Client(runner.coordinator).generate(request):
            chunks.append(chunk)
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
    return {
        "bytes": path.stat().st_size,
        "decoded_frames": decoded,
        "native_generation_seconds": item.get("generation_time"),
        "native_peak_memory_mb": item.get("peak_memory_mb"),
    }


@pytest.mark.asyncio
async def test_super_visual_generation_and_fresh_restart(tmp_path):
    from PIL import Image, ImageDraw

    from sglang_omni.client import GenerateRequest

    model, devices, revision, native_revision = _settings()
    config = _config("generation", model, devices, tmp_path / "media")
    report = _report("generation", devices, revision, native_revision, config)
    reference = tmp_path / "synthetic.png"
    image = Image.new("RGB", (832, 480), "white")
    ImageDraw.Draw(image).rectangle((300, 140, 530, 340), fill="red")
    image.save(reference)
    try:
        first = {}
        report["runs"].append(first)
        async with _running(config, first) as runner:
            first["requests"] = []
            for mode, frames in (("t2i", 1), ("t2v", 17), ("i2v", 17)):
                inputs = {
                    "prompt": "A red box on a white table, stationary camera.",
                    "width": 832,
                    "height": 480,
                    "num_frames": frames,
                    "fps": 24,
                    "seed": 0,
                    "num_inference_steps": 4,
                }
                if mode == "i2v":
                    inputs["image_path"] = str(reference)
                started = time.perf_counter()
                chunk = await _generate(
                    runner, GenerateRequest(prompt=inputs, stream=False)
                )
                assert chunk.media and len(chunk.media) == 1
                first["requests"].append(
                    {
                        "mode": mode,
                        "seed": 0,
                        "steps": 4,
                        "width": 832,
                        "height": 480,
                        "frames": frames,
                        "request_seconds": time.perf_counter() - started,
                        **_check_media(chunk.media[0], frames, 832, 480),
                    }
                )
        # A new runner must load and serve after the first owner fully exits.
        second = {}
        report["runs"].append(second)
        async with _running(config, second) as runner:
            chunk = await _generate(
                runner,
                GenerateRequest(
                    prompt={
                        "prompt": "A red ceramic mug.",
                        "width": 832,
                        "height": 480,
                        "num_frames": 1,
                        "seed": 0,
                        "num_inference_steps": 4,
                    },
                    stream=False,
                ),
            )
            assert chunk.media and len(chunk.media) == 1
            second["media"] = _check_media(chunk.media[0], 1, 832, 480)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        _save_report(report)


@pytest.mark.asyncio
async def test_super_reasoner_text_and_image(tmp_path):
    import base64

    from PIL import Image

    from sglang_omni.client import GenerateRequest, SamplingParams

    model, devices, revision, native_revision = _settings()
    config = _config("reasoner", model, devices, tmp_path)
    report = _report("reasoner", devices, revision, native_revision, config)
    reference = tmp_path / "red.png"
    Image.new("RGB", (224, 224), "red").save(reference)
    image_url = (
        "data:image/png;base64," + base64.b64encode(reference.read_bytes()).decode()
    )
    try:
        run = {}
        report["runs"].append(run)
        async with _running(config, run) as runner:
            run["requests"] = []
            cases = [
                ("text", "What is two plus two? Give a short answer."),
                (
                    "image",
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Name the dominant color.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url},
                                    },
                                ],
                            }
                        ]
                    },
                ),
            ]
            for mode, prompt in cases:
                started = time.perf_counter()
                chunk = await _generate(
                    runner,
                    GenerateRequest(
                        prompt=prompt,
                        sampling=SamplingParams(temperature=0.0, max_new_tokens=128),
                        stream=False,
                    ),
                )
                assert chunk.text and chunk.text.strip()
                run["requests"].append(
                    {
                        "mode": mode,
                        "text_characters": len(chunk.text),
                        "request_seconds": time.perf_counter() - started,
                    }
                )
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        _save_report(report)
