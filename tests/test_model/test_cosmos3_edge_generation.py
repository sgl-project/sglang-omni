# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Edge visual generation on one CUDA GPU; set RUN_COSMOS3_EDGE_TESTS=1."""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from sglang_omni.client.client import Client
from sglang_omni.client.types import CompletionResult, GenerateRequest
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.utils.checkpoint import resolve_checkpoint

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

EDGE = "nvidia/Cosmos3-Edge@a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
CASES = (
    "t2i_landscape",
    "t2i_portrait",
    "t2v_53",
    "t2v_121",
    "i2v_53",
    "i2v_121",
    "video_continuation_53",
)
SamplingFields = dict[str, str | bool | int | float]


@dataclass(kw_only=True)
class VisualResult:
    path: str
    shape: list[int]
    sha256: str
    request_wall_seconds: float
    result: CompletionResult


def build_requests(checkpoint: Path, directory: Path) -> dict[str, SamplingFields]:
    assets = checkpoint / "assets"
    prompt = json.dumps(json.loads((assets / "example_i2v_prompt.json").read_text()))
    negative = json.dumps(json.loads((assets / "negative_prompt.json").read_text()))
    cases: dict[str, SamplingFields] = {}
    for name in CASES:
        large = name.endswith("121")
        width, height = (832, 480) if large else (448, 256)
        frames = 121 if large else 53
        text = prompt
        if name.startswith("t2i"):
            frames = 1
            text = "A red ceramic mug stands on a wooden table in daylight."
            if name.endswith("portrait"):
                width, height = 256, 448
                text = "A tall green plant in a white pot beside a window."
            else:
                pass
        else:
            pass
        sampling_fields: SamplingFields = {
            "prompt": text,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "num_frames": frames,
            "fps": 24,
            "num_inference_steps": 20,
            "guidance_scale": 6.0,
            "flow_shift": 12.0,
            "seed": 0,
            "use_system_prompt": False,
            "use_resolution_template": False,
            "use_duration_template": False,
            "use_guardrails": False,
            "sound_duration": 0.0,
            "output_compression": 100,
        }
        if name.startswith("i2v"):
            image_path = directory / f"conditioning_{width}x{height}.png"
            with Image.open(assets / "example_i2v_input.jpg") as image:
                ImageOps.fit(
                    image.convert("RGB"),
                    (width, height),
                    method=Image.Resampling.BILINEAR,
                ).save(image_path)
            sampling_fields["image_path"] = str(image_path)
        elif name.startswith("video_continuation"):
            sampling_fields.update(
                video_path=str(assets / "example_action_id_av_0_input.mp4"),
                condition_video_keep="first",
            )
        else:
            pass
        cases[name] = sampling_fields
    return cases


async def run_arm(
    checkpoint: Path,
    directory: Path,
    requests: dict[str, SamplingFields],
    layerwise_offload: bool,
) -> dict[str, VisualResult]:
    config = Cosmos3PipelineConfig(model_path=str(checkpoint))
    options = {"attention_backend": "torch_sdpa", "warmup_mode": "off"}
    if layerwise_offload:
        options.update(
            component_residency={"transformer": "layerwise-offload"},
            layerwise_resident_layers={"transformer": 32},
        )
    else:
        pass
    config.stages[0].factory.server_args_overrides = options
    config.stages[0].factory.output_dir = str(directory / "media")
    directory.mkdir()
    (directory / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    runner = MultiProcessPipelineRunner(config)
    records: dict[str, VisualResult] = {}
    try:
        await runner.start(timeout=900)
        client = Client(runner.coordinator)
        for name, sampling_fields in requests.items():
            started = time.perf_counter()
            result = await asyncio.wait_for(
                client.completion(
                    GenerateRequest(prompt=sampling_fields, stream=False),
                    request_id=name,
                ),
                timeout=900,
            )
            elapsed = time.perf_counter() - started
            assert result.finish_reason == "stop"
            assert result.media and len(result.media) == 1
            path = Path(result.media[0]["path"])
            pixels = iio.imread(path)
            if pixels.ndim == 3:
                pixels = pixels[None]
            else:
                pass
            assert pixels.shape == (
                sampling_fields["num_frames"],
                sampling_fields["height"],
                sampling_fields["width"],
                3,
            )
            assert pixels.dtype == np.uint8 and pixels.std() > 1
            records[name] = VisualResult(
                path=str(path),
                shape=list(pixels.shape),
                sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                request_wall_seconds=elapsed,
                result=result,
            )
            (directory / f"{name}.json").write_text(
                json.dumps(asdict(records[name]), indent=2) + "\n"
            )
    finally:
        await asyncio.wait_for(runner.stop(), timeout=90)
    for record in records.values():
        assert Path(record.path).is_file(), "Successful output lost on shutdown"
    return records


@pytest.fixture(scope="module")
def visual_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, VisualResult], dict[str, VisualResult]]:
    if os.environ.get("RUN_COSMOS3_EDGE_TESTS") != "1":
        pytest.skip("Set RUN_COSMOS3_EDGE_TESTS=1 to load the official Edge checkpoint")
    else:
        pass
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 Edge generation requires CUDA")
    else:
        pass
    checkpoint = Path(resolve_checkpoint(os.environ.get("COSMOS3_EDGE_MODEL", EDGE)))
    directory = Path(
        os.environ.get("COSMOS3_TEST_OUTPUT")
        or tmp_path_factory.mktemp("cosmos3-edge-generation")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    requests = build_requests(checkpoint, directory)
    (directory / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    baseline = asyncio.run(run_arm(checkpoint, directory / "baseline", requests, False))
    layerwise_offload = asyncio.run(
        run_arm(checkpoint, directory / "layerwise_offload", requests, True)
    )
    return baseline, layerwise_offload


@pytest.mark.parametrize("case", CASES)
def test_visual_residency_preserves_decoded_pixels(
    visual_runs: tuple[dict[str, VisualResult], dict[str, VisualResult]], case: str
) -> None:
    """Layerwise offload must not change the decoded pixels of any visual task."""
    baseline, layerwise_offload = visual_runs
    assert layerwise_offload[case].sha256 == baseline[case].sha256
