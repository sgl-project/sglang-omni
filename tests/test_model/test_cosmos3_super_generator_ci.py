# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Super generation quality CI (T2I / T2V / I2V / T2V+sound).

Reproduces the checkpoint's official 1280x720x189 quality recipe using the
structured prompts bundled in the checkpoint's ``assets/`` directory, so the run
needs no external data. Generation is driven in-process through
``MultiProcessPipelineRunner`` and the SDK ``Client`` (the native media path),
not an HTTP server.

Super's DiT does not fit two 80GB GPUs while resident, so this bakes in DiT
layerwise offload (the same override the 2-GPU validation used).

Opt in on the GPU node, matching tests/integration/cosmos3/test_super_gpu.py:

    COSMOS3_SUPER_RUN_GPU=1 pytest tests/test_model/test_cosmos3_super_generator_ci.py -s

Each case takes several minutes at full quality; the request/startup timeouts
default high and are overridable via COSMOS3_SUPER_REQUEST_TIMEOUT and
COSMOS3_SUPER_STARTUP_TIMEOUT.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("COSMOS3_SUPER_RUN_GPU") != "1",
    reason="Set COSMOS3_SUPER_RUN_GPU=1 on the GPU node to opt in",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATION_CONFIG = REPO_ROOT / "examples/configs/cosmos3_super_generation.yaml"

# The checkpoint's official example recipe (see the model's cookbook page).
RECIPE = {
    # 1280x720 (the largest resolution we support, so lower-res users hit no
    # surprises) but ~5s instead of the full 189 frames, to keep the CI run
    # tractable on two GPUs. 121 = 4*30+1: the temporal VAE only accepts 4n+1
    # frame counts, and 121 is the nearest valid count to 120 (5.04s @ 24fps).
    "width": 1280,
    "height": 720,
    "num_frames": 121,
    "adjust_frames": False,
    "fps": 24,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "max_sequence_length": 4096,
    "flow_shift": 10.0,
    "seed": 17,
    "use_resolution_template": False,
    "use_duration_template": False,
}
SOUND_DURATION = 121 / 24  # match video length: num_frames / fps (~5.04s)
# T2I reuses the T2V path at a single frame. The checkpoint ships no T2I example
# asset, so follow Cosmos3's structured T2I prompt schema (per the model report)
# instead of a raw caption; the negative prompt reuses the checkpoint asset.
T2I_PROMPT = {
    "subjects": [
        {
            "description": "a small autonomous warehouse robot carrying a blue cardboard box",
            "appearance_details": (
                "compact wheeled mobile robot with a matte white and yellow "
                "chassis, black rubber wheels, a flat top deck, and a thin "
                "status light bar"
            ),
            "relationship": "transporting the box across the floor",
            "location": "center of a clean warehouse aisle",
            "relative_size": "knee-height, occupying the lower-central third of the frame",
            "orientation": "three-quarter front view, moving toward camera-right",
            "pose": "in motion with the box centered on its top deck",
            "number_of_subjects": 1,
        }
    ],
    "background_setting": (
        "a clean modern warehouse with a polished concrete floor, tall metal "
        "shelving racks in soft focus, and neatly stacked pallets"
    ),
    "lighting": {
        "conditions": "bright, even industrial lighting",
        "direction": "overhead",
        "shadows": "soft contact shadow beneath the robot",
        "illumination_effect": "neutral daylight-balanced illumination",
    },
    "aesthetics": {
        "composition": "centered subject with aisle leading lines receding to a vanishing point",
        "color_scheme": "neutral grays and whites with a blue box accent",
        "mood_atmosphere": "calm, orderly, industrial",
        "patterns": "repeating shelving and floor seams",
    },
    "cinematography": {
        "framing": "wide establishing shot",
        "camera_angle": "slightly low eye-level",
        "depth_of_field": "moderate; subject sharp with background gently soft",
        "focus": "on the robot and its box",
        "lens_focal_length": "35mm",
    },
    "style_medium": "photorealistic photograph",
    "artistic_style": "clean commercial product photography",
    "context": "an automated logistics facility during operation",
    "text_and_signage_elements": [],
    "quadrant_scan": {
        "top_left": "shelving racks in soft focus",
        "top_right": "shelving racks with stacked boxes",
        "bottom_left": "polished concrete floor with subtle reflections",
        "bottom_right": "floor seams receding toward the shelving",
        "absolute_center": "the warehouse robot carrying the blue box",
    },
    "comprehensive_t2i_caption": (
        "A small autonomous warehouse robot moves a blue box across a clean, "
        "polished concrete floor in a modern logistics warehouse, flanked by "
        "tall shelving racks under bright, even overhead lighting; photorealistic "
        "wide establishing shot at eye level with a neutral gray-and-white "
        "palette accented by the blue box."
    ),
    "resolution": {"H": 720, "W": 1280},
    "aspect_ratio": 1.7778,
}
MODES = ["t2i", "t2v", "i2v", "t2vs"]


def _devices() -> list[int]:
    devices = [int(v) for v in os.environ.get("COSMOS3_SUPER_GPU_IDS", "0,1").split(",")]
    assert len(devices) in (1, 2, 4, 8) and len(set(devices)) == len(devices)
    return devices


def _resolve_model() -> Path:
    """Return a local checkpoint snapshot dir (assets/ included).

    Honors COSMOS3_SUPER_MODEL_PATH for an offline snapshot; otherwise resolves
    the config's pinned ``repo@revision`` through the shared checkpoint resolver,
    which downloads the snapshot on first use.
    """
    override = os.environ.get("COSMOS3_SUPER_MODEL_PATH")
    if override:
        path = Path(override)
        assert path.is_dir(), "COSMOS3_SUPER_MODEL_PATH must be a local snapshot dir"
        return path

    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    model_path = ConfigManager.from_file(str(GENERATION_CONFIG)).config.model_path
    return Path(resolve_checkpoint(model_path)).resolve()


def _compact(path: Path) -> str:
    """Load a structured-prompt JSON asset and re-serialize it compactly."""
    return json.dumps(json.loads(path.read_text()), separators=(",", ":"))


def _config(model: Path, devices: list[int], output: Path):
    from sglang_omni.config.manager import ConfigManager

    config = ConfigManager.from_file(str(GENERATION_CONFIG)).config
    config.model_path = str(model)
    stage = config.stages[0]
    stage.gpu, stage.runtime_gpu_ids = devices[0], devices
    stage.factory.output_dir = str(output)
    overrides = stage.factory.server_args_overrides
    overrides.update(
        use_fsdp_inference=len(devices) > 1,
        hsdp_shard_dim=len(devices),
        # Super's DiT is excluded from auto offload and does not fit 2x80GB fully
        # resident, so offload it layerwise. Keeping 24 of 128 layers resident is
        # the tuned 2-GPU default: it fits (~67 GiB/GPU peak at this recipe, ~11%
        # faster than resident=1) while leaving headroom; ~40 OOMs.
        component_residency={"transformer": "layerwise-offload"},
        layerwise_resident_layers={"transformer": 24},
    )
    return config


@asynccontextmanager
async def _running(config):
    import psutil

    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    runner = MultiProcessPipelineRunner(config)
    owned: list = []
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


async def _generate(runner, inputs: dict):
    from sglang_omni.client import Client, GenerateRequest

    async def collect():
        chunks = []
        async for chunk in Client(runner.coordinator).generate(
            GenerateRequest(prompt=inputs, stream=False)
        ):
            chunks.append(chunk)
        return chunks

    chunks = await asyncio.wait_for(
        collect(),
        timeout=float(os.environ.get("COSMOS3_SUPER_REQUEST_TIMEOUT", "1800")),
    )
    assert len(chunks) == 1 and chunks[0].finish_reason in ("stop", "length")
    return chunks[0]


def _check_media(item: dict, frames: int, width: int, height: int) -> None:
    path = Path(item["path"])
    assert path.is_file() and path.stat().st_size > 0
    if frames == 1:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            assert image.size == (width, height)
            assert np.asarray(image.convert("RGB")).std() > 0, "Image is constant"
        return
    import av

    with av.open(str(path)) as container:
        decoded = 0
        for frame in container.decode(video=0):
            assert (frame.width, frame.height) == (width, height)
            decoded += 1
        assert decoded == frames


def _check_audio(path: str, expected_seconds: float) -> None:
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


def _case_inputs(mode: str, assets: Path) -> dict:
    if mode == "t2i":
        return {
            **RECIPE,
            "num_frames": 1,
            "prompt": json.dumps(T2I_PROMPT, separators=(",", ":")),
            "negative_prompt": _compact(assets / "negative_prompt.json"),
        }
    inputs = {
        **RECIPE,
        "negative_prompt": _compact(assets / "negative_prompt.json"),
        "prompt": _compact(assets / f"example_{mode}_prompt.json"),
    }
    if mode == "i2v":
        inputs["image_path"] = str(assets / "example_i2v_input.jpg")
    if mode == "t2vs":
        inputs["sound_duration"] = SOUND_DURATION
    return inputs


@pytest.mark.asyncio
async def test_generation_quality(tmp_path):
    """Serve the generation stage once and validate T2I, T2V, I2V, and T2V+sound."""
    devices = _devices()
    model = _resolve_model()
    assets = model / "assets"
    required = (
        "negative_prompt.json",
        "example_t2v_prompt.json",
        "example_i2v_prompt.json",
        "example_t2vs_prompt.json",
        "example_i2v_input.jpg",
    )
    missing = [name for name in required if not (assets / name).is_file()]
    assert not missing, f"Missing checkpoint assets: {missing}"

    config = _config(model, devices, tmp_path / "media")
    async with _running(config) as runner:
        for mode in MODES:
            inputs = _case_inputs(mode, assets)
            chunk = await _generate(runner, inputs)
            assert chunk.media and len(chunk.media) == 1
            item = chunk.media[0]
            _check_media(item, inputs["num_frames"], RECIPE["width"], RECIPE["height"])
            if mode == "t2vs":
                _check_audio(item["path"], SOUND_DURATION)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
