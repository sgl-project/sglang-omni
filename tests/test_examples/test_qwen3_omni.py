# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


async def main_async(img_path, video_path: str, audio_path: str) -> None:
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config_from_model_path(
        model_path=MODEL_PATH
    )
    config = config_cls(model_path=MODEL_PATH, relay_backend="nixl")
    coordinator, stages = compile_pipeline(config)
    runner = PipelineRunner(coordinator, stages)

    await runner.start()
    try:
        images = [img_path] if img_path else []
        videos = [video_path] if video_path else []
        audios = [audio_path] if audio_path else []
        request = {
            "messages": [
                {"role": "user", "content": "What is this about?"},
            ],
            "images": images,
            "videos": videos,
            "video_fps": 2.0,
            "use_audio_in_video": True,
            "audios": audios,
            "audio_target_sr": 16000,
        }
        result = await coordinator.submit(
            "qwen3-omni",
            OmniRequest(
                inputs=request,
                params={
                    "max_new_tokens": 32,
                    "temperature": 0.8,
                },
            ),
        )
        print(result)
    finally:
        await runner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "img_path, video_path, audio_path",
    [
        (None, None, None),
        ("tests/data/cars.jpg", None, None),
        (None, "tests/data/draw.mp4", None),
        (None, None, "tests/data/cough.wav"),
    ],
)
def test_qwen3_omni(img_path, video_path, audio_path):
    asyncio.run(main_async(img_path, video_path, audio_path))


if __name__ == "__main__":
    test_qwen3_omni()
