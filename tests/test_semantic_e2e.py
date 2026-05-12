# SPDX-License-Identifier: Apache-2.0
"""E2E test for semantic text conditioning with ZImage.

Run on remote:
    cd /sgl-workspace/sglang-omni-dev
    source .venv/bin/activate
    PYTHONPATH=. python tests/test_semantic_e2e.py [--device cuda:1]
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get(
    "MING_MODEL_PATH",
    "inclusionAI/Ming-flash-omni-2.0",
)
DEVICE = os.environ.get("CUDA_DEVICE", "cuda:1")


def main():
    from sglang_omni.models.ming_omni.diffusion.backend import ImageGenParams
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()

    logger.info(
        "=== Loading ZImageBackend (with semantic encoder) from %s ===",
        MODEL_PATH,
    )
    t0 = time.time()
    backend.load_models(MODEL_PATH, torch.device(DEVICE))
    logger.info("Backend loaded in %.1fs", time.time() - t0)

    if backend._semantic_encoder is not None:
        logger.info("Semantic encoder: LOADED (LLM + connector)")
    else:
        logger.warning("Semantic encoder: NOT LOADED (ByT5-only fallback)")

    prompts = [
        (
            "A cat sitting on a windowsill watching the sunset",
            ImageGenParams(seed=42),
        ),
        (
            "A watercolor painting of bamboo and distant mountains",
            ImageGenParams(seed=43),
        ),
        (
            "A futuristic city skyline at night with neon lights",
            ImageGenParams(seed=44, width=1024, height=1024),
        ),
    ]

    for prompt, params in prompts:
        logger.info(
            "Generating: %r (size=%dx%d, steps=%d)",
            prompt[:60],
            params.width,
            params.height,
            params.num_inference_steps,
        )
        t0 = time.time()
        image = backend.generate(prompt, params)
        elapsed = time.time() - t0
        logger.info("Generated in %.1fs: %dx%d", elapsed, image.width, image.height)

        assert image.width == params.width
        assert image.height == params.height
        arr = np.array(image)
        logger.info("Pixel stats: mean=%.1f, std=%.1f", arr.mean(), arr.std())
        assert arr.std() > 5.0, "Image appears blank"

        tag = prompt[:20].replace(" ", "_")
        out = f"/tmp/test_semantic_{tag}.png"
        image.save(out)
        logger.info("Saved to %s", out)

    backend.unload()
    logger.info("=== Semantic conditioning E2E test PASSED ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.model_path:
        MODEL_PATH = args.model_path
    if args.device:
        DEVICE = args.device
    main()
