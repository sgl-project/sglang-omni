# SPDX-License-Identifier: Apache-2.0
"""E2E test for ZImageBackend with real model weights.

Run:
    cd /sgl-workspace/sglang-omni-dev
    source .venv/bin/activate
    python tests/test_zimage_backend_e2e.py [--device cuda:1]
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

    # Load
    logger.info("=== Loading ZImageBackend from %s ===", MODEL_PATH)
    t0 = time.time()
    backend.load_models(MODEL_PATH, torch.device(DEVICE))
    logger.info("Backend loaded in %.1fs", time.time() - t0)

    # Generate
    prompts = [
        ("A cat sitting on a windowsill watching the sunset", ImageGenParams(seed=42)),
        ("一幅水墨画，画中有竹子和远山", ImageGenParams(seed=43)),
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

        # Validate
        assert image.width == params.width
        assert image.height == params.height
        arr = np.array(image)
        logger.info("Pixel stats: mean=%.1f, std=%.1f", arr.mean(), arr.std())
        assert arr.std() > 5.0, "Image appears blank"

        out = f"/tmp/test_backend_{prompt[:10].replace(' ', '_')}.png"
        image.save(out)
        logger.info("Saved to %s", out)

    # Cleanup
    backend.unload()
    logger.info("=== ZImageBackend E2E test PASSED ===")


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
