# SPDX-License-Identifier: Apache-2.0
"""E2E test: semantic-conditioned image generation.

Verifies that MingSemanticEncoder produces embeddings that drive
ZImage to generate meaningful images (not text rendering).

Strategy: load LLM → encode → unload LLM → load ZImage pipeline → generate.
This keeps peak GPU memory manageable.

Run on remote:
    cd /sgl-workspace/sglang-omni-dev
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 PYTHONPATH=. \
        python tests/test_semantic_image_gen_e2e.py --device cuda:0
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
OUTPUT_DIR = "/tmp/semantic_image_gen_e2e"


def main():
    global MODEL_PATH, DEVICE
    device = torch.device(DEVICE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompts = [
        "A cat sitting on a windowsill watching the sunset",
        "A futuristic city at night with neon lights",
        "一幅水墨画，画中有竹子和远山",
        "A golden retriever playing in a field of sunflowers",
    ]

    # ================================================================
    # Phase 1: Semantic encoding (LLM + connector)
    # ================================================================
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        MingSemanticEncoder,
    )

    encoder = MingSemanticEncoder()

    logger.info("=== Phase 1: Loading semantic encoder ===")
    t0 = time.time()
    encoder.load(MODEL_PATH, device)
    load_time = time.time() - t0
    logger.info("Semantic encoder loaded in %.1fs", load_time)

    all_pos = []
    all_neg = []
    for prompt in prompts:
        logger.info("Encoding: %r", prompt)
        t0 = time.time()
        pos, neg = encoder.encode(prompt)
        logger.info("Encoded in %.2fs, shape=%s", time.time() - t0, pos[0].shape)
        all_pos.append(pos[0])
        all_neg.append(neg[0])

    # Move embeddings to CPU to survive LLM unload
    all_pos = [e.cpu() for e in all_pos]
    all_neg = [e.cpu() for e in all_neg]

    # Free LLM memory
    logger.info("Unloading semantic encoder to free GPU memory...")
    encoder.unload()
    torch.cuda.empty_cache()

    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0]
        logger.info("GPU %d: %.1f GiB free after unload", i, free / (1 << 30))

    # ================================================================
    # Phase 2: ZImage generation (transformer + VAE)
    # ================================================================
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    logger.info("=== Phase 2: Loading ZImage pipeline ===")
    t0 = time.time()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )
    scheduler.config["use_dynamic_shifting"] = True

    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
    ).to(device)
    logger.info("ZImage pipeline loaded in %.1fs", time.time() - t0)

    # ================================================================
    # Phase 3: Generate images
    # ================================================================
    logger.info("=== Phase 3: Generating images ===")

    results = []
    for i, prompt in enumerate(prompts):
        pos_emb = all_pos[i].to(device)
        neg_emb = all_neg[i].to(device)

        logger.info("Generating image %d/%d: %r", i + 1, len(prompts), prompt[:60])
        t0 = time.time()

        result = pipe(
            prompt_embeds=[pos_emb],
            negative_prompt_embeds=[neg_emb],
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=2.0,  # Ming default; ZImage saturates white at >=4.0
            generator=torch.Generator(device=device).manual_seed(42),
            max_sequence_length=512,
        )
        image = result.images[0]
        elapsed = time.time() - t0

        # Validate dimensions
        assert image.width == 1024, f"Wrong width: {image.width}"
        assert image.height == 1024, f"Wrong height: {image.height}"

        arr = np.array(image)
        pixel_mean = arr.mean()
        pixel_std = arr.std()

        # Save
        safe_name = prompt[:30].replace(" ", "_").replace(",", "").replace("，", "")
        out_path = os.path.join(OUTPUT_DIR, f"img_{i}_{safe_name}.png")
        image.save(out_path)

        logger.info(
            "Image %d: %dx%d in %.1fs, pixel mean=%.1f std=%.1f → %s",
            i,
            image.width,
            image.height,
            elapsed,
            pixel_mean,
            pixel_std,
            out_path,
        )
        results.append((prompt, pixel_mean, pixel_std, out_path))

    # ================================================================
    # Phase 4: Summary
    # ================================================================
    del pipe
    torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("=== RESULTS SUMMARY ===")
    all_pass = True
    for prompt, mean, std, path in results:
        status = "OK" if std > 2.0 else "BLANK?"
        if std <= 2.0:
            all_pass = False
        logger.info("  [%s] std=%5.1f mean=%5.1f  %s", status, std, mean, prompt[:50])
        logger.info("        → %s", path)

    logger.info("=" * 60)
    logger.info("Encoder load: %.1fs", load_time)
    logger.info("Output dir: %s", OUTPUT_DIR)

    if all_pass:
        logger.info("=== ALL IMAGES GENERATED SUCCESSFULLY ===")
    else:
        logger.warning("=== SOME IMAGES MAY BE BLANK — CHECK VISUALLY ===")


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
