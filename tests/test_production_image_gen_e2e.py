# SPDX-License-Identifier: Apache-2.0
"""E2E test: production-level image generation via thinker → conditioner → ZImage.

Tests the full production path where the SGLang thinker (BailingMoeV2)
runs a prefill-only forward pass with injected query tokens, captures
hidden states, and a lightweight SemanticConditioner (~2.9GB) projects
them into condition embeddings for the ZImage diffusion pipeline.

This is the Phase 2 path that eliminates the 200GB MingSemanticEncoder.

GPU layout (6× H200, CUDA_VISIBLE_DEVICES=1,2,3,4,5,7):
  - Thinker: TP=4, cuda:0-3  (physical GPUs 1,2,3,4)
  - Conditioner + ZImage: cuda:4  (physical GPU 5)

Run on remote:
    cd /sgl-workspace/sglang-omni-dev
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 PYTHONPATH=. \
        python tests/test_production_image_gen_e2e.py
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from typing import Any

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
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
THINKER_GPU = int(os.environ.get("THINKER_GPU", "1"))
DIFFUSION_GPU = os.environ.get("DIFFUSION_GPU", "cuda:5")
OUTPUT_DIR = "/tmp/production_image_gen_e2e"


async def run_thinker_prefill(
    executor,
    payload,
    request_id: str,
) -> Any:
    """Submit a request to the thinker executor and return the result."""
    await executor.add_request(payload)
    result = await executor.get_result()
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="E2E test: thinker → conditioner → ZImage production path"
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--thinker-gpu", type=int, default=None)
    parser.add_argument("--diffusion-gpu", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    global MODEL_PATH, TP_SIZE, THINKER_GPU, DIFFUSION_GPU, OUTPUT_DIR
    if args.model_path:
        MODEL_PATH = args.model_path
    if args.tp_size:
        TP_SIZE = args.tp_size
    if args.thinker_gpu is not None:
        THINKER_GPU = args.thinker_gpu
    if args.diffusion_gpu:
        DIFFUSION_GPU = args.diffusion_gpu
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompts = [
        "A cat sitting on a windowsill watching the sunset",
        "一幅水墨画，画中有竹子和远山",
    ]

    asyncio.run(_run_test(prompts))


async def _run_test(prompts: list[str]):
    diffusion_device = torch.device(DIFFUSION_GPU)

    # ==================================================================
    # Phase 1: Load SemanticConditioner (~2.9GB)
    # ==================================================================
    logger.info("=== Phase 1: Loading SemanticConditioner on %s ===", DIFFUSION_GPU)
    from sglang_omni.models.ming_omni.diffusion.semantic_conditioner import (
        SemanticConditioner,
    )

    conditioner = SemanticConditioner()
    t0 = time.time()
    conditioner.load(MODEL_PATH, diffusion_device)
    cond_load_time = time.time() - t0
    logger.info("SemanticConditioner loaded in %.1fs", cond_load_time)
    logger.info(
        "  query_tokens: %s, img_gen_scales: %s",
        list(conditioner.query_tokens.shape),
        conditioner.img_gen_scales,
    )

    # ==================================================================
    # Phase 2: Create MingPreprocessor with conditioner
    # ==================================================================
    logger.info("=== Phase 2: Creating MingPreprocessor ===")
    from sglang_omni.models.ming_omni.components.preprocessor import MingPreprocessor

    preprocessor = MingPreprocessor(model_path=MODEL_PATH, conditioner=conditioner)
    logger.info(
        "Preprocessor created, image_patch_token_id=%s",
        preprocessor._image_patch_token_id,
    )

    # ==================================================================
    # Phase 3: Run preprocessor on image_gen requests
    # ==================================================================
    logger.info("=== Phase 3: Running preprocessor on %d prompts ===", len(prompts))
    from sglang_omni.proto import OmniRequest, StagePayload

    preprocessed_payloads = []
    for i, prompt in enumerate(prompts):
        request = OmniRequest(
            inputs={
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "image_generation": {
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 28,
                    "guidance_scale": 2.0,
                },
            },
            params={"max_new_tokens": 2048, "temperature": 0.0},
        )
        payload = StagePayload(
            request_id=f"img-{i}",
            request=request,
            data={},
        )
        result = await preprocessor(payload)
        preprocessed_payloads.append(result)

        # Verify preprocessing
        from sglang_omni.models.ming_omni.io import PipelineState

        state = PipelineState.from_dict(result.data)
        image_gen = state.mm_inputs.get("image_gen", {})
        gen_mask = image_gen.get("gen_mask")
        query_tokens = image_gen.get("query_tokens")
        prefill_only = image_gen.get("prefill_only")

        assert gen_mask is not None, "gen_mask not set by preprocessor"
        assert query_tokens is not None, "query_tokens not set by preprocessor"
        assert prefill_only is True, "prefill_only not set"

        num_query = sum(gen_mask)
        input_ids = state.prompt["input_ids"]
        logger.info(
            "  [%d] input_ids: %s, gen_mask sum=%d, prefill_only=%s",
            i,
            list(input_ids.shape),
            num_query,
            prefill_only,
        )

    # ==================================================================
    # Phase 4: Load SGLang thinker with capture_hidden=True
    # ==================================================================
    logger.info(
        "=== Phase 4: Loading SGLang thinker (TP=%d, gpu=%d, capture_hidden=True) ===",
        TP_SIZE,
        THINKER_GPU,
    )
    from sglang_omni.models.ming_omni.pipeline.stages import (
        create_sglang_thinker_executor_from_config,
    )

    t0 = time.time()
    thinker_executor = create_sglang_thinker_executor_from_config(
        model_path=MODEL_PATH,
        gpu_id=THINKER_GPU,
        thinker_max_seq_len=8192,
        server_args_overrides=(
            {"tp_size": TP_SIZE, "base_gpu_id": THINKER_GPU} if TP_SIZE > 1 else None
        ),
        capture_hidden=True,
    )
    thinker_load_time = time.time() - t0
    logger.info("SGLang thinker loaded in %.1fs", thinker_load_time)

    # Start the executor (initializes the engine)
    await thinker_executor.start()
    logger.info("SGLang thinker executor started")

    # ==================================================================
    # Phase 5: Run thinker prefill-only, capture hidden states
    # ==================================================================
    logger.info(
        "=== Phase 5: Running thinker prefill-only for %d prompts ===", len(prompts)
    )

    thinker_results = []
    for i, payload in enumerate(preprocessed_payloads):
        logger.info("  [%d] Submitting prefill-only request...", i)
        t0 = time.time()

        await thinker_executor.add_request(payload)
        result = await thinker_executor.get_result()
        elapsed = time.time() - t0

        # Extract thinker output
        from sglang_omni.models.ming_omni.io import PipelineState

        state = PipelineState.from_dict(result.data)
        thinker_out = state.thinker_out
        assert thinker_out is not None, "No thinker output"

        extra = thinker_out.get("extra_model_outputs", {})
        hidden_states = extra.get("hidden_states")

        if hidden_states is None:
            logger.error(
                "  [%d] NO hidden_states captured! Keys: %s", i, list(extra.keys())
            )
            raise RuntimeError(f"Hidden states not captured for prompt {i}")

        if isinstance(hidden_states, torch.Tensor):
            hs_shape = list(hidden_states.shape)
        elif isinstance(hidden_states, dict):
            hs_shape = {
                k: list(v.shape) if hasattr(v, "shape") else type(v).__name__
                for k, v in hidden_states.items()
            }
        else:
            hs_shape = type(hidden_states).__name__

        logger.info(
            "  [%d] Thinker done in %.2fs, hidden_states: %s, output_ids: %d tokens",
            i,
            elapsed,
            hs_shape,
            len(thinker_out.get("output_ids", [])),
        )
        thinker_results.append(result)

    # Stop thinker to free GPU memory before loading ZImage
    logger.info("Stopping thinker executor to free GPU memory...")
    await thinker_executor.stop()
    del thinker_executor
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0]
        logger.info("GPU %d: %.1f GiB free", i, free / (1 << 30))

    # ==================================================================
    # Phase 6: Project hidden states through conditioner
    # ==================================================================
    logger.info("=== Phase 6: Projecting hidden states through conditioner ===")

    condition_embeds_list = []
    for i, result in enumerate(thinker_results):
        state = PipelineState.from_dict(result.data)
        thinker_out = state.thinker_out
        extra = thinker_out.get("extra_model_outputs", {})
        hidden_states = extra.get("hidden_states")

        # Resolve hidden_states to a tensor
        if isinstance(hidden_states, dict):
            numeric_keys = [
                k
                for k in hidden_states
                if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
            ]
            if numeric_keys:
                last_key = max(numeric_keys, key=lambda k: int(k))
                hs = hidden_states[last_key]
            elif "_single" in hidden_states:
                hs = hidden_states["_single"]
            else:
                hs = next(iter(hidden_states.values()))
        elif isinstance(hidden_states, torch.Tensor):
            hs = hidden_states
        else:
            raise TypeError(f"Unexpected hidden_states type: {type(hidden_states)}")

        logger.info(
            "  [%d] Hidden states tensor: shape=%s, dtype=%s",
            i,
            list(hs.shape),
            hs.dtype,
        )

        # Extract query token positions using gen_mask
        image_gen = state.mm_inputs.get("image_gen", {})
        gen_mask_list = image_gen.get("gen_mask")
        assert gen_mask_list is not None, "gen_mask missing from state"

        gen_mask = torch.tensor(gen_mask_list, dtype=torch.bool, device=hs.device)

        # Hidden states may be shorter than gen_mask when radix cache
        # reuses a common prefix across prompts — only the new (extended)
        # tokens are captured.  Align by keeping the tail of gen_mask.
        if hs.dim() == 2 and gen_mask.shape[0] > hs.shape[0]:
            offset = gen_mask.shape[0] - hs.shape[0]
            logger.info(
                "  [%d] gen_mask len=%d > hs seq_len=%d, trimming prefix (%d tokens cached)",
                i,
                gen_mask.shape[0],
                hs.shape[0],
                offset,
            )
            gen_mask = gen_mask[offset:]

        if hs.dim() == 2:
            # [seq_len, hidden_dim] → [1, num_query, hidden_dim]
            query_hidden = hs[gen_mask].unsqueeze(0)
        elif hs.dim() == 3:
            # [batch, seq_len, hidden_dim]
            query_hidden = hs[:, gen_mask, :]
        else:
            raise ValueError(f"Unexpected hidden_states dim={hs.dim()}")

        logger.info(
            "  [%d] Query hidden: %s → projecting through conditioner",
            i,
            list(query_hidden.shape),
        )

        t0 = time.time()
        cond_embeds = conditioner.project(query_hidden)
        proj_time = time.time() - t0

        # Validate output
        norms = cond_embeds[0].float().norm(dim=-1)
        logger.info(
            "  [%d] Condition embeds: %s, norm mean=%.4f std=%.4f, time=%.3fs",
            i,
            list(cond_embeds.shape),
            norms.mean().item(),
            norms.std().item(),
            proj_time,
        )

        condition_embeds_list.append(cond_embeds[0].cpu())

    # Free conditioner before loading ZImage
    conditioner.unload()
    del conditioner
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Phase 7: Load ZImage pipeline and generate images
    # ==================================================================
    logger.info("=== Phase 7: Loading ZImage pipeline on %s ===", DIFFUSION_GPU)
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

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
    ).to(diffusion_device)
    zimage_load_time = time.time() - t0
    logger.info("ZImage pipeline loaded in %.1fs", zimage_load_time)

    # ==================================================================
    # Phase 8: Generate images
    # ==================================================================
    logger.info("=== Phase 8: Generating images ===")
    results = []
    for i, prompt in enumerate(prompts):
        pos_emb = condition_embeds_list[i].to(diffusion_device)
        neg_emb = pos_emb * 0.0

        logger.info("Generating image %d: %r", i, prompt[:60])
        t0 = time.time()

        result = pipe(
            prompt_embeds=[pos_emb],
            negative_prompt_embeds=[neg_emb],
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=2.0,
            generator=torch.Generator(device=diffusion_device).manual_seed(42),
            max_sequence_length=512,
        )
        image = result.images[0]
        elapsed = time.time() - t0

        arr = np.array(image)
        pixel_mean = arr.mean()
        pixel_std = arr.std()

        safe_name = prompt[:20].replace(" ", "_").replace(",", "").replace("，", "")
        out_path = os.path.join(OUTPUT_DIR, f"prod_{i}_{safe_name}.png")
        image.save(out_path)

        logger.info(
            "  [%d] %dx%d in %.1fs, pixel mean=%.1f std=%.1f → %s",
            i,
            image.width,
            image.height,
            elapsed,
            pixel_mean,
            pixel_std,
            out_path,
        )
        results.append((prompt, pixel_mean, pixel_std, out_path))

    # ==================================================================
    # Summary
    # ==================================================================
    del pipe
    torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("=== FINAL SUMMARY ===")
    logger.info("SemanticConditioner load: %.1fs", cond_load_time)
    logger.info("SGLang thinker load (TP=%d): %.1fs", TP_SIZE, thinker_load_time)
    logger.info("ZImage pipeline load: %.1fs", zimage_load_time)

    all_pass = True
    for i, (prompt, mean, std, path) in enumerate(results):
        status = "OK" if std > 2.0 else "BLANK?"
        if std <= 2.0:
            all_pass = False
        logger.info("  [%s] std=%5.1f mean=%5.1f  %s", status, std, mean, prompt[:50])
        logger.info("        → %s", path)

    logger.info("=" * 60)
    logger.info("Output dir: %s", OUTPUT_DIR)

    if all_pass:
        logger.info("=== TEST PASSED: Production image gen path works ===")
    else:
        logger.warning("=== TEST FAILED: Some images may be blank ===")


if __name__ == "__main__":
    main()
