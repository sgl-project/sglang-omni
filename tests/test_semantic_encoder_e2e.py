# SPDX-License-Identifier: Apache-2.0
"""E2E test for MingSemanticEncoder (LLM + connector).

Tests that the standalone semantic encoder can:
1. Load BailingMoeV2 LLM + connector + projections
2. Encode text prompts into [B, 256, 2560] condition embeddings
3. Produce non-zero, normalized embeddings

Run on remote:
    cd /sgl-workspace/sglang-omni-dev
    source .venv/bin/activate
    PYTHONPATH=. python tests/test_semantic_encoder_e2e.py --device cuda:1
"""

from __future__ import annotations

import logging
import os
import time

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
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        MingSemanticEncoder,
    )

    device = torch.device(DEVICE)
    encoder = MingSemanticEncoder()

    # --- Phase 1: Load ---
    logger.info("=== Loading MingSemanticEncoder from %s ===", MODEL_PATH)
    t0 = time.time()
    encoder.load(MODEL_PATH, device)
    load_time = time.time() - t0
    logger.info("Encoder loaded in %.1fs", load_time)

    # Verify components
    assert encoder._llm is not None, "LLM not loaded"
    assert encoder._connector is not None, "Connector not loaded"
    assert encoder._proj_in is not None, "proj_in not loaded"
    assert encoder._proj_out is not None, "proj_out not loaded"
    assert encoder._query_tokens is not None, "query_tokens not loaded"
    logger.info("All components verified")

    # --- Phase 2: Encode single prompt ---
    prompt = "A cat sitting on a windowsill watching the sunset"
    logger.info("Encoding: %r", prompt)
    t0 = time.time()
    pos, neg = encoder.encode(prompt)
    encode_time = time.time() - t0
    logger.info("Encoded in %.2fs", encode_time)

    assert len(pos) == 1, f"Expected 1 embedding, got {len(pos)}"
    assert len(neg) == 1, f"Expected 1 negative, got {len(neg)}"

    emb = pos[0]
    logger.info("Embedding shape: %s, dtype: %s", emb.shape, emb.dtype)
    assert emb.shape == (256, 2560), f"Wrong shape: {emb.shape}"

    # Check non-zero
    emb_std = emb.float().std().item()
    emb_mean = emb.float().mean().item()
    logger.info("Embedding stats: mean=%.4f, std=%.4f", emb_mean, emb_std)
    assert emb_std > 0.001, f"Embedding appears zero (std={emb_std})"

    # Check L2 normalization (each row should have norm ~1.0)
    norms = emb.float().norm(dim=-1)
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()
    logger.info("L2 norms: mean=%.4f, std=%.4f", norm_mean, norm_std)
    assert abs(norm_mean - 1.0) < 0.1, f"Norms not ~1.0: mean={norm_mean}"

    # Check negative is zero
    neg_emb = neg[0]
    assert neg_emb.abs().max().item() < 1e-6, "Negative embeds not zero"
    logger.info("Negative embeddings verified (all zero)")

    # --- Phase 3: Encode batch ---
    prompts = [
        "A watercolor painting of bamboo and mountains",
        "A futuristic city at night with neon lights",
    ]
    logger.info("Encoding batch of %d prompts", len(prompts))
    t0 = time.time()
    pos_batch, neg_batch = encoder.encode(prompts)
    batch_time = time.time() - t0
    logger.info("Batch encoded in %.2fs", batch_time)

    assert len(pos_batch) == 2, f"Expected 2 embeddings, got {len(pos_batch)}"
    for i, e in enumerate(pos_batch):
        assert e.shape == (256, 2560), f"Batch[{i}] wrong shape: {e.shape}"
        std = e.float().std().item()
        logger.info("Batch[%d] std=%.4f", i, std)
        assert std > 0.001, f"Batch[{i}] appears zero"

    # --- Phase 4: Different prompts should produce different embeddings ---
    cos_sim = torch.nn.functional.cosine_similarity(
        pos_batch[0].float().flatten().unsqueeze(0),
        pos_batch[1].float().flatten().unsqueeze(0),
    ).item()
    logger.info("Cosine similarity between different prompts: %.4f", cos_sim)
    assert cos_sim < 0.99, f"Different prompts too similar: cos_sim={cos_sim}"

    # --- Cleanup ---
    encoder.unload()
    logger.info("=== MingSemanticEncoder E2E test PASSED ===")
    logger.info(
        "Summary: load=%.1fs, encode=%.2fs, batch=%.2fs",
        load_time,
        encode_time,
        batch_time,
    )


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
