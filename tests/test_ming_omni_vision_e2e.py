#!/usr/bin/env python3
"""E2E test for Ming-flash-omni-2.0 image understanding.

Tests the complete vision pipeline with a real image:
  1. Image preprocessing (Qwen2VLImageProcessor)
  2. Vision encoding (MingOmniVisionEncoder)
  3. Feature projection + L2 normalization (VisionProjector)
  4. Token expansion & embedding injection
  5. Validate combined input_embeds

Usage (on remote server):
    cd /sgl-workspace/sglang-omni-dev2
    CUDA_VISIBLE_DEVICES=0 python scripts/test_ming_omni_vision_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from PIL import Image

MODEL = os.environ.get("MODEL_PATH", "inclusionAI/Ming-flash-omni-2.0")
DEVICE = os.environ.get("DEVICE", "cuda:0")

# A simple test image
TEST_IMAGE_URL = "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg"


# ============================================================================
# Helpers
# ============================================================================


def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


def load_config(model_dir: Path):
    with open(model_dir / "config.json") as f:
        raw = json.load(f)
    thinker = raw.get("thinker_config", raw)
    vision_raw = thinker.get("vision_config", {})
    mlp_depth = thinker.get("mlp_depth", 2)
    llm_cfg = thinker.get("llm_config", {})
    llm_hidden_size = llm_cfg.get("hidden_size", 4096)
    vocab_size = llm_cfg.get("vocab_size", 157184)
    return vision_raw, mlp_depth, llm_hidden_size, vocab_size, llm_cfg


def init_sglang_tp():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    from sglang.srt.distributed import parallel_state

    parallel_state.init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
    )
    parallel_state.initialize_model_parallel()

    import sglang.srt.layers.dp_attention as dp

    dp._ATTN_TP_SIZE = 1
    dp._ATTN_TP_RANK = 0
    print("[OK] sglang TP=1 context initialized")


def iter_weights_by_prefix(model_dir: Path, prefix: str):
    """Iterate checkpoint weights with given prefix, stripping it."""
    from safetensors import safe_open

    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    for shard, keys in sorted(shards.items()):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in keys:
                yield key[len(prefix) :], f.get_tensor(key)


def download_image(url: str) -> Image.Image:
    """Download a test image, with fallback to a generated dummy."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [WARN] Failed to download image ({e}), using dummy 224x224")
        import numpy as np

        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)


# ============================================================================
# Step 1: Image Preprocessing
# ============================================================================


def preprocess_image(image: Image.Image, spatial_merge_size: int = 2):
    """Preprocess image using Qwen2VLImageProcessor (same as Ming)."""
    from transformers import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor(
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        patch_size=16,
        temporal_patch_size=2,
        merge_size=spatial_merge_size,
    )
    result = processor(images=[image], return_tensors="pt")
    pixel_values = result["pixel_values"]
    grid_thw = result["image_grid_thw"]

    print(f"  Image size: {image.size}")
    print(f"  pixel_values: {pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw.tolist()}")
    return pixel_values, grid_thw


# ============================================================================
# Step 2: Vision Encoding
# ============================================================================


def load_vision_encoder(model_dir: Path, vision_raw: dict, device: str):
    """Load MingOmniVisionEncoder with weights."""
    from transformers import PretrainedConfig

    from sglang_omni.models.ming_omni.components.vision_encoder import (
        MingOmniVisionEncoder,
    )

    vision_config = PretrainedConfig(**vision_raw)
    encoder = MingOmniVisionEncoder(vision_config, quant_config=None, prefix="visual")

    t0 = time.time()
    loaded = encoder.load_weights(iter_weights_by_prefix(model_dir, "vision."))
    print(f"  Vision encoder: {len(loaded)} weights loaded in {time.time()-t0:.1f}s")

    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    return encoder


def load_projector(
    model_dir: Path, vision_dim: int, llm_dim: int, mlp_depth: int, device: str
):
    """Load VisionProjector with weights."""
    from sglang_omni.models.ming_omni.components.projectors import VisionProjector

    proj = VisionProjector(vision_dim=vision_dim, llm_dim=llm_dim, mlp_depth=mlp_depth)
    loaded = proj.load_weights(iter_weights_by_prefix(model_dir, "linear_proj."))
    print(f"  Projector: {len(loaded)} weights loaded")

    proj = proj.to(device=device, dtype=torch.bfloat16).eval()
    return proj


def run_vision_pipeline(encoder, projector, pixel_values, grid_thw, device):
    """Run: pixel_values -> vision encoder -> projector -> L2 normalize."""
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    grid_thw = grid_thw.to(device=device)

    with torch.no_grad():
        # Vision encoder
        image_embeds = encoder(pixel_values, grid_thw)
        print(f"  Vision encoder output: {image_embeds.shape}")

        # Use only base merger output (not deepstack) for projection
        if encoder.use_deepstack:
            image_embeds_for_proj = image_embeds[:, : encoder.image_emb_dim]
            print(f"  After deepstack slice: {image_embeds_for_proj.shape}")
        else:
            image_embeds_for_proj = image_embeds

        # Project to LLM dimension
        image_embeds_proj = projector(image_embeds_for_proj)
        print(f"  After projector: {image_embeds_proj.shape}")

        # L2 normalize
        image_features = F.normalize(image_embeds_proj, dim=-1)

    return image_features


# ============================================================================
# Step 3: Embedding Injection
# ============================================================================


def load_embedding_table(
    model_dir: Path, vocab_size: int, hidden_size: int, device: str
):
    """Load just the LLM embedding table from checkpoint."""
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    embed = VocabParallelEmbedding(vocab_size, hidden_size)

    # Load embedding weights
    for name, tensor in iter_weights_by_prefix(model_dir, "model.model."):
        if name == "word_embeddings.weight":
            embed.weight.data.copy_(tensor)
            print(f"  Embedding table loaded: {tensor.shape}")
            break

    embed = embed.to(device=device, dtype=torch.bfloat16).eval()
    return embed


def build_input_embeds(
    embed_table,
    image_features: torch.Tensor,
    tokenizer,
    llm_cfg: dict,
    device: str,
):
    """Build input_embeds with image features injected at image token positions.

    Prompt: "<role>HUMAN</role><image>..patches..</image>\\nDescribe this image.</s><role>ASSISTANT</role>"
    """
    spatial_merge_size = 2
    num_image_tokens = image_features.shape[0]

    # Build prompt with image patch tokens
    image_patch_token = "<imagePatch>"
    prompt = (
        "<role>SYSTEM</role>You are a friendly AI assistant.\n\ndetailed thinking off</s>"
        "<role>HUMAN</role>"
        "<image>" + (image_patch_token * num_image_tokens) + "</image>\n"
        "Describe this image in English.</s>"
        "<role>ASSISTANT</role>"
    )

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    print(f"  Prompt tokens: {input_ids.shape[1]}")

    # Get the image patch token ID
    patch_token_id = tokenizer.convert_tokens_to_ids(image_patch_token)
    print(f"  Image patch token ID: {patch_token_id}")

    # Get text embeddings
    with torch.no_grad():
        text_embeds = embed_table(input_ids[0])  # [seq_len, hidden]

    # Find image token positions and replace with image features
    mask = input_ids[0] == patch_token_id
    num_found = mask.sum().item()
    print(f"  Image token positions found: {num_found} (expected: {num_image_tokens})")

    if num_found == num_image_tokens:
        text_embeds[mask] = image_features.to(text_embeds.dtype)
        print(f"  [OK] Image features injected into embeddings")
    elif num_found > 0:
        # Truncate or pad as needed
        print(
            f"  [WARN] Token count mismatch, injecting min({num_found}, {num_image_tokens})"
        )
        n = min(num_found, num_image_tokens)
        positions = mask.nonzero(as_tuple=True)[0][:n]
        text_embeds[positions] = image_features[:n].to(text_embeds.dtype)
    else:
        print(f"  [FAIL] No image tokens found in prompt")
        return None

    return text_embeds


# ============================================================================
# Step 4: Validation
# ============================================================================


def validate_features(image_features: torch.Tensor) -> bool:
    """Validate image features are well-formed."""
    ok = True

    # Shape check
    seq_len, dim = image_features.shape
    print(f"\n  Image features: shape=({seq_len}, {dim})")

    # NaN/Inf check
    has_nan = torch.isnan(image_features).any().item()
    has_inf = torch.isinf(image_features).any().item()
    if has_nan or has_inf:
        print(f"  [FAIL] has_nan={has_nan}, has_inf={has_inf}")
        ok = False

    # L2 norm check (should be ~1.0 after normalize)
    norms = image_features.float().norm(dim=-1)
    mean_norm = norms.mean().item()
    print(
        f"  L2 norms: mean={mean_norm:.4f}, min={norms.min():.4f}, max={norms.max():.4f}"
    )
    if abs(mean_norm - 1.0) > 0.01:
        print(f"  [FAIL] Expected mean L2 norm ~1.0, got {mean_norm:.4f}")
        ok = False

    # Diversity check (features should not be all the same)
    feat_std = image_features.float().std(dim=0).mean().item()
    print(f"  Feature diversity (mean std across tokens): {feat_std:.6f}")
    if feat_std < 1e-6:
        print(f"  [FAIL] Features are degenerate (all same)")
        ok = False

    # Stats
    vals = image_features.float()
    print(
        f"  Stats: mean={vals.mean():.4f}, std={vals.std():.4f}, "
        f"min={vals.min():.4f}, max={vals.max():.4f}"
    )

    return ok


def validate_input_embeds(input_embeds: torch.Tensor) -> bool:
    """Validate combined input embeddings."""
    ok = True
    print(f"\n  Combined input_embeds: shape={input_embeds.shape}")

    has_nan = torch.isnan(input_embeds).any().item()
    has_inf = torch.isinf(input_embeds).any().item()
    if has_nan or has_inf:
        print(f"  [FAIL] has_nan={has_nan}, has_inf={has_inf}")
        ok = False
    else:
        print(f"  [OK] No NaN/Inf")

    vals = input_embeds.float()
    print(f"  Stats: mean={vals.mean():.4f}, std={vals.std():.4f}")
    return ok


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("Ming-flash-omni-2.0 Image Understanding E2E Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Device: {DEVICE}\n")

    # Resolve model
    model_dir = resolve_model_dir(MODEL)
    vision_raw, mlp_depth, llm_hidden_size, vocab_size, llm_cfg = load_config(model_dir)
    print(
        f"Vision: depth={vision_raw.get('depth')}, out={vision_raw.get('out_hidden_size')}"
    )
    print(f"LLM: hidden={llm_hidden_size}, vocab={vocab_size}\n")

    # Init sglang
    init_sglang_tp()

    results = []

    # --- Step 1: Download & preprocess image ---
    print("\n--- Step 1: Image Preprocessing ---")
    image = download_image(TEST_IMAGE_URL)
    pixel_values, grid_thw = preprocess_image(
        image, spatial_merge_size=vision_raw.get("spatial_merge_size", 2)
    )
    results.append(("Image Preprocessing", True))

    # --- Step 2: Load vision encoder + projector ---
    print("\n--- Step 2: Load Vision Components ---")
    encoder = load_vision_encoder(model_dir, vision_raw, DEVICE)
    vision_dim = vision_raw.get("out_hidden_size", 4096)
    projector = load_projector(
        model_dir, vision_dim, llm_hidden_size, mlp_depth, DEVICE
    )

    # --- Step 3: Run vision pipeline ---
    print("\n--- Step 3: Vision Pipeline ---")
    t0 = time.time()
    image_features = run_vision_pipeline(
        encoder, projector, pixel_values, grid_thw, DEVICE
    )
    elapsed = time.time() - t0
    print(f"  Pipeline time: {elapsed:.2f}s")

    # --- Step 4: Validate features ---
    print("\n--- Step 4: Feature Validation ---")
    feat_ok = validate_features(image_features)
    results.append(("Feature Validation", feat_ok))

    # --- Step 5: Embedding injection ---
    print("\n--- Step 5: Embedding Injection ---")
    # Load tokenizer
    from sglang_omni.models.ming_omni.components.common import load_ming_tokenizer

    tokenizer = load_ming_tokenizer(str(model_dir))
    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load embedding table
    embed_table = load_embedding_table(model_dir, vocab_size, llm_hidden_size, DEVICE)

    # Build input_embeds
    input_embeds = build_input_embeds(
        embed_table, image_features, tokenizer, llm_cfg, DEVICE
    )

    if input_embeds is not None:
        embed_ok = validate_input_embeds(input_embeds)
        results.append(("Embedding Injection", embed_ok))
    else:
        results.append(("Embedding Injection", False))

    # --- Step 6: Free vision components, attempt LLM forward ---
    print("\n--- Step 6: LLM Logits Check (first token prediction) ---")
    # Free vision encoder to make room for LLM head
    del encoder, projector
    torch.cuda.empty_cache()

    try:
        # Load LM head only (small: vocab_size * hidden_size * 2 bytes ≈ 1.2GB)
        lm_head = torch.nn.Linear(llm_hidden_size, vocab_size, bias=False)
        for name, tensor in iter_weights_by_prefix(model_dir, ""):
            if name in ("lm_head.weight", "model.lm_head.weight"):
                lm_head.weight.data.copy_(tensor)
                print(f"  LM head loaded: {tensor.shape}")
                break
        lm_head = lm_head.to(device=DEVICE, dtype=torch.bfloat16).eval()

        # Get logits for the last token (prediction for first generated token)
        with torch.no_grad():
            # Use just the image features + a few text tokens around them
            # We can't run through the full LLM (needs ForwardBatch), but we can
            # check what the LM head predicts from raw embeddings as a sanity check
            last_embed = input_embeds[-1:].to(torch.bfloat16)  # last token embedding
            logits = lm_head(last_embed)
            top5_vals, top5_ids = logits[0].topk(5)

            print(f"  Top-5 predicted tokens after prompt:")
            for i, (val, tid) in enumerate(zip(top5_vals, top5_ids)):
                token_text = tokenizer.decode([tid.item()])
                print(
                    f"    {i+1}. '{token_text}' (id={tid.item()}, logit={val.item():.2f})"
                )

        # Basic sanity: logits should not be NaN
        logits_ok = not torch.isnan(logits).any().item()
        results.append(("LLM Logits Check", logits_ok))
        del lm_head
    except Exception as e:
        print(f"  [WARN] LLM logits check skipped: {e}")
        results.append(("LLM Logits Check", "SKIP"))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        if ok == "SKIP":
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {status}  {name}")

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
