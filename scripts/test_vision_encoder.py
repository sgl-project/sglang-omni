#!/usr/bin/env python3
"""Test MingOmniVisionEncoder correctness.

Tests:
  1. Model instantiation from checkpoint config
  2. Weight loading completeness (no missing weights)
  3. Weight spot-check against raw checkpoint
  4. Forward pass shape correctness
  5. Output sanity (no NaN, reasonable values)

Usage (on remote server):
    cd /sgl-workspace/sglang-omni-dev2
    CUDA_VISIBLE_DEVICES=0 python scripts/test_vision_encoder.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL = os.environ.get("MODEL_PATH", "inclusionAI/Ming-flash-omni-2.0")
DEVICE = os.environ.get("DEVICE", "cuda:0")


def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


def load_config(model_dir: Path):
    """Load vision config from checkpoint config.json."""
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        raw = json.load(f)

    # Ming-flash-omni-2.0 has thinker_config.vision_config
    thinker = raw.get("thinker_config", raw)
    vision_raw = thinker.get("vision_config", {})
    mlp_depth = thinker.get("mlp_depth", 2)
    llm_hidden_size = thinker.get("llm_config", {}).get("hidden_size", 4096)
    return vision_raw, mlp_depth, llm_hidden_size


def init_sglang_tp():
    """Initialize sglang distributed context for TP=1 standalone testing.

    Sets up: global server args, distributed process group, model parallel,
    and dp_attention module-level variables.
    """
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # 1. Global server args (needed by get_rope -> get_global_server_args)
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    dummy_args = ServerArgs(model_path="dummy")
    set_global_server_args_for_scheduler(dummy_args)

    # 2. Distributed process group
    from sglang.srt.distributed import parallel_state

    parallel_state.init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
    )
    parallel_state.initialize_model_parallel()

    # 3. dp_attention variables (needed by VisionAttention -> get_attention_tp_size)
    import sglang.srt.layers.dp_attention as dp

    dp._ATTN_TP_SIZE = 1
    dp._ATTN_TP_RANK = 0

    print("[OK] sglang TP=1 context initialized")


def get_vision_weight_keys(model_dir: Path) -> dict[str, str]:
    """Get mapping of vision weight keys to their shard files."""
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        return {}
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    # Collect keys with "vision." or "linear_proj." prefix
    # Checkpoint uses "model.vision.*" and "model.linear_proj.*"
    result = {}
    for key, shard in weight_map.items():
        if key.startswith("vision.") or key.startswith("linear_proj."):
            result[key] = shard
    return result


def load_checkpoint_weights(
    model_dir: Path, keys: list[str]
) -> dict[str, torch.Tensor]:
    """Load specific weights from safetensors checkpoint."""
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    # Group keys by shard file
    shards: dict[str, list[str]] = {}
    for key in keys:
        shard = weight_map.get(key)
        if shard:
            shards.setdefault(shard, []).append(key)

    result = {}
    for shard, shard_keys in shards.items():
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in shard_keys:
                result[key] = f.get_tensor(key)
    return result


def iter_vision_weights(model_dir: Path):
    """Iterate over vision encoder weights from checkpoint, with prefix stripped."""
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = "vision."
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    for shard, keys in sorted(shards.items()):
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                stripped = key[len(prefix) :]
                yield stripped, f.get_tensor(key)


def iter_proj_weights(model_dir: Path):
    """Iterate over vision projector weights from checkpoint, with prefix stripped."""
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = "linear_proj."
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    for shard, keys in sorted(shards.items()):
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                stripped = key[len(prefix) :]
                yield stripped, f.get_tensor(key)


# ============================================================================
# Test 1: Instantiation
# ============================================================================


def test_instantiation(vision_raw: dict):
    """Test that MingOmniVisionEncoder can be created from config."""
    from transformers import PretrainedConfig

    from sglang_omni.models.ming_omni.components.vision_encoder import (
        MingOmniVisionEncoder,
    )

    vision_config = PretrainedConfig(**vision_raw)
    encoder = MingOmniVisionEncoder(vision_config, quant_config=None, prefix="visual")

    # Verify key attributes
    assert encoder.hidden_size == vision_raw.get("hidden_size", 1152)
    assert encoder.num_heads == vision_raw.get("num_heads", 16)
    assert len(encoder.blocks) == vision_raw.get("depth", 27)
    assert encoder.use_deepstack is True
    assert len(encoder.merger_list) == len(
        vision_raw.get("deepstack_visual_indexes", [8, 16, 24])
    )

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"[OK] MingOmniVisionEncoder instantiated: {num_params / 1e6:.1f}M params")
    print(
        f"     hidden_size={encoder.hidden_size}, num_heads={encoder.num_heads}, "
        f"depth={len(encoder.blocks)}, out_hidden_size={encoder.image_emb_dim}"
    )
    print(f"     deepstack_indexes={encoder.deepstack_visual_indexes}")
    return encoder


# ============================================================================
# Test 2: Weight Loading Completeness
# ============================================================================


def test_weight_loading(encoder, model_dir: Path):
    """Test that all weights load without missing or unexpected keys."""
    # Collect all model parameter names (before loading)
    all_params = set(dict(encoder.named_parameters()).keys())
    print(f"\n     Model has {len(all_params)} parameters")

    # Load weights
    t0 = time.time()
    loaded = encoder.load_weights(iter_vision_weights(model_dir))
    elapsed = time.time() - t0

    missing = all_params - loaded
    # Some params might be handled by TP sharding or have special loaders
    # Filter out rotary_pos_emb (not in checkpoint — computed at init)
    missing = {k for k in missing if "rotary_pos_emb" not in k}

    print(
        f"[{'OK' if not missing else 'FAIL'}] Weight loading: "
        f"{len(loaded)}/{len(all_params)} loaded in {elapsed:.1f}s"
    )

    if missing:
        print(f"     MISSING weights ({len(missing)}):")
        for m in sorted(missing):
            print(f"       - {m}")
        return False
    return True


# ============================================================================
# Test 3: Weight Spot-Check
# ============================================================================


def test_weight_spotcheck(encoder, model_dir: Path):
    """Spot-check a few loaded weights against checkpoint values."""
    model_params = dict(encoder.named_parameters())

    # Define checkpoint -> model weight name mappings to check.
    # The encoder's load_weights applies _remap_ming_vision_weight internally.
    # So we check the final model param name vs the raw checkpoint value.
    spot_checks = [
        # (checkpoint_key_with_prefix, model_param_name, description)
        (
            "vision.patch_embed.proj.weight",
            "patch_embed.proj.weight",
            "patch_embed Conv3d",
        ),
        ("vision.blocks.0.norm1.weight", "blocks.0.norm1.weight", "block0 norm1"),
        ("vision.blocks.26.norm2.weight", "blocks.26.norm2.weight", "block26 norm2"),
        ("vision.merger.norm.weight", "merger.ln_q.weight", "merger ln_q (remapped)"),
    ]

    ckpt_keys = [s[0] for s in spot_checks]
    ckpt_data = load_checkpoint_weights(model_dir, ckpt_keys)

    passed = 0
    for ckpt_key, model_key, desc in spot_checks:
        if ckpt_key not in ckpt_data:
            print(f"  SKIP {desc}: {ckpt_key} not in checkpoint")
            continue
        if model_key not in model_params:
            print(f"  FAIL {desc}: {model_key} not in model")
            continue

        ckpt_w = ckpt_data[ckpt_key].to("cpu").float()
        model_w = model_params[model_key].data.to("cpu").float()

        if ckpt_w.shape != model_w.shape:
            print(
                f"  FAIL {desc}: shape mismatch ckpt={ckpt_w.shape} model={model_w.shape}"
            )
            continue

        max_diff = (ckpt_w - model_w).abs().max().item()
        status = "OK  " if max_diff < 1e-5 else "FAIL"
        print(f"  {status} {desc}: max_diff={max_diff:.2e}")
        if max_diff < 1e-5:
            passed += 1

    total = len([s for s in spot_checks if s[0] in ckpt_data])
    print(
        f"[{'OK' if passed == total else 'FAIL'}] Spot-check: {passed}/{total} passed"
    )
    return passed == total


# ============================================================================
# Test 4: Forward Pass
# ============================================================================


def test_forward_pass(encoder, device: str):
    """Test forward pass with dummy input."""
    encoder = encoder.to(device=device, dtype=torch.bfloat16)
    encoder.eval()

    # Simulate a single 224x224 image
    # grid_thw: [1, 14, 14] — 1 temporal frame, 14x14 spatial grid
    # total_patches = 1 * 14 * 14 = 196
    # Each patch: C * temporal_patch_size * patch_size * patch_size = 3*2*16*16 = 1536
    t, h, w = 1, 14, 14
    total_patches = t * h * w
    in_channels = 3
    temporal_patch_size = encoder.temporal_patch_size
    patch_size = encoder.patch_size
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

    pixel_values = torch.randn(
        total_patches, patch_dim, dtype=torch.bfloat16, device=device
    )
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.int64, device=device)

    with torch.no_grad():
        output = encoder(pixel_values, grid_thw)

    # Expected output shape after spatial merge:
    # seq_len = total_patches / (spatial_merge_size^2) = 196 / 4 = 49
    merge_size = encoder.spatial_merge_size
    expected_seq_len = total_patches // (merge_size**2)

    # With deepstack: output dim = out_hidden_size * (1 + num_deepstack)
    if encoder.use_deepstack:
        expected_dim = encoder.out_hidden_size
    else:
        expected_dim = encoder.image_emb_dim

    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    all_zero = (output == 0).all().item()

    shape_ok = output.shape == (expected_seq_len, expected_dim)
    sanity_ok = not has_nan and not has_inf and not all_zero

    print(
        f"\n  output shape: {output.shape} (expected: ({expected_seq_len}, {expected_dim}))"
    )
    print(f"  has_nan={has_nan}, has_inf={has_inf}, all_zero={all_zero}")
    print(
        f"  output stats: mean={output.float().mean():.4f}, std={output.float().std():.4f}, "
        f"min={output.float().min():.4f}, max={output.float().max():.4f}"
    )
    print(f"[{'OK' if shape_ok else 'FAIL'}] Forward pass shape")
    print(f"[{'OK' if sanity_ok else 'FAIL'}] Forward pass sanity")
    return shape_ok and sanity_ok


# ============================================================================
# Test 5: Vision Projector
# ============================================================================


def test_projector(
    model_dir: Path, vision_raw: dict, mlp_depth: int, llm_hidden_size: int, device: str
):
    """Test VisionProjector instantiation, weight loading, and forward pass."""
    from sglang_omni.models.ming_omni.components.projectors import VisionProjector

    vision_dim = vision_raw.get("out_hidden_size", 3584)
    proj = VisionProjector(
        vision_dim=vision_dim, llm_dim=llm_hidden_size, mlp_depth=mlp_depth
    )

    num_params = sum(p.numel() for p in proj.parameters())
    print(
        f"\n  VisionProjector: {num_params / 1e6:.1f}M params "
        f"({vision_dim} -> {llm_hidden_size}, depth={mlp_depth})"
    )

    # Load weights
    loaded = proj.load_weights(iter_proj_weights(model_dir))
    all_params = set(dict(proj.named_parameters()).keys())
    missing = all_params - loaded

    print(f"  Loaded {len(loaded)}/{len(all_params)} projector weights")
    if missing:
        print(f"  MISSING: {missing}")

    # Forward pass
    proj = proj.to(device=device, dtype=torch.bfloat16)
    proj.eval()
    dummy = torch.randn(49, vision_dim, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = proj(dummy)

    shape_ok = out.shape == (49, llm_hidden_size)
    sanity_ok = not torch.isnan(out).any().item()

    print(f"  output shape: {out.shape} (expected: (49, {llm_hidden_size}))")
    print(
        f"[{'OK' if shape_ok and sanity_ok and not missing else 'FAIL'}] VisionProjector"
    )
    return shape_ok and sanity_ok and not missing


# ============================================================================
# Main
# ============================================================================


def main():
    print(f"=== MingOmniVisionEncoder Test ===")
    print(f"Model: {MODEL}")
    print(f"Device: {DEVICE}\n")

    # Resolve model directory
    print("Resolving model directory...")
    model_dir = resolve_model_dir(MODEL)
    print(f"  {model_dir}\n")

    # Load config
    vision_raw, mlp_depth, llm_hidden_size = load_config(model_dir)
    print(
        f"Vision config: depth={vision_raw.get('depth')}, "
        f"hidden={vision_raw.get('hidden_size')}, "
        f"out={vision_raw.get('out_hidden_size')}"
    )
    print(f"Projector: mlp_depth={mlp_depth}, llm_hidden={llm_hidden_size}\n")

    # Init sglang TP
    init_sglang_tp()

    results = []

    # Test 1: Instantiation
    print("\n--- Test 1: Instantiation ---")
    encoder = test_instantiation(vision_raw)
    results.append(("Instantiation", True))

    # Test 2: Weight Loading
    print("\n--- Test 2: Weight Loading ---")
    ok = test_weight_loading(encoder, model_dir)
    results.append(("Weight Loading", ok))

    # Test 3: Spot-Check
    print("\n--- Test 3: Weight Spot-Check ---")
    ok = test_weight_spotcheck(encoder, model_dir)
    results.append(("Weight Spot-Check", ok))

    # Test 4: Forward Pass
    print("\n--- Test 4: Forward Pass ---")
    ok = test_forward_pass(encoder, DEVICE)
    results.append(("Forward Pass", ok))

    # Test 5: Projector
    print("\n--- Test 5: Vision Projector ---")
    ok = test_projector(model_dir, vision_raw, mlp_depth, llm_hidden_size, DEVICE)
    results.append(("Vision Projector", ok))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
