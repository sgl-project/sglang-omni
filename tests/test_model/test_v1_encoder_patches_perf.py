# SPDX-License-Identifier: Apache-2.0
"""Performance regression test for the HF-parity patches.

Issue https://github.com/sgl-project/sglang-omni/issues/434 sets a 1% target
for ``(time(fast_pos_embed_interpolate) + time(rot_pos_emb)) / total encoder
forward`` so the patches stay invisible to long-video / batched throughput.
Reference H100 measurements with the current patch are 100×–1000× under that
target (0.0007%–0.034%), so this test asserts only an anti-pathological
ceiling (5%) — looser than the target on purpose, because the prep time is
mostly CPU-bound while the encoder forward time scales with GPU SKU. Trend
tracking against the 1% target lives in CI dashboards, fed from the
per-scenario PERF_JSON artifacts emitted below.

Run::

    pytest -v -m benchmark tests/test_model/test_v1_encoder_patches_perf.py

Requires CUDA and the ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` checkpoint in the
local HF cache (skips if absent — never auto-downloads).
"""
from __future__ import annotations

import json

import pytest
import torch

from tests.test_model.conftest import QWEN3_OMNI_TEST_MODEL_PATH as MODEL_PATH
from tests.test_model.conftest import resolve_qwen3_omni_model_dir

# (label, t, h, w, n_grids, n_iters)
SCENARIOS = [
    ("S1_image", 1, 16, 16, 1, 30),
    ("S2_short_video", 5, 16, 16, 1, 20),
    ("S3_long_video", 60, 16, 16, 1, 5),
    ("S4_batch32_short_video", 5, 16, 16, 32, 5),
]

# Anti-pathological ceiling. The 1% target from issue #434 was set against
# H100 numbers where actual prep_ratio is 100x-1000x lower (0.0007%-0.034%).
# Across GPU SKUs the encoder forward time changes much faster than the
# (mostly CPU-bound) prep time, so a tighter assertion would be brittle.
# This 5% gate only fires on a real regression; tighter trend tracking
# happens via the per-scenario PERF_JSON CI artifacts instead.
PREP_RATIO_GATE = 0.05


@pytest.fixture(scope="module")
def sg_visual(cuda_device, qwen3_omni_vision_sglang_env):
    """One patched SGLang Qwen3-Omni vision encoder loaded with real weights;
    shared across all scenario benchmarks. SGLang dist + DP-attention bring-up
    lives in the session-scoped ``qwen3_omni_vision_sglang_env`` fixture in
    ``tests/test_model/conftest.py`` so the combined benchmark command does
    not re-initialize the process-global TP group.
    """
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeVisionEncoder as SGVE
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vision_cfg = cfg.thinker_config.vision_config

    dtype = torch.bfloat16
    with set_default_torch_dtype(dtype):
        with torch.device("cuda:0"):
            sg_v = SGVE(vision_cfg)

    raw = _load_visual_safetensors()
    sg_input = {
        (k.replace(".attn.qkv.", ".attn.qkv_proj.") if ".attn.qkv." in k else k): v
        for k, v in raw.items()
    }
    sg_v.load_state_dict(sg_input, strict=True)
    sg_v.eval()

    # Apply the patches once; stays applied for every scenario.
    from sglang_omni.model_runner._sglang_qwen3_vl_patches import (
        apply_qwen3_vl_hf_parity_patches,
    )

    apply_qwen3_vl_hf_parity_patches()

    return sg_v, vision_cfg, dtype


def _load_visual_safetensors():
    from safetensors import safe_open

    md = resolve_qwen3_omni_model_dir(MODEL_PATH)
    weight_map = json.loads((md / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    keys = [k for k in weight_map if k.startswith("thinker.visual.")]
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    sd: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(str(md / shard), framework="pt", device="cpu") as f:
            for k in ks:
                sd[k[len("thinker.visual.") :]] = f.get_tensor(k)
    return sd


def _time_fn(fn, n_warmup: int, n_iters: int) -> float:
    """Time a callable in ms (CUDA-synchronized)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def _make_input(t_g: int, h_g: int, w_g: int, n_grids: int, vision_cfg, dtype):
    in_d = (
        vision_cfg.in_channels
        * vision_cfg.temporal_patch_size
        * vision_cfg.patch_size**2
    )
    grids = [[t_g, h_g, w_g]] * n_grids
    num_p = sum(t * h * w for t, h, w in grids)
    torch.manual_seed(42)
    pv = torch.randn(num_p, in_d, dtype=dtype, device="cuda:0")
    gw = torch.tensor(grids, dtype=torch.long, device="cuda:0")
    return pv, gw


@pytest.mark.benchmark
@pytest.mark.parametrize("label,t_g,h_g,w_g,n_grids,n_iters", SCENARIOS)
def test_patch_overhead_below_pathological_ceiling(
    sg_visual, label, t_g, h_g, w_g, n_grids, n_iters
):
    """Each scenario: ``(fpei + rope) / total ≤ PREP_RATIO_GATE``."""
    sg_v, vision_cfg, dtype = sg_visual

    pv, gw = _make_input(t_g, h_g, w_g, n_grids, vision_cfg, dtype)
    gw_cpu = gw.cpu()
    gw_list = gw_cpu.tolist()

    n_warmup = 3
    with torch.no_grad():
        t_fpei = _time_fn(
            lambda: sg_v.fast_pos_embed_interpolate(gw_cpu),
            n_warmup=n_warmup,
            n_iters=n_iters,
        )
        t_rope = _time_fn(
            lambda: sg_v.rot_pos_emb(gw_list), n_warmup=n_warmup, n_iters=n_iters
        )
        t_total = _time_fn(lambda: sg_v(pv, gw_cpu), n_warmup=n_warmup, n_iters=n_iters)

    prep_ratio = (t_fpei + t_rope) / t_total
    print(
        f"\n[{label}] fpei={t_fpei:.4f}ms rope={t_rope:.4f}ms "
        f"total={t_total:.2f}ms prep_ratio={prep_ratio*100:.4f}%"
    )

    # Emit JSON line for CI artifact aggregation.
    artifact = {
        "scenario": label,
        "t_g": t_g,
        "h_g": h_g,
        "w_g": w_g,
        "n_grids": n_grids,
        "fpei_ms": t_fpei,
        "rope_ms": t_rope,
        "total_ms": t_total,
        "prep_ratio": prep_ratio,
    }
    print(f"PERF_JSON: {json.dumps(artifact)}")

    assert prep_ratio <= PREP_RATIO_GATE, (
        f"[{label}] interpolation prep ratio {prep_ratio*100:.4f}% exceeds "
        f"the {PREP_RATIO_GATE*100:.1f}% gate. See issue #434 for context. "
        f"Remediation order: (1) cache idx_tensor/weight_tensor by grid shape; "
        f"(2) vectorize the per-grid loop while preserving accumulation order "
        f"and dtype semantics."
    )
