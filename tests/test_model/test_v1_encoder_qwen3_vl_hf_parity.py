# SPDX-License-Identifier: Apache-2.0
"""GPU parity test: sglang ``Qwen3OmniMoeVisionEncoder`` (post-patch) vs HF
transformers reference, on the real Qwen3-Omni-30B-A3B-Instruct checkpoint.

Validates that ``apply_qwen3_vl_hf_parity_patches()`` restores HF-equivalent
output up to bf16 noise across a 27-layer encoder forward.

Acceptance:
    cosine sim ≥ 0.999, max abs ≤ 0.2 (bf16 27-layer accumulation noise)

Without the patches, cosine sim is ~0.67 — see issue
https://github.com/sgl-project/sglang-omni/issues/434.

Run:
    pytest -v -m benchmark tests/test_model/test_v1_encoder_qwen3_vl_hf_parity.py

Requires:
    - 1× H100 (or equivalent ≥ 80GB GPU) with sglang==0.5.8 and transformers
    - HF cache populated for ``Qwen/Qwen3-Omni-30B-A3B-Instruct``
"""
from __future__ import annotations

import json

import pytest
import torch

from tests.test_model.conftest import QWEN3_OMNI_TEST_MODEL_PATH as MODEL_PATH
from tests.test_model.conftest import resolve_qwen3_omni_model_dir


@pytest.fixture(scope="module")
def dist_init(cuda_device, qwen3_omni_vision_sglang_env):
    """Module-level alias of the session-scoped SGLang vision-encoder env.

    Existed as a per-module fixture before the bring-up was hoisted to
    ``tests/test_model/conftest.py``; kept so the test signature is
    unchanged.
    """


def _load_visual_safetensors():
    """Read all ``thinker.visual.*`` weights from the model checkpoint."""
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


@pytest.mark.benchmark
def test_qwen3_omni_vision_encoder_hf_parity(cuda_device, dist_init):
    """Patched sglang Qwen3-Omni vision encoder must match HF within bf16 noise."""
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeVisionEncoder as SGVE
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder as HFVE,
    )

    from sglang_omni.model_runner._sglang_qwen3_vl_patches import (
        apply_qwen3_vl_hf_parity_patches,
    )

    dtype = torch.bfloat16
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vision_cfg = cfg.thinker_config.vision_config

    # Build both encoders on cuda:0 in bf16
    with torch.device("cuda:0"):
        hf_v = HFVE(vision_cfg).to(dtype=dtype)
    with set_default_torch_dtype(dtype):
        with torch.device("cuda:0"):
            sg_v = SGVE(vision_cfg)

    # Load identical weights into both. SGLang uses fused QKV naming
    # (``attn.qkv_proj.*``); HF uses ``attn.qkv.*``. One-line rename.
    raw = _load_visual_safetensors()
    hf_miss, hf_unexp = hf_v.load_state_dict(raw, strict=False)
    assert (
        not hf_miss and not hf_unexp
    ), f"HF load: missing={len(hf_miss)} unexpected={len(hf_unexp)}"

    sg_input = {
        (k.replace(".attn.qkv.", ".attn.qkv_proj.") if ".attn.qkv." in k else k): v
        for k, v in raw.items()
    }
    sg_miss, sg_unexp = sg_v.load_state_dict(sg_input, strict=True)
    assert (
        not sg_miss and not sg_unexp
    ), f"SG load: missing={len(sg_miss)} unexpected={len(sg_unexp)}"

    hf_v.eval()
    sg_v.eval()

    # Apply HF-parity patches before forward.
    apply_qwen3_vl_hf_parity_patches()

    # Fixed-seed input: 32 patches arranged as t=2, h=4, w=4
    T_g, H_g, W_g = 2, 4, 4
    num_patches = T_g * H_g * W_g
    in_d = (
        vision_cfg.in_channels
        * vision_cfg.temporal_patch_size
        * vision_cfg.patch_size**2
    )
    torch.manual_seed(42)
    pixel_values = torch.randn(num_patches, in_d, dtype=dtype, device="cuda:0")
    grid_thw = torch.tensor([[T_g, H_g, W_g]], dtype=torch.long, device="cuda:0")

    with torch.no_grad():
        hf_out = hf_v(pixel_values.clone(), grid_thw.clone())
        # SGLang requires grid_thw on CPU (compute_cu_seqlens_from_grid_numpy assertion).
        sg_out = sg_v(pixel_values.clone(), grid_thw.cpu().clone())

    # HF returns (main, deepstack_list); SGLang returns a single tensor with
    # main + deepstack concatenated along the last dim.
    hf_main, hf_deepstack = hf_out[0], hf_out[1]
    n_deep = len(hf_deepstack)
    H = vision_cfg.out_hidden_size

    assert sg_out.shape[-1] == H * (1 + n_deep), (
        f"SGLang output last-dim {sg_out.shape[-1]} does not match expected "
        f"{H} × (1 + {n_deep} deepstack)"
    )
    sg_main = sg_out[..., :H]
    sg_deepstack = [sg_out[..., H * (i + 1) : H * (i + 2)] for i in range(n_deep)]

    # Tolerance derived from bf16 27-layer accumulation noise floor on H100
    # (measured via patched-vs-HF parity audit, see issue #434).
    COS_GATE = 0.999
    MAX_ABS_GATE = 0.2

    def _check(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
        assert (
            a.shape == b.shape
        ), f"{label}: shape diff HF={tuple(a.shape)} SG={tuple(b.shape)}"
        diff = (a.float() - b.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
        max_abs = diff.max().item()
        print(
            f"{label}: shape={tuple(a.shape)} max_abs={max_abs:.4e} "
            f"mean_abs={diff.mean().item():.4e} cos={cos:.6f}"
        )
        assert cos >= COS_GATE, (
            f"{label}: cosine sim {cos:.6f} below gate {COS_GATE} — "
            f"patches degraded HF parity. See issue #434."
        )
        assert max_abs <= MAX_ABS_GATE, (
            f"{label}: max abs diff {max_abs:.4e} above gate {MAX_ABS_GATE} — "
            f"bf16 noise exceeded; investigate before relaxing gate."
        )

    _check("MAIN", hf_main, sg_main)
    for i in range(n_deep):
        _check(f"DEEPSTACK[{i}]", hf_deepstack[i], sg_deepstack[i])
