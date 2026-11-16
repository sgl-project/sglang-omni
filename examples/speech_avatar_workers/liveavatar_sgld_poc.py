# SPDX-License-Identifier: Apache-2.0
"""PoC: Live Avatar's streaming DiT step (liveavatar_fast.StreamStep) on SGLang Diffusion's kernels, against torch's.

Runs in a venv with ``sglang[diffusion]`` (it pins its own torch), on one GPU:

    PYTHONPATH=LiveAvatar python liveavatar_sgld_poc.py --ckpt-dir ckpt/Wan2.2-S2V-14B/ [--blocks 8] [--no-compile]

The same model (Wan2.2-S2V-14B DiT, fp8 linears as the worker has them, no
LoRA: the kernels do not care) renders the same blocks from the same inputs
(random latents, speech features and prompt embedding, a sink prefilled from
random reference latents) twice:

* ``torch``: cuDNN attention, the model's own RMSNorm (what the worker runs);
* ``sglang``: SGLang Diffusion's ``LocalAttention`` (its backend selector; FA3
  from sgl-kernel on Hopper) for self-attention, and its ``RMSNorm`` (JIT
  kernel) for the q/k norms of self- and cross-attention.

Prints the steady GPU ms of a block (4 steps), the attention kernel alone at
the block's shape, and the relative L2 difference of each block's velocity.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MOTION_FRAMES = 73
LAT_MOTION_FRAMES = 19
LFB = 3
H, W = 88, 48  # latent of the worker's 704x384 picture


def import_dit_only() -> None:
    """Load Live Avatar's DiT modules without the package __init__s, which pull in T5, wav2vec2 and the pipelines (and transformers 4 APIs this venv's transformers 5 lacks)."""
    import importlib
    import types

    import importlib.util

    root = list(importlib.util.find_spec("liveavatar").submodule_search_locations)[0]
    for name in ("liveavatar.models", "liveavatar.models.wan", "liveavatar.models.wan.wan_2_2", "liveavatar.models.wan.wan_2_2.modules", "liveavatar.models.wan.wan_2_2.modules.s2v", "liveavatar.models.wan.wan_2_2.distributed"):
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(root, *name.split(".")[1:])]
        sys.modules[name] = module


def init_sglang():
    import torch

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    try:
        from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

        set_global_server_args(ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", attention_backend="fa", num_gpus=1))
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"server_args": repr(error)[:300]}), flush=True)
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.cuda.set_device(0)


def sglang_attention(heads: int, head_dim: int):
    import torch
    from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    layer = LocalAttention(
        num_heads=heads,
        head_size=head_dim,
        compute_dtype=torch.bfloat16,
        supported_attention_backends={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
    )

    def attend(q, k, v):
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return layer(q, k, v)

    return attend, str(layer.backend)


def swap_norms(model):
    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

    count = 0
    for block in model.blocks:
        for attn in (block.self_attn, block.cross_attn):
            for name in ("norm_q", "norm_k"):
                old = getattr(attn, name)
                new = RMSNorm(old.weight.numel(), eps=old.eps, cast_x_before_out_mul=True).to(old.weight.device)
                new.weight.data.copy_(old.weight.data)
                setattr(attn, name, new)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-fp8", action="store_true")
    args = parser.parse_args()
    import torch

    init_sglang()
    import_dit_only()
    import liveavatar_fast
    from liveavatar.models.wan import causal_model_s2v
    from liveavatar.models.wan.causal_model_s2v import CausalWanModel_S2V
    from liveavatar.models.wan.wan_2_2.modules import model as wan_model

    # The sink prefill (the authors' code) calls flash-attn 2/3's varlen API, which this venv's flash-attn 4 lacks; its lengths are always whole.
    def flash_attention(q, k, v, k_lens=None, **kwargs):
        return liveavatar_fast.sdpa(q, k, v).type_as(q)

    wan_model.flash_attention = flash_attention
    causal_model_s2v.flash_attention = flash_attention

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    report = {"torch": torch.__version__}
    try:
        import sglang

        report["sglang"] = sglang.__version__
    except Exception:  # noqa: BLE001
        pass
    model = CausalWanModel_S2V.from_pretrained(args.ckpt_dir, torch_dtype=dtype).to(device).eval().requires_grad_(False)
    model.num_frame_per_block = LFB
    if not args.no_fp8:
        from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8

        replace_linear_with_scaled_fp8(
            model,
            ignore_keys=["text_embedding", "time_embedding", "time_projection", "head.head", "casual_audio_encoder.encoder.final_linear"],
        )
    g = torch.Generator(device=device).manual_seed(0)
    fsl = H * W // 4
    ref = torch.randn(1, 16, 1, H, W, generator=g, device=device, dtype=dtype)
    motion = torch.randn(1, 16, LAT_MOTION_FRAMES, H, W, generator=g, device=device, dtype=dtype)
    context = [torch.randn(20, 4096, generator=g, device=device, dtype=dtype) * 0.1]
    cond_states = torch.zeros(1, 16, LFB, H, W, device=device, dtype=dtype)
    noise = [torch.randn(16, LFB, H, W, generator=g, device=device, dtype=dtype) for _ in range(args.blocks)]
    audio = [torch.randn(1, 25, 1024, 4 * LFB, generator=g, device=device, dtype=dtype) for _ in range(args.blocks)]
    t = torch.tensor([[999.0] * LFB], device=device)

    def new_cache():
        cache = []
        for _ in range(model.num_layers):
            cache.append(
                {
                    "k": torch.zeros([1, 12 * fsl, 40, 128], dtype=dtype, device=device),
                    "v": torch.zeros([1, 12 * fsl, 40, 128], dtype=dtype, device=device),
                    "cond_k": torch.zeros([1, 3000, 40, 128], dtype=dtype, device=device),
                    "cond_v": torch.zeros([1, 3000, 40, 128], dtype=dtype, device=device),
                    "cond_end": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        return cache

    crossattn = [{"k": torch.zeros([1, 0, 40, 128], dtype=dtype, device=device), "v": torch.zeros([1, 0, 40, 128], dtype=dtype, device=device), "is_init": False} for _ in range(model.num_layers)]

    def run(label: str, step: "liveavatar_fast.StreamStep"):
        cache = new_cache()
        outs, times = [], []
        with torch.amp.autocast("cuda", dtype=dtype), torch.no_grad():
            model(
                [noise[0]], t=torch.zeros([1, LFB], device=device, dtype=dtype), context=context, seq_len=None,
                cond_states=cond_states, motion_latents=motion, ref_latents=ref, audio_input=audio[0],
                motion_frames=[MOTION_FRAMES, LAT_MOTION_FRAMES], drop_motion_frames=False, sink_flag=True,
                kv_cache=cache, crossattn_cache=crossattn, current_start=0, current_end=LFB * fsl,
            )
            for block in range(args.blocks):
                latents = noise[block]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for s in range(4):
                    random.seed(block * 64 + s)
                    v = step(latents, t, audio[block], cond_states, context, cache, block * LFB * fsl)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - begin) * 1000)
                outs.append(v.float().cpu())
        steady = sorted(times[3:])
        report[label] = {"block_ms_p50": round(steady[len(steady) // 2], 1), "block_ms": [round(x, 1) for x in times]}
        return outs

    def make_step():
        return liveavatar_fast.StreamStep(model, None, LFB, fsl, H // 2, W // 2, LAT_MOTION_FRAMES, MOTION_FRAMES, compile=not args.no_compile)

    reference = run("torch_cudnn", make_step())
    step = make_step()
    attend, backend = sglang_attention(40, 128)
    report["sglang_attention_backend"] = backend
    step.attend = attend
    sgl_attn = run("sglang_attention", step)
    report["norms_swapped"] = swap_norms(model)
    step = make_step()
    step.attend = attend
    sgl_all = run("sglang_attention_rmsnorm", step)
    for label, outs in (("sglang_attention", sgl_attn), ("sglang_attention_rmsnorm", sgl_all)):
        report[label]["rel_l2_vs_torch"] = [round(float((o - r).norm() / r.norm()), 5) for o, r in zip(outs, reference)]

    # The attention kernel alone at the block's shape: 3168 queries over 12 frames of cache plus the sink.
    q = torch.randn(1, LFB * fsl, 40, 128, device=device, dtype=dtype)
    k = torch.randn(1, 12 * fsl + 2640, 40, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    kernels = {"cudnn": liveavatar_fast.cudnn, "torch_sdpa_default": liveavatar_fast.sdpa, "sglang_" + backend: attend}
    base = liveavatar_fast.cudnn(q, k, v).float()
    for name, fn in kernels.items():
        for _ in range(3):
            fn(q, k, v)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(20):
            out = fn(q, k, v)
        torch.cuda.synchronize()
        report.setdefault("attention_kernel_ms", {})[name] = round((time.perf_counter() - begin) * 1000 / 20, 3)
        report.setdefault("attention_kernel_rel_l2_vs_cudnn", {})[name] = round(float((out.float() - base).norm() / base.norm()), 5)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
