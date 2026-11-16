# SPDX-License-Identifier: Apache-2.0
"""Chunk worker of Avatar Forcing (KlingAIResearch/AvatarForcing; README Apache-2.0, LICENSE.txt Tencent RollingForcing academic terms): a talking head at 25 fps, 480x832, sixteen frames per chunk.

Runs in the model's own environment from its checkout:

  cd AvatarForcing && python /path/to/avatarforcing_worker.py --port 8121 --avatar ref.png

The release denoises a window of four 4-latent blocks at noise levels
(200, 400, 800, 1000) and slides it one block per step: a block leaves the
window clean after four steps, so its 16 frames need the audio of the three
blocks after it. A Session here runs the release's window loop one step per
chunk:

- the first chunk primes the cache with the portrait (the release's
  `_prefill_cache_for_rolling`) and runs the three partial windows;
- each chunk runs one full window (one 16-latent and one 4-latent forward)
  and returns the block that left it, decoded with the VAE's causal cache
  kept per Session (the release decodes the whole latent video at the end;
  the cached decode is the same computation one latent at a time);
- each chunk's audio features are wav2vec2's 14 hidden states over a 5 s
  window ending `context` past the newest block (the release runs it once
  over the whole clip on CPU), on the GPU;
- the prompt is encoded once at load and T5 leaves the GPU;
- LoRA weights are merged into the base layers (`--merge-lora`), and
  flash-attn 3 is used when installed (the release picks it on its own);
- a Session restarts from the portrait before the release's RoPE table
  (1024 latents, 164 s) runs out.

Speed options (H100, all off by default):

- `--fuse` compiles the blocks' element-wise work (RoPE from shared cos/sin
  tables, norms with their modulation, gated residuals, the q/k/v projections)
  and keeps the KV-cache offsets on the host instead of reading them back from
  the GPU in every block; `--cuda-graph` then replays the steady-state window
  and cache forwards from CUDA graphs captured at warm-up (one Session at a
  time; not with `--attention sage`, whose sm90 kernel replays as noise);
- `--vae-device cuda:1` decodes on a second GPU, one chunk behind
  (`--vae-mode behind`, the look-ahead grows by a chunk) or alongside the
  cache forward of the same chunk (`overlap`); `--vae-compile` compiles the Wan
  VAE's causal convolutions (same pixels); `--taehv taew2_1.pth` decodes with
  TAEHV's small Wan 2.1 decoder instead (a little softer, ~12x faster);
- `--jpeg-threads` encodes a chunk's frames in parallel;
- quality trades: `--denoise-steps` (fewer levels: a smaller window and a
  chunk less of look-ahead), `--skip-cache-update` (no clean re-run of the
  finished block), `--fp8` (FP8 linears).
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_server import (  # noqa: E402
    SAMPLE_RATE_HZ,
    SessionModel,
    base_parser,
    logger,
    serve,
    setup_logging,
    warm_up,
)

FPS = 25
FRAME_SAMPLES = SAMPLE_RATE_HZ // FPS
LATENTS_PER_BLOCK = 4
FRAMES_PER_BLOCK = 16
CHUNK_SAMPLES = FRAMES_PER_BLOCK * FRAME_SAMPLES
FEATURE_WINDOW_FRAMES = 125
MAX_LATENTS = 1000


class AvatarForcing(SessionModel):
    max_sessions = 2

    def __init__(self, args) -> None:
        super().__init__()
        os.chdir(args.repo)
        sys.path.insert(0, str(Path(args.repo).resolve()))
        import torch
        from omegaconf import OmegaConf
        from PIL import Image
        from pipeline import AvatarForcingInferencePipeline
        from torchvision import transforms
        from transformers import Wav2Vec2FeatureExtractor
        from utils.inject import _apply_lora, slice_conditional_dict
        from wan.models.wav2vec import Wav2VecModel

        self.torch = torch
        self.slice = slice_conditional_dict
        self.fit_flash_attn_3()
        if args.attention == "cudnn":
            self.use_cudnn_attention()
        elif args.attention == "sage":
            self.use_sage_attention()
        if args.rope32:
            self.use_rope32()
        self.args = args
        torch.set_grad_enabled(False)
        torch.manual_seed(args.seed)
        config = OmegaConf.merge(
            OmegaConf.load("configs/default_config.yaml"),
            OmegaConf.load(args.config_path),
        )
        if args.denoise_steps:
            config.denoise_steps = [int(t) for t in args.denoise_steps.split(",")]
        # The window holds one block per noise level: a block leaves it after len(steps) chunks.
        self.window_blocks = len(config.denoise_steps)
        logger.info(
            "denoise steps %s (window of %d blocks)",
            list(config.denoise_steps),
            self.window_blocks,
        )
        pipeline = AvatarForcingInferencePipeline(config, device="cuda")
        state = torch.load(args.checkpoint_path, map_location="cpu")["generator"]
        pipeline.generator.model = _apply_lora(
            pipeline.generator.model, config["models"]["lora"]
        )
        pipeline.generator.load_state_dict(state)
        pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
        if args.merge_lora:
            merged = 0
            for module in pipeline.generator.modules():
                if hasattr(module, "merge") and hasattr(module, "lora_A"):
                    module.merge()
                    merged += 1
            logger.info("merged %d LoRA layers", merged)
        self.pipeline = pipeline
        if args.fp8:
            self.use_fp8_linears()
        self.fused = False
        if args.fuse:
            self.use_fused_blocks()
        if args.cuda_graph:
            if not args.fuse:
                raise ValueError("--cuda-graph needs --fuse")
            if args.attention == "sage":
                # SageAttention 2.2.0's sm90 kernel is not captured: replays leave its
                # output unwritten and the pictures turn to noise.
                raise ValueError("--cuda-graph does not work with --attention sage")
            # One set of caches, captured into the graphs at warm-up (see `start`).
            self.max_sessions = 1
        if args.compile_blocks:
            import torch._dynamo

            torch._dynamo.config.cache_size_limit = 64
            for block in pipeline.generator.model.blocks:
                block.ffn = torch.compile(block.ffn, dynamic=False)
                if hasattr(block, "cross_attn"):
                    block.cross_attn = torch.compile(block.cross_attn, dynamic=False)
            logger.info("compiled the blocks' ffn and cross attention")
        # A second VAE decodes on its own GPU so the two overlap: one chunk behind the DiT
        # (`behind`, the look-ahead grows by a chunk), or within the chunk while the DiT
        # writes the finished block into its cache (`overlap`, no extra look-ahead).
        import concurrent.futures

        self.vae_device = args.vae_device
        self.vae_mode = args.vae_mode if self.vae_device else "inline"
        if self.vae_device:
            from utils.wan_wrapper import WanVAEWrapper

            self.decoder = WanVAEWrapper().to(
                device=self.vae_device, dtype=torch.bfloat16
            )
            self.decode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.taehv = (
            self.load_taehv(args.taehv, self.vae_device or "cuda")
            if args.taehv
            else None
        )
        if args.vae_compile:
            import torch._dynamo

            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 256
            )
            compiled = 0
            for module in self.decoding_vae.model.decoder.modules():
                # The causal 3D convolutions with their cache concatenation and padding.
                if type(module).__name__ == "CausalConv3d":
                    module.forward = torch.compile(module.forward, dynamic=False)
                    compiled += 1
            logger.info("compiled %d VAE decoder convolutions", compiled)
        if args.vae_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        self.jpeg_pool = (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.jpeg_threads)
            if args.jpeg_threads > 1
            else None
        )
        self.args.context_noise = getattr(config, "context_noise", 0)
        self.context_noise = config.get("context_noise", 0)
        self.prompt = pipeline.text_encoder(text_prompts=[args.prompt])
        pipeline.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "wan_models/wav2vec2-base-960h"
        )
        self.wav2vec = (
            Wav2VecModel.from_pretrained(
                "wan_models/wav2vec2-base-960h",
                local_files_only=True,
                attn_implementation="eager",
            )
            .to("cuda")
            .eval()
        )
        self.width, self.height = args.width, args.height
        image = Image.open(args.avatar).convert("RGB")
        scale = max(self.width / image.width, self.height / image.height)
        image = image.resize(
            (math.ceil(image.width * scale), math.ceil(image.height * scale)),
            Image.BICUBIC,
        )
        left, top = (image.width - self.width) // 2, (image.height - self.height) // 2
        image = image.crop((left, top, left + self.width, top + self.height))
        pixels = transforms.Normalize([0.5] * 3, [0.5] * 3)(
            transforms.ToTensor()(image)
        )
        pixels = pixels.unsqueeze(0).unsqueeze(2).to("cuda", torch.bfloat16)
        self.initial_latent = pipeline.vae.encode_to_latent(pixels).to(
            "cuda", torch.bfloat16
        )
        img_lat = self.initial_latent.permute(0, 2, 1, 3, 4)
        mask = torch.ones_like(img_lat[:, :1]).repeat(
            1, 1, self.window_blocks * LATENTS_PER_BLOCK + 1, 1, 1
        )
        mask[:, :, 0] = 0
        # The y condition of latent 0 (the portrait, mask 0) and of every later latent (mask 1).
        self.y = torch.cat([img_lat.repeat(1, 1, mask.shape[2], 1, 1), mask], dim=1).to(
            torch.bfloat16
        )
        self.latent_shape = self.initial_latent.shape[-3:]
        steps = pipeline.denoise_steps.to("cuda", torch.float32)
        self.steps = steps
        self.shared_timestep = (
            steps.flip(0).repeat_interleave(LATENTS_PER_BLOCK).unsqueeze(0)
        )
        lookahead = (
            self.window_blocks - 1 + (1 if self.vae_mode == "behind" else 0)
        ) * CHUNK_SAMPLES + args.context_frames * FRAME_SAMPLES
        self.info = {
            "model": "avatarforcing",
            "fps": FPS,
            "chunk_frames": FRAMES_PER_BLOCK,
            "lookahead_samples": lookahead,
            "listens": False,
            "media": "jpeg_frame",
            "width": self.width,
            "height": self.height,
        }
        self.timings: list = []
        if args.vae_compile:
            # Compile the decoder here, on this thread: dynamo cannot trace on the
            # decode thread while the DiT's first chunks trace on this one.
            vae = self.decoding_vae.model
            vae.clear_cache()
            warm = {"vae": (vae._feat_map, vae._conv_idx), "tae": {}}
            self.decode(warm, self.initial_latent)
            for _ in range(3):
                self.decode(
                    warm, torch.randn_like(self.initial_latent).repeat(1, 4, 1, 1, 1)
                )
            logger.info("VAE decoder compiled")

    def fit_flash_attn_3(self) -> None:
        """The release indexes flash-attn 3's varlen output as an (out, lse) pair; flash_attn_3 3.0.1 returns the output alone."""
        import types

        from wan.modules import attention as wan_attention

        if not wan_attention.FLASH_ATTN_3_AVAILABLE:
            logger.info("attention: flash-attn 2")
            return
        varlen = wan_attention.flash_attn_interface.flash_attn_varlen_func

        def flash_attn_varlen_func(*args, **kwargs):
            out = varlen(*args, **kwargs)
            return out if isinstance(out, tuple) else (out,)

        dense = wan_attention.flash_attn_interface.flash_attn_func
        wan_attention.flash_attn_interface = types.SimpleNamespace(
            flash_attn_varlen_func=flash_attn_varlen_func
        )
        from wan.modules import model as wan_model

        original = wan_attention.flash_attention

        def flash_attention(q, k, v, q_lens=None, k_lens=None, *args, **kwargs):
            # Without lengths, padding or options the release's varlen call is a dense
            # one; the dense API also skips the host-built cu_seqlens (CUDA graphs).
            if (
                q_lens is None
                and k_lens is None
                and not args
                and not any(
                    kwargs.get(name)
                    for name in ("dropout_p", "softmax_scale", "q_scale", "causal")
                )
                and tuple(kwargs.get("window_size", (-1, -1))) == (-1, -1)
            ):
                out = dense(q, k, v)
                return (out[0] if isinstance(out, tuple) else out).type(q.dtype)
            return original(q, k, v, q_lens, k_lens, *args, **kwargs)

        wan_attention.flash_attention = flash_attention
        wan_model.flash_attention = flash_attention
        logger.info("attention: flash-attn 3")

    def use_sage_attention(self) -> None:
        """Self-attention through SageAttention 2 (INT8 QK, FP8 PV); the cross-attention to the 512 text tokens stays on flash-attn.

        SageAttention 2.2.0's sm90 module defines the fake (meta) function under the
        name of the real op, so `core.py` calls the fake one and returns an
        uninitialized buffer (upstream fix d970424, "fix same name issue"). The
        installed package is left as it is: the module attribute is pointed back
        at the registered op here, and the result is checked against flash-attn.
        """
        import sageattention
        import torch
        from wan.modules import attention as wan_attention

        if torch.cuda.get_device_capability() == (9, 0):
            from sageattention import sm90_compile

            sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf = (
                torch.ops.sageattention_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
            )
        original = wan_attention.flash_attention

        def flash_attention(q, k, v, q_lens=None, k_lens=None, *args, **kwargs):
            plain = (
                q_lens is None
                and k_lens is None
                and not args
                and not any(
                    kwargs.get(name)
                    for name in ("dropout_p", "softmax_scale", "q_scale", "causal")
                )
                and tuple(kwargs.get("window_size", (-1, -1))) == (-1, -1)
            )
            if not plain or k.shape[1] < 2048:
                return original(q, k, v, q_lens, k_lens, *args, **kwargs)
            dtype = torch.bfloat16
            return sageattention.sageattn(
                q.to(dtype), k.to(dtype), v.to(dtype), tensor_layout="NHD"
            ).type(q.dtype)

        generator = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = (
            torch.randn(
                1, n, 12, 128, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            for n in (6240, 12480, 12480)
        )
        k = k + 2  # an offset in the keys, as the smoothing has to remove
        reference = original(q, k, v).float().flatten()
        cos = torch.nn.functional.cosine_similarity(
            flash_attention(q, k, v).float().flatten(), reference, dim=0
        ).item()
        if cos < 0.99:
            raise RuntimeError(
                f"SageAttention disagrees with flash-attn (cos {cos:.4f}); not using it"
            )
        wan_attention.flash_attention = flash_attention
        logger.info(
            "attention: SageAttention %s (cos %.4f to flash-attn)",
            sageattention.__file__,
            cos,
        )

    def use_fp8_linears(self) -> None:
        """The DiT blocks' linear layers in FP8 (e4m3): weights per output channel, activations per token, scaled once per call, through torch._scaled_mm."""
        import torch

        fp8 = torch.float8_e4m3fn
        top = torch.finfo(fp8).max

        @torch.compile(dynamic=True)
        def quantize(x):
            scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-6) / top
            return (x.float() / scale).clamp(-top, top).to(fp8), scale

        class Fp8Linear(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                weight = linear.weight.detach().float()
                scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / top
                self.register_buffer("weight", (weight / scale).to(fp8))
                self.register_buffer("scale", scale.t().contiguous())
                self.register_buffer(
                    "bias",
                    (
                        None
                        if linear.bias is None
                        else linear.bias.detach().to(torch.bfloat16)
                    ),
                )
                self.out_features = linear.out_features

            def forward(self, x):
                rows, scale = quantize(x.reshape(-1, x.shape[-1]))
                out = torch._scaled_mm(
                    rows,
                    self.weight.t(),
                    scale_a=scale,
                    scale_b=self.scale,
                    bias=self.bias,
                    out_dtype=torch.bfloat16,
                )
                return out.reshape(*x.shape[:-1], self.out_features)

        def plain(module):
            # A merged LoRA layer forwards to its base layer alone.
            if hasattr(module, "base_layer"):
                if not getattr(module, "merged", False):
                    return None
                module = module.base_layer
            return module if isinstance(module, torch.nn.Linear) else None

        converted = 0
        for block in self.pipeline.generator.model.blocks:
            for parent, names in (
                (block.self_attn, ("q", "k", "v", "o")),
                (block.cross_attn, ("q", "o")),
                (block.ffn, ("0", "2")),
            ):
                for name in names:
                    linear = plain(getattr(parent, name))
                    if linear is None:
                        continue
                    setattr(parent, name, Fp8Linear(linear).to(linear.weight.device))
                    converted += 1
        torch.cuda.empty_cache()
        logger.info("FP8 linears: %d", converted)

    def use_fused_blocks(self) -> None:
        """The DiT blocks' element-wise work compiled into a few kernels per block (the release spends about as much GPU time there as in its GEMMs):

        - RoPE from cos/sin tables built once per (start frame, grid) and shared by
          the 30 blocks, applied in float32 like `--rope32` (no complex tensors);
        - each self-attention's q/k/v projections, RMS norms and RoPE in one graph;
        - the blocks' layer norms with their modulation, and the gated residuals.

        The KV-cache bookkeeping is the release's, line for line.
        """
        import torch
        import torch._dynamo
        from wan.modules import causal_model

        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 4096
        freq_table = self.pipeline.generator.model.freqs
        c = freq_table.shape[1]
        cos_t = freq_table.real.float().cuda()
        sin_t = freq_table.imag.float().cuda()
        split = [c - 2 * (c // 3), c // 3, c // 3]
        # One (cos, sin) buffer per grid (f, h, w), refilled in place when its start
        # frame changes: a CUDA graph captured over them reads the current values.
        tables: dict = {}
        self.rope_log = None

        def fill(table, start_frame: int, f: int, h: int, w: int):
            parts = table.split(split, dim=1)
            return torch.cat(
                [
                    parts[0][start_frame : start_frame + f]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(f * h * w, 1, -1)

        def refresh(key, start_frame: int):
            entry = tables.get(key)
            if entry is None:
                entry = tables[key] = [
                    None,
                    fill(cos_t, start_frame, *key),
                    fill(sin_t, start_frame, *key),
                ]
            elif entry[0] != start_frame:
                entry[1].copy_(fill(cos_t, start_frame, *key))
                entry[2].copy_(fill(sin_t, start_frame, *key))
            entry[0] = start_frame
            return entry[1], entry[2]

        def rope_table(start_frame: int, f: int, h: int, w: int, relative_to=None):
            """`relative_to`: the call's current frame when the start moves with it (logged for graph replays)."""
            key = (f, h, w)
            if self.rope_log is not None:
                # Capturing: log the start, leave the buffers to the replay's refresh.
                self.rope_log.append(
                    (
                        key,
                        None if relative_to is None else start_frame - relative_to,
                        start_frame,
                    )
                )
                return tables[key][1], tables[key][2]
            return refresh(key, start_frame)

        self.rope_refresh = refresh

        def rope(x, cos, sin):
            x = x.float().unflatten(-1, (-1, 2))
            real, imag = x[..., 0], x[..., 1]
            return torch.stack(
                [real * cos - imag * sin, real * sin + imag * cos], dim=-1
            ).flatten(-2)

        @torch.compile(dynamic=False)
        def qkv_rope(attn, x, cos, sin):
            b, s = x.shape[:2]
            n, d = attn.num_heads, attn.head_dim
            q = attn.norm_q(attn.q(x)).view(b, s, n, d)
            k = attn.norm_k(attn.k(x)).view(b, s, n, d)
            v = attn.v(x).view(b, s, n, d)
            return (
                rope(q, cos, sin).type_as(v),
                k,
                rope(k, cos, sin).type_as(v),
                v,
            )

        @torch.compile(dynamic=False)
        def rope_only(x, cos, sin):
            return rope(x, cos, sin).type_as(x)

        @torch.compile(dynamic=False)
        def modulated(norm, x, scale, shift, frames: int):
            y = norm(x).unflatten(dim=1, sizes=(frames, x.shape[1] // frames))
            return (y * (1 + scale) + shift).flatten(1, 2)

        @torch.compile(dynamic=False)
        def gated(x, y, gate, frames: int):
            return x + (
                y.unflatten(dim=1, sizes=(frames, x.shape[1] // frames)) * gate
            ).flatten(1, 2)

        @torch.compile(dynamic=False)
        def gated_norm(norm, x, y, gate, frames: int):
            x = gated(x, y, gate, frames)
            return x, norm(x)

        attention = causal_model.attention
        attn_forward = causal_model.CausalWanSelfAttention.forward

        def self_attn_forward(
            self,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache=None,
            current_start=0,
            cache_start=None,
            updating_cache=False,
        ):
            if kv_cache is None:
                return attn_forward(
                    self,
                    x,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    block_mask,
                    kv_cache,
                    current_start,
                    cache_start,
                    updating_cache,
                )
            if cache_start is None:
                cache_start = current_start
            f, h, w = grid_sizes[0].tolist()
            frame_seqlen = h * w
            current_start_frame = current_start // frame_seqlen
            roped_query, k, roped_key, v = qkv_rope(
                self,
                x,
                *rope_table(
                    current_start_frame, f, h, w, relative_to=current_start_frame
                ),
            )
            cache_write_len = min(self.block_length, k.shape[1])
            cache_end = cache_start + cache_write_len
            global_end, local_end = kv_cache["ends"]
            num_new_tokens = cache_end - global_end
            kv_cache_size = kv_cache["k"].shape[1]
            sink_tokens = self.sink_size * self.frame_length
            if num_new_tokens > 0 and num_new_tokens + local_end > kv_cache_size:
                num_evicted_tokens = num_new_tokens + local_end - kv_cache_size
                num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
                src = slice(
                    sink_tokens + num_evicted_tokens,
                    sink_tokens + num_evicted_tokens + num_rolled_tokens,
                )
                dst = slice(sink_tokens, sink_tokens + num_rolled_tokens)
                kv_cache["k"][:, dst] = kv_cache["k"][:, src].clone()
                kv_cache["v"][:, dst] = kv_cache["v"][:, src].clone()
                local_end_index = (
                    local_end + cache_end - global_end - num_evicted_tokens
                )
                local_start_index = local_end_index - cache_write_len
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key[
                    :, :cache_write_len
                ]
                kv_cache["v"][:, local_start_index:local_end_index] = v[
                    :, :cache_write_len
                ]
            else:
                local_end_index = local_end + cache_end - global_end
                local_start_index = local_end_index - cache_write_len
                # The first block is kept without RoPE: it is roped wherever it is read.
                kv_cache["k"][:, local_start_index:local_end_index] = (
                    k if local_start_index == 0 else roped_key
                )[:, :cache_write_len]
                kv_cache["v"][:, local_start_index:local_end_index] = v[
                    :, :cache_write_len
                ]
            if num_new_tokens > 0:
                kv_cache["ends"] = (cache_end, local_end_index)
            if local_start_index == 0:
                out = attention(roped_query, roped_key, v)
            elif updating_cache:
                start = max(0, local_end_index - self.max_attention_size)
                key = kv_cache["k"][:, start:local_end_index]
                if start == 0:
                    key = torch.cat(
                        [
                            rope_only(
                                key[:, :sink_tokens],
                                *rope_table(0, self.sink_size, h, w),
                            ),
                            key[:, sink_tokens:],
                        ],
                        dim=1,
                    )
                out = attention(
                    roped_query, key, kv_cache["v"][:, start:local_end_index]
                )
            else:
                working_max = (
                    self.max_attention_size - roped_query.shape[1] - sink_tokens
                )
                start = max(sink_tokens, local_start_index - working_max)
                working_frames = (local_start_index - start) // self.frame_length
                anchor = rope_only(
                    kv_cache["k"][:, :sink_tokens],
                    *rope_table(
                        current_start_frame - working_frames - self.sink_size,
                        self.sink_size,
                        h,
                        w,
                        relative_to=current_start_frame,
                    ),
                )
                out = attention(
                    roped_query,
                    torch.cat(
                        [anchor, kv_cache["k"][:, start:local_start_index], roped_key],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            kv_cache["v"][:, :sink_tokens],
                            kv_cache["v"][:, start:local_start_index],
                            v,
                        ],
                        dim=1,
                    ),
                )
            return self.o(out.flatten(2))

        def block_forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask,
            updating_cache=False,
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            cache_start=None,
        ):
            frames = e.shape[1]
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
            y = self.self_attn(
                modulated(self.norm1, x, e[1], e[0], frames),
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                updating_cache=updating_cache,
            )
            x, normed = gated_norm(self.norm3, x, y, e[2], frames)
            x = x + self.cross_attn(
                normed, context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(modulated(self.norm2, x, e[4], e[3], frames))
            return gated(x, y, e[5], frames)

        def sinusoidal_embedding_1d(dim, position):
            # The release builds its frequencies on the host and copies them over each call.
            half = dim // 2
            position = position.type(torch.float64)
            scale = torch.pow(
                10000,
                -torch.arange(half, device=position.device, dtype=torch.float64).div(
                    half
                ),
            )
            sinusoid = torch.outer(position, scale)
            return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

        causal_model.sinusoidal_embedding_1d = sinusoidal_embedding_1d
        scheduler = self.pipeline.generator.scheduler
        for name in ("sigmas", "timesteps"):
            setattr(scheduler, name, getattr(scheduler, name).cuda())
        causal_model.CausalWanSelfAttention.forward = self_attn_forward
        causal_model.CausalWanAttentionBlock.forward = block_forward
        self.fused = True
        logger.info("fused blocks: compiled qkv+RoPE, modulated norms, gated residuals")

    def use_rope32(self) -> None:
        """The release's RoPE in complex64 instead of complex128: positions stay below 1024 frames, well inside float32."""
        import torch
        from wan.modules import causal_model

        def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
            n, c = x.size(2), x.size(3) // 2
            freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            output = []
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                seq_len = f * h * w
                x_i = torch.view_as_complex(
                    x[i, :seq_len].float().reshape(seq_len, n, -1, 2)
                )
                freqs_i = (
                    torch.cat(
                        [
                            freqs[0][start_frame : start_frame + f]
                            .view(f, 1, 1, -1)
                            .expand(f, h, w, -1),
                            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                        ],
                        dim=-1,
                    )
                    .reshape(seq_len, 1, -1)
                    .to(torch.complex64)
                )
                x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
                output.append(torch.cat([x_i.type_as(x), x[i, seq_len:]]))
            return torch.stack(output).type_as(x)

        causal_model.causal_rope_apply = causal_rope_apply
        logger.info("RoPE in complex64")

    def use_cudnn_attention(self) -> None:
        """Route the release's flash-attn 2 calls without padding or windows through PyTorch's cuDNN attention, which on Hopper runs close to flash-attn 3 (the backend the release picks when flash_attn_interface is installed)."""
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from wan.modules import attention as wan_attention

        original = wan_attention.flash_attention

        def flash_attention(
            q,
            k,
            v,
            q_lens=None,
            k_lens=None,
            dropout_p=0.0,
            softmax_scale=None,
            q_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            dtype=torch.bfloat16,
            version=None,
        ):
            if (
                q_lens is not None
                or k_lens is not None
                or tuple(window_size) != (-1, -1)
                or dropout_p
            ):
                return original(
                    q,
                    k,
                    v,
                    q_lens,
                    k_lens,
                    dropout_p,
                    softmax_scale,
                    q_scale,
                    causal,
                    window_size,
                    deterministic,
                    dtype,
                    version,
                )
            out_dtype = q.dtype
            half = lambda x: (
                x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
            )
            q, k, v = half(q), half(k), half(v)
            q = q.to(v.dtype) * (q_scale if q_scale is not None else 1)
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                x = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.to(v.dtype).transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal,
                    scale=softmax_scale,
                )
            return x.transpose(1, 2).type(out_dtype)

        wan_attention.flash_attention = flash_attention
        logger.info("attention: PyTorch cuDNN SDPA")

    def mark(self, name: str) -> None:
        """With --profile, the GPU time since the last mark, summed per name over the chunk."""
        if not self.args.profile:
            return
        self.torch.cuda.synchronize()
        now = time.perf_counter()
        self.profile[name] = self.profile.get(name, 0.0) + now - self.profile_t
        self.profile_t = now

    # ---- per-Session state the release keeps on the pipeline and the VAE ----
    @property
    def decoding_vae(self):
        return self.decoder if self.vae_device else self.pipeline.vae

    def bind(self, session) -> None:
        self.pipeline.kv_cache_clean = session["kv"]
        self.pipeline.crossattn_cache = session["cross"]

    def load_taehv(self, path: str, device: str):
        """TAEHV's Wan 2.1 decoder (madebyollin/taehv, MIT, `taew2_1.pth`): a small causal 2D-conv decoder distilled from the Wan VAE, here as the release's nn.Sequential so its weights load as they are."""
        import torch
        from torch import nn

        def conv(n_in, n_out, **kwargs):
            return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

        class Clamp(nn.Module):
            def forward(self, x):
                return torch.tanh(x / 3) * 3

        class MemBlock(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.conv = nn.Sequential(
                    conv(n * 2, n),
                    nn.ReLU(inplace=True),
                    conv(n, n),
                    nn.ReLU(inplace=True),
                    conv(n, n),
                )
                self.skip = nn.Identity()
                self.act = nn.ReLU(inplace=True)

            def forward(self, x, past):
                return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

        class TGrow(nn.Module):
            def __init__(self, n, stride):
                super().__init__()
                self.stride = stride
                self.conv = nn.Conv2d(n, n * stride, 1, bias=False)

            def forward(self, x):
                c = x.shape[1]
                return self.conv(x).reshape(-1, c, *x.shape[2:])

        layers = [Clamp(), conv(16, 256), nn.ReLU(inplace=True)]
        for n_in, n_out, stride in ((256, 128, 1), (128, 64, 2), (64, 64, 2)):
            layers += [MemBlock(n_in), MemBlock(n_in), MemBlock(n_in)]
            layers += [
                nn.Upsample(scale_factor=2),
                TGrow(n_in, stride),
                conv(n_in, n_out, bias=False),
            ]
        layers += [nn.ReLU(inplace=True), conv(64, 3)]
        decoder = nn.Sequential(*layers)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {
            k[len("decoder.") :]: v
            for k, v in state.items()
            if k.startswith("decoder.")
        }
        own = decoder.state_dict()
        for key, value in state.items():
            # A 1x TGrow keeps the last frame's channels.
            if value.shape != own[key].shape:
                state[key] = value[-own[key].shape[0] :]
        decoder.load_state_dict(state)
        self.memblock = MemBlock
        return decoder.to(device=device, dtype=torch.bfloat16).eval()

    def taehv_decode(self, session, latents):
        """Frames [T, 3, H, W] in [0, 1] of latents [1, T', 16, h, w], each MemBlock fed the previous chunk's last frame; the stream's first three frames are dropped (TAEHV's causal alignment with the Wan VAE)."""
        torch = self.torch
        memory = session["tae"]
        x = latents[0]
        for i, layer in enumerate(self.taehv):
            if isinstance(layer, self.memblock):
                past = memory.get(i)
                past = torch.zeros_like(x[:1]) if past is None else past
                memory[i] = x[-1:]
                x = layer(x, torch.cat([past, x[:-1]]))
            else:
                x = layer(x)
        trim = session.get("tae_trim", 3)
        session["tae_trim"] = max(0, trim - x.shape[0])
        return x[trim:].clamp(0, 1)

    def decode(self, session, latents):
        """The block's frames as uint8 [16, H, W, 3] on the CPU, decoded with the Session's VAE cache."""
        torch = self.torch
        if self.taehv is not None:
            with torch.inference_mode():
                video = self.taehv_decode(
                    session, latents.to(self.vae_device or "cuda")
                )
                pixels = (video.float() * 255).round().to(torch.uint8)
                return pixels.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        vae = self.decoding_vae
        with torch.inference_mode():
            vae.model._feat_map, vae.model._conv_idx = session["vae"]
            video = vae.decode_to_pixel(
                latents.to(self.vae_device or "cuda"), use_cache=True
            )
            session["vae"] = (vae.model._feat_map, vae.model._conv_idx)
            pixels = (
                ((video[0].float() * 0.5 + 0.5).clamp(0, 1) * 255)
                .round()
                .to(torch.uint8)
            )
            return pixels.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def cond(self, frame_s: int, frame_e: int, emb) -> dict:
        """The release's conditional dict over latents [frame_s, frame_e): the prompt, y and the audio embeddings of their frames."""
        n = frame_e - frame_s
        y = self.y[:, :, :n] if frame_s == 0 else self.y[:, :, 1 : 1 + n]
        return {**self.prompt, "y": y, "audio_emb": emb}

    def audio_emb(self, session, frame_s: int, frame_e: int):
        """Audio embeddings of latents [frame_s, frame_e) laid out as the release slices them: latent 0 one zero frame, latent k frames 4(k-1)..4k-1."""
        torch = self.torch
        low = 0 if frame_s == 0 else 1 + 4 * (frame_s - 1)
        high = 1 if frame_e <= 1 else 1 + 4 * (frame_e - 1)
        rows = [session["emb"].get(index) for index in range(low, high)]
        zero = next(row for row in session["emb"].values())
        return torch.stack(
            [zero * 0 if row is None else row for row in rows]
        ).unsqueeze(0)

    def encode_frames(self, session, first_frame: int, last_frame: int) -> None:
        """wav2vec2 features of pixel frames [first_frame, last_frame) (embedding indices, 1-based) from a 5 s window of the Session's audio ending `context` frames past them."""
        torch = self.torch
        end = last_frame - 1 + self.args.context_frames
        start = max(end - FEATURE_WINDOW_FRAMES, 0)
        audio = session["audio"][start * FRAME_SAMPLES : end * FRAME_SAMPLES]
        audio = np.pad(audio, (0, (end - start) * FRAME_SAMPLES - audio.size))
        values = self.extractor(audio, sampling_rate=SAMPLE_RATE_HZ).input_values[0]
        values = torch.from_numpy(np.asarray(values)).float().unsqueeze(0).to("cuda")
        hs = self.wav2vec(values, seq_len=end - start, output_hidden_states=True)
        emb = hs.last_hidden_state
        for h in hs.hidden_states:
            emb = torch.cat([emb, h], dim=-1)
        emb = emb[0].to(torch.bfloat16)
        for index in range(first_frame, last_frame):
            session["emb"][index] = emb[index - 1 - start]
        if 0 not in session["emb"]:
            session["emb"][0] = torch.zeros_like(emb[0])
        for index in [
            i
            for i in session["emb"]
            if 0 < i < first_frame - 4 * self.window_blocks * LATENTS_PER_BLOCK
        ]:
            del session["emb"][index]

    def start(self):
        torch = self.torch
        pipeline = self.pipeline
        slot = getattr(self, "slot", None)
        if slot is not None:
            # CUDA graphs read the caches at fixed addresses: every Session reuses the
            # one set of caches (and graphs) the warm-up captured; the prompt, and so the
            # cross-attention cache, is the same for all of them.
            pipeline.kv_cache_clean, pipeline.crossattn_cache = (
                slot["kv"],
                slot["cross"],
            )
        else:
            pipeline.kv_cache_clean = None
            pipeline._reset_or_init_caches(
                batch_size=1, dtype=torch.bfloat16, device="cuda"
            )
        if self.fused:
            # (global end, local end) of each cache as host ints: no device sync per block.
            for cache in pipeline.kv_cache_clean:
                cache["ends"] = (0, 0)
        if self.args.cuda_graph and slot is None:
            slot = self.slot = {
                "kv": pipeline.kv_cache_clean,
                "cross": pipeline.crossattn_cache,
                "graphs": {},
                "graph_warm": set(),
                "graph_pool": torch.cuda.graph_pool_handle(),
            }
        vae = self.decoding_vae.model
        vae.clear_cache()
        session = {
            "kv": pipeline.kv_cache_clean,
            "cross": pipeline.crossattn_cache,
            "vae": (vae._feat_map, vae._conv_idx),
            "audio": np.zeros(0, np.float32),
            "emb": OrderedDict(),
            "noisy": {},
            "block": 0,
            "primed": False,
            "generator": torch.Generator(device="cuda").manual_seed(self.args.seed),
            "decoding": None,
            "tae": {},
        }
        if slot is not None:
            session.update(
                graphs=slot["graphs"],
                graph_warm=slot["graph_warm"],
                graph_pool=slot["graph_pool"],
            )
        return session

    def prime(self, session) -> None:
        """The release's cache prefill with the portrait as latent 0, and the VAE's first frame."""
        torch = self.torch
        zero = torch.zeros([1, 1], device="cuda", dtype=torch.int64)
        self.pipeline.generator(
            noisy_image_or_video=self.initial_latent,
            conditional_dict=self.cond(0, 1, self.audio_emb(session, 0, 1)),
            timestep=zero,
            kv_cache=self.pipeline.kv_cache_clean,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=0,
        )
        self.decode(session, self.initial_latent)

    def window(self, session, wi: int, on_done=None):
        """The release's window `wi`: returns the latents of the block that leaves it, or None while it ramps up; `on_done` gets that block before the cache update."""
        torch = self.torch
        pipeline = self.pipeline
        n = LATENTS_PER_BLOCK
        sb, eb = max(0, wi - self.window_blocks + 1), wi
        frame_s, frame_e = 1 + sb * n, 1 + (eb + 1) * n
        cur = frame_e - frame_s
        noise = torch.randn(
            (1, n, *self.latent_shape),
            device="cuda",
            dtype=torch.bfloat16,
            generator=session["generator"],
        )
        cached = [session["noisy"][b] for b in range(sb, eb)]
        noisy_input = torch.cat([*cached, noise], dim=1)
        timestep = (
            self.shared_timestep
            if cur == self.window_blocks * n
            else self.shared_timestep[:, -cur:]
        )
        cond = self.cond(frame_s, frame_e, self.audio_emb(session, frame_s, frame_e))
        current_start = frame_s * pipeline.frame_seq_length
        # The steady state (full window, full cache) has fixed shapes and cache offsets:
        # with --cuda-graph its two forwards are replayed from CUDA graphs.
        graph = (
            self.args.cuda_graph
            and cur == self.window_blocks * n
            and session["primed"]
            and all(c["ends"][1] == c["k"].shape[1] for c in session["kv"])
        )

        def window_forward(inputs):
            return pipeline.generator(
                noisy_image_or_video=inputs["x"],
                conditional_dict={**cond, "audio_emb": inputs["a"]},
                timestep=timestep,
                kv_cache=pipeline.kv_cache_clean,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start,
            )[1]

        self.mark("pre")
        inputs = {"x": noisy_input, "a": cond["audio_emb"]}
        denoised = (
            self.graphed(session, "window", frame_s, inputs, window_forward)
            if graph
            else window_forward(inputs)
        )
        self.mark("dit16")
        done = None
        last = len(self.steps) - 1
        blocks = cur // n
        for blk in range(sb, eb + 1):
            offset = (blk - sb) * n
            # The window's blocks sit at the last `blocks` noise levels, newest noisiest.
            level = last - (blk - sb) - (len(self.steps) - blocks)
            block = denoised[:, offset : offset + n]
            if level == last:
                done = block.clone() if graph else block
                session["noisy"].pop(blk, None)
                continue
            next_ts = self.steps[level + 1]
            session["noisy"][blk] = pipeline.scheduler.add_noise(
                block.flatten(0, 1),
                torch.randn(
                    block.flatten(0, 1).shape,
                    device="cuda",
                    dtype=block.dtype,
                    generator=session["generator"],
                ),
                next_ts * torch.ones([n], device="cuda", dtype=torch.long),
            ).unflatten(0, block.shape[:2])
        if done is not None and on_done is not None:
            on_done(done)
        if self.args.skip_cache_update:
            # The window's forward already wrote the oldest block's keys and values
            # (from its input at the last noise level) where the clean pass would.
            self.mark("dit4")
            return done
        clean_cond = self.cond(frame_s, frame_s + n, None)
        clean_timestep = torch.ones_like(timestep[:, :n]) * self.context_noise

        def cache_forward(inputs):
            pipeline.generator(
                noisy_image_or_video=inputs["x"],
                conditional_dict={**clean_cond, "audio_emb": inputs["a"]},
                timestep=clean_timestep,
                kv_cache=pipeline.kv_cache_clean,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start,
                updating_cache=True,
            )

        inputs = {
            "x": denoised[:, :n],
            "a": self.audio_emb(session, frame_s, frame_s + n),
        }
        if graph:
            self.graphed(session, "cache", frame_s, inputs, cache_forward)
        else:
            cache_forward(inputs)
        self.mark("dit4")
        return done

    def graphed(self, session, name: str, frame: int, inputs: dict, run):
        """`run(inputs)` replayed from a CUDA graph of the Session: the first steady call runs eagerly (warm-up), the second captures, later ones copy `inputs` into the captured buffers, refill the RoPE tables for `frame` and replay."""
        torch = self.torch
        graphs = session.setdefault("graphs", {})
        entry = graphs.get(name)
        if entry is None:
            if name not in session.setdefault("graph_warm", set()):
                session["graph_warm"].add(name)
                return run(inputs)
            static = {key: value.clone() for key, value in inputs.items()}
            before = [cache["ends"] for cache in session["kv"]]
            pool = session.setdefault("graph_pool", torch.cuda.graph_pool_handle())
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            self.rope_log = []
            try:
                with torch.cuda.graph(
                    graph, pool=pool, capture_error_mode="thread_local"
                ):
                    out = run(static)
            finally:
                log, self.rope_log = self.rope_log, None
            after = [cache["ends"] for cache in session["kv"]]
            if (
                len({a[1] for a in after}) != 1
                or len({a[0] - b[0] for a, b in zip(after, before)}) != 1
            ):
                raise RuntimeError(f"graph {name}: the caches moved unevenly")
            entry = graphs[name] = {
                "graph": graph,
                "static": static,
                "out": out,
                "rope": log,
                "advance": after[0][0] - before[0][0],
                "local_end": after[0][1],
            }
            logger.info(
                "captured the %s forward as a CUDA graph (rope tables %s)",
                name,
                [(key[0], rel, start) for key, rel, start in log],
            )
        else:
            for key, value in inputs.items():
                entry["static"][key].copy_(value)
            for cache in session["kv"]:
                cache["ends"] = (
                    cache["ends"][0] + entry["advance"],
                    entry["local_end"],
                )
        for key, rel, start in entry["rope"]:
            self.rope_refresh(key, start if rel is None else frame + rel)
        entry["graph"].replay()
        return entry["out"]

    def step(self, session, speech, listen):
        if self.args.torch_profile and session["block"] == 3:
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                out = self.step_chunk(session, speech, listen)
            logger.info(
                "profile:\n%s",
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=25),
            )
            return out
        return self.step_chunk(session, speech, listen)

    def step_chunk(self, session, speech, listen):
        torch = self.torch
        started = time.perf_counter()
        self.profile, self.profile_t = {}, started
        k = session["block"]
        if (k + self.window_blocks + 1) * LATENTS_PER_BLOCK >= MAX_LATENTS:
            logger.info("restarting a session at its RoPE limit")
            session.update(self.start())
            k = 0
        session["audio"] = np.concatenate(
            [session["audio"][: k * CHUNK_SAMPLES], speech]
        )
        # Decoding one chunk behind on another GPU, the DiT runs one block ahead: chunk k's window finishes block k + 1 while block k decodes.
        ahead = 1 if self.vae_mode == "behind" else 0
        overlap = self.vae_mode == "overlap"
        decoding = []

        def start_decode(block):
            decoding.append(self.decode_pool.submit(self.decode, session, block))

        self.bind(session)
        with torch.inference_mode():
            first = 1 + k * FRAMES_PER_BLOCK
            if not session["primed"]:
                self.encode_frames(
                    session, 1, 1 + (self.window_blocks + ahead) * FRAMES_PER_BLOCK
                )
                self.prime(session)
                for wi in range(self.window_blocks - 1 + ahead):
                    block = self.window(session, wi)
                if ahead:
                    session["decoding"] = self.decode_pool.submit(
                        self.decode, session, block
                    )
                session["primed"] = True
            else:
                newest = first + (self.window_blocks - 1 + ahead) * FRAMES_PER_BLOCK
                self.encode_frames(session, newest, newest + FRAMES_PER_BLOCK)
            self.mark("audio")
            block = self.window(
                session,
                k + self.window_blocks - 1 + ahead,
                on_done=start_decode if overlap else None,
            )
            self.mark("window")
            if ahead:
                pixels = session["decoding"].result()
                session["decoding"] = self.decode_pool.submit(
                    self.decode, session, block
                )
            elif overlap:
                pixels = decoding[0].result()
            else:
                pixels = self.decode(session, block)
            self.mark("vae")
        self.pipeline.kv_cache_clean, self.pipeline.crossattn_cache = None, None
        session["block"] = k + 1
        generated = time.perf_counter()
        jpegs = (
            list(self.jpeg_pool.map(self.encode, pixels))
            if self.jpeg_pool
            else [self.encode(frame) for frame in pixels]
        )
        self.timings.append(
            (
                generated - started,
                time.perf_counter() - generated,
                {k: round(v * 1000, 1) for k, v in self.profile.items()},
            )
        )
        if len(jpegs) != FRAMES_PER_BLOCK:
            raise RuntimeError(f"the VAE decoded {len(jpegs)} frames of a block")
        return jpegs

    def encode(self, rgb: np.ndarray) -> bytes:
        import cv2

        ok, data = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(rgb[:, :, ::-1]),
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )
        assert ok
        return data.tobytes()


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--prompt", default="A person is talking to the camera.")
    parser.add_argument("--config-path", default="configs/avatarforcing.yaml")
    parser.add_argument("--checkpoint-path", default="checkpoints/model.pt")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="wav2vec2 right context past the newest block, in frames",
    )
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument(
        "--rope32", action="store_true", help="RoPE in complex64 instead of complex128"
    )
    parser.add_argument(
        "--compile-blocks",
        action="store_true",
        help="torch.compile each block's FFN and cross attention",
    )
    parser.add_argument(
        "--vae-device",
        default=None,
        help="decode on this device (e.g. cuda:1), one chunk behind the DiT; the look-ahead grows by a chunk",
    )
    parser.add_argument(
        "--vae-mode",
        default="behind",
        choices=["behind", "overlap"],
        help="with --vae-device: decode one chunk behind the DiT (the look-ahead grows by a chunk), or within the chunk alongside the DiT's cache update",
    )
    parser.add_argument(
        "--taehv",
        default=None,
        help="decode with TAEHV's Wan 2.1 decoder from this taew2_1.pth instead of the Wan VAE (much faster, a little softer)",
    )
    parser.add_argument(
        "--vae-compile",
        action="store_true",
        help="torch.compile the Wan VAE decoder's causal 3D convolutions (same output, about 13%% faster)",
    )
    parser.add_argument(
        "--vae-cudnn-benchmark",
        action="store_true",
        help="let cuDNN pick the VAE's 3D convolution algorithms",
    )
    parser.add_argument(
        "--attention",
        default="release",
        choices=["release", "cudnn", "sage"],
        help="release: flash-attn 3 when installed, else 2; cudnn: PyTorch's cuDNN attention; sage: SageAttention 2 for the self-attention",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="compile the blocks' element-wise work (RoPE, norms, modulation, residuals) with the q/k/v projections",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="with --fuse: replay the steady-state window and cache forwards from CUDA graphs captured per Session",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="the DiT blocks' linear layers in FP8 (per-channel weights, per-token activations)",
    )
    parser.add_argument(
        "--denoise-steps",
        default=None,
        help="comma-separated noise levels instead of the config's (1000,800,400,200); the window holds one block per level",
    )
    parser.add_argument(
        "--skip-cache-update",
        action="store_true",
        help="keep the keys and values the window's forward wrote for the finished block instead of re-running it clean",
    )
    parser.add_argument(
        "--jpeg-threads",
        type=int,
        default=1,
        help="encode a chunk's frames on this many threads",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="log the kernel table of a Session's fourth chunk",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="synchronize and time each part of a chunk, reported with /stats",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    args.avatar = str(Path(args.avatar).resolve())
    setup_logging()

    def load():
        model = AvatarForcing(args)
        times = warm_up(model, args.warmup_chunks)
        logger.info("warm-up chunk times (ms): %s", [round(t) for t in times])
        return model

    serve(load, args.host, args.port)


if __name__ == "__main__":
    main()
