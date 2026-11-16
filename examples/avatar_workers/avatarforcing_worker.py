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
WINDOW_BLOCKS = 4
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
        if args.attention == "cudnn":
            self.use_cudnn_attention()
        if args.rope32:
            self.use_rope32()
        self.fit_flash_attn_3()
        self.args = args
        torch.set_grad_enabled(False)
        torch.manual_seed(args.seed)
        config = OmegaConf.merge(
            OmegaConf.load("configs/default_config.yaml"),
            OmegaConf.load(args.config_path),
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
        if args.compile_blocks:
            import torch._dynamo

            torch._dynamo.config.cache_size_limit = 64
            for block in pipeline.generator.model.blocks:
                block.ffn = torch.compile(block.ffn, dynamic=False)
                if hasattr(block, "cross_attn"):
                    block.cross_attn = torch.compile(block.cross_attn, dynamic=False)
            logger.info("compiled the blocks' ffn and cross attention")
        # A second VAE decodes on its own GPU, one chunk behind the DiT, so the two overlap.
        self.vae_device = args.vae_device
        if self.vae_device:
            import concurrent.futures

            from utils.wan_wrapper import WanVAEWrapper

            self.decoder = WanVAEWrapper().to(
                device=self.vae_device, dtype=torch.bfloat16
            )
            self.decode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
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
            1, 1, WINDOW_BLOCKS * LATENTS_PER_BLOCK + 1, 1, 1
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
            WINDOW_BLOCKS - 1 + (1 if args.vae_device else 0)
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

        wan_attention.flash_attn_interface = types.SimpleNamespace(
            flash_attn_varlen_func=flash_attn_varlen_func
        )
        logger.info("attention: flash-attn 3")

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

    def decode(self, session, latents):
        """The block's frames as uint8 [16, H, W, 3] on the CPU, decoded with the Session's VAE cache."""
        torch = self.torch
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
            if 0 < i < first_frame - 4 * WINDOW_BLOCKS * LATENTS_PER_BLOCK
        ]:
            del session["emb"][index]

    def start(self):
        torch = self.torch
        pipeline = self.pipeline
        pipeline.kv_cache_clean = None
        pipeline._reset_or_init_caches(
            batch_size=1, dtype=torch.bfloat16, device="cuda"
        )
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
        }
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

    def window(self, session, wi: int):
        """The release's window `wi`: returns the latents of the block that leaves it, or None while it ramps up."""
        torch = self.torch
        pipeline = self.pipeline
        n = LATENTS_PER_BLOCK
        sb, eb = max(0, wi - WINDOW_BLOCKS + 1), wi
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
            if cur == WINDOW_BLOCKS * n
            else self.shared_timestep[:, -cur:]
        )
        cond = self.cond(frame_s, frame_e, self.audio_emb(session, frame_s, frame_e))
        self.mark("pre")
        _, denoised = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond,
            timestep=timestep,
            kv_cache=pipeline.kv_cache_clean,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=frame_s * pipeline.frame_seq_length,
        )
        self.mark("dit16")
        done = None
        last = len(self.steps) - 1
        for blk in range(sb, eb + 1):
            offset = (blk - sb) * n
            level = int(torch.argmin(torch.abs(self.steps - timestep[0, offset])))
            block = denoised[:, offset : offset + n]
            if level == last:
                done = block
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
        pipeline.generator(
            noisy_image_or_video=denoised[:, :n],
            conditional_dict=self.cond(
                frame_s, frame_s + n, self.audio_emb(session, frame_s, frame_s + n)
            ),
            timestep=torch.ones_like(timestep[:, :n]) * self.context_noise,
            kv_cache=pipeline.kv_cache_clean,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=frame_s * pipeline.frame_seq_length,
            updating_cache=True,
        )
        self.mark("dit4")
        return done

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
        if (k + WINDOW_BLOCKS + 1) * LATENTS_PER_BLOCK >= MAX_LATENTS:
            logger.info("restarting a session at its RoPE limit")
            session.update(self.start())
            k = 0
        session["audio"] = np.concatenate(
            [session["audio"][: k * CHUNK_SAMPLES], speech]
        )
        # Decoding on another GPU, the DiT runs one block ahead: chunk k's window finishes block k + 1 while block k decodes.
        ahead = 1 if self.vae_device else 0
        self.bind(session)
        with torch.inference_mode():
            first = 1 + k * FRAMES_PER_BLOCK
            if not session["primed"]:
                self.encode_frames(
                    session, 1, 1 + (WINDOW_BLOCKS + ahead) * FRAMES_PER_BLOCK
                )
                self.prime(session)
                for wi in range(WINDOW_BLOCKS - 1 + ahead):
                    block = self.window(session, wi)
                if ahead:
                    session["decoding"] = self.decode_pool.submit(
                        self.decode, session, block
                    )
                session["primed"] = True
            else:
                newest = first + (WINDOW_BLOCKS - 1 + ahead) * FRAMES_PER_BLOCK
                self.encode_frames(session, newest, newest + FRAMES_PER_BLOCK)
            self.mark("audio")
            block = self.window(session, k + WINDOW_BLOCKS - 1 + ahead)
            self.mark("window")
            if ahead:
                pixels = session["decoding"].result()
                session["decoding"] = self.decode_pool.submit(
                    self.decode, session, block
                )
            else:
                pixels = self.decode(session, block)
            self.mark("vae")
        self.pipeline.kv_cache_clean, self.pipeline.crossattn_cache = None, None
        session["block"] = k + 1
        generated = time.perf_counter()
        jpegs = [self.encode(frame) for frame in pixels]
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
        "--attention",
        default="release",
        choices=["release", "cudnn"],
        help="release: flash-attn 3 when installed, else 2; cudnn: PyTorch's cuDNN attention",
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
