# SPDX-License-Identifier: Apache-2.0
"""A SoulX-LiveAct speech-avatar worker: a talking head from one portrait, 32 frames (1.6 s at 20 fps) per request, on the authors' two-GPU real-time setup.

Run it with torchrun in LiveAct's own environment (sequence parallel over the
two GPUs, as the authors' generate.py does); rank 0 serves the HTTP protocol
of chunk_server.py and rank 1 follows its every step:

    USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 \
        liveact_worker.py --repo SoulX-LiveAct --port 9520 \
        --ckpt-dir <Soul-AILab/LiveAct> --wav2vec-dir <chinese-wav2vec2-base> \
        --cond-image SoulX-LiveAct/examples/image/1.png

The model is autoregressive: its blocks attend to a KV cache (with ConvKV
memory) of the blocks before, which lives on the GPUs and cannot travel with
a request. So, unlike the stateless workers, the state this worker returns is
a ticket: the Session's token and the number of chunks rendered. A request
whose ticket is not the worker's current one (an empty state, another
Session, a worker restart) starts a new Session there; one worker therefore
holds one Session (sessions_per_worker: 1).

Geometry, following generate.py --steam_audio: the model's block b >= 1
generates 8 latent frames (32 video frames, model frames 21 + 32 (b - 1) on)
from the audio of model frames 32 (b - 1) to 32 (b - 1) + 55. Chunk k of a
Session is block k + 1, its frames placed at Session frames 32 k on, so its
window reads 21 frames (1.05 s) of speech before the chunk and 2 frames
(0.1 s) after it. A Session's first request also runs block 0 (6 latent
frames, 21 video frames) on the same window and drops its frames, as
generate.py's first block precedes the audio.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_server import (  # noqa: E402
    SAMPLE_RATE_HZ,
    ChunkModel,
    base_parser,
    serve,
    setup_logging,
    warm_up,
)

logger = logging.getLogger("liveact_worker")

FPS = 20
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
# generate.py's block sizes in latent frames: the first block, then every block after it.
BLOCK_LATENTS = (6, 8)
# The 14 latent frames of the model's window are 53 video frames; the audio window reads 2 frames more.
WINDOW_FRAMES = (sum(BLOCK_LATENTS) - 1) * VAE_STRIDE[0] + 1
AUDIO_WINDOW_FRAMES = WINDOW_FRAMES + 2
CHUNK_FRAMES = BLOCK_LATENTS[1] * VAE_STRIDE[0]
CONTEXT_FRAMES = WINDOW_FRAMES - CHUNK_FRAMES
LOOKAHEAD_FRAMES = AUDIO_WINDOW_FRAMES - WINDOW_FRAMES
FRAME_SAMPLES = SAMPLE_RATE_HZ // FPS
# The previous block's last latent frames the decoder sees again, and the video frames they make.
OVERLAP_LATENTS = 3
OVERLAP_FRAMES = 9
TIMESTEPS = (1000.0, 937.5, 833.33333333, 0.0)
TICKET = struct.Struct("<16sI")


def geometry_info(width: int, height: int, half_blocks: bool = False) -> dict:
    """A chunk is a block (32 frames); with half_blocks it is half a block (16 frames), whose window reaches back to its block's start."""
    chunk = CHUNK_FRAMES // 2 if half_blocks else CHUNK_FRAMES
    return {
        "kind": "video",
        "model": "soulx-liveact",
        "fps": FPS,
        "chunk_frames": chunk,
        "context_samples": (WINDOW_FRAMES - chunk) * FRAME_SAMPLES
        if half_blocks
        else CONTEXT_FRAMES * FRAME_SAMPLES,
        "lookahead_samples": LOOKAHEAD_FRAMES * FRAME_SAMPLES,
        "width": width,
        "height": height,
    }


def model_unit(unit: int, half_blocks: bool) -> tuple:
    """A Session's model unit as (block, half): unit 0 is block 0, then a block per unit, or with half_blocks half a block per unit (half None: the whole block)."""
    if half_blocks and unit > 0:
        return (unit + 1) // 2, (unit + 1) % 2
    else:
        return unit, None


def model_window(unit: int, speech: np.ndarray, half_blocks: bool) -> np.ndarray:
    """The model window's 55 frames of speech for a unit from a request's window: a block's first half sees up to its own lookahead, silence after."""
    _, half = model_unit(unit, half_blocks)
    if half == 0:
        known = speech[(CHUNK_FRAMES // 2) * FRAME_SAMPLES :]
        return np.concatenate(
            [known, np.zeros(AUDIO_WINDOW_FRAMES * FRAME_SAMPLES - len(known), np.float32)]
        )
    else:
        return speech


@dataclass
class Ticket:
    """Which Session the worker's KV cache holds and how many of its chunks it rendered."""

    token: bytes
    chunks: int

    def encode(self) -> bytes:
        return TICKET.pack(self.token, self.chunks)

    @classmethod
    def decode(cls, state: bytes) -> Ticket | None:
        if len(state) != TICKET.size:
            return None
        else:
            token, chunks = TICKET.unpack(state)
            return cls(token, chunks)


class Tickets:
    """The worker's one Session: whether a request continues it or starts a new one."""

    def __init__(self, new_token=lambda: os.urandom(16)) -> None:
        self.new_token = new_token
        self.current: Ticket | None = None

    def blocks_for(self, state: bytes) -> list[int]:
        """The model blocks a request with `state` runs, the last one making its chunk; moves the worker to the next ticket."""
        ticket = Ticket.decode(state)
        if ticket is not None and ticket == self.current:
            self.current = Ticket(ticket.token, ticket.chunks + 1)
            return [ticket.chunks + 1]
        else:
            if state:
                logger.warning(
                    "a request for another Session's chunk (%s), not %s: starting a new Session",
                    ticket,
                    self.current,
                )
            else:
                pass
            self.current = Ticket(self.new_token(), 1)
            return [0, 1]

    def state(self) -> bytes:
        assert self.current is not None
        return self.current.encode()


class GraphedSteps:
    """The DiT's denoising steps replayed as CUDA graphs, one per kind of step (its latent shape, step, place in the block and cache update).

    A kind's first call runs normally (it compiles and warms up); its second
    is captured, then replayed, and so is every later one. Capturing does not
    run anything, so the model's in-place KV cache updates happen once a
    call, as without graphs. Inputs that change between calls (the latent
    and the speech) are copied into the graph's buffers; everything else
    (the cache, timestep, prompt, image and y tensors) is the same tensor on
    every call. Every rank captures the same kinds in the same order, so the
    sequence-parallel collectives inside line up.
    """

    def __init__(self, model, torch) -> None:
        self.model, self.torch = model, torch
        self.seen = set()
        self.graphs = {}
        # One memory pool for every graph: they replay one at a time and each one's output is used before the next replays, so
        # their temporaries can share memory.
        self.pool = torch.cuda.graph_pool_handle()

    def __call__(self, x, **kwargs):
        torch = self.torch
        latent, audio = x[0], kwargs["audio"]
        key = (
            tuple(latent.shape),
            id(kwargs["kv_cache"]),
            kwargs["start_idx"],
            kwargs["update_cache"],
            kwargs["skip_audio"],
        )
        graph = self.graphs.get(key)
        if graph is None and key not in self.seen:
            self.seen.add(key)
            return self.model(x, **kwargs)
        elif graph is None:
            static_latent, static_audio = latent.clone(), audio.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self.model([static_latent], **{**kwargs, "audio": static_audio})
            self.graphs[key] = graph = (graph, static_latent, static_audio, out)
            logger.info("captured a CUDA graph for %s (%d graphs)", key, len(self.graphs))
        else:
            pass
        graph, static_latent, static_audio, out = graph
        static_latent.copy_(latent)
        static_audio.copy_(audio)
        graph.replay()
        return out


class LiveActModel(ChunkModel):
    """generate.py's model setup and block loop, one request a chunk; every rank builds it and runs each step."""

    def __init__(self, args, rank: int, world_size: int) -> None:
        os.chdir(args.repo)
        sys.path.insert(0, args.repo)
        import torch
        import torch.distributed as dist
        import torchaudio
        import torchaudio.transforms as T
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
        from PIL import Image
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        from torchvision import transforms
        from transformers import Wav2Vec2FeatureExtractor
        from util_liveact import (
            center_rescale_crop_keep_ratio,
            get_audio_emb,
            get_embedding,
            get_msk,
        )
        from wan.modules.clip import CLIPModel
        from wan.modules.t5 import T5EncoderModel

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.allow_tf32 = True
        self.torch, self.dist, self.Image = torch, dist, Image
        self.torchaudio, self.Resample = torchaudio, T.Resample
        self.get_audio_emb, self.get_embedding = get_audio_emb, get_embedding
        self.rank, self.world_size = rank, world_size
        self.seed = args.seed
        self.sp_cache = args.sp_cache if world_size > 1 else "seq"
        self.jpeg_quality = args.jpeg_quality
        device = self.device = rank
        if world_size > 1:
            from model_liveact import model_memory_sp
            from model_liveact.model_memory_sp import WanModel

            if args.sp_cache == "heads":
                import liveact_sp

                liveact_sp.patch(model_memory_sp, world_size, args.attention)
            else:
                assert world_size == 2, "the authors' sequence-split cache only works on 2 GPUs: use --sp-cache heads"
        else:
            from model_liveact.model_memory import WanModel
        width, height = (int(side) for side in args.size.split("*"))
        self.width, self.height = width, height
        self.frame_len = (height // (PATCH_SIZE[1] * VAE_STRIDE[1])) * (
            width // (PATCH_SIZE[2] * VAE_STRIDE[2])
        )
        self.timesteps = [
            torch.tensor([step]).to(device, dtype=torch.float32)
            for step in args.timesteps
        ]
        self.audio_steps = set(args.audio_steps)
        if world_size > 1 and args.sp_cache == "heads":
            kv_shape = liveact_sp.kv_cache_shape(self.frame_len, world_size)
        else:
            kv_shape = [1, self.frame_len * sum(BLOCK_LATENTS) // world_size, 40, 128]
        self.kv_cache = {
            step: {
                layer: {
                    "k": torch.zeros(kv_shape, dtype=torch.bfloat16, device=device),
                    "v": torch.zeros(kv_shape, dtype=torch.bfloat16, device=device),
                    "k_scale": None,
                    "v_scale": None,
                    "mean_memory": False,
                    "offload_cache": False,
                    "fp8_kv_cache": False,
                }
                for layer in range(40)
            }
            for step in range(len(self.timesteps) - 1)
        }
        model = WanModel.from_pretrained(
            args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        ).to(dtype=torch.bfloat16)
        for block in range(40):
            model.blocks[block].self_attn.init_kvidx(self.frame_len, world_size)
        self.vae = LightVAE(
            vae_path=os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"),
            dtype=torch.bfloat16,
            device=device,
            use_lightvae=False,
            parallel=world_size > 1,
        )
        clip = CLIPModel(
            checkpoint_path=os.path.join(
                args.ckpt_dir,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            tokenizer_path=os.path.join(args.ckpt_dir, "xlm-roberta-large"),
            dtype=torch.bfloat16,
            device=device,
        )
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device="cpu",
            checkpoint_path=os.path.join(
                args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"
            ),
            tokenizer_path=os.path.join(args.ckpt_dir, "google/umt5-xxl"),
        )
        self.audio_encoder = (
            Wav2Vec2Model.from_pretrained(
                args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
            )
            .to(device, dtype=torch.bfloat16)
            .eval()
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_dir, local_files_only=True
        )
        self.audio_encoder.feature_extractor._freeze_parameters()
        model.freqs = model.freqs.to(device)
        for module in (model, clip.model, self.audio_encoder, self.vae.model):
            for parameter in module.parameters():
                parameter.requires_grad = False
        if args.fp8_gemm:
            from fp8_gemm import FP8GemmOptions, enable_fp8_gemm

            enable_fp8_gemm(model, options=FP8GemmOptions())
        else:
            pass
        model = model.to(device).eval()
        self.model = torch.compile(model)
        if args.cuda_graphs:
            assert world_size > 1 and args.sp_cache == "heads" and args.attention == "fa3", "--cuda-graphs needs --sp-cache heads --attention fa3 (Sage's kernels do not replay)"
            self.model = GraphedSteps(self.model, torch)
        else:
            pass
        self.vae.model.eval()
        self.vae.encode = torch.compile(self.vae.encode)
        self.vae_stream = args.vae_stream
        if args.vae_stream:
            # The decoder's causal cache carries over from block to block (a Session's latents decode as one video), so a block
            # decodes its own 8 latents, not 3 of the block before as well, and every frame it makes is kept.
            self.vae.model.decode = self.vae.model.cached_decode
        else:
            pass
        if args.vae_compile:
            self.vae.model.decoder = torch.compile(self.vae.model.decoder, dynamic=False)
        else:
            pass

        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda picture: center_rescale_crop_keep_ratio(
                        picture, (height, width)
                    )
                ),
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        with torch.no_grad():
            # The prompt is fixed: encode it once on the CPU, then free T5.
            self.context = [
                text_encoder(texts=args.prompt, device="cpu")[0].to(
                    device, dtype=torch.bfloat16
                )
            ]
            del text_encoder
            cond_image = (
                transform(Image.open(args.cond_image).convert("RGB"))
                .unsqueeze(1)
                .unsqueeze(0)
                .to(device, torch.bfloat16)
            )
            clip.model.to(device)
            self.clip_context = clip.visual(cond_image)
            del clip
            self.ref_target_masks = torch.ones(
                3, height // VAE_STRIDE[1], width // VAE_STRIDE[2]
            ).to(device, torch.bfloat16)
            msk = get_msk(WINDOW_FRAMES, cond_image, VAE_STRIDE, device)
            frames = torch.zeros(
                1, cond_image.shape[1], WINDOW_FRAMES - 1, height, width
            ).to(cond_image.device, cond_image.dtype)
            y = (
                self.vae.encode(
                    torch.concat([cond_image, frames], dim=2).to(self.vae.device)
                )
                .to(device)
                .unsqueeze(0)
            )
            self.y = torch.concat([msk, y], dim=1)[:, :, : WINDOW_FRAMES // 4 + 1]
        torch.cuda.empty_cache()
        self.tickets = Tickets()
        self.pre_latent = None
        self.half_blocks = args.half_blocks
        assert not args.half_blocks or (args.vae_stream and args.sp_cache == "heads"), "--half-blocks needs --vae-stream and --sp-cache heads"
        self.info = geometry_info(width, height, args.half_blocks)
        self.encoder = ThreadPoolExecutor(max_workers=CHUNK_FRAMES)
        try:
            import cv2

            self.cv2 = cv2
        except ImportError:
            self.cv2 = None
        self.idle = None
        if world_size > 1:
            from datetime import timedelta

            # The ranks meet on every request; an idle worker waits for days, not gloo's default 30 min.
            self.control = dist.new_group(backend="gloo", timeout=timedelta(days=30))
        else:
            self.control = None
        # Every step runs on this one thread, warm-up included: torch.compile's guards read thread-local state, and a model compiled on one thread
        # but called from the HTTP server's handler threads falls back to slow paths (9.8 s a chunk instead of 1.5 s on two H200s).
        self.gpu = ThreadPoolExecutor(
            max_workers=1, initializer=torch.cuda.set_device, initargs=(device,)
        )
        self.primed = None
        if args.prime:
            self.gpu.submit(self.prime).result()
        else:
            pass
        if rank == 0:
            if args.cuda_graphs and args.warmup_chunks > 0:
                # every kind of step is captured on its second sighting: enough chunks for each to come twice, and a
                # second Session for the first chunk's kind
                times = warm_up(self, max(args.warmup_chunks, 6 if args.half_blocks else 4))
                times += warm_up(self, 2)
            else:
                times = warm_up(self, args.warmup_chunks)
            logger.info("warm-up chunk times (ms): %s", [round(t) for t in times])
        else:
            pass

    # ---- one request on every rank

    def broadcast(self, message):
        """Rank 0 sends the request (or None to stop) to the other ranks; they return it."""
        if self.control is None:
            return message
        else:
            box = [message]
            self.dist.broadcast_object_list(box, src=0, group=self.control)
            return box[0]

    def render(self, state: bytes, speech: np.ndarray):
        """Rank 0, from the HTTP server's threads: hand the chunk to the one thread that runs the model."""
        return self.gpu.submit(self.render_here, state, speech).result()

    def render_here(self, state: bytes, speech: np.ndarray):
        """Rank 0, on the model thread: decide the blocks from the ticket, run them on every rank, then encode the frames."""
        start = time.perf_counter()
        blocks = self.tickets.blocks_for(state)
        self.broadcast((blocks, speech))
        frames = self.step(blocks, speech)
        stepped = time.perf_counter()
        jpegs = list(self.encoder.map(self.encode, frames))
        logger.info(
            "chunk %d blocks %s: model %.1f ms, jpeg %.1f ms, %d KB",
            self.tickets.current.chunks - 1,
            blocks,
            (stepped - start) * 1000,
            (time.perf_counter() - stepped) * 1000,
            sum(map(len, jpegs)) // 1024,
        )
        if self.idle is None:
            self.idle = jpegs[0]
        else:
            pass
        return self.tickets.state(), jpegs

    def follow(self) -> None:
        """The other ranks: run every step rank 0 sends until it sends None."""
        while (message := self.broadcast(None)) is not None:
            self.gpu.submit(self.step, *message).result()

    def audio_embedding(self, speech: np.ndarray):
        """generate.py --steam_audio's window: sox tempo 25/fps, 16 kHz, x3, wav2vec2 hidden states."""
        torch = self.torch
        audio = torch.from_numpy(np.ascontiguousarray(speech, dtype=np.float32))[None]
        audio, rate = self.torchaudio.sox_effects.apply_effects_tensor(
            audio, SAMPLE_RATE_HZ, [["tempo", f"{25 / FPS}"]]
        )
        audio = self.Resample(rate, SAMPLE_RATE_HZ)(audio) * 3.0
        embedding = self.get_embedding(
            audio[0], self.feature_extractor, self.audio_encoder, device=self.device
        )
        return self.get_audio_emb(embedding, 0, WINDOW_FRAMES, self.device)

    def prime(self) -> None:
        """Every rank, once at start: run block 0 on silence and keep what it leaves behind for every Session's start.

        Block 0 is the reference block: its frames are dropped and its only
        audio is the 1.05 s before a Session's first chunk, which is silence
        (the agent had not spoken). What it leaves is the same for every
        Session: the KV cache's memory frames of every step and layer, the
        latent the next decode overlaps, and the random generator's state. A
        Session's first chunk restores them (a copy on the GPU) and runs
        block 1 only, instead of blocks 0 and 1.
        """
        torch = self.torch
        torch.manual_seed(self.seed)
        silence = np.zeros(self.window_samples(), dtype=np.float32)
        with torch.no_grad():
            self.run_block(0, self.audio_embedding(silence))
        memory = self.memory_tokens()
        self.primed = {
            "kv": {
                (step, layer): (
                    cache["k"][:, :memory].clone(),
                    cache["v"][:, :memory].clone(),
                )
                for step, layers in self.kv_cache.items()
                for layer, cache in layers.items()
            },
            "pre_latent": self.pre_latent.clone(),
            "vae_cache": [
                t.clone() if isinstance(t, torch.Tensor) else t
                for t in self.vae.model._feat_map
            ],
            "cuda_rng": torch.cuda.get_rng_state(),
            "cpu_rng": torch.get_rng_state(),
        }
        torch.cuda.synchronize()
        logger.info(
            "primed block 0: %.1f GB of KV cache kept on rank %d",
            sum(k.numel() + v.numel() for k, v in self.primed["kv"].values())
            * 2
            / 2**30,
            self.rank,
        )

    def memory_tokens(self) -> int:
        """How many of a rank's cache tokens block 0 writes: its six latent frames, split by sequence or kept whole (split by heads)."""
        frames = BLOCK_LATENTS[0] * self.frame_len
        return frames if self.sp_cache == "heads" else frames // self.world_size

    def restore_primed(self) -> None:
        torch = self.torch
        memory = self.memory_tokens()
        for (step, layer), (k, v) in self.primed["kv"].items():
            cache = self.kv_cache[step][layer]
            cache["k"][:, :memory].copy_(k)
            cache["v"][:, :memory].copy_(v)
        self.pre_latent = self.primed["pre_latent"].clone()
        if self.vae_stream:
            self.vae.model._feat_map = [
                t.clone() if isinstance(t, torch.Tensor) else t
                for t in self.primed["vae_cache"]
            ]
        else:
            pass
        torch.cuda.set_rng_state(self.primed["cuda_rng"])
        torch.set_rng_state(self.primed["cpu_rng"])

    def step(self, blocks: list[int], speech: np.ndarray):
        """Run the chunk's model blocks (rank 0 decided them from the ticket); returns its frames, uint8 [32, H, W, 3], on rank 0."""
        torch = self.torch
        if blocks[0] == 0 and self.primed is not None:
            self.restore_primed()
            blocks = blocks[1:]
        elif blocks[0] == 0:
            torch.manual_seed(self.seed)
        else:
            pass
        with torch.no_grad():
            audio = self.audio_embedding(
                model_window(blocks[-1], speech, self.half_blocks)
            )
            for block in blocks:
                video = self.run_block(block, audio)
        frames = None
        if self.rank == 0:
            frames = (
                ((video[0].float().permute(1, 2, 3, 0) + 1.0) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        else:
            pass
        return frames

    def run_block(self, block: int, audio):
        """generate.py's loop body for block `block` (its `_`): three denoising steps, then the VAE decode."""
        torch = self.torch
        frame_len = self.frame_len
        block, half = model_unit(block, self.half_blocks)
        first = min(block, 1)
        start = sum(BLOCK_LATENTS[:first])
        end = sum(BLOCK_LATENTS[: first + 1])
        update_cache = block > 1
        if half is not None:
            # half a block: its 4 latents attend to the memory, the block's latents before them and themselves
            start += half * BLOCK_LATENTS[1] // 2
            end = start + BLOCK_LATENTS[1] // 2
            update_cache = block > 1 and half == 0
        else:
            pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = torch.randn(
                16,
                end - start,
                self.height // VAE_STRIDE[1],
                self.width // VAE_STRIDE[2],
                dtype=torch.bfloat16,
                device=self.device,
            )
            for step in range(len(self.timesteps) - 1):
                timestep = self.timesteps[step]
                noise_pred = self.model(
                    [latent],
                    t=timestep,
                    kv_cache=self.kv_cache[step],
                    skip_audio=step not in self.audio_steps,
                    context=self.context,
                    clip_fea=self.clip_context,
                    ref_target_masks=self.ref_target_masks,
                    audio=audio,
                    y=self.y[:, :, start:end],
                    start_idx=start * frame_len,
                    end_idx=end * frame_len,
                    update_cache=update_cache,
                )[0]
                x0_pred = latent + (-noise_pred) * (timestep[0] / 1000 - 0.0)
                after = self.timesteps[step + 1][0] / 1000
                latent = (1 - after) * x0_pred + torch.randn_like(x0_pred) * after
            if self.vae_stream:
                if first == 0:
                    self.vae.model.clear_decode_cache()
                else:
                    pass
                video = self.vae.decode(latent)
            elif first == 0:
                video = self.vae.decode(latent.squeeze(0))
            else:
                video = self.vae.decode(
                    torch.concat(
                        [self.pre_latent[:, -OVERLAP_LATENTS:], latent], dim=1
                    ).squeeze(0)
                )[:, :, OVERLAP_FRAMES:]
        self.pre_latent = latent
        return video

    def encode(self, frame: np.ndarray) -> bytes:
        """One RGB frame as a JPEG; OpenCV releases the GIL, so the pool encodes a chunk's 32 frames at once (PIL's encoder holds it: about 150 ms a chunk)."""
        if self.cv2 is not None:
            ok, data = self.cv2.imencode(
                ".jpg",
                np.ascontiguousarray(frame[:, :, ::-1]),
                [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            assert ok
            return data.tobytes()
        else:
            out = io.BytesIO()
            self.Image.fromarray(frame).save(
                out, format="JPEG", quality=self.jpeg_quality
            )
            return out.getvalue()

    def idle_jpeg(self) -> bytes | None:
        return self.idle


def parse_args(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--wav2vec-dir", required=True)
    parser.add_argument("--cond-image", required=True)
    parser.add_argument(
        "--size",
        default="416*720",
        help="width*height as generate.py's --size: 416*720 (the authors' real-time setting) or 512*512.",
    )
    parser.add_argument("--prompt", default="一个人在说话")
    parser.add_argument(
        "--fp8-gemm",
        action="store_true",
        help="generate.py --fp8_gemm: FP8 GEMMs through vLLM's kernels.",
    )
    parser.add_argument(
        "--timesteps",
        type=lambda text: [float(step) for step in text.split(",")],
        default=list(TIMESTEPS),
        help="The denoising schedule, comma separated, ending in 0 (the distilled model's is 1000,937.5,833.33333333,0).",
    )
    parser.add_argument(
        "--audio-steps",
        type=lambda text: [int(step) for step in text.split(",")],
        default=[1, 2],
        help="The steps that attend to the speech (generate.py: every step but the first).",
    )
    parser.add_argument(
        "--sp-cache",
        choices=("seq", "heads"),
        default="seq",
        help="How the GPUs split the KV cache: by sequence (the authors' code, 2 GPUs only) or by heads (liveact_sp.py, any GPU count that divides 40).",
    )
    parser.add_argument(
        "--prime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run block 0 once at start and restore it at every Session's start (its first chunk then costs one block, not two).",
    )
    parser.add_argument(
        "--vae-stream",
        action="store_true",
        help="Keep the VAE decoder's causal cache across a Session's blocks: decode 8 latents a block instead of 11, no seam re-decode.",
    )
    parser.add_argument(
        "--vae-compile",
        action="store_true",
        help="torch.compile the VAE decoder (its norms and activations are most of its time uncompiled).",
    )
    parser.add_argument(
        "--half-blocks",
        action="store_true",
        help="0.8 s chunks: each block runs as two halves of 4 latents, the second attending to the first (needs --vae-stream, --sp-cache heads).",
    )
    parser.add_argument(
        "--attention",
        choices=("sage", "fa3"),
        default="sage",
        help="Self-attention kernel with --sp-cache heads: SageAttention FP8 (the authors') or FlashAttention 3 BF16 (as fast, exact, CUDA-graph safe).",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Replay the DiT's denoising steps as CUDA graphs (the steps are launch-bound, most of all on 4 GPUs).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args(argv)
    for name in ("repo", "ckpt_dir", "wav2vec_dir", "cond_image"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    return args


def init_distributed(rank: int, world_size: int) -> None:
    """torchrun's process group and xDiT's Ulysses sequence parallel over every rank, as generate.py sets them up."""
    import torch

    if world_size > 1:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        init_distributed_environment(rank=rank, world_size=world_size)
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=1,
            ulysses_degree=world_size,
        )
    else:
        pass


def main() -> None:
    args = parse_args()
    setup_logging()
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    init_distributed(rank, world_size)
    if rank == 0:
        serve(lambda: LiveActModel(args, rank, world_size), args.host, args.port)
    else:
        LiveActModel(args, rank, world_size).follow()
        time.sleep(1)


if __name__ == "__main__":
    main()
