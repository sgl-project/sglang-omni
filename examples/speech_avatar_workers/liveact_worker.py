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


def geometry_info(width: int, height: int) -> dict:
    return {
        "kind": "video",
        "model": "soulx-liveact",
        "fps": FPS,
        "chunk_frames": CHUNK_FRAMES,
        "context_samples": CONTEXT_FRAMES * FRAME_SAMPLES,
        "lookahead_samples": LOOKAHEAD_FRAMES * FRAME_SAMPLES,
        "width": width,
        "height": height,
    }


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
        self.jpeg_quality = args.jpeg_quality
        device = self.device = rank
        if world_size > 1:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel
        width, height = (int(side) for side in args.size.split("*"))
        self.width, self.height = width, height
        self.frame_len = (height // (PATCH_SIZE[1] * VAE_STRIDE[1])) * (
            width // (PATCH_SIZE[2] * VAE_STRIDE[2])
        )
        self.timesteps = [
            torch.tensor([step]).to(device, dtype=torch.float32) for step in TIMESTEPS
        ]
        kv_tokens = self.frame_len * sum(BLOCK_LATENTS) // world_size
        self.kv_cache = {
            step: {
                layer: {
                    "k": torch.zeros(
                        [1, kv_tokens, 40, 128], dtype=torch.bfloat16, device=device
                    ),
                    "v": torch.zeros(
                        [1, kv_tokens, 40, 128], dtype=torch.bfloat16, device=device
                    ),
                    "k_scale": None,
                    "v_scale": None,
                    "mean_memory": False,
                    "offload_cache": False,
                    "fp8_kv_cache": False,
                }
                for layer in range(40)
            }
            for step in range(len(TIMESTEPS) - 1)
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
        self.vae.model.eval()
        self.vae.encode = torch.compile(self.vae.encode)

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
        self.info = geometry_info(width, height)
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
        if rank == 0:
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

    def step(self, blocks: list[int], speech: np.ndarray):
        """Run the chunk's model blocks (rank 0 decided them from the ticket); returns its frames, uint8 [32, H, W, 3], on rank 0."""
        torch = self.torch
        if blocks[0] == 0:
            torch.manual_seed(self.seed)
        else:
            pass
        with torch.no_grad():
            audio = self.audio_embedding(speech)
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
        first = min(block, 1)
        start = sum(BLOCK_LATENTS[:first])
        end = sum(BLOCK_LATENTS[: first + 1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = torch.randn(
                16,
                BLOCK_LATENTS[first],
                self.height // VAE_STRIDE[1],
                self.width // VAE_STRIDE[2],
                dtype=torch.bfloat16,
                device=self.device,
            )
            for step in range(len(TIMESTEPS) - 1):
                timestep = self.timesteps[step]
                noise_pred = self.model(
                    [latent],
                    t=timestep,
                    kv_cache=self.kv_cache[step],
                    skip_audio=step not in (1, 2),
                    context=self.context,
                    clip_fea=self.clip_context,
                    ref_target_masks=self.ref_target_masks,
                    audio=audio,
                    y=self.y[:, :, start:end],
                    start_idx=start * frame_len,
                    end_idx=end * frame_len,
                    update_cache=block > 1,
                )[0]
                x0_pred = latent + (-noise_pred) * (timestep[0] / 1000 - 0.0)
                after = self.timesteps[step + 1][0] / 1000
                latent = (1 - after) * x0_pred + torch.randn_like(x0_pred) * after
            if first == 0:
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


def main() -> None:
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    for name in ("repo", "ckpt_dir", "wav2vec_dir", "cond_image"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    setup_logging()
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
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
    if rank == 0:
        serve(lambda: LiveActModel(args, rank, world_size), args.host, args.port)
    else:
        LiveActModel(args, rank, world_size).follow()
        time.sleep(1)


if __name__ == "__main__":
    main()
